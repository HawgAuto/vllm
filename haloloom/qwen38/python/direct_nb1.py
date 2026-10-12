# SPDX-License-Identifier: Apache-2.0
"""nb1: NON-BITWISE prefill arm (user-approved 2026-10-02, quality-gated, not exact).

Gate: a separate phase file (DIRECT_NB_PHASE, schema {'nb': 0|1, 'case': str}) so every existing v11 'wmma5' gate stays on
in both arms. Only prefill-sized calls (M >= DIRECT_NB1_MIN_TOKENS, default 256) can take the nb1 path, so decode CUDA-graph
captures (M <= 4) always record the original kernels.

  MoE  : x -> fp16 (saturating) -> moe13_up (fp16 WMMA, magic-number dequant) -> silu_mul_bf16_f16 -> moe13_down (fp16 act)
  HC   : gate = input_mix_weight_up(lora); y = hc_gate_mix(xn, gate)   ==>   y = hc_ugm_xn(lora, W_up, xn) (gate never stored)
Reference path of the HC op calls the original module + original gate_mix op, unchanged.
"""
import collections, ctypes, json, os
from pathlib import Path
import torch
import direct_hip_module_v1 as hm
from vllm.utils.torch_utils import direct_register_custom_op

SP = '/opt/venv/lib/python3.14/site-packages/'
AUX = os.environ.get('DIRECT_NB1_AUX_HSACO', SP + 'direct_nb1_aux.hsaco')
V13 = os.environ.get('DIRECT_NB1_MOE_HSACO', SP + 'direct_moe_wna16_wmma_v13.hsaco')
PHASE = Path(os.environ.get('DIRECT_NB_PHASE', '/workspace/nb-phase.json'))
MIN_TOKENS = int(os.environ.get('DIRECT_NB1_MIN_TOKENS', '256'))
ENABLE_MOE = os.environ.get('DIRECT_NB1_MOE', '1') == '1'
ENABLE_HC = os.environ.get('DIRECT_NB1_HC', '1') == '1'
HCV3_MODEL = os.environ.get('DIRECT_HCV3', '1') == '1'   # route combine_and_mix through the dual-arm custom op
V3E = os.environ.get('DIRECT_HCV3_HSACO', SP + 'direct_hc_fused_v3e.hsaco')
V3G = os.environ.get('DIRECT_HCV3G_HSACO', SP + 'direct_hc_fused_v3g.hsaco')  # nb 3: v3e + conflict-free dn5 LDS swizzle (bitwise == v3e)
COUNTS = collections.Counter()
_mods = {}
_phase_cache = [None, None]  # (stat key, nb level 0/1/2)
_reported = set()
_HC_LAYERS = []


def _report(tag, **kw):
    if tag in _reported:
        return
    _reported.add(tag)
    print('DIRECT_NB1 ' + json.dumps(dict(tag=tag, **kw), sort_keys=True, default=str), flush=True)


def nb_level() -> int:
    # nb2-hcv3: 0 = incumbent reference, 1 = nb1 (accepted), 2 = nb1 + HC fused v3 (combine_and_mix), 3 = level 2 with the v3g hsaco,
    # 4 = level 3 + GDN post_conv v-view (overlay nb2-gdnvv-ab-v1; HC behaviour identical to level 3),
    # 5 = level 4 + QSA union capacity bound (overlay nb2-qsacap-ab-v1)
    # 6 = level 5 + native bitwise QSA indexer scores qsa_mqa_scores_v4 (overlay nb2-mqa4-ab-v1)
    # 7 = level 6 + exact native QSA union build qsa_union_build_v1 (overlay nb2-union1-ab-v1)
    # 8 = level 7 + canonical QSA selection v2: kth/nsel once per row (overlay nb2-canon2-ab-v1)
    # 9 = level 8 + safe bitwise QSA indexer I1; experimental isolated overlay only
    try:
        st = PHASE.stat()
    except FileNotFoundError:
        return 0
    key = (st.st_mtime_ns, st.st_size, st.st_ino)
    if _phase_cache[0] != key:
        data = json.loads(PHASE.read_text())
        if not isinstance(data, dict) or set(data) != {'nb', 'case'} or data['nb'] not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            raise ValueError('invalid nb phase schema')
        _phase_cache[0], _phase_cache[1] = key, int(data['nb'])
    return _phase_cache[1]


def nb_on() -> bool:
    return nb_level() >= 1


def _mod(path):
    m = _mods.get(path)
    if m is None:
        m = _mods[path] = hm.Module(path)
    return m


# ------------------------------------------------------------------------------------------------------------- MoE
def moe_active(m: int) -> bool:
    return ENABLE_MOE and m >= MIN_TOKENS and nb_on()


def moe_run(W, x, sid, eid, ntpp, tab, nmb, m, t):
    """Same routing/alignment tensors as the incumbent unified route; returns per-slot unweighted bf16 [S, 2560]."""
    v13, aux = _mod(V13), _mod(AUX)
    c_ = ctypes.c_int; ll = ctypes.c_longlong
    S = m * t
    x16 = torch.empty((m, x.shape[1]), dtype=torch.float16, device=x.device)
    if x.is_contiguous():
        n = x.numel()
        v13.launch('cvt_bf16_f16_sat', ((n // 8 + 255) // 256,), (256,), [hm.ptr(x), hm.ptr(x16), ll(n), ctypes.c_void_p(0)])
    else:
        x16.copy_(x.clamp(-65504.0, 65504.0))

    def launch(name, A, C, w, s, top_k):
        B4, S4, B8, S8 = W[w + '_4'], W[s + '_4'], W[w + '_8'], W[s + '_8']
        N = B4.size(1); K = A.size(1)
        v13.launch(name, (N // 128, nmb), (256,), [hm.ptr(A), hm.ptr(B4), hm.ptr(S4), hm.ptr(B8), hm.ptr(S8), hm.ptr(C), hm.ptr(sid), hm.ptr(eid),
                   hm.ptr(ntpp), hm.ptr(tab), c_(N), c_(K), c_(S), c_(top_k), ll(A.stride(0)),
                   ll(B4.stride(0)), ll(B4.stride(2)), ll(B4.stride(1)), ll(S4.stride(0)), ll(S4.stride(2)), ll(S4.stride(1)),
                   ll(B8.stride(0)), ll(B8.stride(2)), ll(B8.stride(1)), ll(S8.stride(0)), ll(S8.stride(2)), ll(S8.stride(1)), ll(C.stride(-2))])

    up = torch.empty((S, 1280), dtype=torch.bfloat16, device=x.device)
    launch('moe13_up', x16, up, 'w13', 's13', t)
    act = torch.empty((S, 640), dtype=torch.float16, device=x.device)
    aux.launch('silu_mul_bf16_f16', ((S * 640 // 8 + 255) // 256,), (256,), [hm.ptr(up), hm.ptr(act), c_(S), c_(640), ll(up.stride(0)),
               ll(act.stride(0)), ctypes.c_void_p(0)])
    out = torch.empty((S, 2560), dtype=torch.bfloat16, device=x.device)
    launch('moe13_down', act, out, 'w2', 's2', 1)
    COUNTS[('moe', m, t)] += 1
    _report('first_moe_nb1', m=m, t=t, hsaco=V13, aux=AUX)
    return out


# -------------------------------------------------------------------------------------------------------------- HC
def register_hc(module) -> int:
    _HC_LAYERS.append(module)
    return len(_HC_LAYERS) - 1


def _hc_ok(xn, lora, w, hc):
    M = xn.shape[0]
    if not (ENABLE_HC and M >= MIN_TOKENS):
        return False
    H = xn.shape[1] // hc; R = lora.shape[1]
    return (xn.dtype == lora.dtype == w.dtype == torch.bfloat16 and xn.stride(1) == 1 and lora.stride(1) == 1 and w.is_contiguous()
            and tuple(w.shape) == (hc * H, R) and H % 128 == 0 and R % 32 == 0 and xn.stride(0) % 8 == 0 and lora.stride(0) % 8 == 0
            and lora.shape[0] == M and nb_on())


def _hc_up_gate_mix(xn: torch.Tensor, lora: torch.Tensor, weight: torch.Tensor, hc_count: int, layer_id: int) -> torch.Tensor:
    if _hc_ok(xn, lora, weight, hc_count):
        M = xn.shape[0]; H = xn.shape[1] // hc_count; R = lora.shape[1]
        y = torch.empty((M, H), dtype=torch.bfloat16, device=xn.device)
        _mod(AUX).launch('hc_ugm_xn', (H // 128, (M + 63) // 64), (256,), [hm.ptr(lora), hm.ptr(weight), hm.ptr(xn), hm.ptr(y),
                         ctypes.c_int(M), ctypes.c_int(H), ctypes.c_int(hc_count), ctypes.c_int(R), ctypes.c_longlong(lora.stride(0)),
                         ctypes.c_longlong(xn.stride(0)), ctypes.c_longlong(y.stride(0))])
        COUNTS[('hc', M)] += 1
        _report('first_hc_nb1', m=M, h=H, r=R, hc=hc_count)
        return y
    gate = _HC_LAYERS[layer_id].input_mix_weight_up(lora)
    return torch.ops.vllm.qwen4_exp_hc_gate_mix(xn, gate, hc_count)


def _hc_up_gate_mix_fake(xn, lora, weight, hc_count, layer_id):
    return xn.new_empty((xn.shape[0], xn.shape[1] // hc_count))


direct_register_custom_op(op_name='qwen4_exp_nb1_hc_up_gate_mix', op_func=_hc_up_gate_mix, fake_impl=_hc_up_gate_mix_fake)


def hc_up_gate_mix(xn, lora, weight, hc_count, layer_id):
    return torch.ops.vllm.qwen4_exp_nb1_hc_up_gate_mix(xn, lora, weight, hc_count, layer_id)



# --------------------------------------------------------------------------------------- nb2: HC fused v3 (packed xn)
# One opaque op for GatedResidual.combine_and_mix.  nb level < 2 (or any guard failing) runs EXACTLY the accepted nb1
# sequence (hc_combine_norm -> merged down/inject ReplicatedLinear -> split -> hc_silu -> nb1 hc_up_gate_mix).  nb == 2
# runs the qualified v3 pipeline tri_b256: Triton combine-norm with B256-packed xn store (direct_hc_v3.py, identical
# arithmetic to production -> out/xn bit-identical), hc_dn5_r8_k1_l1 (packed down GEMM + bf16/silu/inject epilogue),
# hc_ugmq_c8_l1 (K2 up+gate-mix reading packed xn).  Injection is returned with the merged-GEMM row stride in BOTH arms.
_WDP = {}
_HV = [None]
_DUMMY = {}
# M-adaptive dispatch (operator sweep hc-v3e-sweep-01, site 5.mlp, every combo gate-PASS at every swept M):
# (M_lo, M_hi_exclusive, down (RG, KS)); K2 = hc_ugmq_c8_l1 everywhere; M outside the table -> exact nb1 sequence.
HCV3_TABLE = ((1200, 1800, (4, 4)), (1800, 2900, (8, 2)), (2900, 3850, (4, 4)), (3850, 5200, (8, 1)),
              (5200, 7168, (4, 2)), (7168, 12289, (8, 2)))
HCV3_MAX_M, HCV3_MAX_KS = 12288, 4


def hcv3_pick(m: int):
    for lo, hi, cfg in HCV3_TABLE:
        if lo <= m < hi:
            return cfg
    return None


def _hv():
    if _HV[0] is None:
        import direct_hc_v3
        _HV[0] = direct_hc_v3
    return _HV[0]


def _hcv3_ok(hidden, blk, inj, norm_w, merged_w, up_w, hc):
    M, K = hidden.shape
    if not (HCV3_MODEL and M >= MIN_TOKENS and M <= HCV3_MAX_M and hcv3_pick(M) is not None):
        return False
    H = K // hc; R = up_w.shape[1] if up_w.dim() == 2 else -1
    return (hidden.dtype == blk.dtype == inj.dtype == norm_w.dtype == merged_w.dtype == up_w.dtype == torch.bfloat16
            and hidden.stride(1) == 1 and hidden.stride(0) == K and blk.stride(1) == 1 and inj.stride(1) == 1
            and norm_w.is_contiguous() and norm_w.numel() in (H, K) and merged_w.is_contiguous() and up_w.is_contiguous()
            and tuple(merged_w.shape) == (336, K) and tuple(up_w.shape) == (K, R) and R == 320 and R + hc <= 336
            and H % 128 == 0 and K % 256 == 0 and tuple(blk.shape) == (M, H) and tuple(inj.shape) == (M, hc)
            and nb_level() in (2, 3, 4, 5, 6, 7, 8, 9))


def _hcv3_run(hidden, blk, inj, norm_w, merged_w, up_w, eps, hc, layer_id):
    M, K = hidden.shape; H = K // hc; R = up_w.shape[1]; NP = merged_w.shape[0]
    out, xb = _hv().combine_norm_pack(hidden, blk, inj, norm_w, eps, hc, ly=1)
    wdp = _WDP.get(layer_id)
    if wdp is None or wdp.data_ptr() == 0:
        wdp = _WDP[layer_id] = merged_w.view(NP, K // 32, 32).permute(1, 0, 2).contiguous()
    dev = hidden.device
    if dev not in _DUMMY:  # persistent split-K partials (fp32, KSmax x Mmax x NP) + per-row-tile counters (self-resetting)
        _DUMMY[dev] = (torch.empty(HCV3_MAX_KS * HCV3_MAX_M * 336, dtype=torch.float32, device=dev),
                       torch.zeros(4096, dtype=torch.int32, device=dev))
    part, cnt = _DUMMY[dev]
    RG, KS = hcv3_pick(M)
    lora = torch.empty((M, R), dtype=torch.bfloat16, device=dev)
    injb = torch.empty((M, NP), dtype=torch.bfloat16, device=dev)
    lvl = nb_level(); hs = V3G if lvl >= 3 else V3E
    m = _mod(hs)
    m.launch(f'hc_dn5_r{RG}_k{KS}_l1', ((M + 16 * RG - 1) // (16 * RG), KS), (96 * RG,), [hm.ptr(xb), hm.ptr(wdp), hm.ptr(lora), hm.ptr(injb), hm.ptr(part),
             hm.ptr(cnt), ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(R), ctypes.c_int(hc), ctypes.c_longlong(0),
             ctypes.c_longlong(R), ctypes.c_longlong(NP)])
    y = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    m.launch('hc_ugmq_c8_l1', (H // 128, (M + 127) // 128), (512,), [hm.ptr(lora), hm.ptr(up_w), hm.ptr(xb), hm.ptr(y),
             ctypes.c_int(M), ctypes.c_int(H), ctypes.c_int(hc), ctypes.c_int(R), ctypes.c_longlong(R), ctypes.c_longlong(0),
             ctypes.c_longlong(H)])
    COUNTS[('hcv3', lvl, M, RG, KS)] += 1
    _audit_hcv3_inheritance(lvl,M,hs)
    _report('first_hcv3g' if lvl >= 3 else 'first_hcv3', m=M, h=H, r=R, hc=hc, np=NP, rg=RG, ks=KS, hsaco=hs)
    return out, y, injb[:, :hc]


def _hc_combine_and_mix(hidden: torch.Tensor, blk: torch.Tensor, inj: torch.Tensor, norm_w: torch.Tensor,
                        merged_w: torch.Tensor, up_w: torch.Tensor, eps: float, hc_count: int,
                        layer_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _hcv3_ok(hidden, blk, inj, norm_w, merged_w, up_w, hc_count):
        return _hcv3_run(hidden, blk, inj, norm_w, merged_w, up_w, eps, hc_count, layer_id)
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm, hc_silu
    mod = _HC_LAYERS[layer_id]
    out, xn = hc_combine_norm(hidden, blk, inj, norm_w, eps, hc_count)
    d = mod.input_mix_weight_down_block_inject(xn)
    lora, injection, _ = d.split([mod.lora_rank, mod.hc_count, mod.pad_size], dim=-1)
    lora = hc_silu(lora, hc_count)
    y = _hc_up_gate_mix(xn, lora, up_w, hc_count, layer_id)
    return out, y, injection


def _hc_combine_and_mix_fake(hidden, blk, inj, norm_w, merged_w, up_w, eps, hc_count, layer_id):
    M, K = hidden.shape
    return (hidden.new_empty((M, K)), hidden.new_empty((M, K // hc_count)),
            torch.empty_strided((M, hc_count), (merged_w.shape[0], 1), dtype=hidden.dtype, device=hidden.device))


direct_register_custom_op(op_name='qwen4_exp_nb2_hc_combine_and_mix', op_func=_hc_combine_and_mix,
                          fake_impl=_hc_combine_and_mix_fake)


def hc_combine_and_mix(hidden, blk, inj, norm_w, merged_w, up_w, eps, hc_count, layer_id):
    return torch.ops.vllm.qwen4_exp_nb2_hc_combine_and_mix(hidden, blk, inj, norm_w, merged_w, up_w, eps, hc_count, layer_id)

# Exact inherited route audit: once per phase/shape, after the actual HCV3 launches.
_hc_inheritance_seen=set()
_hc_inheritance_case=[None,None]
def _audit_hcv3_inheritance(level,rows,hsaco):
    if _hc_inheritance_case[0] != _phase_cache[0]:
        _hc_inheritance_case[:]=[_phase_cache[0],json.loads(PHASE.read_text())['case']]
    case=_hc_inheritance_case[1];key=(case,level,rows)
    if key not in _hc_inheritance_seen:
        _hc_inheritance_seen.add(key)
        print('DIRECT_HCV3_INHERITANCE '+json.dumps(dict(case=case,nb=level,rows=rows,hsaco=hsaco),sort_keys=True),flush=True)
