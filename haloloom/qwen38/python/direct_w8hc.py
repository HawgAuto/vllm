# SPDX-License-Identifier: Apache-2.0
"""Dense W8A16 K5d-A (k5-gemm-v1/k5d-v1/K5D-PLAN.md, Amendment 1): INT8 HC layers take the accepted HC prefill fusions.

w8k5 INT8 HC layers fail the BF16 dtype guards of direct_nb1 (HCV3 dn5+ugmq for M in the HCV3 table, nb1 hc_ugm_xn for
M >= DIRECT_NB1_MIN_TOKENS) and run the reference chain (combine_norm -> W8 hc_mixed -> silu -> W8 up GEMM -> gate_mix).
Here they dequantize their HC weights to BF16 ON THE FLY, per call, into allocator temporaries (no persistent BF16 copy),
and feed the UNMODIFIED direct_nb1 kernels:
  * HCV3 range (direct_nb1 guards, weights checked as int8/fp32/bf16-inject): one Triton pass packs [down int8 x scale |
    inject/pad bf16] straight into the HCV3 [K/32][336][32] layout; up [K,320] via the sha-pinned K5 fdq; then the
    direct_nb1._hcv3_run launches verbatim (wdp passed in, no _WDP cache).
  * 256 <= M outside HCV3 with nb level >= 1: the same pass writes row-major merged [336,K]; _bf16_gemm (= the ctl merged
    ReplicatedLinear call, torch.ops.vllm.rocm_unquantized_gemm); split; hc_silu; direct_nb1._hc_up_gate_mix with the
    dequantized up (-> hc_ugm_xn).
  * anything else (M < 256: decode, CUDA-graph sizes 1..4; nb level 0; a guard failing): the EXACT current w8k5 call.
Dequant arithmetic is the K5c one (fp32(q) * s, RTNE downcast) and every layer's first dequantizing call is checked
BITWISE against the torch reference (fail closed), so the HC math equals the screen arm's BF16-weight HC math.
"""
import collections, json
import torch
import triton
import triton.language as tl
import direct_nb1 as _nb1
import direct_w8a16 as _w8
from vllm.utils.torch_utils import direct_register_custom_op

NP, R_UP = 336, 320
COUNTS = collections.Counter()
_CHECKED = set()
_reported = set()


def _emit(tag, **kw):
    if tag in _reported:
        return
    _reported.add(tag)
    print('DIRECT_W8HC ' + json.dumps(dict(tag=tag, **kw), sort_keys=True, default=str), flush=True)


@triton.jit
def _w8hc_pack_kernel(q_ptr, s_ptr, inj_ptr, o_ptr, K, N_DQ: tl.constexpr, NPAD: tl.constexpr, P32: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = offs_k[None, :] < K
    dq = offs_n[:, None] < N_DQ
    inn = offs_n[:, None] < NPAD
    q = tl.load(q_ptr + offs_n[:, None] * K + offs_k[None, :], mask=dq & kmask, other=0)
    s = tl.load(s_ptr + offs_n, mask=offs_n < N_DQ, other=0.0)
    wq = (q.to(tl.float32) * s[:, None]).to(tl.bfloat16, fp_downcast_rounding='rtne')
    wi = tl.load(inj_ptr + (offs_n[:, None] - N_DQ) * K + offs_k[None, :], mask=(offs_n[:, None] >= N_DQ) & inn & kmask, other=0.0)
    w = tl.where(dq, wq, wi)
    if P32:  # [K/32][NPAD][32]: the HCV3 dn5 weight layout (merged_w.view(NP, K//32, 32).permute(1, 0, 2))
        dst = (offs_k[None, :] // 32) * (NPAD * 32) + offs_n[:, None] * 32 + (offs_k[None, :] % 32)
    else:     # row-major [NPAD, K] == torch.cat([dequant(down), inject_weight])
        dst = offs_n[:, None] * K + offs_k[None, :]
    tl.store(o_ptr + dst, w, mask=inn & kmask)


def pack(qd, sd, injw, p32):
    """[down int8 [320,K] x fp32 scale [320] | inject bf16 [16,K]] -> bf16, P32 layout [K/32,336,32] or row-major [336,K]."""
    n_dq, K = qd.shape; npad = n_dq + injw.shape[0]
    out = torch.empty((K // 32, npad, 32) if p32 else (npad, K), dtype=torch.bfloat16, device=qd.device)
    BN, BK = 16, 256
    _w8hc_pack_kernel[(triton.cdiv(npad, BN), triton.cdiv(K, BK))](qd, sd, injw, out, K, N_DQ=n_dq, NPAD=npad, P32=bool(p32),
                                                                    BLOCK_N=BN, BLOCK_K=BK, num_warps=4)
    return out


def dq_up(qu, su, m):
    """Up [K,320] int8 -> bf16 row-major via the K5 dispatch's fdq choice at this M (bitwise == _w8._dequant)."""
    path, cfg = _w8._route(m, qu.shape[0], qu.shape[1]) if m >= 5 else ('dequant', None)
    if path == 'fdq':
        return _w8._k5()['dq'].dequant(qu, su, torch.bfloat16, False, cfg)
    return _w8._dequant(qu, su, torch.bfloat16, False)


def _wts_ok(qd, sd, injw, qu, su, K):
    return (qd.dtype == qu.dtype == torch.int8 and sd.dtype == su.dtype == torch.float32 and injw.dtype == torch.bfloat16
            and qd.is_contiguous() and qu.is_contiguous() and sd.is_contiguous() and su.is_contiguous() and injw.is_contiguous()
            and tuple(qd.shape) == (R_UP, K) and tuple(sd.shape) == (R_UP,) and tuple(injw.shape) == (NP - R_UP, K)
            and tuple(qu.shape) == (K, R_UP) and tuple(su.shape) == (K,) and K % 32 == 0)


def _selfcheck(layer_id, qd, sd, injw, qu, su, m):
    """First dequantizing call of each HC layer: both pack layouts and the up dequant BITWISE == torch reference."""
    if layer_id in _CHECKED:
        return
    K = qd.shape[1]
    ref = torch.cat([_w8._dequant(qd, sd, torch.bfloat16, False), injw], dim=0)
    a = pack(qd, sd, injw, False); b = pack(qd, sd, injw, True)
    refp = ref.view(NP, K // 32, 32).permute(1, 0, 2).contiguous()
    u = dq_up(qu, su, m); ur = _w8._dequant(qu, su, torch.bfloat16, False)
    ok = (a.shape == ref.shape and bool(torch.equal(a, ref)) and b.shape == refp.shape and bool(torch.equal(b, refp))
          and u.shape == ur.shape and bool(torch.equal(u, ur)))
    if not ok:
        raise RuntimeError(f'DIRECT_W8HC selfcheck not bitwise layer_id={layer_id} m={m}')
    _CHECKED.add(layer_id)
    COUNTS['selfcheck'] += 1
    if len(_CHECKED) in (1, 24, 48, 72, 96):
        _emit(f'selfcheck_{len(_CHECKED)}', layers=len(_CHECKED), m=m)
    _emit(f'selfcheck_l{layer_id}', layer_id=layer_id, layers=len(_CHECKED), m=m)  # r2 (K5D-PLAN.md Amendment 3): exact per-layer coverage receipt


def _hcv3_ok_w8(hidden, blk, inj, norm_w, qd, sd, injw, qu, su, hc):
    """direct_nb1._hcv3_ok with the weight checks done on the int8/fp32/bf16-inject tensors."""
    M, K = hidden.shape
    if not (_nb1.HCV3_MODEL and M >= _nb1.MIN_TOKENS and M <= _nb1.HCV3_MAX_M and _nb1.hcv3_pick(M) is not None):
        return False
    H = K // hc
    return (hidden.dtype == blk.dtype == inj.dtype == norm_w.dtype == torch.bfloat16 and _wts_ok(qd, sd, injw, qu, su, K)
            and hidden.stride(1) == 1 and hidden.stride(0) == K and blk.stride(1) == 1 and inj.stride(1) == 1
            and norm_w.is_contiguous() and norm_w.numel() in (H, K) and R_UP + hc <= NP
            and H % 128 == 0 and K % 256 == 0 and tuple(blk.shape) == (M, H) and tuple(inj.shape) == (M, hc)
            and _nb1.nb_level() in (2, 3, 4, 5, 6, 7, 8, 9))


def _nb1_branch_ok(m, K, qd, sd, injw, qu, su):
    return _nb1.ENABLE_HC and m >= _nb1.MIN_TOKENS and _nb1.nb_on() and _wts_ok(qd, sd, injw, qu, su, K)


def _hcv3_run_w8(hidden, blk, inj, norm_w, wdp, upb, eps, hc, layer_id):
    """direct_nb1._hcv3_run verbatim except that the packed down/inject weight wdp is supplied (no _WDP cache)."""
    import ctypes
    import direct_hip_module_v1 as hm
    M, K = hidden.shape; H = K // hc; R = upb.shape[1]
    out, xb = _nb1._hv().combine_norm_pack(hidden, blk, inj, norm_w, eps, hc, ly=1)
    dev = hidden.device
    if dev not in _nb1._DUMMY:
        _nb1._DUMMY[dev] = (torch.empty(_nb1.HCV3_MAX_KS * _nb1.HCV3_MAX_M * 336, dtype=torch.float32, device=dev),
                            torch.zeros(4096, dtype=torch.int32, device=dev))
    part, cnt = _nb1._DUMMY[dev]
    RG, KS = _nb1.hcv3_pick(M)
    lora = torch.empty((M, R), dtype=torch.bfloat16, device=dev)
    injb = torch.empty((M, NP), dtype=torch.bfloat16, device=dev)
    lvl = _nb1.nb_level(); hs = _nb1.V3G if lvl >= 3 else _nb1.V3E
    m = _nb1._mod(hs)
    m.launch(f'hc_dn5_r{RG}_k{KS}_l1', ((M + 16 * RG - 1) // (16 * RG), KS), (96 * RG,), [hm.ptr(xb), hm.ptr(wdp), hm.ptr(lora), hm.ptr(injb), hm.ptr(part),
             hm.ptr(cnt), ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(R), ctypes.c_int(hc), ctypes.c_longlong(0),
             ctypes.c_longlong(R), ctypes.c_longlong(NP)])
    y = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    m.launch('hc_ugmq_c8_l1', (H // 128, (M + 127) // 128), (512,), [hm.ptr(lora), hm.ptr(upb), hm.ptr(xb), hm.ptr(y),
             ctypes.c_int(M), ctypes.c_int(H), ctypes.c_int(hc), ctypes.c_int(R), ctypes.c_longlong(R), ctypes.c_longlong(0),
             ctypes.c_longlong(H)])
    _nb1.COUNTS[('hcv3', lvl, M, RG, KS)] += 1
    _nb1._audit_hcv3_inheritance(lvl, M, hs)
    COUNTS[('w8hcv3', lvl, M, RG, KS)] += 1
    _emit('first_w8hcv3', m=M, h=H, r=R, hc=hc, np=NP, rg=RG, ks=KS, hsaco=hs, nb=lvl)
    return out, y, injb[:, :hc]


def _w8_combine_and_mix(hidden: torch.Tensor, blk: torch.Tensor, inj: torch.Tensor, norm_w: torch.Tensor,
                        carrier: torch.Tensor, qd: torch.Tensor, sd: torch.Tensor, injw: torch.Tensor, qu: torch.Tensor,
                        su: torch.Tensor, eps: float, hc_count: int, layer_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K = hidden.shape
    if _hcv3_ok_w8(hidden, blk, inj, norm_w, qd, sd, injw, qu, su, hc_count):
        _selfcheck(layer_id, qd, sd, injw, qu, su, M)
        wdp = pack(qd, sd, injw, True); upb = dq_up(qu, su, M)
        return _hcv3_run_w8(hidden, blk, inj, norm_w, wdp, upb, eps, hc_count, layer_id)
    if _nb1_branch_ok(M, K, qd, sd, injw, qu, su):
        from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm, hc_silu
        _selfcheck(layer_id, qd, sd, injw, qu, su, M)
        out, xn = hc_combine_norm(hidden, blk, inj, norm_w, eps, hc_count)
        d = _w8._bf16_gemm(xn, pack(qd, sd, injw, False))
        lora, injection, _ = d.split([R_UP, hc_count, NP - R_UP - hc_count], dim=-1)
        lora = hc_silu(lora, hc_count)
        y = _nb1._hc_up_gate_mix(xn, lora, dq_up(qu, su, M), hc_count, layer_id)
        COUNTS[('w8nb1', M)] += 1
        _emit('first_w8nb1', m=M, k=K, hc=hc_count, nb=_nb1.nb_level())
        return out, y, injection
    COUNTS[('w8ref', M)] += 1
    return _nb1._hc_combine_and_mix(hidden, blk, inj, norm_w, carrier, qu, eps, hc_count, layer_id)


def _w8_combine_and_mix_fake(hidden, blk, inj, norm_w, carrier, qd, sd, injw, qu, su, eps, hc_count, layer_id):
    M, K = hidden.shape
    return (hidden.new_empty((M, K)), hidden.new_empty((M, K // hc_count)),
            torch.empty_strided((M, hc_count), (carrier.shape[0], 1), dtype=hidden.dtype, device=hidden.device))


def _w8_up_gate_mix(xn: torch.Tensor, lora: torch.Tensor, qu: torch.Tensor, su: torch.Tensor, hc_count: int,
                    layer_id: int) -> torch.Tensor:
    M = xn.shape[0]
    if (_nb1.ENABLE_HC and M >= _nb1.MIN_TOKENS and _nb1.nb_on() and qu.dtype == torch.int8 and qu.is_contiguous()
            and su.dtype == torch.float32 and tuple(su.shape) == (qu.shape[0],)):
        COUNTS[('w8ugm', M)] += 1
        _emit('first_w8ugm', m=M)
        _emit(f'ugm_l{layer_id}', layer_id=layer_id, m=M)  # r2: INT8 up-gate-mix layers (mix() path) exact receipt
        return _nb1._hc_up_gate_mix(xn, lora, dq_up(qu, su, M), hc_count, layer_id)
    return _nb1._hc_up_gate_mix(xn, lora, qu, hc_count, layer_id)


def _w8_up_gate_mix_fake(xn, lora, qu, su, hc_count, layer_id):
    return xn.new_empty((xn.shape[0], xn.shape[1] // hc_count))


direct_register_custom_op(op_name='direct_w8hc_combine_and_mix', op_func=_w8_combine_and_mix, fake_impl=_w8_combine_and_mix_fake)
direct_register_custom_op(op_name='direct_w8hc_up_gate_mix', op_func=_w8_up_gate_mix, fake_impl=_w8_up_gate_mix_fake)


def is_int8(layer) -> bool:
    w = getattr(layer, 'weight', None)
    return w is not None and w.dtype == torch.int8 and getattr(layer, 'weight_scale', None) is not None


def hc_combine_and_mix(hidden, blk, inj, norm_w, dn, up, eps, hc_count, layer_id):
    return torch.ops.vllm.direct_w8hc_combine_and_mix(hidden, blk, inj, norm_w, dn.w8_ops_merged_w, dn.weight, dn.weight_scale,
                                                      dn.inject_weight, up.weight, up.weight_scale, eps, hc_count, layer_id)


def hc_up_gate_mix(xn, lora, up, hc_count, layer_id):
    return torch.ops.vllm.direct_w8hc_up_gate_mix(xn, lora, up.weight, up.weight_scale, hc_count, layer_id)


def state():
    return {'checked_layers': len(_CHECKED), 'counts': {str(k): v for k, v in COUNTS.items()}}
