# SPDX-License-Identifier: Apache-2.0
"""[nb1 overlay copy: + direct_nb1 branch] Unified-pool native gfx1151 WMMA MoE prefill route (v2: kernel family selectable; default moe_wna16_wmma_v6.hsaco, unfused SiLU).

Replaces, for prefill-sized direct-input batches under benchmark arm 'wmma5', the two-pool sequence
  per pool: moe_align(expert_map) -> WNA16 gate_up -> _C.silu_and_mul -> WNA16 down  (other pool's slots = exact zeros)
with ONE global-expert alignment, ONE fused gate_up+SiLU launch and ONE down launch over all slots
(per-workgroup W4/W8 dispatch via tab[e] = (pool << 16) | local). Returns per-slot UNWEIGHTED bf16 outputs; the
caller reduces with reduce_slots(slots, None, ...) which is bit-identical to reduce_slots(a, b) when one of a/b is +0.
Kernel-level bitwise gate: attempts/moe-wmma-v5-gate-01 (MOE_V5_UNIFIED_BITWISE_PASS, all SILUV variants exact).
Any layout/dtype mismatch returns None and the caller keeps the original path.
"""
import collections, ctypes, json, os
import torch
import direct_hip_module_v1 as hm
import direct_moe_ab_adapter_v1 as ab

SP = '/opt/venv/lib/python3.14/site-packages/'
HSACO = os.environ.get('DIRECT_MOE_UNIFIED_HSACO', SP + 'direct_moe_wna16_wmma_v6.hsaco')
PREFIX = os.environ.get('DIRECT_MOE_UNIFIED_PREFIX', 'moe6')
MIN_TOKENS = int(os.environ.get('DIRECT_MOE_UNIFIED_MIN_TOKENS', '256'))
SILUV = int(os.environ.get('DIRECT_MOE_UNIFIED_SILUV', '-1'))  # -1: unfused gate_up + torch.ops._C.silu_and_mul
ARM = 'wmma5'
COUNTS = collections.Counter()
_mod = None
_finite = set()
_reported = set()


def active(m, t):
    if m < MIN_TOKENS:
        return False
    ph = ab.read_phase()
    return ph is not None and ph['arm'] == ARM


def _module():
    global _mod
    if _mod is None:
        _mod = hm.Module(HSACO)
    return _mod


def _report(tag, **kw):
    if tag in _reported:
        return
    _reported.add(tag)
    print('DIRECT_MOE_UNIFIED ' + json.dumps(dict(tag=tag, **kw), sort_keys=True, default=str), flush=True)


def _tab(layer):
    tab = getattr(layer, '_direct_unified_tab', None)
    if tab is None:
        l4, l8 = layer.mixed_lookup_4, layer.mixed_lookup_8
        own4, own8 = l4 >= 0, l8 >= 0
        if l4.shape != l8.shape or not bool((own4 ^ own8).all()):  # one-time sync per layer at first use
            raise ValueError('every global expert must be owned by exactly one precision pool')
        tab = torch.where(own4, l4.to(torch.int32), l8.to(torch.int32) | (1 << 16)).to(torch.int32).contiguous()
        layer._direct_unified_tab = tab
    return tab


def _finite_once(t):
    k = (t.data_ptr(), tuple(t.shape))
    if k not in _finite:
        if not bool(torch.isfinite(t).all()):
            raise ValueError('non-finite MoE scale (unified dequant omits NaN propagation)')
        if not bool((t >= 0).all()):
            raise ValueError('negative MoE scale (v6 FMA dequant identity requires s >= 0)')
        _finite.add(k)


def run(layer, x, gids, natives):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    p4, p8 = layer.mixed_pools['4'], layer.mixed_pools['8']
    n4, n8 = natives.get(4), natives.get(8)
    if n4 is None or n8 is None:
        _report('fallback_missing_pool'); return None
    W = dict(w13_4=p4.w13_weight, w2_4=p4.w2_weight, w13_8=p8.w13_weight, w2_8=p8.w2_weight,
             s13_4=n4.w1_scale, s2_4=n4.w2_scale, s13_8=n8.w1_scale, s2_8=n8.w2_scale)
    m, t = gids.shape
    ok = (x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.shape[1] == 2560
          and all(v is not None for v in W.values())
          and W['w13_4'].dtype == torch.uint8 and W['w13_8'].dtype == torch.uint8
          and tuple(W['w13_4'].shape[1:]) == (1280, 1280) and tuple(W['w2_4'].shape[1:]) == (2560, 320)
          and tuple(W['w13_8'].shape[1:]) == (1280, 2560) and tuple(W['w2_8'].shape[1:]) == (2560, 640)
          and W['w13_4'].stride(1) == 1 and W['w2_4'].stride(1) == 1           # W4 [E][K/2][N]
          and W['w13_8'].stride(1) == 1 and W['w2_8'].stride(2) == 1           # W8 gate_up [E][K][N], down [E][N][K]
          and W['s13_4'].stride(1) == 1 and W['s2_4'].stride(1) == 1
          and all(W[k].dtype == torch.float32 for k in ('s13_4', 's2_4', 's13_8', 's2_8')))
    if not ok:
        _report('fallback_layout', shapes={k: (tuple(v.shape), tuple(v.stride()), str(v.dtype)) for k, v in W.items() if v is not None})
        return None
    for k in ('s13_4', 's2_4', 's13_8', 's2_8'):
        _finite_once(W[k])
    tab = _tab(layer)
    S = m * t
    sid, eid, ntpp = moe_align_block_size(gids, 64, tab.numel(), None)[:3]
    mod = _module(); c_ = ctypes.c_int; ll = ctypes.c_longlong
    nmb = (sid.numel() + 63) // 64
    import direct_nb1 as nb  # nb1 NON-BITWISE arm (separate phase file); off => incumbent path below, unchanged
    if nb.moe_active(m):
        out = nb.moe_run(W, x, sid, eid, ntpp, tab, nmb, m, t)
        COUNTS[(m, t)] += 1
        return out

    def launch(name, A, C, w, s, top_k, fuse):
        B4, S4, B8, S8 = W[w + '_4'], W[s + '_4'], W[w + '_8'], W[s + '_8']
        N = B4.size(1); K = A.size(1)
        grid = ((N // 2) // 64 if fuse else N // 128, nmb)
        mod.launch(name, grid, (256,), [hm.ptr(A), hm.ptr(B4), hm.ptr(S4), hm.ptr(B8), hm.ptr(S8), hm.ptr(C), hm.ptr(sid), hm.ptr(eid), hm.ptr(ntpp), hm.ptr(tab),
                   c_(N), c_(K), c_(S), c_(top_k), ll(A.stride(0)),
                   ll(B4.stride(0)), ll(B4.stride(2)), ll(B4.stride(1)), ll(S4.stride(0)), ll(S4.stride(2)), ll(S4.stride(1)),
                   ll(B8.stride(0)), ll(B8.stride(2)), ll(B8.stride(1)), ll(S8.stride(0)), ll(S8.stride(2)), ll(S8.stride(1)), ll(C.stride(-2))])

    act = torch.empty((S, 640), dtype=torch.bfloat16, device=x.device)
    if SILUV >= 0:
        launch(f'{PREFIX}_up_silu{SILUV}', x, act, 'w13', 's13', t, True)
    else:
        up = torch.empty((S, 1280), dtype=torch.bfloat16, device=x.device)
        launch(f'{PREFIX}_up', x, up, 'w13', 's13', t, False)
        torch.ops._C.silu_and_mul(act, up)
    out = torch.empty((S, 2560), dtype=torch.bfloat16, device=x.device)
    launch(f'{PREFIX}_down', act, out, 'w2', 's2', 1, False)
    COUNTS[(m, t)] += 1
    _report('first_unified', m=m, t=t, siluv=SILUV, prefix=PREFIX, hsaco=HSACO)
    return out
