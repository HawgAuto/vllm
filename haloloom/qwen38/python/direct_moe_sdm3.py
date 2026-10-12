# Operator-only multi-row successor; unchanged arithmetic helpers in sdm3.hip.
# No production dispatch. Grid z selects independent rows; no expert-union reuse.
"""decode23 moe-hip-v2: two-launch bit-exact HIP WMMA scalar decode MoE route (gfx1151).

Same contract as direct_moe_sdm_v1 (bitwise vs the incumbent scalar pool-pair route), with the per-layer route glue
folded into the two kernels:
  * scalar_route_guards + three torch._assert_async  -> in-kernel guard in sdm3_up (block 0,0 wave 0; s_trap on violation)
  * _slots_reduce (+ shared-expert fold)               -> last-arriver epilogue in sdm3_dn
Per decode MoE layer: 7 kernels (v1) -> 2 kernels (v2).
"""
import ctypes, json, os, sys
import torch

import direct_hip_module_v1 as hm
import direct_scalar_route as parent

_MODS = {}
_FUSED_CHECKED = {}


def _mod(path):
    m = _MODS.get(path)
    if m is None:
        m = _MODS[path] = hm.Module(path)
    return m


def up_geom(name):
    # sdm3_up_m2_c{CPL}_p{PF}_l{L}
    cpl = int(name.split('_c')[1].split('_')[0])
    return (640 // (32 * cpl), 10), (64,)


def dn_geom(name):
    cpl = int(name.split('_c')[1].split('_')[0])
    return (2560 // (32 * cpl), 10), (32,)


class Cfg:
    def __init__(self, hsaco, up, dn, siluv):
        self.hsaco, self.up, self.dn, self.siluv = hsaco, up, dn, int(siluv)
        self.up_grid, self.up_block = up_geom(up)
        self.dn_grid, self.dn_block = dn_geom(dn)

    def __repr__(self):
        return f'Cfg({os.path.basename(self.hsaco)},{self.up},{self.dn},silu{self.siluv})'


def validate_layer(layer):
    p4, p8 = layer.mixed_pools['4'], layer.mixed_pools['8']
    want = {
        ('4', 'w13_weight'): ((1280, 1280), (1280 * 1280, 1, 1280), torch.uint8),
        ('4', 'w2_weight'): ((2560, 320), (2560 * 320, 1, 2560), torch.uint8),
        ('4', 'w13_scale'): ((1280, 20), (25600, 1, 1280), torch.float32),
        ('4', 'w2_scale'): ((2560, 5), (12800, 1, 2560), torch.float32),
        ('8', 'w13_weight'): ((1280, 2560), (1280 * 2560, 1, 1280), torch.uint8),
        ('8', 'w2_weight'): ((2560, 640), (2560 * 640, 640, 1), torch.uint8),
        ('8', 'w13_scale'): ((1280, 20), (25600, 20, 1), torch.float32),
        ('8', 'w2_scale'): ((2560, 5), (12800, 5, 1), torch.float32),
    }
    for (b, n), (shape, stride, dt) in want.items():
        p = getattr(layer.mixed_pools[b], n)
        if tuple(p.shape[1:]) != shape or tuple(p.stride()) != stride or p.dtype != dt or p.storage_offset() != 0:
            raise ValueError(f'sdm3 layout mismatch {b}.{n}: {tuple(p.shape)} {tuple(p.stride())} {p.dtype}')
    for b in ('4', '8'):
        lut = getattr(layer, 'mixed_lookup_' + b)
        if lut.dtype != torch.long or lut.shape != (512,) or not lut.is_contiguous():
            raise ValueError('sdm3 lookup mismatch')
    return p4, p8


def _counter(layer, device):
    c = getattr(layer, '_sdm3_cnt', None)
    if c is None or c.device != device:
        # one zeroed counter per column tile (max 80 tiles); the last arriver of each tile resets its entry
        c = torch.zeros(5 * 128, dtype=torch.int32, device=device)
        layer._sdm3_cnt = c
    return c


def launch_up(cfg, layer, x, ids, act, weights, padding, flags, name=None, grid=None, block=None):
    p4, p8 = layer.mixed_pools['4'], layer.mixed_pools['8']
    mod = _mod(cfg.hsaco)
    has_pad = padding is not None
    pad_t = padding if has_pad else flags  # never dereferenced when has_padding == 0
    mod.launch(name or cfg.up, grid or (*cfg.up_grid, x.shape[0]), block or cfg.up_block,
               [hm.ptr(x), hm.ptr(ids), ctypes.c_int(int(ids.dtype == torch.int64)), hm.ptr(layer.mixed_lookup_4),
                hm.ptr(layer.mixed_lookup_8), ctypes.c_int(512), hm.ptr(p4.w13_weight), hm.ptr(p4.w13_scale),
                hm.ptr(p8.w13_weight), hm.ptr(p8.w13_scale), hm.ptr(act), ctypes.c_int(cfg.siluv),
                hm.ptr(weights), hm.ptr(pad_t), ctypes.c_int(int(has_pad)), hm.ptr(flags)])


def launch_dn(cfg, layer, act, ids, down, weights, shared, gate, mask_padding, y, cnt, name=None, grid=None, block=None):
    p4, p8 = layer.mixed_pools['4'], layer.mixed_pools['8']
    mod = _mod(cfg.hsaco)
    has_s = shared is not None
    mod.launch(name or cfg.dn, grid or (*cfg.dn_grid, ids.shape[0]), block or cfg.dn_block,
               [hm.ptr(act), hm.ptr(ids), ctypes.c_int(int(ids.dtype == torch.int64)), hm.ptr(layer.mixed_lookup_4),
                hm.ptr(layer.mixed_lookup_8), ctypes.c_int(512), hm.ptr(p4.w2_weight), hm.ptr(p4.w2_scale),
                hm.ptr(p8.w2_weight), hm.ptr(p8.w2_scale), hm.ptr(down), hm.ptr(weights),
                hm.ptr(shared if has_s else y), hm.ptr(gate if has_s else y), ctypes.c_int(int(has_s)),
                ctypes.c_int(int(bool(mask_padding))), hm.ptr(y), hm.ptr(cnt)])


def _shared_fold(method, x):
    # identical acceptance rules as QuarkMixedMoEMethod._direct_fold_or_reduce
    try:
        from direct_moe_reduction import fold_shared_for_reduce
    except ImportError:
        return None
    fs = fold_shared_for_reduce(getattr(method, '_direct_fold_shared', None))
    if fs is None:
        return None
    ctx, s, g = fs
    m = x.shape[0]
    if (s.shape == (m, 2560) and s.is_contiguous() and s.dtype == torch.bfloat16 and s.device == x.device
            and g.shape == (m, 1) and g.is_contiguous() and g.dtype == torch.bfloat16):
        return ctx, s, g
    return None


def run(method, layer, x, weights, ids, cfg, collect=None, poison=False, fold=True, return_parts=False):
    assert supported(x, ids, weights)
    parent.validate(method, layer, x)
    validate_layer(layer)
    padding = parent.padding_for(x)
    if padding is not None and (padding.shape[0] < x.shape[0] or not padding.is_contiguous()): raise ValueError("invalid multi-row padding")
    dev = x.device
    fill = float('nan') if poison else None
    act = torch.empty((x.shape[0] * 10, 640), device=dev, dtype=torch.bfloat16)
    down = torch.empty((x.shape[0] * 10, 2560), device=dev, dtype=torch.bfloat16)
    y = torch.empty((x.shape[0], 2560), device=dev, dtype=torch.bfloat16)
    if fill is not None:
        act.fill_(fill); down.fill_(fill); y.fill_(fill)
    flags = torch.empty(4 * x.shape[0], device=dev, dtype=torch.int32)
    cnt = _counter(layer, dev)
    wts = weights if (weights.dtype == torch.float32 and weights.is_contiguous()) else weights.float().contiguous()
    fs = _shared_fold(method, x) if fold else None
    s = g = None
    if fs is not None:
        ctx, s, g = fs
    launch_up(cfg, layer, x, ids, act, wts, padding, flags)
    if collect is not None:
        collect('act', act)
    launch_dn(cfg, layer, act, ids, down, wts, s, g, padding is not None, y, cnt)
    if collect is not None:
        collect('down', down)
    key = (cfg.hsaco, cfg.dn, x.shape[0], fs is not None, padding is not None)
    if key not in _FUSED_CHECKED and not torch.cuda.is_current_stream_capturing():
        # one-time eager self-check of the fused epilogue against the installed Triton reduction on the same slots
        from direct_moe_reduction import reduce_slots
        refs = []
        if fs is not None:
            refs.append(reduce_slots(down, None, weights, ids, padding is not None, shared=s, gate=g))
            refs.append((g * s) + reduce_slots(down, None, weights, ids, padding is not None))  # eager scale + runner add
        else:
            refs.append(reduce_slots(down, None, weights, ids, padding is not None))
        if not all(torch.equal(r.view(torch.int16), y.view(torch.int16)) for r in refs):
            raise RuntimeError('sdm3 fused slot reduction self-check mismatch')
        _FUSED_CHECKED[key] = True
        print('DIRECT_SDM2_FUSED_REDUCE_SELFCHECK ' + json.dumps({'pid': os.getpid(), 'dn': cfg.dn, 'shared_fold': fs is not None,
              'padding': padding is not None}, sort_keys=True), file=sys.stderr, flush=True)
    if fs is not None:
        ctx['folded'] = True
    if return_parts:
        return y, act, down, flags
    return y


def supported(x, ids, w):
    return (x.device.type == 'cuda' and x.ndim == 2 and 1 <= x.shape[0] <= 5 and x.shape[1] == 2560 and x.dtype == torch.bfloat16 and x.is_contiguous()
            and ids.shape == (x.shape[0], 10) and ids.dtype in (torch.int32, torch.int64) and ids.is_contiguous()
            and w.shape == ids.shape and w.dtype == torch.float32 and w.is_contiguous() and x.device == ids.device == w.device)
