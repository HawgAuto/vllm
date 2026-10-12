"""direct_ddn: skinny-M native launch for mtp_dense.MTPDenseMethod W8A16 g64 (ddn_v1.hsaco).

Arithmetic contract is the stock fused_moe_kernel_gptq_awq int8_w8a16 contract (see ddn_v1.hip).  This module only
launches; it never dequantizes on the host and never changes resident weights.  Unsupported geometry, dtype, layout
or alignment returns None so the caller keeps the stock route.  Fail closed: hsaco hash mismatch raises.
"""
import ctypes, hashlib, json, os, re
from pathlib import Path
import torch

_SYM = re.compile(r'^ddn_m(\d)_p(\d)_w(\d)$')
_LDS_MAX = 65536
_mod = None


def parse(sym):
    m = _SYM.match(sym)
    if not m: raise ValueError('kernel symbol ' + sym)
    mr, pg, wpg = map(int, m.groups())
    if mr not in (1, 2, 4) or pg not in (2, 4, 8) or wpg not in (1, 4): raise ValueError('kernel symbol ' + sym)
    return mr, pg, wpg


def module(path=None, sha=None):
    """Load the pinned HSACO once (process-wide)."""
    global _mod
    path = path or os.environ['DIRECT_DDN_HSACO']; sha = sha or os.environ['DIRECT_DDN_HSACO_SHA256']
    if _mod is None:
        got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if got != sha: raise RuntimeError('DIRECT_DDN hsaco sha256 mismatch: ' + got)
        import direct_hip_module_v1 as hm
        _mod = (hm, hm.Module(path), path)
    elif _mod[2] != path: raise RuntimeError('DIRECT_DDN hsaco path changed')
    return _mod


def supported(a, w, s, n, k, sym):
    """Pure-shape guard; True only if the native kernel can run this call exactly as the stock route."""
    mr, _, _ = parse(sym)
    m = a.shape[0] if a.ndim == 2 else -1
    return (a.ndim == 2 and a.dtype == torch.bfloat16 and a.is_contiguous() and a.shape[1] == k and m >= 1
            and w.dtype == torch.uint8 and w.is_contiguous() and tuple(w.shape) == (n, k)
            and s.dtype == torch.float32 and s.is_contiguous() and tuple(s.shape) == (n, k // 64)
            and k % 64 == 0 and n >= 1 and mr * k * 2 <= _LDS_MAX and n * k < 2 ** 31
            and a.data_ptr() % 16 == 0 and a.device.type == 'cuda' and w.device == a.device and s.device == a.device)


def launch(sym, a, w, s, out, n, k):
    """out[m, n] (bf16, contiguous) = native(a[m, k]) with resident W [n, k] uint8 and S [n, k/64] f32."""
    hm, mod, _ = module()
    mr, _, wpg = parse(sym)
    m = a.shape[0]
    grid = ((n + 32 * wpg - 1) // (32 * wpg), (m + mr - 1) // mr)
    mod.launch(sym, grid, (32 * wpg,), [hm.ptr(a), hm.ptr(w), hm.ptr(s), hm.ptr(out), ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k)],
               shared=mr * k * 2)
    return out


def load_table(path=None, sha=None):
    """Frozen dispatch table: {"rows": [{"n":..,"k":..,"m":..,"kernel":..}, ...]} -> {(n,k,m): kernel}."""
    path = path or os.environ['DIRECT_DDN_TABLE']; sha = sha or os.environ['DIRECT_DDN_TABLE_SHA256']
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha: raise RuntimeError('DIRECT_DDN table sha256 mismatch')
    rows = json.loads(raw)['rows']; table = {}
    for r in rows:
        key = (int(r['n']), int(r['k']), int(r['m']))
        if key in table: raise ValueError('duplicate table row')
        parse(r['kernel']); table[key] = r['kernel']
    return table
