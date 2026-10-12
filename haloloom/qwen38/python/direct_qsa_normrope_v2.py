"""decode23 qsa-normrope v2 route: exact QSA-indexer GemmaRMSNorm + cos/sin gather in ONE HIP launch, then the installed
Triton triton_mrope on bit-identical inputs (gfx1151).

Native eager path per call (q or k):  w.float(); +1.0; x.float(); pow; mean; +eps; rsqrt; mul; mul; to(bf16)   (10)
                                      cache[positions] gather; cos.contiguous(); sin.contiguous(); _triton_mrope   (4)
This route:                           qng_r2 (norm + gather + contiguous cos/sin)                                  (1)
                                      _triton_mrope_forward (installed, same call as apply_qsa_rope)              (1)
Norm order/rsqrt identified by attempts/decode23-qsa-normrope-03 (sum-of-squares V1 = torch reduce order, var bits
6000/6000 both shapes; rsqrt RS2 = (float)(1/sqrt(double)) == torch.rsqrt on all 2139095039 positive finite floats).
Scope: bf16, head_dim 128, rotary_dim 64 MRotaryEmbedding, positions [3,T] int32/int64, rows in {1,4} (decode T=1).
Anything else returns None (caller uses the native path). The first call per key is compared bitwise with the native
path and raises on mismatch (fail-closed).
"""
import ctypes, json, os
import torch

HSACO = os.environ.get('DIRECT_QNR_HSACO', '')
_M = None
_CHECKED = set()


def enabled():
    return bool(HSACO)


def _mod():
    global _M
    if _M is None:
        import direct_hip_module_v1 as hm
        _M = (hm, hm.Module(HSACO))
    return _M


def supported(rotary, x2d, positions, heads):
    return (enabled() and x2d.dtype == torch.bfloat16 and x2d.dim() == 2 and x2d.shape[1] == 128 and x2d.stride(1) == 1
            and x2d.shape[0] in (1, 4) and x2d.shape[0] % heads == 0 and not torch.is_grad_enabled()
            and getattr(rotary, 'mrope_section', None) and int(rotary.rotary_dim) == 64
            and positions.dim() == 2 and positions.shape[0] == 3 and positions.shape[1] == x2d.shape[0] // heads
            and positions.dtype in (torch.int64, torch.int32) and x2d.is_cuda)


def norm_gather(x2d, weight, eps, cache, positions, heads):
    hm, m = _mod()
    rows = x2d.shape[0]; T = rows // heads
    out = torch.empty((rows, 128), dtype=torch.bfloat16, device=x2d.device)
    cos = torch.empty((3, T, 32), dtype=torch.bfloat16, device=x2d.device)
    sin = torch.empty((3, T, 32), dtype=torch.bfloat16, device=x2d.device)
    assert cache.dtype == torch.bfloat16 and cache.is_contiguous() and cache.shape[1] == 64 and weight.is_contiguous()
    m.launch('qng_r2', (rows, 1, 1), (128, 1, 1),
             [hm.ptr(x2d), ctypes.c_long(x2d.stride(0)), hm.ptr(weight), ctypes.c_float(float(eps)), hm.ptr(cache),
              ctypes.c_long(cache.stride(0)), hm.ptr(positions), ctypes.c_int(1 if positions.dtype == torch.int64 else 0),
              ctypes.c_long(positions.stride(0)), ctypes.c_long(positions.stride(1)), ctypes.c_int(heads), ctypes.c_int(T),
              hm.ptr(out), hm.ptr(cos), hm.ptr(sin)])
    return out, cos, sin


def fused(norm, rotary, x2d, positions, heads):
    from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
    cache = rotary._match_cos_sin_cache_dtype(x2d)
    out, cos, sin = norm_gather(x2d, norm.weight, norm.variance_epsilon, cache, positions, heads)
    T = out.shape[0] // heads
    t3 = out.view(T, heads, 128)
    q, _ = triton_mrope(t3.reshape(T, -1), t3.new_empty((T, 128)), cos, sin, rotary.mrope_section, 128,
                        rotary.rotary_dim, rotary.mrope_interleaved, rotary.is_neox_style)
    return q.reshape(t3.shape)


def normrope(norm, rotary, x2d, positions, heads, native_fn, key):
    if not supported(rotary, x2d, positions, heads):
        return None
    y = fused(norm, rotary, x2d, positions, heads)
    if key not in _CHECKED:
        ref = native_fn()
        if not torch.equal(y.contiguous().view(torch.int16), ref.contiguous().view(torch.int16)):
            raise RuntimeError(f'DIRECT_QNR self-check mismatch at {key}')
        _CHECKED.add(key)
        print('DIRECT_QNR_SELFCHECK_PASS ' + json.dumps({'key': list(key), 'rows': x2d.shape[0], 'kernel': 'qng_r2+triton_mrope'}), flush=True)
    return y
