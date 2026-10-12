"""Private native-QSA producer adapter; general attention keeps native semantics.
No duplicate-set inference from tensor values. Admission binds the producer's
canonical selection/expansion source and the one modified owner callsite.
"""
import torch


def eligible(q, k_cache, v_cache, logical_indices, block_table, token_to_req,
             out, *, qualified_unique=False):
    if qualified_unique is not True or not q.is_cuda:
        return False
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        return False
    if not (65 <= q.shape[0] <= 12288 and q.shape[1:] == (24, 256)):
        return False
    if k_cache.shape[2:] != (2, 256) or not all(k_cache.shape[:2]):
        return False
    # Bound workspace to the retained page geometry/context. Other geometries
    # retain the native implementation, rather than receiving a broad claim.
    if k_cache.shape[1] != 1568 or block_table.ndim != 2:
        return False
    if block_table.shape != (1, 84) or logical_indices.shape != (q.shape[0], 2051):
        return False
    if token_to_req.shape != (q.shape[0],):
        return False
    if not (q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16):
        return False
    if not (logical_indices.dtype == block_table.dtype == token_to_req.dtype == torch.int32):
        return False
    if not all(t.device == q.device for t in (k_cache, v_cache, logical_indices, block_table, token_to_req)):
        return False
    if not all(t.stride(-1) == 1 for t in (q, k_cache, v_cache, logical_indices, block_table, token_to_req)):
        return False
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype
                            or out.device != q.device or out.stride(-1) != 1):
        return False
    arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(':')[0]
    return arch == 'gfx1151'


def _native(*args):
    from vllm.models.qwen4_exp.amd.ops.qsa import qsa_sparse_paged_attention
    return qsa_sparse_paged_attention(*args)


def _union(*args, context_len=None):
    from direct_qsa_union_kernel_v1 import native_unique_attention
    import os
    if os.environ.get('DIRECT_QSA_UNION_AUDIT') == '1':
        import json, time
        print('DIRECT_QSA_UNION_DISPATCH ' + json.dumps({
            'unix_ns': time.time_ns(), 'pid': os.getpid(),
            'query_rows': args[0].shape[0], 'query_heads': args[0].shape[1],
            'page_table': list(args[4].shape), 'query_group': 4,
        }), flush=True)
    return native_unique_attention(*args, query_group=4, context_len=context_len)


_POISON_CLASSES = set()


def qsa_native_unique_paged_attention(q, k_cache, v_cache, logical_indices,
                                      block_table, token_to_req, out=None, *,
                                      qualified_unique=False, context_len=None, zero_fallback=False):
    args = (q, k_cache, v_cache, logical_indices, block_table, token_to_req, out)
    if not eligible(*args, qualified_unique=qualified_unique):
        if zero_fallback and out is not None:
            out.zero_()   # caller skipped its zero-init (DIRECT_QSA_OUT_TAILZERO); restore it for the native path
        return _native(*args)
    if out is None:
        out = torch.empty_like(q)
    cls = 'long' if q.shape[0] >= 2048 else 'short'
    if zero_fallback and cls not in _POISON_CLASSES:
        # one-time per size class: prove the union path writes every element of out (fail closed)
        _POISON_CLASSES.add(cls)
        out.fill_(float('nan'))
        r = _union(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, context_len=context_len)
        bad = int(torch.isnan(out).sum())
        import json, os
        print('DIRECT_QSA_OUT_TAILZERO ' + json.dumps({'tag': 'poison_check', 'class': cls, 'rows': int(q.shape[0]),
                                                       'nan_left': bad, 'pid': os.getpid()}), flush=True)
        if bad:
            raise RuntimeError('DIRECT_QSA_OUT_TAILZERO: union path left %d unwritten output elements' % bad)
        return r
    return _union(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, context_len=context_len)
