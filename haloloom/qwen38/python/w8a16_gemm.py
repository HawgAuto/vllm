# SPDX-License-Identifier: Apache-2.0
"""Dense W8A16 K5: INT8-weight x BF16-activation GEMM for the prefill path (tokens > 4) on gfx1151 (k5-gemm-v1/K5-PLAN.md).

Faithful port of vLLM `_triton_w4a16_skinny_fmt_kernel` + `triton_w4a16_skinny_fmt_gemm`, the Triton prefill half of
RDNAHybridW4A16LinearKernel (image aa04e1247a69, vLLM 0.29.1.dev0+qwen38.21ee743f4,
vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py; source sha256 recorded in K5-PLAN.md).
Kept verbatim: program grid (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N)), K loop, load masks, tl.trans + tl.dot with an FP32
accumulator, contiguity asserts and the gfx1151 branch of the tile heuristic.
Deliberate changes, all forced by the weight format:
  1. weight fetch: int8 codes [N, K] row-major -- the K2 skinny-kernel buffer, shared, no second weight copy -- loaded as an
     int8 [BLOCK_N, BLOCK_K] tile and converted exactly (|q| <= 127) to the activation dtype; replaces the int32
     ExLlama-packed int4 load + interleave/shift unpack and the zero-point bias;
  2. scale: per-output-channel FP32 scale (group = K, symmetric) applied once to the FP32 accumulator before the store
     (the K1 skinny kernel's numerics), replacing the per-group activation-dtype multiply inside the K loop; upstream's
     BLOCK_K = min(BLOCK_K, group_size) clamp existed only for that per-tile scale load and is dropped with it;
  3. upstream's per-shape override table is not carried (its keys are other models' (group, K, N)); a K5-measured table
     (W8A16-GEMM-CONFIGS.json) may replace the heuristic per (N, K, M bucket).
Offsets are int32 as upstream: the launcher refuses M*K, N*K or M*N >= 2**31 (callers fall back to the dequant path).
"""
import json
import torch
import triton
import triton.language as tl

INT32_LIMIT = 2 ** 31


@triton.jit
def _w8a16_gemm_kernel(
    a_ptr,  # [M, K]  bf16/fp16 activations
    b_ptr,  # [N, K]  int8 codes (row-major, K contiguous)
    scales_ptr,  # [N]  fp32 per-output-channel scale
    c_ptr,  # [M, N]  bf16/fp16 output
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
        mask_b = (offs_n[:, None] < N) & mask_k[None, :]
        b = tl.load(b_ptrs, mask=mask_b, other=0)

        b_fp_t = tl.trans(b.to(a.dtype))
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = (accumulator * scales[None, :]).to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def upstream_gfx1151_config(M: int, N: int, K: int):
    """Verbatim gfx1151 branch of upstream triton_w4a16_skinny_fmt_gemm (minus the override table): (BM, BN, BK, warps, stages)."""
    if M <= 32:
        return (32, 32, 128, 4, None)
    elif M <= 64:
        return (64, 64, 32, 4, None)
    elif M <= 128:
        if K >= 2 * N:  # tall K (e.g. down_proj)
            return (64, 16, 64, 1, None)
        elif N > K:  # wide N (e.g. qkv_proj, gate_up_proj)
            return (64, 64, 64, 4, None)
        else:  # N ~= K (e.g. o_proj)
            return (64, 32, 64, 4, None)
    elif M <= 1024:
        if K >= 2 * N:
            return (64, 64, 64, 4, None)
        elif N >= 4 * K:
            return (128, 64, 64, 8, None)
        else:
            return (64, 128, 32, 4, None)
    else:
        if K >= 2 * N:
            return (128, 512, 32, 16, None)
        else:
            return (128, 64, 64, 8, None)


def fits(M: int, N: int, K: int) -> bool:
    return max(M * K, N * K, M * N) < INT32_LIMIT


def launch(a, b_q, scales, cfg=None, return_kernel=False):
    assert a.dim() == 2 and b_q.dim() == 2, (a.shape, b_q.shape)
    assert a.dtype in (torch.bfloat16, torch.float16) and b_q.dtype == torch.int8 and scales.dtype == torch.float32
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"
    M, K = a.shape
    N = b_q.shape[0]
    assert b_q.shape[1] == K and tuple(scales.shape) == (N,), (b_q.shape, scales.shape, K)
    assert fits(M, N, K), ('int32 offsets', M, N, K)
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = cfg if cfg is not None else upstream_gfx1151_config(M, N, K)
    # Upstream clamps BLOCK_K = min(BLOCK_K, group_size) because it loads one scale per K tile; with the scale moved to the
    # epilogue (change 2) there is no per-tile scale, so the clamp is dropped with it (K tails are handled by mask_k).
    assert all(v > 0 and v & (v - 1) == 0 for v in (BLOCK_M, BLOCK_N, BLOCK_K)) and min(BLOCK_M, BLOCK_N, BLOCK_K) >= 16, cfg
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    extra_kwargs = {} if num_stages is None else {"num_stages": num_stages}
    k = _w8a16_gemm_kernel[grid](a, b_q, scales, c, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                 num_warps=num_warps, **extra_kwargs)
    return (c, k) if return_kernel else c


# ------------------------------------------------------------------------------------------------- measured table
def load_table(path):
    """W8A16-GEMM-CONFIGS.json -> {(N, K): [(m_max, path, cfg_tuple_or_None), ...] ascending m_max}."""
    t = json.loads(open(path).read())
    assert t['schema'] == 'dense-w8a16-k5-gemm-configs-v1', t.get('schema')
    out = {}
    for key, rows in t['table'].items():
        n, k = (int(v) for v in key.split('x'))
        ent = []
        for r in rows:
            assert r['path'] in ('k5', 'dequant'), r
            cfg = tuple(r['cfg']) if r['path'] == 'k5' else None
            if cfg is not None:
                assert len(cfg) == 5 and all(isinstance(v, int) for v in cfg[:4]) and (cfg[4] is None or isinstance(cfg[4], int)), r
            ent.append((int(r['m_max']), r['path'], cfg))
        ent.sort(key=lambda e: e[0])
        assert ent and len({e[0] for e in ent}) == len(ent), key
        out[(n, k)] = ent
    return out


def lookup(table, M, N, K):
    """(path, cfg) for an M-row call: first bucket with M <= m_max, else the last bucket.  None when (N, K) is absent."""
    ent = table.get((N, K)) if table else None
    if not ent:
        return None
    for m_max, path, cfg in ent:
        if M <= m_max:
            return path, cfg
    return ent[-1][1], ent[-1][2]
