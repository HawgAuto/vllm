# SPDX-License-Identifier: Apache-2.0
"""Dense W8A16 K5d-B fused dequant (k5-gemm-v1/k5d-v1/K5D-PLAN.md Amendment 1, lever (b)).

= k5-gemm-v1/w8a16_dequant.py (K5c) with ONE change: trans=True ([K, N] contiguous output for the NN [2560,6144] projections)
runs an output-space-tiled kernel (`_w8_dequant_t_kernel`, w8dqt-bench-02 winner: BLOCK_K=64, BLOCK_N=128, 4 warps, 322 us
vs 513 us for the K5c (16,512,16) tl.trans kernel, 146 vs 92 GB/s).  The row-major path (trans=False) and `_w8_dequant_kernel`
are byte-identical to K5c.  Same IEEE ops in the same order as native `_dequant`: fp32(q) * s (one RTNE FP32 multiply), RTNE
downcast -> BITWISE equal (bench-01/02 on 2-3 seeds incl. extreme codes; re-gated at load by the runtime fdq self-check on the
real weights for every dispatched cfg).
cfg = (BLOCK_N, BLOCK_K, num_warps) for both kernels.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _w8_dequant_kernel(q_ptr, s_ptr, o_ptr, N, K, TRANS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    q = tl.load(q_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask, other=0)
    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
    w = (q.to(tl.float32) * s[:, None]).to(o_ptr.type.element_ty, fp_downcast_rounding='rtne')
    if TRANS:  # [K, N] row-major == the [N, K] weight's transpose made contiguous
        tl.store(o_ptr + offs_k[:, None] * N + offs_n[None, :], tl.trans(w), mask=tl.trans(mask))
    else:
        tl.store(o_ptr + offs_n[:, None] * K + offs_k[None, :], w, mask=mask)


@triton.jit
def _w8_dequant_t_kernel(q_ptr, s_ptr, o_ptr, N, K, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # tile in OUTPUT space [BLOCK_K, BLOCK_N]: loads q[n, k] at offs_n * K + offs_k (strided), stores contiguous along n
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    q = tl.load(q_ptr + offs_n[None, :] * K + offs_k[:, None], mask=mask, other=0)
    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
    w = (q.to(tl.float32) * s[None, :]).to(o_ptr.type.element_ty, fp_downcast_rounding='rtne')
    tl.store(o_ptr + offs_k[:, None] * N + offs_n[None, :], w, mask=mask)


DEFAULT_CFG = (64, 128, 4)  # (BLOCK_N, BLOCK_K, num_warps); the dispatch table supplies the per-shape choice
DEFAULT_T_CFG = (128, 64, 4)  # trans: w8dqt-bench-02 winner (BLOCK_N=128, BLOCK_K=64, 4 warps)


def dequant(q, s, dtype, trans, cfg=None):
    """int8 [N, K] codes x FP32 per-row scale -> dtype; [N, K] (trans=False) or contiguous [K, N] (trans=True)."""
    assert q.dim() == 2 and q.dtype == torch.int8 and q.is_contiguous(), (q.shape, q.dtype)
    assert s.dtype == torch.float32 and s.is_contiguous() and s.shape == (q.shape[0],), (s.shape, s.dtype)
    assert dtype in (torch.bfloat16, torch.float16), dtype
    N, K = q.shape
    assert N * K < 2 ** 31, 'int32 offsets'
    if trans:
        BN, BK, nw = cfg or DEFAULT_T_CFG
        assert BN & (BN - 1) == 0 and BK & (BK - 1) == 0, cfg
        out = torch.empty((K, N), dtype=dtype, device=q.device)
        _w8_dequant_t_kernel[(triton.cdiv(K, BK), triton.cdiv(N, BN))](q, s, out, N, K, BLOCK_K=BK, BLOCK_N=BN, num_warps=nw)
        return out
    BN, BK, nw = cfg or DEFAULT_CFG
    assert BN & (BN - 1) == 0 and BK & (BK - 1) == 0, cfg
    out = torch.empty((N, K), dtype=dtype, device=q.device)
    _w8_dequant_kernel[(triton.cdiv(N, BN), triton.cdiv(K, BK))](q, s, out, N, K, TRANS=False, BLOCK_N=BN,
                                                                  BLOCK_K=BK, num_warps=nw)
    return out
