"""HC fused prefill v3 (NON-BITWISE arm) host-side ops.

_hc_combine_norm_pack_kernel: production vllm qwen4_exp amd/ops/hc.py::_hc_combine_norm_kernel (lines 270-336 of
overlays/same-process-moe-abc-v1/hc.py, copied verbatim by script) with ONLY the y (= xn) store address changed to a
packed layout.  out (residual) and xn values are therefore produced by the identical Triton arithmetic.
"""
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _hc_combine_norm_pack_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
    SWIZZLE: tl.constexpr = False,
    LY: tl.constexpr = 1,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    row = tl.program_id(1) if SWIZZLE else tl.program_id(0)
    stream = tl.program_id(0) if SWIZZLE else tl.program_id(1)
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    # Shared norm weights repeat across streams; per-branch weights use the
    # same flattened HC layout as the residual.
    w_offs = offs_inner if W_SHARED else offs

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Start the uncached residual load first, then issue the other combine
    # loads before consuming any of them.
    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    inj = tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    # Round the materialized combine result before normalization. This matches
    # the unfused combine -> RMSNorm boundary.
    out = (res.to(tl.float32) + block.to(tl.float32) * inj).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    # Keep the two-axis reduction: flattening the padded tile is ~40% slower
    # at decode sizes.
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    # Loading the weight earlier helps decode but keeps the tile live across
    # the reduction and regresses larger batches, so defer it to the norm.
    w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    # v3: identical arithmetic; only the xn (y) store address changes to the packed layout consumed by the
    # hc_fused_v3e GEMMs.  LY 0 = P64 [row/64][col/32][64][32], LY 1 = B256 [row/64][col/256][64][256].
    K_ALL: tl.constexpr = HC * HC_DIM
    rowl = row.to(tl.int64)
    if LY == 1:
        yoff = ((rowl // 64) * (K_ALL // 256) + offs // 256) * (64 * 256) + (rowl % 64) * 256 + (offs % 256)
    else:
        yoff = ((rowl // 64) * (K_ALL // 32) + offs // 32) * (64 * 32) + (rowl % 64) * 32 + (offs % 32)
    tl.store(y_ptr + yoff, y, mask_inner)


def combine_norm_pack(residual, block_output, injection_logits, norm_weight, eps, hc_count, ly=1):
    """Returns (out [N, DIM] row-major, xb packed flat buffer of ceil(N/128)*128 * DIM bf16)."""
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim) and injection_logits.shape == (N, hc_count)
    assert residual.stride(1) == 1 and block_output.stride(1) == 1 and injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous() and norm_weight.numel() in (hc_dim, DIM)
    assert DIM % 256 == 0
    out = residual.new_empty(residual.shape)
    Mp = (N + 127) // 128 * 128
    xb = residual.new_empty((Mp * DIM,))
    _hc_combine_norm_pack_kernel[(hc_count, N) if N >= 64 else (N, hc_count)](
        block_output, residual, injection_logits, norm_weight, out, xb,
        block_output.stride(0), residual.stride(0), injection_logits.stride(0), out.stride(0),
        hc_dim, hc_count, W_SHARED=norm_weight.numel() == hc_dim, EPS=eps, BLOCK_SIZE=512,
        launch_pdl=current_platform.is_arch_support_pdl(), SWIZZLE=N >= 64, LY=ly)
    return out, xb
