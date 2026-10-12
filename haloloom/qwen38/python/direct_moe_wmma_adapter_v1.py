# SPDX-License-Identifier: Apache-2.0
"""Native gfx1151 WMMA MoE WNA16 adapter (moe_wna16_wmma_v4.hsaco), drop-in for invoke_fused_moe_wna16_triton_kernel.

Numeric contract (bit-exact with the vLLM WNA16 Triton kernel, gated in attempts/moe-wmma-gate-*):
  b = bf16_rne((float(q) - zp) * scale_fp32); acc_fp32 += WMMA_bf16(a, b) over K ascending in 16-chunks; out = bf16_rne(acc).
Scope: prefill-sized calls only (tokens >= MIN_TOKENS), 64-row moe_align_block_size layout, symmetric (no zp),
no routed-weight multiply (slot outputs). Anything else returns the caller's fallback unchanged.
"""
import ctypes, os
import torch
import direct_hip_module_v1 as hm

HSACO = os.environ.get('DIRECT_MOE_WMMA_HSACO', '/opt/venv/lib/python3.14/site-packages/direct_moe_wna16_wmma_v4.hsaco')
MIN_TOKENS = int(os.environ.get('DIRECT_MOE_WMMA_MIN_TOKENS', '256'))
_mod = None
_finite_checked = set()


def qualified_policy(*, enabled, tokens, bf16, block_shape, w4, w8, projection, config):
    return (bool(enabled) and bf16 and (bool(w4) != bool(w8)) and tokens >= MIN_TOKENS
            and projection in ('gate_up', 'down') and tuple(block_shape or ()) == (0, 128)
            and config.get('BLOCK_SIZE_M') == 64)


def _module():
    global _mod
    if _mod is None:
        _mod = hm.Module(HSACO)
    return _mod


def _kernel_name(use_int4, B):
    if use_int4:
        return 'moe4_w4' if B.stride(1) == 1 else None
    if B.stride(1) == 1:
        return 'moe4_w8kn'
    if B.stride(2) == 1:
        return 'moe4_w8nk'
    return None


def make_invoke(fallback):
    def invoke(A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
               mul_routed_weight, top_k, config, compute_type, use_int8_w8a16, use_int4_w4a16, block_shape):
        args = (A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
                mul_routed_weight, top_k, config, compute_type, use_int8_w8a16, use_int4_w4a16, block_shape)
        name = _kernel_name(use_int4_w4a16, B)
        N, K = B.size(1), A.size(1)
        if (name is None or mul_routed_weight or B_zp is not None or A.dtype != torch.bfloat16 or C.dtype != torch.bfloat16
                or A.stride(1) != 1 or C.stride(-1) != 1 or B_scale is None or B_scale.dtype != torch.float32
                or (use_int4_w4a16 and B_scale.stride(1) != 1) or N % 128 or K % 32
                or sorted_token_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32
                or expert_ids.numel() != (sorted_token_ids.numel() + 63) // 64
                or (use_int4_w4a16 and B.size(2) * 2 != K) or (use_int8_w8a16 and B.size(2) != K)):
            return fallback(*args)
        key = B_scale.data_ptr()
        if key not in _finite_checked:  # one-time per weight tensor; dequant omits the NaN branch
            if not bool(torch.isfinite(B_scale).all()):
                return fallback(*args)
            _finite_checked.add(key)
        c = ctypes.c_int; ll = ctypes.c_longlong
        grid = (N // 128, (sorted_token_ids.numel() + 63) // 64)
        _module().launch(name, grid, (256,), [
            hm.ptr(A), hm.ptr(B), hm.ptr(B_scale), hm.ptr(C), hm.ptr(sorted_token_ids), hm.ptr(expert_ids),
            hm.ptr(num_tokens_post_padded), c(N), c(K), c(A.size(0) * top_k), c(top_k),
            ll(A.stride(0)), ll(B.stride(0)), ll(B.stride(2)), ll(B.stride(1)),
            ll(B_scale.stride(0)), ll(B_scale.stride(2)), ll(B_scale.stride(1)), ll(C.stride(-2))])
    invoke.__name__ = 'invoke_moe_wna16_wmma_v4'
    return invoke
