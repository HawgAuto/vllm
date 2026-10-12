# SPDX-License-Identifier: Apache-2.0
"""Private opt-in scale broadcast. Unsupported shapes retain native dispatch."""
import json
import os
from typing import Any
import torch
from vllm.triton_utils import triton, tl
from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel as _native_invoke,
    get_moe_wna16_block_config,
)
from direct_moe_scale_broadcast_kernel_v1 import fused_moe_scale_broadcast
AUDIT = os.environ.get('DIRECT_MOE_BROADCAST_AUDIT') == '1'
W4_TOKENS = frozenset((960,1568,3136,10976))
W8_DOWN_TOKENS = frozenset((3136,10976))

def qualified_policy(*, enabled, tokens, bf16, block_shape, w4, w8, projection, config):
    return (bool(enabled) and bf16 and bool(w4) != bool(w8)
        and tuple(block_shape or ()) == (0,128)
        and ((w4 and tokens in W4_TOKENS and projection in ('gate_up','down'))
             or (w8 and tokens in W8_DOWN_TOKENS and projection == 'down'))
        and all(config.get(key,default)==value for key,default,value in (
            ('BLOCK_SIZE_M',None,64),('BLOCK_SIZE_N',64,64),
            ('BLOCK_SIZE_K',32,32),('GROUP_SIZE_M',None,1),
            ('SPLIT_K',1,1),('num_warps',4,2),('num_stages',2,2))))

def select_invoke(native, **meta):
    return _invoke_broadcast if qualified_policy(**meta) else native

def supported_tensors(A,B,C,B_scale,B_zp,compute_type,w4,w8,top_k):
    if B_zp is not None or B_scale is None or bool(w4)==bool(w8):return False
    if compute_type != tl.bfloat16 or any(t.device != A.device for t in (B,C,B_scale)):return False
    if A.device.type != 'cuda' or A.dtype != torch.bfloat16 or C.dtype != torch.bfloat16:return False
    if B.dtype != torch.uint8 or B_scale.dtype != torch.float32:return False
    if A.ndim != 2 or B.ndim != 3 or B_scale.ndim != 3 or C.ndim != 3:return False
    if not A.is_contiguous() or not C.is_contiguous():return False
    e,n,k=B.shape; pack=2 if w4 else 1
    if not (0 < e <= 512) or (A.shape[1],n) not in ((2560,1280),(640,2560)):return False
    if k*pack != A.shape[1] or B_scale.shape != (e,n,A.shape[1]//128):return False
    if not (1 <= top_k <= 10) or C.numel() != A.shape[0]*top_k*n:return False
    for tensor in (B,B_scale):
        ee,nn,kk=tensor.shape
        if tensor.stride() not in ((nn*kk,kk,1),(nn*kk,1,nn)) or tensor.storage_offset()!=0:return False
    return True

def _invoke_broadcast(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    # Defense in depth: geometry/data-layout guard, without reading device data.
    config_ok = tuple(block_shape or ()) == (0,128) and all(
        config.get(key,default)==value for key,default,value in (
            ('BLOCK_SIZE_M',None,64),('BLOCK_SIZE_N',64,64),
            ('BLOCK_SIZE_K',32,32),('GROUP_SIZE_M',None,1),
            ('SPLIT_K',1,1),('num_warps',4,2),('num_stages',2,2)))
    if not config_ok or not supported_tensors(A,B,C,B_scale,B_zp,compute_type,use_int4_w4a16,use_int8_w8a16,top_k):
        return _native_invoke(A,B,C,B_scale,B_zp,topk_weights,sorted_token_ids,
            expert_ids,num_tokens_post_padded,mul_routed_weight,top_k,config,
            compute_type,use_int8_w8a16,use_int4_w4a16,block_shape)
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    compiled = fused_moe_scale_broadcast[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )
    if AUDIT:
        print('DIRECT_MOE_BROADCAST_LAUNCH_RETURNED '+json.dumps({
            'A':list(A.shape),'B':list(B.shape),'top_k':top_k,
            'bits':4 if use_int4_w4a16 else 8,'config':config,
            'compiled_kernel_name':compiled.name,'compiled_hash':compiled.hash,
            'registers':compiled.n_regs,'spills':compiled.n_spills,
            'scope':'native Triton launch returned; not independent hardware trace geometry'
        },sort_keys=True),flush=True)
