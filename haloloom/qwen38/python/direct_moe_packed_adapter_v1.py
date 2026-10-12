# SPDX-License-Identifier: Apache-2.0
"""Private packed-W4 stage1 adapter; other format/layout/config stays native."""
import json,os
from typing import Any
import torch
from vllm.triton_utils import triton,tl
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_wna16_triton_kernel as _native_invoke,get_moe_wna16_block_config
from direct_moe_scale_broadcast_adapter_v1 import supported_tensors
from direct_moe_w4_packed_once_v1 import fused_moe_w4_packed_once
AUDIT=os.environ.get('DIRECT_MOE_BROADCAST_AUDIT')=='1'

def qualified_policy(*,enabled,tokens,bf16,block_shape,w4,w8,projection,config):
    return (bool(enabled) and bf16 and bool(w4) and not w8
        and tokens in (504,960,1568,3136,10976)
        and projection in ('gate_up','down') and tuple(block_shape or ())==(0,128)
        and all(config.get(k,d)==v for k,d,v in (
            ('BLOCK_SIZE_M',None,64),('BLOCK_SIZE_N',64,64),('BLOCK_SIZE_K',32,32),
            ('GROUP_SIZE_M',None,1),('SPLIT_K',1,1),('num_stages',2,2),
            ('num_warps',4,4 if tokens==504 else 2))))

def _invoke_packed(
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
    config_ok = bool(use_int4_w4a16) and not use_int8_w8a16 and config.get('num_warps',4) in (2,4) and tuple(block_shape or ()) == (0,128) and all(
        config.get(key,default)==value for key,default,value in (
            ('BLOCK_SIZE_M',None,64),('BLOCK_SIZE_N',64,64),
            ('BLOCK_SIZE_K',32,32),('GROUP_SIZE_M',None,1),
            ('SPLIT_K',1,1),('num_stages',2,2)))
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

    config['num_stages']=1

    compiled = fused_moe_w4_packed_once[grid](
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
        print('DIRECT_MOE_PACKED_LAUNCH_RETURNED '+json.dumps({
            'A':list(A.shape),'B':list(B.shape),'top_k':top_k,
            'bits':4 if use_int4_w4a16 else 8,'config':config,
            'compiled_kernel_name':compiled.name,'compiled_hash':compiled.hash,
            'registers':compiled.n_regs,'spills':compiled.n_spills,
            'scope':'native Triton launch returned; not independent hardware trace geometry'
        },sort_keys=True),flush=True)
