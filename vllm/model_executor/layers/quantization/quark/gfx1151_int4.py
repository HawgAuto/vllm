# SPDX-License-Identifier: Apache-2.0
"""Experimental, explicit Quark file-to-file INT4 consumer for gfx1151.

Uses the unchanged packed RDNA3 backend. No BF16 expert expansion/fallback.
"""
import torch
from compressed_tensors.quantization import QuantizationArgs
from .quark_moe import QuarkMoEMethod
from .gfx1151_int4_wire import convert_projection
from vllm.model_executor.layers.fused_moe.experts import gfx1151_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend, _process_weights_rdna3, make_wna16_moe_kernel,
    make_wna16_moe_quant_config, select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4Static
from vllm.model_executor.utils import replace_parameter


class QuarkGfx1151Int4MoEMethod(QuarkMoEMethod):
    supported_activation_quant_keys = [None]
    supported_weight_quant_keys = [kInt4Static]

    def __init__(self, moe, weight_quant_key, activation_quant_key):
        super().__init__(moe, weight_quant_key, activation_quant_key)
        p=moe.moe_parallel_config
        if not gfx1151_moe.enabled() or moe.in_dtype != torch.bfloat16:
            raise ValueError('explicit gfx1151 opt-in and BF16 activations required')
        if (p.tp_size,p.ep_size,p.dp_size,p.pcp_size)!=(1,1,1,1) or p.enable_eplb:
            raise ValueError('only single-rank non-EPLB Quark INT4 is qualified')
        if moe.has_bias or moe.activation != MoEActivation.SILU or moe.w13_num_shards!=2:
            raise ValueError('only bias-free gated SiLU is supported')
        self._loaded=set()
        self._ready=False
        self.conversion_metrics=[]

    @property
    def loaded_piece_count(self):
        return len(self._loaded)

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        if (hidden_size,intermediate_size_per_partition)!=(2560,640) or params_dtype!=torch.bfloat16:
            raise ValueError('this exact packed kernel requires K=2560/I=640/BF16')
        if not 1<=num_experts<=512 or num_experts!=self.moe.num_experts:
            raise ValueError('expected all 1..512 experts on one rank')
        self.num_experts=num_experts
        definitions={
            'w13_weight':((num_experts,2560,160),torch.int32),
            'w2_weight':((num_experts,640,320),torch.int32),
            'w13_weight_scale':((num_experts,1280,20),torch.float32),
            'w2_weight_scale':((num_experts,2560,5),torch.float32),
        }
        for name,(shape,dtype) in definitions.items():
            param=torch.nn.Parameter(torch.empty(shape,dtype=dtype,device=self.moe.device),requires_grad=False)
            param.weight_loader=self.weight_loader
            param.quark_parameter_name=name
            layer.register_parameter(name,param)
        layer.orig_dtype=params_dtype
        layer.weight_block_size=None

    def weight_loader(self,param,loaded_weight,weight_name,shard_id,expert_id,return_success=False):
        name=param.quark_parameter_name
        if self._ready or not 0<=expert_id<self.num_experts:
            raise ValueError('late weight load or invalid expert')
        allowed=('w2',) if name.startswith('w2_') else ('w1','w3')
        if shard_id not in allowed:
            raise ValueError('projection/parameter mismatch')
        key=(name,expert_id,shard_id)
        if key in self._loaded:
            raise ValueError('duplicate Quark expert piece')
        dest=param.data[expert_id]
        if name.startswith('w13_'):
            if name.endswith('_scale'):
                dest=dest[:640] if shard_id=='w1' else dest[640:]
            else:
                dest=dest[:,:80] if shard_id=='w1' else dest[:,80:]
        if loaded_weight.shape!=dest.shape or loaded_weight.dtype!=dest.dtype:
            raise ValueError(f'wrong Quark tensor wire for {weight_name}: {loaded_weight.shape}/{loaded_weight.dtype}')
        dest.copy_(loaded_weight)
        self._loaded.add(key)
        return True if return_success else None

    def process_weights_after_loading(self,layer):
        if self._ready or len(self._loaded)!=self.num_experts*6:
            raise ValueError('incomplete or already processed Quark expert payload')
        if layer.w13_weight.device.type!='cuda':
            raise ValueError('GPU processing requires a real gfx1151 device')
        gfx1151_moe.operator()  # validates source/library/ABI; native wrapper validates architecture
        quant=QuantizationArgs(num_bits=4,type='int',symmetric=True,strategy='group',group_size=128)
        backend,cls=select_wna16_moe_backend(self.moe,kInt4Static,quant,False,False)
        if backend!=WNA16MoEBackend.RDNA3:
            raise ValueError('no alternative kernel or dequantized fallback permitted')
        device=layer.w13_weight.device
        w1=torch.empty(self.num_experts,320,1280,device=device,dtype=torch.int32)
        w2=torch.empty(self.num_experts,80,2560,device=device,dtype=torch.int32)
        s1=torch.empty(self.num_experts,20,1280,device=device,dtype=torch.bfloat16)
        s2=torch.empty(self.num_experts,5,2560,device=device,dtype=torch.bfloat16)
        for e in range(self.num_experts):
            for start,end,offset in ((0,80,0),(80,160,640)):
                q,s,metrics=convert_projection(layer.w13_weight[e,:,start:end],layer.w13_weight_scale[e,offset:offset+640])
                w1[e,:,offset:offset+640].copy_(q);s1[e,:,offset:offset+640].copy_(s)
                self.conversion_metrics.append(metrics)
            q,s,metrics=convert_projection(layer.w2_weight[e],layer.w2_weight_scale[e])
            w2[e].copy_(q);s2[e].copy_(s);self.conversion_metrics.append(metrics)
        w1,w2,s1,s2,z1,z2,*_=_process_weights_rdna3(w1,w2,s1,s2,128)
        for name,tensor in [('w13_weight',w1),('w2_weight',w2),('w13_weight_scale',s1),('w2_weight_scale',s2)]:
            replace_parameter(layer,name,tensor)
        layer.register_buffer('w13_weight_zero_point',z1)
        layer.register_buffer('w2_weight_zero_point',z2)
        self.moe_quant_config=make_wna16_moe_quant_config(s1,s2,128,4,z1,z2)
        self.moe_kernel=make_wna16_moe_kernel(self.moe_quant_config,self.moe,cls,backend,
                                             routing_tables=layer._expert_routing_tables())
        self._ready=True

    def get_fused_moe_quant_config(self,layer):
        if not self._ready:
            raise ValueError('weights not yet processed')
        return self.moe_quant_config

    def apply(self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
        if not self._ready:
            raise ValueError('Quark weights not ready')
        return self.moe_kernel.apply(hidden_states=x,w1=layer.w13_weight,w2=layer.w2_weight,
            topk_weights=topk_weights,topk_ids=topk_ids,activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,shared_experts=shared_experts,
            shared_experts_input=shared_experts_input)
