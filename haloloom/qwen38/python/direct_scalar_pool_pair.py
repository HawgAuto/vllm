"""Scalar mixed-pool co-dispatch; no math/weight-layout replacement.
Two projections instead of four pool-specific launches. Both pool outputs and
native activation/reduction boundaries remain materialized exactly as before.
Each block calls the unchanged native masked-slot JIT with a static W4/W8 mode.
"""
import torch
from vllm.triton_utils import triton,tl
from vllm.model_executor.layers.fused_moe.fused_moe import write_zeros_to_output
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from direct_scalar_masked_kernel import scalar_masked_slot_kernel
import direct_scalar_route as parent

@triton.jit
def scalar_pool_pair_kernel(A4,A8,B4,B8,C4,C8,S4,S8,Unit,IDs,L4,L8,
 N:tl.constexpr,K:tl.constexpr,AM4,AM8,BE4,BK4,BN4,BE8,BK8,BN8,
 SE4,SK4,SN4,SE8,SK8,SN8,CM:tl.constexpr,DOWN:tl.constexpr):
 slot=tl.program_id(1)
 eid=tl.load(IDs+slot).to(tl.int64)
 local4=tl.load(L4+eid,mask=(eid>=0)&(eid<512),other=-1)
 # Static per-branch precision, original body, unchanged K order and BF16
 # materialization. Native async route guards remain in the caller.
 if local4>=0:
  scalar_masked_slot_kernel(A4,B4,C4,S4,None,Unit,IDs,L4,None,N,K,160,10,
   AM4,1,BE4,BK4,BN4,CM,1,SE4,SK4,SN4,0,0,0,
   True,128,16,64,32,1,1,DOWN,1,tl.bfloat16,False,True,False)
  write_zeros_to_output(C8,CM,1,tl.program_id(0),N,tl.full((16,),slot,tl.int64),tl.arange(0,16)==0,16,64,tl.bfloat16)
 else:
  scalar_masked_slot_kernel(A8,B8,C8,S8,None,Unit,IDs,L8,None,N,K,160,10,
   AM8,1,BE8,BK8,BN8,CM,1,SE8,SK8,SN8,0,0,0,
   True,128,16,64,32,1,1,DOWN,1,tl.bfloat16,False,False,True)
  write_zeros_to_output(C4,CM,1,tl.program_id(0),N,tl.full((16,),slot,tl.int64),tl.arange(0,16)==0,16,64,tl.bfloat16)

def invoke(layer,aa,cc,ids,unit,down,collect):
 p4,p8=layer.mixed_pools['4'],layer.mixed_pools['8']
 b4,b8=(p4.w2_weight,p8.w2_weight)if down else(p4.w13_weight,p8.w13_weight)
 s4,s8=(p4.w2_scale,p8.w2_scale)if down else(p4.w13_scale,p8.w13_scale)
 n,k=b4.shape[1],aa[0].shape[-1]
 assert(n,k)in((1280,2560),(2560,640))and all(a.stride(-1)==1 for a in aa)
 scalar_pool_pair_kernel[(n//64,10)](aa[0],aa[1],b4,b8,cc[0],cc[1],s4,s8,unit,ids,
  layer.mixed_lookup_4,layer.mixed_lookup_8,n,k,
  aa[0].stride(0)if down else 0,aa[1].stride(0)if down else 0,
  b4.stride(0),b4.stride(2),b4.stride(1),b8.stride(0),b8.stride(2),b8.stride(1),
  s4.stride(0),s4.stride(2),s4.stride(1),s8.stride(0),s8.stride(2),s8.stride(1),n,down,
  num_warps=4,num_stages=2)
 if collect is not None:
  for bits,c in zip((4,8),cc):collect(bits,'down'if down else'gate_up',c)

def run(method,layer,x,weights,ids,collect=None,poison=False):
 assert parent.supported(x,ids,weights)
 parent.validate(method,layer,x);padding=parent.padding_for(x)
 flags=torch.empty(3,dtype=torch.bool,device=x.device);unit=torch.empty((10,1),device=x.device,dtype=torch.float32)
 parent.scalar_route_guards[(1,)](ids,weights,padding if padding is not None else ids,flags,unit,padding is not None,num_warps=4)
 torch._assert_async(flags[0],'invalid global expert ID')
 torch._assert_async(flags[1],'nonfinite router weights')
 torch._assert_async(flags[2],'duplicate top-k ID')
 up=[torch.empty((10,1,1280),dtype=x.dtype,device=x.device)for _ in(4,8)]
 act=[torch.empty((10,640),dtype=x.dtype,device=x.device)for _ in(4,8)]
 down=[torch.empty((10,1,2560),dtype=x.dtype,device=x.device)for _ in(4,8)]
 out=[torch.empty((10,2560),dtype=x.dtype,device=x.device)for _ in(4,8)]
 if poison:
  for t in up+act+down+out:t.fill_(float('nan'))
 invoke(layer,[x,x],up,ids,unit,False,collect)
 for i,bits in enumerate((4,8)):method.native[bits].activation(MoEActivation.SILU,act[i],up[i].view(10,1280))
 invoke(layer,act,down,ids,unit,True,collect)
 for i,bits in enumerate((4,8)):method.native[bits].moe_sum(down[i],out[i])
 return parent.reduce_slots(out[0],out[1],weights,ids,padding is not None)
