"""Private scalar route fusion; native prefill/unsupported fallback stays outside.
Manually authored guard/address fusion, not a new quantizer or GEMV arithmetic.
"""
import torch
from vllm.triton_utils import triton,tl
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from direct_scalar_masked_kernel import scalar_masked_slot_kernel
from direct_moe_reduction import reduce_slots
CONFIG={'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':1,'SPLIT_K':1,'num_warps':4,'num_stages':2}

@triton.jit
def scalar_route_guards(IDs,W,P,Flags,Unit,HAS_PADDING:tl.constexpr):
    r=tl.arange(0,16);live=r<10
    ids=tl.load(IDs+r,live,other=-1).to(tl.int64)
    w=tl.load(W+r,live,other=0).to(tl.float32)
    valid=(ids>=0)&(ids<512)
    if HAS_PADDING:valid=valid|((ids==-1)&tl.load(P))
    valid=tl.sum((live&(~valid)).to(tl.int32),0)==0
    finite=tl.sum((live&((w!=w)|(tl.abs(w)==float('inf')))).to(tl.int32),0)==0
    dup=(ids[:,None]==ids[None,:])&(r[:,None]<r[None,:])&(r[None,:]<10)&(ids[:,None]!=-1)
    unique=tl.sum(tl.sum(dup.to(tl.int32),0),0)==0
    tl.store(Flags+0,valid);tl.store(Flags+1,finite);tl.store(Flags+2,unique)
    tl.store(Unit+r,1.,live)

def supported(x,ids,w):
    return (x.device.type=='cuda' and x.shape==(1,2560) and x.dtype==torch.bfloat16 and x.is_contiguous()
        and ids.shape==(1,10) and ids.dtype in (torch.int32,torch.int64) and ids.is_contiguous()
        and w.shape==ids.shape and w.dtype==torch.float32 and w.is_contiguous() and x.device==ids.device==w.device)

def padding_for(x):
    from vllm.forward_context import get_forward_context,is_forward_context_available
    p=get_forward_context().is_padding if is_forward_context_available()else None
    if p is not None and (not isinstance(p,torch.Tensor) or p.ndim!=1 or p.dtype!=torch.bool or p.device!=x.device or p.shape[0]<1):
        raise ValueError('invalid native padding mask geometry/dtype/device')
    return p

def validate(method,layer,x):
    if not method._ready:raise ValueError('mixed weights not ready')
    if method.plan.total!=512 or set(method.native)!={4,8}:raise ValueError('unsupported scalar pools')
    from vllm.model_executor.layers.fused_moe import get_config
    if get_config()is not None:raise ValueError('global kernel overrides not allowed on isolated route')
    for bits,experts in ((4,method.plan.four),(8,method.plan.eight)):
        pool=layer.mixed_pools[str(bits)];native=method.native[bits];qc=native.quant_config;pack=2 if bits==4 else 1
        assert native.quant_dtype is None and not native.per_act_token_quant and list(native.block_shape)==[0,128]
        assert qc.w1_zp is None and qc.w2_zp is None and bool(qc.use_int4_w4a16)==(bits==4) and bool(qc.use_int8_w8a16)==(bits==8)
        assert pool.w13_weight.shape==(len(experts),1280,2560//pack) and pool.w2_weight.shape==(len(experts),2560,640//pack)
        for weight,scale,ns in ((pool.w13_weight,pool.w13_scale,native.w1_scale),(pool.w2_weight,pool.w2_scale,native.w2_scale)):
            assert weight.dtype==torch.uint8 and scale.dtype==torch.float32 and scale.shape==(weight.shape[0],weight.shape[1],weight.shape[2]*pack//128)
            assert ns.data_ptr()==scale.data_ptr()
            for t in(weight,scale):
                e,n,k=t.shape;assert t.stride()in((n*k,k,1),(n*k,1,n)) and t.storage_offset()==0
                assert t.device==x.device and t.untyped_storage().nbytes()==t.numel()*t.element_size()
        lut=getattr(layer,'mixed_lookup_'+str(bits));assert lut.shape==(512,) and lut.dtype==torch.int64 and lut.is_contiguous()and lut.device==x.device

def invoke(a,b,c,s,ids,lut,unit,w4,down,collect=None):
    assert a.stride(-1)==1 and (a.shape[-1],b.shape[1])in((2560,1280),(640,2560))
    scalar_masked_slot_kernel[(b.shape[1]//64,10)](a,b,c,s,None,unit,ids,lut,None,b.shape[1],a.shape[-1],160,10,
        a.stride(0)if down else 0,a.stride(1),b.stride(0),b.stride(2),b.stride(1),c.stride(1),c.stride(2),s.stride(0),s.stride(2),s.stride(1),0,0,0,
        block_k_diviable=True,group_size=128,MUL_ROUTED_WEIGHT=down,top_k=1,compute_type=tl.bfloat16,
        has_zp=False,use_int4_w4a16=w4,use_int8_w8a16=not w4,**CONFIG)
    if collect is not None:collect(4 if w4 else 8,'down'if down else'gate_up',c)

def run(method,layer,x,weights,ids,collect=None,poison=False):
    assert supported(x,ids,weights);validate(method,layer,x);padding=padding_for(x)
    flags=torch.empty(3,dtype=torch.bool,device=x.device);unit=torch.empty((10,1),device=x.device,dtype=torch.float32)
    scalar_route_guards[(1,)](ids,weights,padding if padding is not None else ids,flags,unit,padding is not None,num_warps=4)
    torch._assert_async(flags[0],'invalid global expert ID')
    torch._assert_async(flags[1],'nonfinite router weights')
    torch._assert_async(flags[2],'duplicate top-k ID')
    outputs=[]
    for bits in(4,8):
        pool=layer.mixed_pools[str(bits)];native=method.native[bits];lut=getattr(layer,'mixed_lookup_'+str(bits))
        up=torch.empty((10,1,1280),dtype=x.dtype,device=x.device);act=torch.empty((10,640),dtype=x.dtype,device=x.device)
        down=torch.empty((10,1,2560),dtype=x.dtype,device=x.device);out=torch.empty((10,2560),dtype=x.dtype,device=x.device)
        if poison:up.fill_(float('nan'));down.fill_(float('nan'));out.fill_(float('nan'))
        invoke(x,pool.w13_weight,up,pool.w13_scale,ids,lut,unit,bits==4,False,collect)
        native.activation(MoEActivation.SILU,act,up.view(10,1280))
        invoke(act,pool.w2_weight,down,pool.w2_scale,ids,lut,unit,bits==4,True,collect)
        native.moe_sum(down,out);outputs.append(out)
    return reduce_slots(outputs[0],outputs[1],weights,ids,padding is not None)
