"""Isolated prefill-only routing fusion; unchanged packed expert GEMMs."""
import torch
from vllm.triton_utils import triton, tl
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts

class SlotExperts(TritonWNA16Experts):
    def moe_sum(self,input,output,topk_ids=None,expert_map=None):
        if input.data_ptr() == output.data_ptr():
            assert topk_ids is None and expert_map is None
            assert output.shape == (input.shape[0]*input.shape[1],input.shape[2])
            return
        return super().moe_sum(input,output,topk_ids,expert_map)

@triton.jit
def _slots_reduce(A,B,W,IDs,Y,S,G,H:tl.constexpr,T:tl.constexpr,HAS_B:tl.constexpr,MASK_PADDING:tl.constexpr,HAS_S:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    a0=tl.full((BLOCK,),0,tl.float32);a1=tl.full((BLOCK,),0,tl.float32)
    a2=tl.full((BLOCK,),0,tl.float32);a3=tl.full((BLOCK,),0,tl.float32)
    # Native Torch's strided sum, vt0=4, no inter-thread input reduction
    # for T<=10. Keep rounded products and the four independent accumulators.
    for slot in tl.static_range(T):
        offsets=(row*T+slot)*H+col
        x=tl.load(A+offsets,col<H,other=0).to(tl.float32)
        if HAS_B:
            b=tl.load(B+offsets,col<H,other=0).to(tl.float32)
            x=(x+b).to(A.dtype.element_ty).to(tl.float32)
        if MASK_PADDING:
            active=tl.load(IDs+row*T+slot)>=0
            x=tl.where(active,x,0.)
        w=tl.load(W+row*T+slot).to(tl.float32)
        product=x*w
        if slot%4==0:a0=a0+product
        elif slot%4==1:a1=a1+product
        elif slot%4==2:a2=a2+product
        else:a3=a3+product
    value=((a0+a1)+a2)+a3
    if HAS_S:
        # DIRECT_MOE_SHARED_FOLD: replicate, with the same roundings, the eager shared-expert scale
        # bf16(f32(sigmoid_gate)*f32(shared)) and the runner's traced add bf16(f32(shared)+f32(routed)).
        r=value.to(Y.dtype.element_ty).to(tl.float32)
        sv=tl.load(S+row*H+col,col<H,other=0).to(tl.float32)
        gv=tl.load(G+row).to(tl.float32)
        sc=(gv*sv).to(Y.dtype.element_ty).to(tl.float32)
        value=sc+r
    tl.store(Y+row*H+col,value.to(Y.dtype.element_ty),col<H)

def reduce_slots(a,b,weights,ids,mask_padding,shared=None,gate=None):
    m,t=ids.shape;h=a.shape[-1]
    assert h==2560 and 0<t<=10 and a.dtype==torch.bfloat16
    assert a.is_contiguous() and weights.is_contiguous() and ids.is_contiguous()
    assert b is None or (b.shape==a.shape and b.dtype==a.dtype and b.is_contiguous())
    has_s=shared is not None
    if has_s:
        assert shared.shape==(m,h) and shared.dtype==a.dtype and shared.is_contiguous()
        assert gate is not None and gate.shape==(m,1) and gate.dtype==a.dtype and gate.is_contiguous()
    y=torch.empty((m,h),dtype=a.dtype,device=a.device)
    _slots_reduce[(m,triton.cdiv(h,512))](a,b if b is not None else a,weights,ids,y,
        shared if has_s else a,gate if has_s else weights,h,t,b is not None,mask_padding,has_s,512,num_warps=4,enable_fp_fusion=False)
    return y


# ---- DIRECT_MOE_SHARED_FOLD context (overlay nb2-moefold-v1) ----
# The FusedMoE runner op wrapper opens a context per moe_forward_shared call; Qwen2MoeMLP (shared expert, eager inside
# the op) stashes its sigmoid gate and returns the UNSCALED output; the routed apply folds bf16(g*s) + routed into
# _slots_reduce when possible; otherwise fold_finish applies g*s and the add eagerly (identical op order), so the op
# ALWAYS returns shared+routed in its fused output and the traced runner add is removed (static flag).
_CTX=[]
_FOLD_CHECKED=[False]
def fold_begin(layer):
    if getattr(layer,'routed_scaling_factor',1.0)!=1.0:
        raise RuntimeError('DIRECT_MOE_SHARED_FOLD requires routed_scaling_factor == 1.0')
    ctx={'gate':None,'folded':False}
    _CTX.append(ctx)
    return ctx
def fold_end(ctx):
    assert _CTX and _CTX[-1] is ctx
    _CTX.pop()
    return ctx
def fold_stash_gate(g):
    if not _CTX or _CTX[-1]['gate'] is not None:
        return False
    _CTX[-1]['gate']=g
    return True
def fold_shared_for_reduce(shared_experts):
    if not _CTX or shared_experts is None:
        return None
    ctx=_CTX[-1]
    if ctx['gate'] is None or ctx['folded']:
        return None
    idx=shared_experts._output_idx
    s=shared_experts._output[idx]
    if s is None:
        return None
    if getattr(shared_experts,'_stream',None) is not None:
        # MULTI_STREAM_OVERLAPPED: s and the stashed gate are produced on the aux stream; make the main stream wait
        # for them before the folded reduce reads them (no-op for NO_OVERLAP: event not recorded / already complete).
        shared_experts._output_ready_event[idx].wait(torch.cuda.current_stream())
    return ctx,s,ctx['gate']
def fold_reduce(ctx,a,b,weights,ids,mask_padding,s,g):
    y=reduce_slots(a,b,weights,ids,mask_padding,shared=s,gate=g)
    if not _FOLD_CHECKED[0]:
        _FOLD_CHECKED[0]=True
        ref=reduce_slots(a,b,weights,ids,mask_padding)
        ref=(g*s)+ref  # eager scale (original: sigmoid(gate)*out) then the runner add (shared + routed)
        eq=bool(torch.equal(y.view(torch.int16),ref.view(torch.int16)))
        import json,os
        print('DIRECT_MOE_SHARED_FOLD '+json.dumps({'tag':'selfcheck','equal':eq,'rows':int(ids.shape[0]),'pid':os.getpid()}),flush=True)
        if not eq:
            raise RuntimeError('DIRECT_MOE_SHARED_FOLD selfcheck: folded reduce != eager scale+add')
    ctx['folded']=True
    return y
def fold_finish(s,f,ctx):
    if s is None or ctx['folded']:
        return (s,f)
    g=ctx['gate']
    s2=(g*s) if g is not None else s
    if not isinstance(f,torch.Tensor) or f.shape!=s2.shape or f.dtype!=s2.dtype:
        raise RuntimeError('DIRECT_MOE_SHARED_FOLD: cannot add shared output eagerly (unexpected fused output type/shape)')
    return (s2,s2+f)


# ---- DIRECT D23 dense-merge M1 (decode23-phase1/dense-merge-v1/m1; opt-in DIRECT_D23_MERGE=1) ----
# Pure append. The first fold_begin call (inside the opaque moe_forward_shared op, never traced) installs the MoE trio
# merge (direct_d23_merge.install), then rebinds the module-level name fold_begin to the original function object above.
if __import__("os").environ.get("DIRECT_D23_MERGE") == "1":
    _d23_fold_begin_orig = fold_begin

    def fold_begin(layer):
        global fold_begin
        import direct_d23_merge as _d23m
        _d23m.install()
        fold_begin = _d23_fold_begin_orig
        return _d23_fold_begin_orig(layer)
