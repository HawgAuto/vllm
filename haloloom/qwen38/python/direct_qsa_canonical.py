"""Exact-score deterministic QSA selection rebuild, no host scalar reads.
Standalone opt-in candidate. Does not change logits or top-k cutoff. Ties choose
lowest logical column, matching the retained Torch canonical reference. Not a
claim of equality to the nondeterministic native selected set or model quality.
"""
import torch
from vllm.triton_utils import triton,tl

@triton.jit
def _count(B,L,V,C,T,S,O,bs0:tl.constexpr,bs1:tl.constexpr,ls0:tl.constexpr,ls1:tl.constexpr,vs:tl.constexpr,W:tl.constexpr,K:tl.constexpr,NT:tl.constexpr,BK:tl.constexpr,BT:tl.constexpr):
 row=tl.program_id(0);tile=tl.program_id(1)
 v=tl.minimum(tl.maximum(tl.load(V+row*vs),0),W)
 ki=tl.arange(0,BK)
 sel=tl.load(B+row*bs0+ki*bs1,ki<K,other=-1)
 chosen=(ki<K)&(sel>=0)&(sel<v)
 picked=tl.load(L+row*ls0+sel*ls1,chosen,other=float('inf'))
 kth=tl.min(picked,0);nsel=tl.sum(chosen.to(tl.int32),0)
 cols=tile*BT+tl.arange(0,BT)
 valid=(cols<W)&(cols<v)
 score=tl.load(L+row*ls0+cols*ls1,valid,other=float('-inf'))
 greater=valid&(score>kth);equal=valid&(score==kth)
 tl.store(C+(row*NT+tile)*2,tl.sum(greater.to(tl.int32),0))
 tl.store(C+(row*NT+tile)*2+1,tl.sum(equal.to(tl.int32),0))
 if tile==0:
  tl.store(T+row,kth);tl.store(S+row,nsel)
  tl.store(O+row*K+ki,-1,ki<K)

@triton.jit
def _scatter(L,V,C,T,S,O,ls0:tl.constexpr,ls1:tl.constexpr,vs:tl.constexpr,W:tl.constexpr,K:tl.constexpr,NT:tl.constexpr,CT:tl.constexpr,BT:tl.constexpr):
 row=tl.program_id(0);tile=tl.program_id(1)
 v=tl.minimum(tl.maximum(tl.load(V+row*vs),0),W)
 tiles=tl.arange(0,CT)
 cg=tl.load(C+(row*NT+tiles)*2,tiles<NT,other=0)
 ce=tl.load(C+(row*NT+tiles)*2+1,tiles<NT,other=0)
 need=tl.maximum(tl.load(S+row)-tl.sum(cg,0),0)
 prior_g=tl.sum(tl.where(tiles<tile,cg,0),0)
 prior_e=tl.sum(tl.where(tiles<tile,ce,0),0)
 cols=tile*BT+tl.arange(0,BT);valid=(cols<W)&(cols<v)
 score=tl.load(L+row*ls0+cols*ls1,valid,other=float('-inf'))
 kth=tl.load(T+row)
 greater=valid&(score>kth);equal=valid&(score==kth)
 eq_rank=prior_e+tl.cumsum(equal.to(tl.int32),0)
 take=greater|(equal&(eq_rank<=need))
 rank=prior_g+tl.minimum(prior_e,need)+tl.cumsum(take.to(tl.int32),0)-1
 tl.store(O+row*K+rank,cols,take&(rank>=0)&(rank<K))

@triton.jit
def _kth(B,L,V,T,S,O,bs0:tl.constexpr,bs1:tl.constexpr,ls0:tl.constexpr,ls1:tl.constexpr,vs:tl.constexpr,W:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
 # [nb2-canon2-ab-v1] _count's per-row prologue verbatim (kth, nsel, O=-1), once per row instead of once per 1024-column tile
 row=tl.program_id(0)
 v=tl.minimum(tl.maximum(tl.load(V+row*vs),0),W)
 ki=tl.arange(0,BK)
 sel=tl.load(B+row*bs0+ki*bs1,ki<K,other=-1)
 chosen=(ki<K)&(sel>=0)&(sel<v)
 picked=tl.load(L+row*ls0+sel*ls1,chosen,other=float('inf'))
 kth=tl.min(picked,0);nsel=tl.sum(chosen.to(tl.int32),0)
 tl.store(T+row,kth);tl.store(S+row,nsel)
 tl.store(O+row*K+ki,-1,ki<K)

@triton.jit
def _count2(L,V,C,T,ls0:tl.constexpr,ls1:tl.constexpr,vs:tl.constexpr,W:tl.constexpr,NT:tl.constexpr,BT:tl.constexpr):
 # [nb2-canon2-ab-v1] _count's per-tile body verbatim, kth read from T[row] (written by _kth from the identical expression)
 row=tl.program_id(0);tile=tl.program_id(1)
 v=tl.minimum(tl.maximum(tl.load(V+row*vs),0),W)
 kth=tl.load(T+row)
 cols=tile*BT+tl.arange(0,BT)
 valid=(cols<W)&(cols<v)
 score=tl.load(L+row*ls0+cols*ls1,valid,other=float('-inf'))
 greater=valid&(score>kth);equal=valid&(score==kth)
 tl.store(C+(row*NT+tile)*2,tl.sum(greater.to(tl.int32),0))
 tl.store(C+(row*NT+tile)*2+1,tl.sum(equal.to(tl.int32),0))

import os as _c2_os, sys as _c2_sys, json as _c2_json
_C2_MIN_NT=int(_c2_os.environ.get('DIRECT_QSA_CANON2_MIN_NT','3'))
_C2_CHECKED=[False]
_C2_USED=[0]

def _c2_on():
 f=_c2_os.environ.get('DIRECT_QSA_CANON2')
 if f is not None:return f=='1'
 try:
  import direct_nb1 as _nb
  return _nb.nb_level()>=8
 except Exception:
  return False

def _c2_log(tag,**kw):
 kw['tag']=tag;kw['pid']=_c2_os.getpid()
 print('DIRECT_QSA_CANON2 '+_c2_json.dumps(kw,sort_keys=True),file=_c2_sys.stderr,flush=True)

def _canon_v1(blocks,logits,visible_blocks,rows,k,w,bt,nt):
 counts=torch.empty((rows,nt,2),device=blocks.device,dtype=torch.int32)
 threshold=torch.empty(rows,device=blocks.device,dtype=torch.float32)
 nsel=torch.empty(rows,device=blocks.device,dtype=torch.int32)
 output=torch.empty((rows,k),device=blocks.device,dtype=torch.int32)
 _count[(rows,nt)](blocks,logits,visible_blocks,counts,threshold,nsel,output,*blocks.stride(),*logits.stride(),visible_blocks.stride(0),w,k,nt,triton.next_power_of_2(k),bt,num_warps=4)
 _scatter[(rows,nt)](logits,visible_blocks,counts,threshold,nsel,output,*logits.stride(),visible_blocks.stride(0),w,k,nt,triton.next_power_of_2(nt),bt,num_warps=4)
 return output,counts,threshold,nsel

def _canon_v2(blocks,logits,visible_blocks,rows,k,w,bt,nt):
 counts=torch.empty((rows,nt,2),device=blocks.device,dtype=torch.int32)
 threshold=torch.empty(rows,device=blocks.device,dtype=torch.float32)
 nsel=torch.empty(rows,device=blocks.device,dtype=torch.int32)
 output=torch.empty((rows,k),device=blocks.device,dtype=torch.int32)
 _kth[(rows,)](blocks,logits,visible_blocks,threshold,nsel,output,*blocks.stride(),*logits.stride(),visible_blocks.stride(0),w,k,triton.next_power_of_2(k),num_warps=4)
 _count2[(rows,nt)](logits,visible_blocks,counts,threshold,*logits.stride(),visible_blocks.stride(0),w,nt,bt,num_warps=4)
 _scatter[(rows,nt)](logits,visible_blocks,counts,threshold,nsel,output,*logits.stride(),visible_blocks.stride(0),w,k,nt,triton.next_power_of_2(nt),bt,num_warps=4)
 return output,counts,threshold,nsel

def canonicalize_selected_blocks(blocks,logits,visible_blocks):
 if blocks.dtype!=torch.int32 or blocks.ndim!=2:raise ValueError('int32 rank2 block indices required')
 if logits.dtype!=torch.float32 or logits.ndim!=2 or logits.shape[0]!=blocks.shape[0]:raise ValueError('aligned float32 rank2 logits required')
 if visible_blocks.dtype!=torch.int32 or visible_blocks.shape!=(blocks.shape[0],):raise ValueError('one int32 visible count per row required')
 if not blocks.is_cuda or logits.device!=blocks.device or visible_blocks.device!=blocks.device:raise ValueError('same GPU required')
 rows,k=blocks.shape;w=logits.shape[1]
 if not rows or not k:return blocks
 if w<1:raise ValueError('nonempty score width required')
 bt=1024;nt=triton.cdiv(w,bt)
 if nt>=_C2_MIN_NT and _c2_on():  # nt<3: v1 already computes kth in <=2 tiles (canon2-ab-01 c2 -0.33%)
  output,counts,threshold,nsel=_canon_v2(blocks,logits,visible_blocks,rows,k,w,bt,nt)
  _C2_USED[0]+=1
  if not _C2_CHECKED[0]:
   o1,c1,t1,n1=_canon_v1(blocks,logits,visible_blocks,rows,k,w,bt,nt)
   ok=bool(torch.equal(o1,output)) and bool(torch.equal(c1,counts)) and bool(torch.equal(t1.view(torch.int32),threshold.view(torch.int32))) and bool(torch.equal(n1,nsel))
   _c2_log('selfcheck',equal=ok,rows=int(rows),k=int(k),w=int(w),nt=int(nt))
   if not ok:raise RuntimeError('DIRECT_QSA_CANON2 selfcheck mismatch vs canonical v1')
   _C2_CHECKED[0]=True
  elif _C2_USED[0] in (2,100):
   _c2_log('used_%d'%_C2_USED[0],rows=int(rows),w=int(w),nt=int(nt))
 else:
  output,counts,threshold,nsel=_canon_v1(blocks,logits,visible_blocks,rows,k,w,bt,nt)
 blocks.copy_(output)
 return blocks
