"""[nb2-loadfast-v3: lf1 + bounded threaded expert-piece repack, DIRECT_LF3_WORKERS] Isolated candidate: Quark mixed W4A16/W8A16 -> STOCK TritonWNA16Experts.

No GPU kernel source, quantization, BF16 expert expansion or fallback. CPU test
callback is deliberately NOT reachable from apply(). Native GPU admission and
serving qualification are external gates. Original Quark/RDNA3 remains intact.
"""
from dataclasses import dataclass, replace
from direct_moe_reduction import SlotExperts, reduce_slots, fold_shared_for_reduce, fold_reduce
import copy
import hashlib
import json
import torch
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig, should_ignore_layer
from vllm.model_executor.layers.quantization.quark.quark_moe import QuarkMoEMethod
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import make_wna16_moe_quant_config
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4Static,kInt8Static

RECIPE_SHA256='aabce8b815655175f7608a356fba2a2ec1c1396c7bbd622737c7eb54e3240ffd'
ROUTE='quark-mixed-stock-triton-wna16-v1'
# Native file2file calls Pack_8_bits.pack(q[N,K], False). Pack_8_bits
# transposes per_group to signed int8[K,N]; scales are saved without transpose.
# This contract is NOT the distinct ExportBuilder/real_quantizer scale wire.
INT8_WIRE='quark-file2file-per-group-int8-KN-fp32-NG-v1'
INT4_WIRE='quark-file2file-int4-no-reorder-KN8-fp32-NG-v1'
NATIVE_INT8_WIRE='quark-native-export-per-group-int8-KN-fp32-GN-v1'

def select_mixed_config(base):
    """Proposed from_config hook; no metadata preserves the old selection."""
    from pathlib import Path
    selection=base.quant_config.get('mixed_moe')
    if selection is None:return base
    if (not isinstance(selection,dict) or set(selection)!={'route','recipe_path','int8_wire'}
        or selection['route']!=ROUTE):
        raise ValueError('unknown mixed candidate selection; no fallback')
    recipe=Path(selection['recipe_path'])
    if not recipe.is_absolute():raise ValueError('absolute locked recipe path required')
    return QuarkMixedConfig(base,plans_from_recipe(recipe.read_bytes()),
                           int8_wire=selection['int8_wire'],route=selection['route'])

def validate_outer_shared_owner(shared):
    from vllm.model_executor.layers.fused_moe import SharedExperts
    if shared is not None and not isinstance(shared,SharedExperts):
        raise ValueError('shared experts must be owned by stock outer runner')
    # Never execute or drain shared.output here: stock runner owns both.


@dataclass(frozen=True)
class PoolPlan:
    four: tuple[int,...]
    eight: tuple[int,...]
    total: int
    def __post_init__(self):
        object.__setattr__(self,'four',tuple(self.four))
        object.__setattr__(self,'eight',tuple(self.eight))
        ids=self.four+self.eight
        if (type(self.total) is not int or not 1<=self.total<=512
            or any(type(e) is not int for e in ids)
            or len(ids)!=self.total or set(ids)!=set(range(self.total))):
            raise ValueError('duplicate, missing or invalid explicit global expert IDs')
    def locate(self,e):
        if type(e) is not int or not 0<=e<self.total:
            raise ValueError('invalid global expert ID')
        return (4,self.four.index(e)) if e in self.four else (8,self.eight.index(e))

def plans_from_recipe(raw: bytes):
    if hashlib.sha256(raw).hexdigest()!=RECIPE_SHA256:
        raise ValueError('approved recipe hash mismatch')
    r=json.loads(raw)
    return {b['prefix']:PoolPlan(tuple(b['INT4_AWQ']+b['INT4_RTN_unknown']),
                                tuple(b['INT8_AWQ']),512) for b in r['trunk']['blocks']}

class QuarkMixedConfig(QuarkConfig):
    """Explicit per-instance selector. Unselected layers delegate to stock Quark.

    Parent installs this wrapper ONLY for a locked recipe. Supplying PoolPlan
    directly supports bounded integer fixtures, not a claim of recipe approval.
    No registry, environment, global factory or selector is monkey-patched.
    """
    def __init__(self, base, plans, *, int8_wire=None, route=ROUTE):
        if type(base) is not QuarkConfig or route!=ROUTE:
            raise ValueError('explicit stock-triton mixed candidate required; no fallback')
        self.__dict__.update(copy.deepcopy(base.__dict__))
        self.mixed_plans=dict(plans)
        self.int8_wire=int8_wire
        q=self.quant_config;g=q['global_quant_config'];w=g['weight'];e=q['export']
        if (q.get('version')!='0.12.post1' or e.get('file2file_pack_reorder') is not False
            or e.get('weight_format')!='real_quantized' or e.get('pack_method')!='reorder'
            or g.get('input_tensors') is not None or g.get('output_tensors') is not None
            or g.get('bias') is not None
            or any(w.get(k)!=v for k,v in {'dtype':'int4','qscheme':'per_group',
                    'group_size':128,'ch_axis':-1,'symmetric':True,'is_dynamic':False}.items())):
            raise ValueError('unsupported mixed file2file base metadata')
        if any(p.eight for p in plans.values()) and int8_wire not in (INT8_WIRE,NATIVE_INT8_WIRE):
            raise ValueError('INT8 producer wire unavailable or not explicitly bound')
        for prefix,p in self.mixed_plans.items():
            if not isinstance(p,PoolPlan) or self._protected(prefix):
                raise ValueError('protected or invalid mixed pool target')
            resolved=self._find_matched_config(prefix,RoutedExperts)
            if resolved!=g:
                raise ValueError('per-layer overrides need explicit resolution, not silent replacement')
    def _protected(self,prefix):
        return (prefix.startswith('mtp.') or '.visual.' in prefix or prefix.startswith('model.visual.')
                or should_ignore_layer(prefix,ignore=self._consumer_exclusions(),
                     fused_mapping=self.packed_modules_mapping,check_children=True))
    def apply_vllm_mapper(self, mapper):
        super().apply_vllm_mapper(mapper)
        self.mixed_plans = mapper.apply_dict(self.mixed_plans)

    def draft_pool_plan(self, prefix):
        policy = getattr(self, '_mtp_dense_policy', None)
        if policy is None:
            return None
        source = policy.original(prefix, namespace='draft_prefix')
        if source != 'mtp.layers.0.mlp.experts':
            return None
        routed = [s for s in policy.roster if len(s['source_shape']) == 3]
        if len(routed) != 2 or any(s['weight_bits'] != 4 or s['group_size'] != 128 or s['algorithm'] != 'RTN' or s['source_shape'][0] != 512 for s in routed):
            raise ValueError('unapproved MTP routed policy')
        return PoolPlan(tuple(range(512)), (), 512)

    def get_quant_method_target(self,prefix,layer_type):
        if self.draft_pool_plan(prefix) is not None:
            if not issubclass(layer_type,RoutedExperts):
                raise ValueError('draft route requires RoutedExperts')
            return None,None,QuarkMixedMoEMethod
        if prefix in self.mixed_plans:
            if not issubclass(layer_type,RoutedExperts) or self._protected(prefix):
                raise ValueError('mixed route is only for explicit routed experts')
            # No misleading homogeneous precision key for a mixed layer.
            return None,None,QuarkMixedMoEMethod
        return super().get_quant_method_target(prefix,layer_type)
    def get_quant_method(self,layer,prefix):
        draft = self.draft_pool_plan(prefix)
        if draft is not None:
            self.get_quant_method_target(prefix,type(layer))
            return QuarkMixedMoEMethod(layer.moe_config,draft,int8_wire=self.int8_wire)
        if prefix in self.mixed_plans:
            self.get_quant_method_target(prefix,type(layer))
            return QuarkMixedMoEMethod(layer.moe_config,self.mixed_plans[prefix],
                                       int8_wire=self.int8_wire)
        return super().get_quant_method(layer,prefix)

def validate_tail(k,group,config):
    bk=config.get('BLOCK_SIZE_K')
    if type(bk) is not int or bk not in (16,32,64,128,256,512) or k%bk:
        raise ValueError(f'unsafe unmasked native K tail: K={k}, BLOCK_SIZE_K={bk}')
    if k%group:
        raise ValueError('incomplete quantization group')

# ---- LOADFAST v3 (nb2-loadfast-v3): bounded multi-threaded expert-piece repack ----
import os as _lf3_os, threading as _lf3_threading, time as _lf3_time, sys as _lf3_sys
from concurrent.futures import ThreadPoolExecutor as _Lf3Pool
class _Lf3Exec:
    def __init__(self,n,depth):
        self.n=n;self.depth=depth;self.pool=_Lf3Pool(n,thread_name_prefix='lf3-repack')
        self.sem=_lf3_threading.BoundedSemaphore(depth);self.tls=_lf3_threading.local();self.lock=_lf3_threading.Lock()
        self.streams=[];self.futs=[];self.submitted=0;self.wait_s=0.0;self.t_first=None;self.reported=False
    def _run(self,fn):
        try:
            s=getattr(self.tls,'s',None)
            if s is None:
                s=torch.cuda.Stream();self.tls.s=s
                with self.lock:self.streams.append(s)
            with torch.cuda.stream(s):fn()
        finally:self.sem.release()
    def _reap(self):
        keep=[]
        for f in self.futs:
            if f.done():f.result()  # re-raise the first worker error (same exception object/message)
            else:keep.append(f)
        self.futs=keep
    def submit(self,fn):
        if self.t_first is None:self.t_first=_lf3_time.perf_counter()
        t=_lf3_time.perf_counter();self.sem.acquire();self.wait_s+=_lf3_time.perf_counter()-t
        self.futs.append(self.pool.submit(self._run,fn));self.submitted+=1
        if len(self.futs)>=4*self.depth:self._reap()
    def drain(self):
        t=_lf3_time.perf_counter()
        for f in self.futs:f.result()
        self.futs=[]
        for s in list(self.streams):s.synchronize()
        torch.cuda.synchronize()
        if not self.reported and self.submitted:
            self.reported=True
            print(f"DIRECT_LF3 workers={self.n} depth={self.depth} pieces={self.submitted} streams={len(self.streams)} "
                  f"main_wait_s={self.wait_s:.3f} drain_s={_lf3_time.perf_counter()-t:.3f} "
                  f"span_s={_lf3_time.perf_counter()-self.t_first:.3f}",file=_lf3_sys.stderr,flush=True)
_LF3_N=int(_lf3_os.environ.get('DIRECT_LF3_WORKERS','0') or 0)
_LF3=_Lf3Exec(_LF3_N,int(_lf3_os.environ.get('DIRECT_LF3_DEPTH',str(4*max(_LF3_N,1))))) if _LF3_N>0 else None

class _Pool(torch.nn.Module):
    def __init__(self,ids,bits,device):
        super().__init__();self.bits=bits;self.global_ids=ids
        e=len(ids);pack=2 if bits==4 else 1
        for name,shape,dtype in (
            ('w13_weight',(e,1280,2560//pack),torch.uint8),
            ('w2_weight',(e,2560,640//pack),torch.uint8),
            ('w13_scale',(e,1280,20),torch.float32),
            ('w2_scale',(e,2560,5),torch.float32)):
            # Direct final backing storage: preserve logical values and FP32 scales.
            # W4: transpose weights+scales. W8: transpose gate/up weight only.
            e,n,k=shape
            strides=(n*k,1,n) if bits==4 or name=='w13_weight' else (n*k,k,1)
            self.register_parameter(name,torch.nn.Parameter(torch.empty_strided(shape,strides,dtype=dtype,device=device),requires_grad=False))

_SDM_CFG=[]
def _sdm_cfg():
    # Frozen per-process kernel selection from the overlay env (fail-closed if any key is missing).
    if not _SDM_CFG:
        import os,direct_moe_sdm_v2 as _sdm
        _SDM_CFG.append(_sdm.Cfg(os.environ['DIRECT_SDM_HSACO'],os.environ['DIRECT_SDM_UP'],os.environ['DIRECT_SDM_DN'],int(os.environ['DIRECT_SDM_SILUV'])))
        import json as _json,sys as _sys
        c=_SDM_CFG[0]
        print('DIRECT_SDM2_ROUTE_ACTIVE '+_json.dumps({'pid':os.getpid(),'hsaco':c.hsaco,'up':c.up,'dn':c.dn,'siluv':c.siluv,
              'up_grid':list(c.up_grid),'up_block':list(c.up_block),'dn_grid':list(c.dn_grid),'dn_block':list(c.dn_block)},sort_keys=True),file=_sys.stderr,flush=True)
    return _SDM_CFG[0]
class QuarkMixedMoEMethod(QuarkMoEMethod):
    supported_activation_quant_keys=[None]
    supported_weight_quant_keys=[kInt4Static,kInt8Static]
    def __init__(self,moe,plan,*,int8_wire=None):
        # Base validates the two separate pool schemes; no factory rejection is caught.
        super().__init__(moe,kInt4Static,None)
        self.weight_quant_key=None  # deliberately not a homogeneous W4 claim
        self.plan=plan
        self.int8_wire=int8_wire
        p=moe.moe_parallel_config
        if (p.tp_size,p.ep_size,p.dp_size,p.pcp_size,p.sp_size)!=(1,1,1,1,1) or p.enable_eplb or p.use_ep:
            raise ValueError('only single TP/EP/DP/PCP/SP without EPLB supported')
        if (moe.hidden_dim,moe.intermediate_size_per_partition)!=(2560,640) or moe.in_dtype!=torch.bfloat16:
            raise ValueError('requires exact K2560/I640/BF16')
        if (moe.num_experts!=plan.total or moe.num_local_experts!=plan.total
            or moe.num_logical_experts!=plan.total or moe.has_bias
            or moe.activation!=MoEActivation.SILU or moe.w13_num_shards!=2
            or moe.is_lora_enabled or moe.defer_moe_finalize
            or any(v is not None for v in (moe.swiglu_limit,moe.swiglu_alpha,moe.swiglu_beta))):
            raise ValueError('unsupported expert geometry, bias, activation, LoRA or deferred finalize')
        if plan.eight and int8_wire not in (INT8_WIRE,NATIVE_INT8_WIRE):
            raise ValueError('INT8 wire unavailable: refuse orientation guess')
        self._loaded=set();self._ready=False;self.native={};self._params={}
    @property
    def loaded_piece_count(self):return len(self._loaded)
    def maybe_roundup_sizes(self,hidden_size,intermediate_size_per_partition,act_dtype,moe_parallel_config):
        if (hidden_size,intermediate_size_per_partition)!=(2560,640):raise ValueError('no silent padding')
        return hidden_size,intermediate_size_per_partition
    @property
    def skip_forward_padding(self):return True
    def create_weights(self,layer,num_experts,hidden_size,intermediate_size_per_partition,params_dtype,**attrs):
        if (num_experts,hidden_size,intermediate_size_per_partition,params_dtype)!=(self.plan.total,2560,640,torch.bfloat16):
            raise ValueError('wrong mixed layer geometry or dtype')
        if hasattr(layer,'mixed_pools'):raise ValueError('weights already created')
        layer.mixed_pools=torch.nn.ModuleDict({str(b):_Pool(ids,b,self.moe.device)
                              for b,ids in ((4,self.plan.four),(8,self.plan.eight))})
        # Zero-length checkpoint entrypoints preserve vLLM's root parameter names.
        # Real storage is ONLY the two compact pools, never a padded/global expert array.
        for name in ('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale'):
            param=torch.nn.Parameter(torch.empty(0,device=self.moe.device),requires_grad=False)
            param.weight_loader=self.weight_loader;param.quark_parameter_name=name
            layer.register_parameter(name,param);self._params[id(param)]=(layer,name)
        for bits,ids in ((4,self.plan.four),(8,self.plan.eight)):
            lookup=torch.full((self.plan.total,),-1,dtype=torch.long,device=self.moe.device)
            if ids:lookup[torch.tensor(ids,device=self.moe.device)]=torch.arange(len(ids),device=self.moe.device)
            layer.register_buffer('mixed_lookup_'+str(bits),lookup)
        layer.orig_dtype=params_dtype;layer.weight_block_size=None
    def weight_loader(self,param,loaded_weight,weight_name,shard_id,expert_id,return_success=False):
        if self._ready:raise ValueError('late weight load')
        if id(param) not in self._params:raise ValueError('foreign parameter')
        layer,name=self._params[id(param)];bits,local=self.plan.locate(expert_id)
        allowed=('w2',) if name.startswith('w2_') else ('w1','w3')
        if shard_id not in allowed:raise ValueError('projection/parameter mismatch')
        key=(expert_id,shard_id,name.endswith('_scale'))
        if key in self._loaded:raise ValueError('duplicate expert piece')
        n,k=(2560,640) if shard_id=='w2' else (640,2560)
        scale=name.endswith('_scale')
        native_scale=scale and bits==8 and self.int8_wire==NATIVE_INT8_WIRE
        expected_shape=((k//128,n) if native_scale else (n,k//128)) if scale else ((k,n//8) if bits==4 else (k,n))
        expected_dtype=torch.float32 if scale else (torch.int32 if bits==4 else torch.int8)
        if loaded_weight.device.type!='cpu':raise ValueError('loader requires bounded CPU wire tensors')
        if tuple(loaded_weight.shape)!=expected_shape or loaded_weight.dtype!=expected_dtype:
            raise ValueError(f'wrong {bits}-bit wire shape/dtype for {weight_name}')
        # LOADFAST v3: the per-piece integer/scale work below is the lf1 sequence verbatim, wrapped in a closure.
        # DIRECT_LF3_WORKERS=0 runs it inline (lf1 order); >0 runs it on a bounded worker pool (own HIP stream
        # per worker). Each piece writes only its own disjoint pool slice, so completion order cannot change bytes.
        def _lf3_piece():
            if scale:
                if not bool((torch.isfinite(loaded_weight)&(loaded_weight>0)).all()):
                    raise ValueError('scales must be finite and positive')
                converted=loaded_weight.T.contiguous() if native_scale else loaded_weight
                # FP32 preserved: no silent BF16 scale rounding.
            elif bits==4:
                # Decode one integer projection only. Source packs N, target packs K.
                # LOADFAST (nb2-loadfast-v1): identical integer op sequence, run on the pool device after one
                # copy of the packed wire tensor (shift/and/where/sub/add/cast/or are exact on any device).
                w=loaded_weight.to(layer.mixed_pools[str(bits)].w13_weight.device)
                q=torch.stack([(w>>(4*j))&15 for j in range(8)],-1).reshape(k,n)
                signed=torch.where(q>=8,q-16,q).T
                unsigned=(signed+8).to(torch.uint8)
                converted=(unsigned[:,0::2]|(unsigned[:,1::2]<<4)).contiguous()
            else:
                w=loaded_weight.to(layer.mixed_pools[str(bits)].w13_weight.device)
                converted=(w.T.to(torch.int16)+128).to(torch.uint8).contiguous()
            pool=layer.mixed_pools[str(bits)]
            target=getattr(pool,('w2' if shard_id=='w2' else 'w13')+('_scale' if scale else '_weight'))[local]
            if shard_id!='w2':target=target[:640] if shard_id=='w1' else target[640:]
            target.copy_(converted)
        if _LF3 is None:_lf3_piece()
        else:_LF3.submit(_lf3_piece)
        self._loaded.add(key)
        return True if return_success else None
    def process_weights_after_loading(self,layer):
        if _LF3 is not None:_LF3.drain()
        if self._ready or len(self._loaded)!=self.plan.total*6:
            raise ValueError('incomplete or already processed payload: six pieces per expert required')
        for bits,ids in ((4,self.plan.four),(8,self.plan.eight)):
            if not ids:continue
            pool=layer.mixed_pools[str(bits)]
            qc=make_wna16_moe_quant_config(pool.w13_scale,pool.w2_scale,128,bits)
            conf=replace(self.moe,num_experts=len(ids),num_local_experts=len(ids),
                         num_logical_experts=len(ids),experts_per_token=1,moe_backend='triton')
            for name,p in pool.named_parameters():
                e,n,k=p.shape
                want=(n*k,1,n) if bits==4 or name=='w13_weight' else (n*k,k,1)
                if p.stride()!=want or p.storage_offset()!=0 or p.untyped_storage().nbytes()!=p.numel()*p.element_size():
                    raise ValueError('noncompact or unexpected direct MoE layout')
            self.native[bits]=SlotExperts(conf,qc)
            self.native[bits]._direct_n_contiguous=True
            self.native[bits]._direct_warp_policy_enabled=True
            self.native[bits]._direct_scale_broadcast_enabled=True
        self._ready=True
    def get_fused_moe_quant_config(self,layer):
        # Pool quant configs are intentionally separate. No homogeneous layer key.
        if not self._ready:raise ValueError('mixed weights not ready')
        return None
    def _arguments(self,layer,bits,x,global_ids):
        pool=layer.mixed_pools[str(bits)];native=self.native[bits];m=x.shape[0]
        t=global_ids.shape[1] if global_ids.ndim==2 else 1
        w13,w2,out=native.workspace_shapes(m,1280,2560,t,self.plan.total,
                                         pool.w13_weight.shape[0],None,MoEActivation.SILU)
        workspace2=torch.empty(w2,dtype=x.dtype,device=x.device)
        output=(workspace2.view(-1)[:m*t*2560].view(m*t,2560) if t>1 or m>=640 else torch.empty(out,device=x.device,dtype=x.dtype))
        return dict(output=output,hidden_states=x,
            w1=pool.w13_weight,w2=pool.w2_weight,
            topk_weights=torch.ones((m,t),dtype=torch.float32,device=x.device),
            topk_ids=global_ids.reshape(m,t).contiguous(),activation=MoEActivation.SILU,
            global_num_experts=self.plan.total,
            expert_map=getattr(layer,'mixed_lookup_'+str(bits)),
            a1q_scale=None,a2_scale=None,
            workspace13=torch.empty(w13,dtype=x.dtype,device=x.device),
            workspace2=workspace2,
            expert_tokens_meta=None,apply_router_weight_on_input=False)
    def _route(self,layer,x,weights,ids,invoke):
        if not self._ready:raise ValueError('mixed weights not ready')
        # Private scalar-only successor; original fallback below is verbatim.
        if x.device.type=='cuda' and x.shape==(1,2560) and self.plan.total==512 and set(self.native)=={4,8}:
            import direct_scalar_route as _scalar
            if _scalar.supported(x,ids,weights):
                # decode23 moe-hip-v2: bit-exact HIP WMMA scalar route, 2 launches (in-kernel fail-closed guards, fused SiLU, last-arriver reduce+shared fold).
                import direct_moe_sdm_v2 as _sdm
                return _sdm.run(self,layer,x,weights,ids,_sdm_cfg())
        # Opt-in multi-row verification only. Scalar M=1 and the fallback remain unchanged.
        import os as _os
        if _os.environ.get('DIRECT_SDM3') == '1' and x.ndim == 2 and 2 <= x.shape[0] <= 5 and self.plan.total == 512 and set(self.native) == {4,8}:
            import direct_moe_sdm3 as _multi
            if _multi.supported(x,ids,weights):
                from direct_sdm3_model import run_checked
                return run_checked(self,layer,x,weights,ids,invoke)
        if (x.ndim!=2 or x.shape[1]!=2560 or x.dtype!=torch.bfloat16
            or ids.ndim!=2 or ids.shape[0]!=x.shape[0] or ids.dtype not in (torch.int32,torch.int64)
            or weights.shape!=ids.shape or weights.dtype not in (torch.float32,torch.float16,torch.bfloat16)
            or ids.device!=x.device or weights.device!=x.device or ids.shape[1]>self.plan.total):
            raise ValueError('invalid routing tensor geometry/dtype/device')
        # Native V2 routing emits -1 for padded tokens. Only the framework's
        # explicit per-token mask authorizes that sentinel; live/undeclared
        # negative IDs retain the original fail-closed contract.
        from vllm.forward_context import get_forward_context,is_forward_context_available
        padding=get_forward_context().is_padding if is_forward_context_available() else None
        valid=(ids>=0)&(ids<self.plan.total)
        if padding is not None:
            if (not isinstance(padding,torch.Tensor) or padding.ndim!=1
                or padding.dtype!=torch.bool or padding.device!=x.device
                or padding.shape[0]<x.shape[0]):
                raise ValueError('invalid native padding mask geometry/dtype/device')
            valid=valid|((ids==-1)&padding[:x.shape[0],None])
        torch._assert_async(valid.all(),'invalid global expert ID')
        torch._assert_async(torch.isfinite(weights).all(),'nonfinite router weights')
        sorted_ids=ids.sort(dim=1).values
        torch._assert_async(((sorted_ids[:,1:]!=sorted_ids[:,:-1])|(sorted_ids[:,1:]==-1)).all(),'duplicate top-k ID')
        m,t=ids.shape
        if m*t==0:return torch.zeros_like(x)
        direct_input=(x.device.type=='cuda' and m>=64 and 0<t<=10)
        expanded=x if direct_input else x[:,None,:].expand(m,t,2560).reshape(m*t,2560).contiguous()
        slots=None;pool_outputs=[]
        fused_reduce=(x.device.type=='cuda' and m>=64 and 0<t<=10 and weights.is_contiguous() and ids.is_contiguous())
        flat=ids.reshape(-1).long().clamp(0,self.plan.total-1)
        if fused_reduce and direct_input:
            # Unified-pool native WMMA prefill route (benchmark arm 'wmma5' only); None keeps the original path.
            import direct_moe_unified_v1 as _unified
            if _unified.active(m,t):
                slots_u=_unified.run(layer,expanded,flat.reshape(m,t),self.native)
                if slots_u is not None:
                    return self._direct_fold_or_reduce(slots_u,None,weights,ids,padding is not None,m,x)
        for bits in (4,8):
            if bits not in self.native:continue
            lookup=getattr(layer,'mixed_lookup_'+str(bits))
            if lookup.device!=x.device:raise ValueError('weights/lookup not on activation device')
            # Alignment maps the other precision's experts to -1; both packed
            # GEMMs write zeros for those slots without dynamic compaction.
            kw=self._arguments(layer,bits,expanded,flat.reshape(m,t) if direct_input else flat)
            invoke(bits,self.native[bits],kw)
            if fused_reduce:pool_outputs.append(kw['output'])
            elif slots is None:slots=kw['output']
            else:slots.add_(kw['output'])
        if fused_reduce:
            return self._direct_fold_or_reduce(pool_outputs[0],pool_outputs[1] if len(pool_outputs)==2 else None,weights,ids,padding is not None,m,x)
        # Native padded -1 IDs were clamped for alignment. Discard their
        # computed values, including NaNs, before the router-weight reduction.
        if padding is not None:
            slots.masked_fill_((ids.reshape(-1)<0).unsqueeze(-1),0)
        # Kernels saw unit weights, so returned per-slot values are UNWEIGHTED
        # w.r.t. the original router. Exactly one original-weight multiplication.
        return (slots.reshape(m,t,2560).float()*weights.float().unsqueeze(-1)).sum(1).to(x.dtype)
    def _sdm3_reference(self,layer,x,weights,ids,invoke):
        if not self._ready:raise ValueError('mixed weights not ready')
        # Private scalar-only successor; original fallback below is verbatim.
        if x.device.type=='cuda' and x.shape==(1,2560) and self.plan.total==512 and set(self.native)=={4,8}:
            import direct_scalar_route as _scalar
            if _scalar.supported(x,ids,weights):
                # decode23 moe-hip-v2: bit-exact HIP WMMA scalar route, 2 launches (in-kernel fail-closed guards, fused SiLU, last-arriver reduce+shared fold).
                import direct_moe_sdm_v2 as _sdm
                return _sdm.run(self,layer,x,weights,ids,_sdm_cfg())
        if (x.ndim!=2 or x.shape[1]!=2560 or x.dtype!=torch.bfloat16
            or ids.ndim!=2 or ids.shape[0]!=x.shape[0] or ids.dtype not in (torch.int32,torch.int64)
            or weights.shape!=ids.shape or weights.dtype not in (torch.float32,torch.float16,torch.bfloat16)
            or ids.device!=x.device or weights.device!=x.device or ids.shape[1]>self.plan.total):
            raise ValueError('invalid routing tensor geometry/dtype/device')
        # Native V2 routing emits -1 for padded tokens. Only the framework's
        # explicit per-token mask authorizes that sentinel; live/undeclared
        # negative IDs retain the original fail-closed contract.
        from vllm.forward_context import get_forward_context,is_forward_context_available
        padding=get_forward_context().is_padding if is_forward_context_available() else None
        valid=(ids>=0)&(ids<self.plan.total)
        if padding is not None:
            if (not isinstance(padding,torch.Tensor) or padding.ndim!=1
                or padding.dtype!=torch.bool or padding.device!=x.device
                or padding.shape[0]<x.shape[0]):
                raise ValueError('invalid native padding mask geometry/dtype/device')
            valid=valid|((ids==-1)&padding[:x.shape[0],None])
        torch._assert_async(valid.all(),'invalid global expert ID')
        torch._assert_async(torch.isfinite(weights).all(),'nonfinite router weights')
        sorted_ids=ids.sort(dim=1).values
        torch._assert_async(((sorted_ids[:,1:]!=sorted_ids[:,:-1])|(sorted_ids[:,1:]==-1)).all(),'duplicate top-k ID')
        m,t=ids.shape
        if m*t==0:return torch.zeros_like(x)
        direct_input=(x.device.type=='cuda' and m>=64 and 0<t<=10)
        expanded=x if direct_input else x[:,None,:].expand(m,t,2560).reshape(m*t,2560).contiguous()
        slots=None;pool_outputs=[]
        fused_reduce=(x.device.type=='cuda' and m>=64 and 0<t<=10 and weights.is_contiguous() and ids.is_contiguous())
        flat=ids.reshape(-1).long().clamp(0,self.plan.total-1)
        if fused_reduce and direct_input:
            # Unified-pool native WMMA prefill route (benchmark arm 'wmma5' only); None keeps the original path.
            import direct_moe_unified_v1 as _unified
            if _unified.active(m,t):
                slots_u=_unified.run(layer,expanded,flat.reshape(m,t),self.native)
                if slots_u is not None:
                    return self._direct_fold_or_reduce(slots_u,None,weights,ids,padding is not None,m,x)
        for bits in (4,8):
            if bits not in self.native:continue
            lookup=getattr(layer,'mixed_lookup_'+str(bits))
            if lookup.device!=x.device:raise ValueError('weights/lookup not on activation device')
            # Alignment maps the other precision's experts to -1; both packed
            # GEMMs write zeros for those slots without dynamic compaction.
            kw=self._arguments(layer,bits,expanded,flat.reshape(m,t) if direct_input else flat)
            invoke(bits,self.native[bits],kw)
            if fused_reduce:pool_outputs.append(kw['output'])
            elif slots is None:slots=kw['output']
            else:slots.add_(kw['output'])
        if fused_reduce:
            return self._direct_fold_or_reduce(pool_outputs[0],pool_outputs[1] if len(pool_outputs)==2 else None,weights,ids,padding is not None,m,x)
        # Native padded -1 IDs were clamped for alignment. Discard their
        # computed values, including NaNs, before the router-weight reduction.
        if padding is not None:
            slots.masked_fill_((ids.reshape(-1)<0).unsqueeze(-1),0)
        # Kernels saw unit weights, so returned per-slot values are UNWEIGHTED
        # w.r.t. the original router. Exactly one original-weight multiplication.
        return (slots.reshape(m,t,2560).float()*weights.float().unsqueeze(-1)).sum(1).to(x.dtype)
    def _native_call(self,bits,native,kw):
        from vllm.model_executor.layers.fused_moe import get_config
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config,get_moe_wna16_block_config
        if get_config() is not None:raise ValueError('global kernel overrides not allowed on isolated route')
        m=kw['hidden_states'].shape[0]
        config=try_get_optimal_moe_config(kw['w1'].shape,kw['w2'].shape,1,
                    native.quant_config.config_name(kw['hidden_states'].dtype),m,block_shape=native.block_shape)
        for k,n in ((2560,1280),(640,2560)):
            effective=dict(config)
            effective.update(get_moe_wna16_block_config(config=config,use_moe_wna16_cuda=False,
                num_valid_tokens=m,size_k=k,size_n=n,num_experts=kw['w1'].shape[0],group_size=128,
                real_top_k=1,block_size_m=config['BLOCK_SIZE_M']))
            validate_tail(k,128,effective)
        # The exact unmodified installed class performs alignment, both GEMMs,
        # SiLU and topk1 reduction. No replacement kernels or emulation path.
        native.apply(**kw)
    def _direct_fold_or_reduce(self,a,b,weights,ids,mask_padding,m,x):
        # DIRECT_MOE_SHARED_FOLD: peek (never drain) the runner-owned shared output for the reduce epilogue.
        fs=fold_shared_for_reduce(getattr(self,'_direct_fold_shared',None))
        if fs is not None:
            ctx,s,g=fs
            if (s.shape==(m,2560) and s.is_contiguous() and s.dtype==torch.bfloat16 and s.device==x.device
                    and g.shape==(m,1) and g.is_contiguous() and g.dtype==torch.bfloat16):
                return fold_reduce(ctx,a,b,weights,ids,mask_padding,s,g)
        return reduce_slots(a,b,weights,ids,mask_padding)
    def apply(self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
        if x.device.type!='cuda':raise ValueError('production candidate requires GPU/CUDA tensor; no CPU fallback')
        if layer.apply_router_weight_on_input:raise ValueError('router-on-input is unsupported')
        # supports_internal_mk=False: stock FusedMoE runner owns SharedExperts.
        # Never run/add a shared expert once per precision pool.
        validate_outer_shared_owner(shared_experts)
        self._direct_fold_shared=shared_experts
        try:
            return self._route(layer,x,topk_weights,topk_ids,self._native_call)
        finally:
            self._direct_fold_shared=None
    def apply_cpu_test(self,layer,x,weights,ids,callback,shared_callback=None):
        """Explicit CPU test boundary. Captures native arguments; NOT dispatch."""
        if x.device.type!='cpu':raise ValueError('test callback is CPU-only')
        out=self._route(layer,x,weights,ids,callback)
        if shared_callback is not None:out=out+shared_callback(x)
        return out
