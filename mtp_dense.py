"""Isolated MTP packed dense bridge. Not GPU/serving-qualified.

Production apply calls ONLY the pinned stock fused_moe_kernel_gptq_awq GEMM,
with one expert/topk1 and no activation/router multiplication. No full-weight
BF16 expansion. dequant_oracle is CPU TEST ONLY and is never called by apply.
Native INT8 file2file tensors use signed [K,N] codes and FP32 [N,G] scales.
"""
from pathlib import Path
from functools import lru_cache
import hashlib,json,re,copy
import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearMethodBase,UnquantizedLinearMethod

RECIPE_SHA='023bddc128e45d56c9fc9bc8dae56dd213d3647b95739c1a23b059ae01c515d8'
KERNEL_SHA='21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7'
WIRE='canonical-offset-k-packed-v1'

class Policy:
    @classmethod
    def from_recipe(cls,path,num_hidden_layers):
        data=Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=RECIPE_SHA: raise ValueError('recipe hash')
        return cls(json.loads(data)['MTP']['tensors'],num_hidden_layers)

    def __init__(self,roster,num_hidden_layers):
        if type(num_hidden_layers) is not int or num_hidden_layers<1: raise ValueError('num_hidden_layers')
        self.start=num_hidden_layers
        self.roster=copy.deepcopy(roster)
        if len(roster)!=31 or len({x['name'] for x in roster})!=31: raise ValueError('approved 31 tensor roster required')
        self.dense=[s for s in self.roster if len(s['source_shape'])==2 and s['weight_bits'] in (4,8)]
        if len(self.dense)!=19:raise ValueError('19 dense matrices required')
        self.by_name={s['name']:s for s in self.roster}

    def map_source(self,name,*,namespace):
        if not name.startswith('mtp.'):raise ValueError('not an original MTP name')
        if namespace not in ('draft_prefix','draft_module'):raise ValueError('namespace')
        result='model.'+name[4:]
        if namespace=='draft_prefix':
            result=re.sub(r'^model\.layers\.(\d+)\.',lambda m:f'model.layers.{self.start+int(m[1])}.',result)
        return result

    def original(self,name,*,namespace):
        if namespace=='source':return name if name.startswith('mtp.') else None
        if namespace not in ('draft_prefix','draft_module'):raise ValueError('namespace')
        if not name.startswith(('model.','mtp.')):return None
        result='mtp.'+name.split('.',1)[1]
        if namespace=='draft_prefix':
            match=re.match(r'mtp\.layers\.(\d+)\.',result)
            if match:
                index=int(match[1])-self.start
                if index!=0:return None
                result=result.replace(f'.layers.{match[1]}.',f'.layers.{index}.',1)
        return result

    def resolve(self,prefix,*,namespace):
        name=self.original(prefix,namespace=namespace)
        if name is None:return None
        name=name.removesuffix('.weight')
        shards=None
        if name.endswith('.self_attn.qkv_proj'):shards=['q_proj','k_proj','v_proj']
        elif name.endswith('.mlp.shared_expert.gate_up_proj'):shards=['gate_proj','up_proj']
        names=[name+'.weight'] if shards is None else [name.rsplit('.',1)[0]+'.'+s+'.weight' for s in shards]
        specs=[self.by_name.get(n) for n in names]
        if any(s is None or s not in self.dense for s in specs):return None
        return copy.deepcopy(specs)

    def hc_plan(self,prefix,*,namespace):
        down=self.resolve(prefix+'.input_mix_weight_down',namespace=namespace)
        up=self.resolve(prefix+'.input_mix_weight_up',namespace=namespace)
        if down is None or up is None:return None
        result={'padding':0,'input_mix_weight_down':down[0],'input_mix_weight_up':up[0]}
        inject=self.resolve(prefix+'.block_inject_weight',namespace=namespace)
        if inject:result['block_inject_weight']=inject[0]
        return result


def metadata(spec):
    return {'wire':WIRE,'source_name':spec['name'],'source_shape':list(spec['source_shape']),
            'weight_bits':spec['weight_bits'],'group_size':spec['group_size'],
            'algorithm':'RTN','symmetric':True,'input_tensors':None,'output_tensors':None,
            'scale_storage_dtype':'FP32','compute_dtype':'BF16','native_int8_producer_verified':True}


def validate_metadata(spec,meta):
    if meta!=metadata(spec):raise ValueError('quant metadata mismatch: wire/group/activation/source')


def _scales(scales,n,k,group):
    if k%group or tuple(scales.shape)!=(n,k//group):raise ValueError('scale shape/group')
    if scales.dtype!=torch.float32:raise ValueError('scale dtype must be FP32')
    if not bool((torch.isfinite(scales)&(scales>0)).all()):raise ValueError('scale must be finite and positive')
    # FP32 scales are retained; reject weights that could overflow BF16 tiles.
    if not bool((scales <= torch.finfo(torch.bfloat16).max/128).all()):raise ValueError('scale tile overflow')


def pack_signed(q,bits):
    if bits not in (4,8) or q.ndim!=2 or q.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64):raise ValueError('signed integer matrix required')
    offset=1<<(bits-1)
    if q.numel()==0 or not bool(((q.to(torch.int64)>=-offset)&(q.to(torch.int64)<offset)).all()):raise ValueError('integer range')
    u=(q.to(torch.int16)+offset).to(torch.uint8)
    if bits==8:return u.contiguous()
    if q.shape[1]%2:raise ValueError('K even required')
    return (u[:,0::2]|(u[:,1::2]<<4)).contiguous()


def dequant_oracle(packed,scales,bits,group):
    """CPU-only oracle, not reachable from the production method."""
    if packed.device.type!='cpu' or scales.device.type!='cpu':raise ValueError('CPU oracle only')
    if bits==4:q=torch.stack((packed&15,packed>>4),dim=-1).reshape(packed.shape[0],-1).float()-8
    elif bits==8:q=packed.float()-128
    else:raise ValueError('bits')
    _scales(scales,*q.shape,group)
    return q*scales.repeat_interleave(group,dim=1)


def convert_quark_int4(packed,scales,*,n,k,group):
    """Qualified native file2file conversion, including explicit logical-N trim."""
    from native_wire_binding import convert_native
    if group!=128:raise ValueError('source group128 required')
    return convert_native(packed,scales,n=n,k=k,bits=4,group=group)


def convert_native_int8(packed,scales,*,n,k,group):
    from native_wire_binding import convert_native
    if group!=64:raise ValueError('dense source group64 required')
    return convert_native(packed,scales,n=n,k=k,bits=8,group=group)


class MTPDenseMethod(LinearMethodBase):
    """TP=1 bounded adapter. Fused QKV/gate-up keep explicit logical row maps."""
    def __init__(self,specs,*,test_recorder=None):
        self.specs=copy.deepcopy(specs);self.test_recorder=test_recorder
        if not specs:raise ValueError('missing specs')
        for s in specs:
            if s['weight_bits'] not in (4,8):raise ValueError('bits')
            if s['group_size']!=(64 if s['weight_bits']==8 else 128):raise ValueError('group')
            if s['algorithm']!='RTN' or s['activations']!='BF16' or not s['symmetric']:raise ValueError('recipe semantics')
            n,k=s['source_shape']
            if min(n,k)<=0 or k%s['group_size']:raise ValueError('group divisibility')
        self.bits=specs[0]['weight_bits'];self.group=specs[0]['group_size'];self.k=specs[0]['source_shape'][1]
        if any((s['weight_bits'],s['group_size'],s['source_shape'][1])!=(self.bits,self.group,self.k) for s in specs):raise ValueError('mixed fused dense formats')
        self.row_map=[];start=0
        for s in specs:
            n=s['source_shape'][0];self.row_map.append((s['name'],start,n));start+=n
        self.n=start
        self.shards=(['q','k','v'] if len(specs)==3 else list(range(len(specs))))
        self.layer=None;self.loaded=set();self.ready=False

    def create_weights(self,layer,input_size_per_partition,output_partition_sizes,input_size,output_size,params_dtype,**extra_weight_attrs):
        if (input_size_per_partition,input_size,output_size)!=(self.k,self.k,self.n) or output_partition_sizes!=[s['source_shape'][0] for s in self.specs] or getattr(layer,'tp_size',1)!=1:raise ValueError('TP/partition geometry not supported')
        if params_dtype!=torch.bfloat16:raise ValueError('BF16 activation required')
        if self.layer is not None:raise ValueError('method already owns a layer')
        self.layer=layer
        for name,shape,dtype in [('weight',(self.n,self.k if self.bits==8 else self.k//2),torch.uint8),('weight_scale',(self.n,self.k//self.group),torch.float32)]:
            p=nn.Parameter(torch.empty(shape,dtype=dtype),requires_grad=False)
            p.weight_loader=self.weight_loader;p.mtp_piece=name
            layer.register_parameter(name,p)

    def weight_loader(self,param,loaded_weight,shard_id=None):
        if self.ready:raise ValueError('duplicate or reload after finalization')
        if self.layer is None or param is not getattr(self.layer,getattr(param,'mtp_piece',''),None):raise ValueError('foreign parameter')
        piece=param.mtp_piece
        if shard_id is None:
            if len(self.specs)!=1:raise ValueError('explicit logical shard required')
            idx=0
        else:
            if shard_id not in self.shards:raise ValueError('unknown shard')
            idx=self.shards.index(shard_id)
        key=(piece,idx)
        if key in self.loaded:raise ValueError('duplicate piece')
        _,start,n=self.row_map[idx];dst=param.data[start:start+n]
        if loaded_weight.dtype!=param.dtype:raise ValueError('payload dtype')
        if loaded_weight.shape!=dst.shape:raise ValueError('payload shape')
        if loaded_weight.device.type=='meta':raise ValueError('meta payload')
        if piece=='weight_scale':_scales(loaded_weight,n,self.k,self.group)
        dst.copy_(loaded_weight)
        self.loaded.add(key)

    def process_weights_after_loading(self,layer):
        expected={(p,i) for p in ('weight','weight_scale') for i in range(len(self.specs))}
        if layer is not self.layer or self.loaded!=expected:raise ValueError('missing weight/scale pieces')
        _scales(layer.weight_scale,self.n,self.k,self.group)
        if self.test_recorder is None:
            native_kernel()
        self.ready=True

    @staticmethod
    def launch_config(k,block_k=64):
        # Stock B load is UNMASKED. Group divisibility alone is not sufficient.
        if block_k!=64 or k%block_k:raise ValueError('K must be divisible by supported BLOCK_SIZE_K=64')
        return {'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':32,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':1,'SPLIT_K':1,'num_warps':4,'num_stages':2}

    def apply(self,layer,x,bias=None):
        if layer is not self.layer or not self.ready:raise ValueError('missing/finalization required')
        if x.dtype!=torch.bfloat16:raise ValueError('BF16 activations required; not W8A8')
        if x.ndim<2 or x.shape[-1]!=self.k:raise ValueError('input shape')
        if tuple(layer.weight.shape)!=(self.n,self.k if self.bits==8 else self.k//2) or tuple(layer.weight_scale.shape)!=(self.n,self.k//self.group):raise ValueError('resident shape changed')
        if layer.weight.dtype!=torch.uint8 or layer.weight_scale.dtype!=torch.float32 or not layer.weight.is_contiguous() or not layer.weight_scale.is_contiguous():raise ValueError('resident dtype/layout changed')
        if bias is not None:raise ValueError('approved matrices have no bias')
        if x.device!=layer.weight.device or x.device!=layer.weight_scale.device:raise ValueError('device mismatch')
        if self.test_recorder is None and x.device.type!='cuda':raise RuntimeError('GPU dispatch only; CPU oracle is not production fallback')
        if self.test_recorder is not None and (not isinstance(self.test_recorder,TestCallRecorder) or x.device.type!='cpu'):raise ValueError('explicit CPU test recorder only')
        torch._assert_async(torch.isfinite(x).all(), 'activations must be finite')
        a=x.reshape(-1,self.k).contiguous();m=a.shape[0]
        out=torch.empty((m,1,self.n),dtype=x.dtype,device=x.device)
        if m==0:return out.reshape(*x.shape[:-1],self.n)
        cfg=self.launch_config(self.k)
        bm=cfg['BLOCK_SIZE_M'];em=((m+bm-1)//bm)*bm
        ids=torch.arange(em,dtype=torch.int32,device=x.device)
        experts=torch.zeros((em//bm,),dtype=torch.int32,device=x.device)
        padded=torch.full((1,),em,dtype=torch.int32,device=x.device)
        b=layer.weight.unsqueeze(0);s=layer.weight_scale.unsqueeze(0)
        import triton.language as tl
        kernel=self.test_recorder if self.test_recorder is not None else native_kernel()
        grid=(em//bm*((self.n+cfg['BLOCK_SIZE_N']-1)//cfg['BLOCK_SIZE_N']),)
        kernel[grid](a,b,out,s,None,None,ids,experts,padded,self.n,self.k,em,m,
                     a.stride(0),a.stride(1),b.stride(0),b.stride(2),b.stride(1),out.stride(1),out.stride(2),
                     s.stride(0),s.stride(2),s.stride(1),0,0,0,
                     block_k_diviable=True,group_size=self.group,MUL_ROUTED_WEIGHT=False,top_k=1,
                     compute_type=tl.bfloat16,has_zp=False,use_int4_w4a16=self.bits==4,use_int8_w8a16=self.bits==8,**cfg)
        return out.reshape(*x.shape[:-1],self.n)


@lru_cache(maxsize=1)
def native_kernel():
    from vllm.model_executor.layers.fused_moe import fused_moe
    if hashlib.sha256(Path(fused_moe.__file__).read_bytes()).hexdigest()!=KERNEL_SHA:raise ValueError('stock kernel source hash mismatch')
    return fused_moe.fused_moe_kernel_gptq_awq


class TestCallRecorder:
    """Explicit CPU dispatch-contract test double. Does NOT compute GEMM."""
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def record(*args,**kwargs):
            import inspect
            kernel=native_kernel()
            signature=inspect.signature(getattr(kernel,'fn',kernel))
            launch_options={'num_warps','num_stages'}
            signature.bind(*args,**{k:v for k,v in kwargs.items() if k not in launch_options})
            self.calls.append({'grid':grid,'kwargs':kwargs,'B_shape':tuple(args[1].shape),'scales_dtype':args[3].dtype,'sorted_ids':args[6].tolist(),'experts':args[7].tolist()})
            args[2].zero_()
        return record


def bind_draft(quant_config,policy,manifest):
    """Copy, never mutate the target/trunk config; require all dense metadata."""
    if manifest.get('recipe_sha256')!=RECIPE_SHA:raise ValueError('recipe metadata hash')
    tensors=manifest.get('tensors',{})
    if set(tensors)!={s['name'] for s in policy.dense}:raise ValueError('missing/extra dense metadata')
    for s in policy.dense:validate_metadata(s,tensors[s['name']])
    result=copy.deepcopy(quant_config)
    result._mtp_dense_policy=copy.deepcopy(policy)
    result._mtp_dense_manifest=copy.deepcopy(manifest)
    return result


def configure_draft_if_requested(quant_config,start):
    if quant_config is None:return None
    request=getattr(quant_config,'quant_config',{}).get('mtp_dense_bridge')
    if request is None:return quant_config
    if set(request) not in ({'recipe_path','manifest'},{'recipe_path','manifest','native_checkpoint'}):raise ValueError('explicit MTP request fields')
    result=bind_draft(quant_config,Policy.from_recipe(request['recipe_path'],start),request['manifest'])
    if 'native_checkpoint' in request:
        from mtp_checkpoint import bind_checkpoint
        result._mtp_native_checkpoint=bind_checkpoint(request['native_checkpoint'])
    return result


def scoped_specs(quant_config,prefix):
    p=getattr(quant_config,'_mtp_dense_policy',None)
    return None if p is None else p.resolve(prefix,namespace='draft_prefix')


def scoped_protected(quant_config,prefix):
    p=getattr(quant_config,'_mtp_dense_policy',None)
    if p is None:return False
    source=p.original(prefix,namespace='draft_prefix')
    spec=p.by_name.get(source.removesuffix('.weight')+'.weight') if source else None
    return spec is not None and spec['weight_bits']==16


def scoped_hc(quant_config,prefix):
    p=getattr(quant_config,'_mtp_dense_policy',None)
    return None if p is None else p.hc_plan(prefix,namespace='draft_prefix')


def wire_shared_gate(block,quant_config,prefix):
    specs=scoped_specs(quant_config,prefix+'.shared_expert_gate')
    if specs is None:return
    if block.is_fused_shared_expert_enabled or block.shared_expert is None:
        raise ValueError('MTP dense requires unfused shared expert; fused shared mode not admitted')
    layer=block.shared_expert_gate
    if isinstance(layer.quant_method,MTPDenseMethod):raise ValueError('duplicate shared gate wiring')
    method=MTPDenseMethod(specs)
    del layer._parameters['weight']
    layer.quant_method=method
    method.create_weights(layer,2560,[1],2560,1,torch.bfloat16)


def consume_dense_payloads(owner,weights,policy,loaded):
    """Intercept canonical logical payloads BEFORE native HC/QKV name mappers.
    Runtime named_modules uses ModuleList index0, not logical prefix index48.
    Caller must exhaust this iterator via AutoWeightsLoader and preserve its
    loaded set. Native post-load hooks still enforce complete per-piece loads.
    """
    modules=dict(owner.named_modules())
    for name,tensor in weights:
        source=name
        for wrapper in ('model.language_model.','language_model.','model.'):
            if source.startswith(wrapper+'mtp.'):
                source=source[len(wrapper):];break
        piece='weight_scale' if source.endswith('.weight_scale') else 'weight'
        logical=source.removesuffix('.'+piece)+'.weight'
        s=policy.by_name.get(logical)
        if s is None or s not in policy.dense:
            yield name,tensor;continue
        prefix=policy.map_source(logical.removesuffix('.weight'),namespace='draft_module')
        shard=None
        for suffix,merged,shard_id in [('.self_attn.q_proj','.self_attn.qkv_proj','q'),('.self_attn.k_proj','.self_attn.qkv_proj','k'),('.self_attn.v_proj','.self_attn.qkv_proj','v'),('.mlp.shared_expert.gate_proj','.mlp.shared_expert.gate_up_proj',0),('.mlp.shared_expert.up_proj','.mlp.shared_expert.gate_up_proj',1)]:
            if prefix.endswith(suffix):prefix=prefix[:-len(suffix)]+merged;shard=shard_id;break
        layer=modules.get(prefix)
        if layer is None or not isinstance(getattr(layer,'quant_method',None),MTPDenseMethod):raise ValueError('missing bound dense module: '+prefix)
        param=getattr(layer,piece)
        param.weight_loader(param,tensor,shard)
        loaded.add(prefix+'.'+piece)


class BoundConfig:
    """Explicit draft-only CPU/integration facade, never installed globally."""
    def __init__(self,policy,*,namespace='draft_prefix'):
        self.policy=policy;self.namespace=namespace;self.packed_modules_mapping={}
        self.online_quantization_config=None
    def get_name(self):return 'mtp_dense_bounded'
    def get_quant_method(self,layer,prefix):
        specs=self.policy.resolve(prefix,namespace=self.namespace)
        return MTPDenseMethod(specs) if specs else UnquantizedLinearMethod()
    def get_quant_method_target(self,prefix,layer_type):
        return None,None,MTPDenseMethod if self.policy.resolve(prefix,namespace=self.namespace) else UnquantizedLinearMethod
