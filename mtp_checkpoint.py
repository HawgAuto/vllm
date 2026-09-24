"""Source-bound native MTP checkpoint adapter. No kernels or requantization."""
import re,json,hashlib
from pathlib import Path
import torch
from native_wire_binding import convert_native
from mtp_dense import consume_dense_payloads,MTPDenseMethod
from quark_mixed_moe import QuarkMixedMoEMethod

def describe_tensor(t):
 if t.device.type!='cpu':raise ValueError('CPU checkpoint staging only')
 return {'dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}

def validate_tensor(t,meta):
 if describe_tensor(t)!=meta:raise ValueError('native source tensor binding mismatch')

def expert_piece(name,policy):
 m=re.fullmatch(r'mtp\.layers\.0\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale)',name)
 if m is None or not 0<=int(m[1])<512:raise ValueError('invalid MTP expert checkpoint name')
 return int(m[1]),{'gate_proj':'w1','up_proj':'w3','down_proj':'w2'}[m[2]],m[3]=='weight_scale'

def bind_checkpoint(request):
 if set(request)!={'path','sha256'}:raise ValueError('native checkpoint request fields')
 p=Path(request['path'])
 if not p.is_absolute() or hashlib.sha256(p.read_bytes()).hexdigest()!=request['sha256']:raise ValueError('native checkpoint manifest digest')
 c=json.loads(p.read_text())
 if c['schema']!='native-mtp-checkpoint-v1' or c['recipe_sha256']!='023bddc128e45d56c9fc9bc8dae56dd213d3647b95739c1a23b059ae01c515d8':raise ValueError('native checkpoint recipe')
 if len(c['tensors'])!=512*6+19*2+10+2:raise ValueError('incomplete native checkpoint manifest')
 return c

def consume_native_payloads(owner,weights,policy,checkpoint,loaded):
 """Called by actual Qwen4ExpMTP.load_weights before native name remapping.

 Dense packets pair by exact logical source name. Experts use actual registered
 RoutedExperts parameters; BF16 tensors continue through native AutoWeightsLoader.
 All checkpoint pieces are bound before copying. Native post-load methods remain
 responsible for finalizing packed execution; no initialized dummy weights.
 """
 expected=checkpoint['tensors'];seen=set();pending={};modules=dict(owner.named_modules())
 param_names={id(p):n for n,p in owner.named_parameters()}
 experts=[(n,m) for n,m in modules.items() if isinstance(getattr(m,'quant_method',None),QuarkMixedMoEMethod) and n.startswith('model.layers.0.mlp.experts')]
 # MoERunner may expose quant_method as a property; only true parameter owners.
 experts=[(n,m) for n,m in experts if 'w13_weight' in m._parameters]
 if len(experts)!=1:raise ValueError('exactly one actual draft RoutedExperts owner required')
 expert_name,expert=experts[0]
 if expert.quant_method.plan.four!=tuple(range(512)) or expert.quant_method.plan.eight:raise ValueError('draft must not inherit trunk mask')
 for name,t in weights:
  source=name
  for wrapper in ('model.language_model.','language_model.','model.'):
   if source.startswith(wrapper+'mtp.'):
    source=source[len(wrapper):];break
  if source not in expected:
   if source.startswith('mtp.'):raise ValueError('unexpected native MTP tensor '+source)
   yield name,t;continue
  if source in seen:raise ValueError('duplicate native checkpoint tensor '+source)
  validate_tensor(t,expected[source]);seen.add(source)
  if '.mlp.experts.' in source:
   e,shard,scale=expert_piece(source,policy)
   key=('w2' if shard=='w2' else 'w13')+'_weight'+('_scale' if scale else '')
   p=getattr(expert,key);p.weight_loader(p,t,key,shard,e)
   loaded.add(param_names[id(p)])
   pool=expert.mixed_pools['4'];target=getattr(pool,('w2' if shard=='w2' else 'w13')+('_scale' if scale else '_weight'))
   loaded.add(param_names[id(target)])
   continue
  logical=source.removesuffix('_scale')
  spec=policy.by_name.get(logical)
  if spec is not None and spec in policy.dense:
   piece='scale' if source.endswith('_scale') else 'weight'
   packet=pending.setdefault(logical,{})
   packet[piece]=t
   if len(packet)==2:
    n,k=spec['source_shape'];q,s=convert_native(packet['weight'],packet['scale'],n=n,k=k,bits=spec['weight_bits'],group=spec['group_size'])
    list(consume_dense_payloads(owner,[(logical,q),(logical+'_scale',s)],policy,loaded))
    del pending[logical]
    # These copies are consumed; do not retain the last packet while draining experts.
    del packet,q,s
   continue
  yield name,t
 if pending or seen!=set(expected):raise ValueError('missing native checkpoint pieces: '+str(sorted(set(expected)-seen)))
 owner._native_checkpoint_receipt={'seen':len(seen),'source_names':sorted(seen),'expert_pieces':expert.quant_method.loaded_piece_count,'expert_owner':expert_name,'dense_row_maps':{n:m.quant_method.row_map for n,m in modules.items() if isinstance(getattr(m,'quant_method',None),MTPDenseMethod)}}
