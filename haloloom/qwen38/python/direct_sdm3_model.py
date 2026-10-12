"""Experimental multirow MoE, no shared fold. Exact same-input selfcheck on each layer/row geometry."""
import hashlib,json,os,sys
import torch
import direct_moe_sdm3 as multi
_CFG=None
_CHECKED=set()
def config():
 global _CFG
 if _CFG is None:
  p=os.environ['DIRECT_SDM3_HSACO'];h=hashlib.sha256(open(p,'rb').read()).hexdigest();assert h==os.environ['DIRECT_SDM3_SHA256']
  _CFG=multi.Cfg(p,'sdm3_up_m2_c2_p4_l1','sdm3_dn_m0_c1_p8_l1',0)
 return _CFG
def run_checked(method,layer,x,weights,ids,invoke):
 y=multi.run(method,layer,x,weights,ids,config(),fold=False)
 key=(id(method),x.shape[0])
 if key not in _CHECKED:
  assert not torch.cuda.is_current_stream_capturing(),'unvalidated multi-row geometry during graph capture'
  ref=method._sdm3_reference(layer,x,weights,ids,invoke)
  torch.cuda.synchronize()
  if not torch.equal(ref.view(torch.int16),y.view(torch.int16)):raise RuntimeError('SDM3 same-input native model-layer selfcheck mismatch')
  _CHECKED.add(key)
  print('DIRECT_SDM3_SELFCHECK '+json.dumps({'pid':os.getpid(),'layer_identity':id(method),'rows':x.shape[0],'exact':True,'hsaco_sha256':os.environ['DIRECT_SDM3_SHA256'],'shared_fold':False}),file=sys.stderr,flush=True)
 return y
