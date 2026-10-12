"""Default-off serving adapter over the unchanged, operator-screened direct_ddn bridge.
No dynamic host branch is traced; custom op fake describes the exact contiguous [M,N] result.
Same-input first-use checks synchronize outside capture; ordinary timed and captured dispatch remain asynchronous.
"""
import json, os
import torch
import direct_ddn as ddn
from vllm.utils.torch_utils import direct_register_custom_op
REGISTRY=[]; TABLE=None; CHECKED=set(); REPORTED=set()
REFERENCE_ONLY=os.environ.get('DIRECT_DDN_REFERENCE_ONLY','0')=='1'
def emit(tag, **kw):
    key=(tag,json.dumps(kw,sort_keys=True))
    if key not in REPORTED:
        REPORTED.add(key);print('DIRECT_DDN '+json.dumps({'tag':tag,**kw},sort_keys=True),flush=True)
def register(method):
    global TABLE
    if os.environ.get('DIRECT_DDN')!='1':raise RuntimeError('DDN disabled')
    if TABLE is None:TABLE=ddn.load_table()
    if method.bits!=8 or method.group!=64 or method.test_recorder is not None:raise ValueError('DDN scope')
    i=len(REGISTRY);REGISTRY.append(method);return i

def run(a:torch.Tensor,w:torch.Tensor,s:torch.Tensor,layer_id:int)->torch.Tensor:
    method=REGISTRY[layer_id];m,k=a.shape;n=w.shape[0];names=[x['name'] for x in method.specs]
    if w.data_ptr()!=method.layer.weight.data_ptr() or s.data_ptr()!=method.layer.weight_scale.data_ptr():raise RuntimeError('DDN foreign resident weights')
    # Runtime M gate inside the opaque op; no shape branch is traced by Dynamo.
    if m > 4:
        return method._ddn_reference(method.layer,a)
    if REFERENCE_ONLY:
        capturing=torch.cuda.is_current_stream_capturing();key=(layer_id,m)
        if key not in CHECKED and capturing:raise RuntimeError('DDN reference capture before eager readiness: '+str(key))
        out=method._ddn_reference(method.layer,a)
        if key not in CHECKED:
            assert torch.cuda.get_device_properties(a.device).gcnArchName.split(':')[0]=='gfx1151'
            torch.cuda.current_stream(a.device).synchronize()
            CHECKED.add(key)
            emit('reference_ready',names=names,m=m,n=n,k=k,implementation='native_reference',completion='caller_stream_synchronize')
        emit('reference_capture' if capturing else 'reference_eager',names=names,m=m,n=n,k=k,implementation='native_reference')
        return out
    sym=TABLE.get((n,k,m))
    if sym is None or not ddn.supported(a,w,s,n,k,sym):
        emit('fallback',names=names,m=m,n=n,k=k,reason='unsupported_geometry_or_alignment')
        return method._ddn_reference(method.layer,a)
    capturing=torch.cuda.is_current_stream_capturing();key=(layer_id,m)
    if key not in CHECKED and capturing:raise RuntimeError('DDN capture before eager exactness: '+str(key))
    out=torch.empty((m,n),device=a.device,dtype=a.dtype)
    out=ddn.launch(sym,a,w,s,out,n,k)
    if key not in CHECKED:
        props=torch.cuda.get_device_properties(a.device)
        assert props.gcnArchName.split(':')[0]=='gfx1151'
        ref=method._ddn_reference(method.layer,a)
        torch.cuda.current_stream(a.device).synchronize()
        if not torch.equal(out.view(torch.int16),ref.contiguous().view(torch.int16)):
            raise RuntimeError('DDN exactness failed '+str((names,m,n,k,sym)))
        CHECKED.add(key)
        emit('selfcheck',names=names,m=m,n=n,k=k,kernel=sym,exact=True,hsaco_sha256=os.environ['DIRECT_DDN_HSACO_SHA256'],completion='caller_stream_synchronize')
    emit('capture' if capturing else 'eager',names=names,m=m,n=n,k=k,kernel=sym)
    return out

def fake(a,w,s,layer_id):return a.new_empty((a.shape[0],w.shape[0]))
direct_register_custom_op(op_name='direct_mtp_ddn',op_func=run,fake_impl=fake)
