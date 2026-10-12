# SPDX-License-Identifier: Apache-2.0
"""Private sequential same-process A/B (v1 + 'wmma' arm). Not a concurrent serving interface.

Only the isolated bench writes PHASE atomically, after each request completes.
Normal decode/capture sizes never qualify. Startup defaults to native.
Arm 'wmma': native gfx1151 WMMA kernel where its policy qualifies, otherwise exactly the 'packed' arm's choice.
Arm 'wmma5': unified-pool route in quark_mixed_moe (direct_moe_unified_v1); any per-pool call left = exactly 'packed'.
No scratch comparisons or synchronization in this performance path.
"""
import collections,functools,json,os
from pathlib import Path
import direct_moe_scale_broadcast_adapter_v1 as broadcast
import direct_moe_packed_adapter_v1 as packed
import direct_moe_wmma_adapter_v1 as wmma
PHASE=Path(os.environ.get('DIRECT_MOE_AB_PHASE','/workspace/moe-ab-phase.json'))
ARMS=('native','broadcast','packed','wmma','wmma5')
COUNTS=collections.Counter()
AUDIT=os.environ.get('DIRECT_MOE_AB_AUDIT','1')=='1'

def choose_strategy(arm,meta):
    if arm not in ARMS:raise ValueError('unknown benchmark arm')
    if arm=='native':return 'native'
    if arm=='wmma' and wmma.qualified_policy(**meta):return 'wmma'
    if arm in ('packed','wmma','wmma5') and packed.qualified_policy(**meta):return 'packed'
    if broadcast.qualified_policy(**meta):return 'broadcast'
    return 'native'

def read_phase():
    try:data=json.loads(PHASE.read_text())
    except FileNotFoundError:return None
    if not isinstance(data,dict) or set(data)!= {'arm','case'}:
        raise ValueError('invalid sequential benchmark phase schema')
    if data['arm'] not in ARMS or not isinstance(data['case'],str) or not 0<len(data['case'])<=128:
        raise ValueError('invalid sequential benchmark phase value')
    return data

def select_invoke(native,**meta):
    if not (packed.qualified_policy(**meta) or broadcast.qualified_policy(**meta) or wmma.qualified_policy(**meta)):
        return native
    phase=read_phase()
    if phase is None:return native
    strategy=choose_strategy(phase['arm'],meta)
    if strategy=='wmma':
        fb={'packed':packed._invoke_packed,'broadcast':broadcast._invoke_broadcast,'native':native}[
            'packed' if packed.qualified_policy(**meta) else ('broadcast' if broadcast.qualified_policy(**meta) else 'native')]
        fn=wmma.make_invoke(fb)
    else:
        fn={'native':native,'broadcast':broadcast._invoke_broadcast,'packed':packed._invoke_packed}[strategy]
    return functools.partial(invoke,fn,strategy,phase,meta)

def invoke(fn,strategy,phase,meta,*args,**kwargs):
    result=fn(*args,**kwargs)
    COUNTS[(phase['arm'],strategy)]+=1
    if AUDIT:
        print('DIRECT_MOE_AB_INVOKE_RETURNED '+json.dumps(dict(
            phase=phase,strategy=strategy,tokens=meta['tokens'],
            bits=4 if meta['w4'] else 8,projection=meta['projection'],
            function=fn.__module__+'.'+fn.__name__,
            scope='invocation returned; not independent physical trace'
        ),sort_keys=True),flush=True)
    return result
