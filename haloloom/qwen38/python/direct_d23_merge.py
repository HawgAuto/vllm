"""decode23 dense-merge M1 (opt-in, DIRECT_D23_MERGE=1): the single-token MoE trio
    shared_expert.gate_up_proj [1280, 2560]  +  router gate [512, 2560]  +  shared_expert_gate [1, 2560]
(all reading the same hidden_states) as ONE launch of d23_grouped: the native wvSplitK_hf_sml_ per-tile bodies lifted
verbatim with each segment's native (YTILE, UNRL) variant, so every output row is bitwise identical to the native GEMV
(operator screen decode23-dense-merge-01: exact on 9 inputs, 55.2 -> 46.3 us per layer at grid 20 x 15 waves).

Install: lazily, from the first direct_moe_reduction.fold_begin call, i.e. inside the opaque moe_forward_shared op
(never traced, so the compile hash and the tuning seed are unchanged). MoERunner._forward_impl is wrapped at class level.
For an eligible runner and a [1, 2560] bf16 input the trio is launched once on the current stream before the original
_forward_impl runs; the router gate, shared gate_up_proj and shared expert_gate *instances* are armed with consume-once
outputs keyed by the exact input tensor (data_ptr/shape/stride/dtype/device). The original code then calls those modules
as usual and receives the precomputed outputs. Fail closed: a different input or any unconsumed output raises.
Self-check: the first eager (non-capturing) activation of every runner compares the grouped outputs bitwise with the
modules' original forwards. Shared outputs consumed on the shared-experts aux stream are record_stream'ed onto it.
validate()/launch argument layout are copied verbatim from the screened bridge (decode23-phase1/dense-merge-v1/bridge.py).
"""
import ctypes, hashlib, json, os, sys
import torch
import direct_hip_module_v1 as hm

YT = {12: (1, 2), 14: (1, 4), 41: (4, 1)}     # variant code -> native (YTILE, UNRL)
MAX_LDS_ELEMS = 65536 // 2
K_TRIO = 2560
TRIO = (('gate_up', 1280, 12), ('router', 512, 12), ('se_gate', 1, 14))   # screened segment order + native variants
GRID, WAVES = 20, 15                                                       # screened arm grouped_g20_w15
STATS = {'installed': False, 'runners_seen': 0, 'eligible': 0, 'ineligible': {}, 'act_capturing': 0, 'act_eager': 0,
         'skip_tokens': 0, 'skip_input': {}, 'selfcheck_pass': 0, 'aux_streams': 0}
_MOD = [None]
_CHECKED = set()
_MISSING = object()


def _emit(tag, **kw):
    print('DIRECT_D23_MERGE ' + json.dumps(dict(tag=tag, pid=os.getpid(), **kw), sort_keys=True, default=str), file=sys.stderr, flush=True)


def validate(x, segs, waves, grid):
    if not (1 <= len(segs) <= 3): raise ValueError('segments')
    if not (isinstance(waves, int) and 1 <= waves <= 16 and isinstance(grid, int) and 1 <= grid <= 64): raise ValueError('geometry')
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[0] != 1 or not x.is_contiguous() or x.data_ptr() % 16: raise ValueError('activation')
    K = x.shape[1]
    if K % 8 or K > MAX_LDS_ELEMS: raise ValueError('K')
    spans = [(x.data_ptr(), x.data_ptr() + x.numel() * 2, 'x')]
    for w, y, v in segs:
        if v not in YT: raise ValueError('variant')
        if w.dtype != torch.bfloat16 or y.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] != K: raise ValueError('weight')
        if not (w.is_contiguous() and y.is_contiguous()) or w.data_ptr() % 16 or y.data_ptr() % 2: raise ValueError('layout')
        M = w.shape[0]
        if tuple(y.shape) != (1, M): raise ValueError('output_shape')
        if M % YT[v][0]: raise ValueError('ytile')
        if not (w.device == y.device == x.device) or not w.is_cuda: raise ValueError('device')
        spans += [(w.data_ptr(), w.data_ptr() + w.numel() * 2, 'w'), (y.data_ptr(), y.data_ptr() + y.numel() * 2, 'y')]
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if not (a[1] <= b[0] or b[1] <= a[0]) and not (a[2] == b[2] == 'w'): raise ValueError('alias')
    return K


def _module():
    if _MOD[0] is None:
        path = os.environ['DIRECT_D23_MERGE_HSACO']; name = os.environ['DIRECT_D23_MERGE_FN']
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if got != os.environ['DIRECT_D23_MERGE_SHA256']:
            raise RuntimeError('DIRECT_D23_MERGE hsaco sha256 mismatch: ' + got)
        mod = hm.Module(path); attrs = mod.attrs(name)
        if not (attrs['shared_bytes'] == 65536 and attrs['local_bytes'] == 0 and attrs['max_threads'] == 512):
            raise RuntimeError('DIRECT_D23_MERGE kernel attributes ' + json.dumps(attrs))
        _MOD[0] = (mod, name)
        _emit('module', hsaco=path, sha256=got, fn=name, attrs=attrs)
    return _MOD[0]


def launch(x, segs, waves=WAVES, grid=GRID):
    K = validate(x, segs, waves, grid)
    mod, name = _module()
    args = [ctypes.c_int(K), ctypes.c_int(K), hm.ptr(x)]
    for i in range(3):
        if i < len(segs):
            w, y, v = segs[i]; M = w.shape[0]; T = M // YT[v][0]
            args += [hm.ptr(w), hm.ptr(y), ctypes.c_int(M), ctypes.c_int(v), ctypes.c_int(T)]
        else:
            args += [ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0), ctypes.c_int(12), ctypes.c_int(0)]
    args += [ctypes.c_int(waves), ctypes.c_int(grid * waves)]
    mod.launch(name, (grid,), (32, 16), args, stream=torch.cuda.current_stream().cuda_stream)


def _key(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _arm_module(mod):
    """Instance-level forward override: returns the armed precomputed output exactly once, else the original forward."""
    if '_d23_orig_forward' in mod.__dict__:
        return
    orig = mod.forward

    def forward(x, *args, **kwargs):
        pre = mod.__dict__.get('_d23_pre')
        if pre is None:
            return orig(x, *args, **kwargs)
        mod.__dict__['_d23_pre'] = None
        if args or kwargs or not isinstance(x, torch.Tensor) or _key(x) != pre[0]:
            raise RuntimeError('DIRECT_D23_MERGE: armed module called with a different input')
        return pre[1], None
    mod.__dict__['_d23_orig_forward'] = orig
    mod.__dict__['_d23_pre'] = None
    mod.__dict__['forward'] = forward


def _linear_ok(mod, M):
    w = getattr(mod, 'weight', None)
    return (isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2 and tuple(w.shape) == (M, K_TRIO)
            and w.is_contiguous() and w.data_ptr() % 16 == 0 and getattr(mod, 'bias', None) is None and getattr(mod, 'return_bias', None) is True
            and not getattr(mod, 'skip_bias_add', False) and getattr(mod, 'tp_size', 1) == 1
            and type(getattr(mod, 'quant_method', None)).__name__ == 'UnquantizedLinearMethod')


def _state(runner):
    st = runner.__dict__.get('_d23_state', _MISSING)
    if st is not _MISSING:
        return st
    STATS['runners_seen'] += 1
    why = None; st = None
    gate = getattr(runner, 'gate', None); se = getattr(runner, '_shared_experts', None)
    mlp = getattr(se, '_layer', None) if se is not None else None
    if gate is None: why = 'no_gate'
    elif getattr(runner, '_fse_fuse_gate', False): why = 'fse_fuse_gate'
    elif se is None or mlp is None: why = 'no_shared'
    elif getattr(se, 'enable_dbo', False): why = 'dbo'
    else:
        gu = getattr(mlp, 'gate_up_proj', None); sg = getattr(mlp, 'expert_gate', None)
        if gu is None or sg is None: why = 'no_shared_linears'
        elif len({id(gu), id(gate), id(sg)}) != 3: why = 'aliased_modules'
        elif not (_linear_ok(gu, TRIO[0][1]) and _linear_ok(gate, TRIO[1][1]) and _linear_ok(sg, TRIO[2][1])): why = 'linear_shape_or_type'
        else:
            for m in (gu, gate, sg):
                _arm_module(m)
            st = (gu, gate, sg, se)
            if getattr(se, '_stream', None) is not None:
                STATS['aux_streams'] += 1
    runner.__dict__['_d23_state'] = st
    if st is None:
        STATS['ineligible'][why] = STATS['ineligible'].get(why, 0) + 1
    else:
        STATS['eligible'] += 1
    if STATS['runners_seen'] in (1, 48) or (why is not None and STATS['ineligible'][why] == 1):
        _emit('state', layer=getattr(runner, 'layer_name', None), why=why, eligible=STATS['eligible'], ineligible=STATS['ineligible'],
              runners_seen=STATS['runners_seen'], aux_streams=STATS['aux_streams'])
    return st


def _activation_ok(x):
    return x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] == K_TRIO and x.data_ptr() % 16 == 0 and x.is_cuda


def _wrap(orig):
    def _forward_impl(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
        st = _state(self)
        x = hidden_states
        if st is None or not isinstance(x, torch.Tensor) or x.dim() != 2 or x.shape[0] != 1:
            if st is not None:
                STATS['skip_tokens'] += 1
            return orig(self, hidden_states, router_logits, shared_experts_input, input_ids)
        why = None
        if not isinstance(shared_experts_input, torch.Tensor) or _key(shared_experts_input) != _key(x): why = 'shared_input_differs'
        elif not _activation_ok(x): why = 'activation'
        if why is not None:
            STATS['skip_input'][why] = STATS['skip_input'].get(why, 0) + 1
            if STATS['skip_input'][why] == 1:
                _emit('skip_input', why=why)
            return orig(self, hidden_states, router_logits, shared_experts_input, input_ids)
        gu, gate, sg, se = st
        capturing = torch.cuda.is_current_stream_capturing()
        outs = [torch.empty((1, M), dtype=torch.bfloat16, device=x.device) for _, M, _ in TRIO]
        launch(x, [(gu.weight, outs[0], TRIO[0][2]), (gate.weight, outs[1], TRIO[1][2]), (sg.weight, outs[2], TRIO[2][2])])
        aux = getattr(se, '_stream', None)
        if aux is not None:
            # gate_up / expert_gate outputs may be read on the shared-experts aux stream: keep the allocator from reusing
            # these main-stream blocks before that stream is done (also defers reuse until the end of a graph capture).
            outs[0].record_stream(aux); outs[2].record_stream(aux)
        if not capturing and id(self) not in _CHECKED:
            refs = [gu.__dict__['_d23_orig_forward'](x)[0], gate.__dict__['_d23_orig_forward'](x)[0], sg.__dict__['_d23_orig_forward'](x)[0]]
            for (nm, _, _), r, o in zip(TRIO, refs, outs):
                if not (r.shape == o.shape and r.dtype == o.dtype and torch.equal(r.view(torch.int16), o.view(torch.int16))):
                    raise RuntimeError('DIRECT_D23_MERGE selfcheck mismatch: ' + nm + ' layer ' + str(getattr(self, 'layer_name', None)))
            _CHECKED.add(id(self)); STATS['selfcheck_pass'] += 1
            if STATS['selfcheck_pass'] in (1, 48):
                _emit('selfcheck', passed=STATS['selfcheck_pass'], layer=getattr(self, 'layer_name', None))
        k = _key(x)
        gu.__dict__['_d23_pre'] = (k, outs[0]); gate.__dict__['_d23_pre'] = (k, outs[1]); sg.__dict__['_d23_pre'] = (k, outs[2])
        try:
            res = orig(self, hidden_states, router_logits, shared_experts_input, input_ids)
        finally:
            left = [nm for (nm, _, _), m in zip(TRIO, (gu, gate, sg)) if m.__dict__.get('_d23_pre') is not None]
            for m in (gu, gate, sg):
                m.__dict__['_d23_pre'] = None
        if left:
            raise RuntimeError('DIRECT_D23_MERGE unconsumed precomputed outputs: ' + ','.join(left))
        STATS['act_capturing' if capturing else 'act_eager'] += 1
        n = STATS['act_capturing'] + STATS['act_eager']
        if (n in (1, 48, 96, 192, 480, 4800) or n % 49152 == 0 or (n % 48 == 0 and n <= 960)
                or (capturing and STATS['act_capturing'] in (1, 48, 96, 144, 192))):
            _emit('stats', **STATS)
        return res
    _forward_impl._d23 = True
    _forward_impl.__wrapped__ = orig
    return _forward_impl


def install():
    if STATS['installed']:
        return
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    cls = mr.MoERunner
    orig = cls.__dict__['_forward_impl']
    if getattr(orig, '_d23', False):
        STATS['installed'] = True
        return
    cls._forward_impl = _wrap(orig)
    STATS['installed'] = True
    _emit('install', cls=cls.__module__ + '.' + cls.__qualname__, grid=GRID, waves=WAVES, trio=TRIO, k=K_TRIO)
