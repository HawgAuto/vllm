# SPDX-License-Identifier: Apache-2.0
"""Dense W8A16 K5b runtime (dense-w8a16-v1/k5-gemm-v1/K5B-PLAN.md) = the K2/K3/K4 runtime (native/direct_w8a16.py
sha256 8fa48361...) with ONE behavioural change: >4-token GEMMs are routed by the K5c-measured per-(N,K) M-bucket dispatch
table (DIRECT_W8A16_K5_DISPATCH, sha-pinned) to either
  'k5'  the K5a-qualified native W8A16 Triton GEMM (k5-gemm-v1/w8a16_gemm.py, port of vLLM's RDNA hybrid W4A16 prefill
        kernel for int8 [N,K] + per-channel FP32 scale; small/mid M), or
  'fdq' the K5c-qualified single-pass fused dequant (k5-gemm-v1/w8a16_dequant.py; BITWISE equal to _dequant) feeding the
        SAME BF16 GEMM call as before (large M; output bitwise equal to the K4-qualified path).
The <=4-token native skinny path, selection, loaders, HC mixed layout and receipts are unchanged.  Added fail-closed checks:
K5 module / table sha256, every target shape covered by the table, and a load-time K5 self-check per layer (every fdq tile
bitwise vs _dequant on the full weight; every k5 tile rel-L2 <= 4e-3 vs FP64, warmed at both Triton M-divisibility
specializations of its bucket on the first layer of each shape so no Triton compile can happen during CUDA-graph capture).
Route receipts: one 'route' line per distinct (shape, path, cfg) at first runtime use.

Original K2 docstring:  Inert unless DIRECT_W8A16=1.

INT8 weight-only (per-output-channel symmetric RTN, FP32 scale) consumer for the 497 dense target matrices = 389 runtime
LinearBase layers (merged qkvz / qkv / gate_up / HC down+inject) + lm_head.
  * <=4 tokens: native skinny kernel torch.ops.direct_w8a16.wvsplitk_i8 (port of vLLM wvSplitK, K1-qualified,
    attempts/w8kernel-k1-01) with the K1 per-(M,K,N) best (YTILE, UNRL);
  * more tokens (prefill / prompt logprobs): dequant bf16(fp32(q) * s) -- bitwise the W8 screen's fake-quant weights --
    then the BF16 GEMM (row-major F.linear; NN mm for the [2560,6144] projections, the DIRECT_DENSE_CM formulation);
  * HC down+inject merged layer ('hc_mixed'): INT8 rows for down (shard 0), BF16 rows for inject (shard 1) + zero pad,
    output = cat(int8 GEMV, BF16 GEMV) with the production BF16 dispatch for the inject rows.
Selection: QuarkConfig.get_quant_method is wrapped (target layers built with the model's Quark config) and W8Config is
passed explicitly where the model builds layers without a quant config (HC, lm_head).  A prefix matches only if, after
stripping leading 'model.'/'language_model.' segments, it is in the targets file (so 'mtp.' draft layers never match).
Fail closed: unknown kinds, shape/dtype mismatches, a missing/mismatched kernel .so, or a failed first-call self-check
raise.  Logs: 'DIRECT_W8A16 {json}' lines on stderr.
"""
import hashlib, importlib.util, json, os, sys
import torch
from torch.nn import Parameter

ENABLE = os.environ.get('DIRECT_W8A16', '0') == '1'
_STATE = {'matched': {}, 'pwal': 0, 'selfcheck': 0, 'unmatched_logged': 0, 'k5_selfcheck': 0, 'k5_warm_shapes': 0,
          'k5_fdq_checks': 0, 'k5_gemm_checks': 0, 'k5_worst_rel_l2': 0.0, 'routes': 0}


def _emit(tag, **kw):
    print('DIRECT_W8A16 ' + json.dumps(dict(tag=tag, pid=os.getpid(), **kw), sort_keys=True, default=str), file=sys.stderr, flush=True)


def canon(prefix: str) -> str:
    parts = prefix.split('.')
    while parts and parts[0] in ('model', 'language_model'):
        parts = parts[1:]
    return '.'.join(parts)


_T = None
_CU = None


def _targets():
    global _T
    if _T is None:
        p = os.environ['DIRECT_W8A16_TARGETS']; raw = open(p, 'rb').read()
        want = os.environ.get('DIRECT_W8A16_TARGETS_SHA256')
        if want and hashlib.sha256(raw).hexdigest() != want:
            raise RuntimeError('DIRECT_W8A16 targets sha256 mismatch')
        _T = json.loads(raw)
        _emit('targets', path=p, layers=len(_T['layers']), expected_layers=_T['expected_runtime_layers'])
    return _T


def _load_kernel():
    path = os.environ['DIRECT_W8A16_SO']; got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if got != os.environ['DIRECT_W8A16_SO_SHA256']:
        raise RuntimeError('DIRECT_W8A16 kernel .so sha256 mismatch: ' + got)
    torch.ops.load_library(path)
    _emit('kernel', path=path, sha256=got)


def _cu():
    global _CU
    if _CU is None:
        _CU = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _CU


def _cfg_for(M, K):
    """Flat [yt1, un1, ..., yt4, un4] from the K1 sweep; default (2,1) for shapes K1 did not cover."""
    tab = _targets()['kernel_configs']
    row = tab.get(f'{M}x{K}')
    out = []
    for n in (1, 2, 3, 4):
        yt, un = (row[str(n)] if row else (2, 1))
        if M % yt:
            yt = 1
        out += [int(yt), int(un)]
    return out


# ------------------------------------------------------------------------------------------------------- custom ops
def _dequant(w, s, dtype, nn):
    if nn:  # [K, N] contiguous, consumed as x @ wT (NN hipBLASLt; the DIRECT_DENSE_CM formulation)
        return (w.t().to(torch.float32) * s.unsqueeze(0)).to(dtype).contiguous()
    return (w.to(torch.float32) * s.unsqueeze(1)).to(dtype)


# ----------------------------------------------------------------------------------------------- K5 dispatch (K5b)
_K5 = None
_ROUTES = set()
K5_PATHS = ('k5', 'fdq', 'dequant')


def _sha_file(path, env_sha):
    raw = open(path, 'rb').read(); got = hashlib.sha256(raw).hexdigest()
    if got != os.environ[env_sha]:
        raise RuntimeError(f'DIRECT_W8A16 K5 sha256 mismatch {path}: {got}')
    return raw, got


def _k5():
    """Sha-pinned K5 GEMM + fused-dequant modules and the dispatch table {(N, K): ((m_max, path, cfg), ...)}."""
    global _K5
    if _K5 is None:
        mods, shas = {}, {}
        for key, name in (('GEMM', 'w8a16_gemm'), ('DEQUANT', 'w8a16_dequant')):
            p = os.environ[f'DIRECT_W8A16_K5_{key}']; _, shas[key] = _sha_file(p, f'DIRECT_W8A16_K5_{key}_SHA256')
            spec = importlib.util.spec_from_file_location('direct_w8a16_k5_' + name, p)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[key] = m
        p = os.environ['DIRECT_W8A16_K5_DISPATCH']; raw, shas['DISPATCH'] = _sha_file(p, 'DIRECT_W8A16_K5_DISPATCH_SHA256')
        t = json.loads(raw)
        if t.get('schema') != 'dense-w8a16-k5b-dispatch-v1':
            raise RuntimeError('DIRECT_W8A16 K5 dispatch schema ' + str(t.get('schema')))
        if t['gemm_kernel_sha256'] != shas['GEMM'] or t['dequant_kernel_sha256'] != shas['DEQUANT']:
            raise RuntimeError('DIRECT_W8A16 K5 dispatch table was measured with different kernel sources')
        table = {}
        for key, ents in t['table'].items():
            N, K = (int(v) for v in key.split('x')); rows = []
            for e in ents:
                path, cfg = e['path'], e['cfg']
                if path not in K5_PATHS:
                    raise RuntimeError('DIRECT_W8A16 K5 unknown path ' + str(path))
                if path == 'k5':
                    if not (len(cfg) == 5 and all(isinstance(v, int) for v in cfg[:4])):
                        raise RuntimeError(f'DIRECT_W8A16 K5 bad k5 cfg {key} {cfg}')
                elif path == 'fdq':
                    if not (len(cfg) == 3 and all(isinstance(v, int) for v in cfg)):
                        raise RuntimeError(f'DIRECT_W8A16 K5 bad fdq cfg {key} {cfg}')
                elif cfg is not None:
                    raise RuntimeError(f'DIRECT_W8A16 K5 dequant entry with cfg {key}')
                rows.append((int(e['m_max']), path, tuple(cfg) if cfg is not None else None))
            rows.sort(key=lambda r: r[0])
            if not rows or rows[0][0] < 5 or len({r[0] for r in rows}) != len(rows):
                raise RuntimeError('DIRECT_W8A16 K5 bad buckets ' + key)
            table[(N, K)] = tuple(rows)
        _K5 = {'gemm': mods['GEMM'], 'dq': mods['DEQUANT'], 'table': table, 'sha256': shas}
        _emit('k5', gemm=os.environ['DIRECT_W8A16_K5_GEMM'], dequant=os.environ['DIRECT_W8A16_K5_DEQUANT'], dispatch=p,
              sha256=shas, shapes=sorted(t['table']))
    return _K5


def _route(n, N, K):
    """(path, cfg) for an n-token (n >= 5) GEMM with an [N, K] int8 weight: first bucket with n <= m_max, else the last."""
    rows = _k5()['table'][(N, K)]
    for m_max, path, cfg in rows:
        if n <= m_max:
            return path, cfg
    return rows[-1][1], rows[-1][2]


def _note_route(n, N, K, path, cfg):
    key = (N, K, path, cfg, n % 16 == 0)
    if key not in _ROUTES:
        _ROUTES.add(key); _STATE['routes'] = len(_ROUTES)
        _emit('route', shape=[N, K], path=path, cfg=cfg, m_div16=n % 16 == 0, first_n=n)


def _w8_core(x2, w, s, cfg, nn):
    n = x2.shape[0]
    if 1 <= n <= 4 and x2.dtype == torch.bfloat16:
        if (not x2.is_contiguous()) or x2.data_ptr() % 16 or x2.stride(0) % 8:
            x2 = x2.clone(memory_format=torch.contiguous_format)
        return torch.ops.direct_w8a16.wvsplitk_i8(w, s, x2, _cu(), cfg[2 * (n - 1)], cfg[2 * (n - 1) + 1])
    path, pc = _route(n, w.shape[0], w.shape[1]) if (n >= 5 and x2.dtype == torch.bfloat16) else ('dequant', None)
    if n >= 5:
        _note_route(n, w.shape[0], w.shape[1], path, pc)
    if path == 'k5':
        return _k5()['gemm'].launch(x2 if x2.is_contiguous() else x2.contiguous(), w, s, pc)
    wd = _k5()['dq'].dequant(w, s, x2.dtype, nn, pc) if path == 'fdq' else _dequant(w, s, x2.dtype, nn)
    return torch.mm(x2, wd) if nn else torch.nn.functional.linear(x2, wd)


def _bucket_ms(rows, i, first_layer):
    """Token counts that exercise bucket i: its m_max, plus (first layer of a shape) the other Triton M%16 class if any."""
    lo = (rows[i - 1][0] + 1) if i else 5; hi = rows[i][0]
    ms = [hi]
    if first_layer:
        want = hi % 16 != 0  # the class hi is NOT in
        alt = next((m for m in range(hi - 1, lo - 1, -1) if (m % 16 == 0) == want), None)
        if alt is not None:
            ms.append(alt)
    return ms


def _selfcheck_k5(layer, first_layer):
    """Load-time K5 check on the loaded device weight (fail closed).  fdq: every distinct fdq tile of this shape BITWISE equal
    to _dequant on the full weight in this layer's formulation.  k5: every distinct k5 bucket at _bucket_ms token counts,
    finite and rel-L2 <= 4e-3 vs FP64 on the first <= 256 output rows; on the first layer of a shape this also compiles
    and loads every Triton specialization the table can route to (before any CUDA-graph capture)."""
    w, s = layer.weight, layer.weight_scale; N, K = w.shape; nn = layer._w8_nn; R = min(N, 256)
    rows = _k5()['table'][(N, K)]
    ref_w = None; worst = 0.0
    for cfg in sorted({r[2] for r in rows if r[1] == 'fdq'}):
        a = _k5()['dq'].dequant(w, s, torch.bfloat16, nn, cfg); b = _dequant(w, s, torch.bfloat16, nn)
        if not (a.shape == b.shape and a.is_contiguous() and bool(torch.equal(a, b))):
            raise RuntimeError(f'DIRECT_W8A16 K5 fdq selfcheck not bitwise {layer._w8_name} cfg={cfg}')
        del a, b; _STATE['k5_fdq_checks'] += 1
    g = torch.Generator(device=w.device); g.manual_seed(20261006 + N + K)
    for i, (m_max, path, cfg) in enumerate(rows):
        if path != 'k5':
            continue
        if ref_w is None:
            ref_w = w[:R].double() * s[:R].double().unsqueeze(1)
        for m in _bucket_ms(rows, i, first_layer):
            x = torch.randn(m, K, device=w.device, generator=g).to(torch.bfloat16)
            out = _k5()['gemm'].launch(x, w, s, cfg)[:, :R]
            ref = x.double() @ ref_w.T
            rel = float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30)); worst = max(worst, rel)
            if not (bool(torch.isfinite(out).all()) and rel <= 4e-3):
                raise RuntimeError(f'DIRECT_W8A16 K5 gemm selfcheck failed {layer._w8_name} m={m} cfg={cfg} rel_l2={rel}')
            _STATE['k5_gemm_checks'] += 1
    _STATE['k5_selfcheck'] += 1; _STATE['k5_warm_shapes'] += int(first_layer)
    _STATE['k5_worst_rel_l2'] = max(_STATE['k5_worst_rel_l2'], worst)
    return worst


def _w8_linear(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, cfg: list[int], nn: bool) -> torch.Tensor:
    shp = x.shape
    out = _w8_core(x.reshape(-1, shp[-1]), w, s, cfg, nn)
    return out.reshape(*shp[:-1], w.shape[0])


def _w8_linear_fake(x, w, s, cfg, nn):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def _bf16_gemm(x2, w):  # production BF16 dispatch (wvSplitK for <=5 tokens, else F.linear); module-level for CPU tests
    return torch.ops.vllm.rocm_unquantized_gemm(x2, w, None)


def _w8_hc_mixed(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, inj: torch.Tensor, cfg: list[int]) -> torch.Tensor:
    shp = x.shape; x2 = x.reshape(-1, shp[-1])
    a = _w8_core(x2, w, s, cfg, False)
    b = _bf16_gemm(x2, inj)
    return torch.cat([a, b], dim=-1).reshape(*shp[:-1], w.shape[0] + inj.shape[0])


def _w8_hc_mixed_fake(x, w, s, inj, cfg):
    return x.new_empty((*x.shape[:-1], w.shape[0] + inj.shape[0]))


# --------------------------------------------------------------------------------------------------- linear methods
if ENABLE:
    from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.utils.torch_utils import direct_register_custom_op
    import vllm.model_executor.layers.utils  # noqa: F401  (registers torch.ops.vllm.rocm_unquantized_gemm)

    _load_kernel()
    _k5()  # K5b: sha-pinned K5 modules + dispatch table (fail closed at install time)
    _K5_WARMED = set()
    direct_register_custom_op(op_name='direct_w8a16_linear', op_func=_w8_linear, fake_impl=_w8_linear_fake)
    direct_register_custom_op(op_name='direct_w8a16_hc_mixed', op_func=_w8_hc_mixed, fake_impl=_w8_hc_mixed_fake)

    def _selfcheck(layer):
        """Load-time check per layer on the loaded device weights (fail closed): the native kernel at 1..4 tokens and both
        dequant formulations at 8 tokens vs an FP64 reference on the first <=256 output rows."""
        w, s = layer.weight, layer.weight_scale; M, K = w.shape; R = min(M, 256)
        g = torch.Generator(device=w.device); g.manual_seed(20261006 + M + K)
        ref_w = w[:R].double() * s[:R].double().unsqueeze(1)
        worst = 0.0
        for n in (1, 2, 3, 4, 8):
            x = torch.randn(n, K, device=w.device, generator=g).to(torch.bfloat16)
            if n <= 4:
                outs = [torch.ops.direct_w8a16.wvsplitk_i8(w, s, x, _cu(), layer._w8_cfg[2 * (n - 1)], layer._w8_cfg[2 * (n - 1) + 1])[:, :R]]
            else:
                outs = [torch.nn.functional.linear(x, _dequant(w[:R], s[:R], x.dtype, False)), torch.mm(x, _dequant(w[:R], s[:R], x.dtype, True))]
            ref = x.double() @ ref_w.T
            for out in outs:
                rel = float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30)); worst = max(worst, rel)
                if not rel <= 4e-3:
                    raise RuntimeError(f'DIRECT_W8A16 selfcheck failed {layer._w8_name} n={n} rel_l2={rel}')
        _STATE['selfcheck'] += 1
        return worst

    class W8A16LinearMethod(LinearMethodBase):
        def __init__(self, name, kind):
            self.name, self.kind = name, kind

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                           params_dtype, **extra_weight_attrs):
            K = input_size_per_partition
            if self.kind == 'hc_mixed':
                assert len(output_partition_sizes) in (2, 3), output_partition_sizes
                N = output_partition_sizes[0]; NI = sum(output_partition_sizes[1:])
                layer.register_buffer('inject_weight', torch.zeros(NI, K, dtype=params_dtype), persistent=False)
                layer._w8_parts = tuple(output_partition_sizes)
            else:
                N = sum(output_partition_sizes)
            w = Parameter(torch.empty(N, K, dtype=torch.int8), requires_grad=False)
            s = Parameter(torch.full((N,), float('nan'), dtype=torch.float32), requires_grad=False)  # NaN = never loaded
            set_weight_attrs(w, {'input_dim': 1, 'output_dim': 0}); set_weight_attrs(s, {'output_dim': 0})
            set_weight_attrs(w, extra_weight_attrs); set_weight_attrs(s, extra_weight_attrs)
            if self.kind == 'hc_mixed':  # down -> int8 rows (shard 0); inject -> bf16 buffer rows (shard 1)
                def _ld_w(param, loaded, shard_id=None):
                    if shard_id == 0:
                        assert loaded.dtype == torch.int8 and tuple(loaded.shape) == tuple(param.shape), (loaded.dtype, loaded.shape)
                        param.data.copy_(loaded)
                    elif shard_id == 1:
                        n_inj = layer._w8_parts[1]
                        assert loaded.dtype == layer.inject_weight.dtype and tuple(loaded.shape) == (n_inj, K), (loaded.dtype, loaded.shape)
                        layer.inject_weight[:n_inj].copy_(loaded)
                    else:
                        raise RuntimeError(f'DIRECT_W8A16 hc_mixed weight shard {shard_id}')

                def _ld_s(param, loaded, shard_id=None):
                    assert shard_id == 0 and loaded.dtype == torch.float32 and tuple(loaded.shape) == tuple(param.shape), (shard_id, loaded.shape)
                    param.data.copy_(loaded)
                w.weight_loader = _ld_w; s.weight_loader = _ld_s
            layer.register_parameter('weight', w); layer.register_parameter('weight_scale', s)
            layer._w8_name, layer._w8_kind = self.name, self.kind

        def process_weights_after_loading(self, layer):
            w, s = layer.weight, layer.weight_scale
            assert w.dtype == torch.int8 and w.is_contiguous() and s.dtype == torch.float32 and s.is_contiguous(), layer._w8_name
            M, K = w.shape
            if not (bool(torch.isfinite(s).all()) and bool((s > 0).all())):
                raise RuntimeError('DIRECT_W8A16 weight_scale not loaded / invalid: ' + layer._w8_name)
            layer._w8_cfg = _cfg_for(M, K)
            layer._w8_nn = (M, K) == (2560, 6144)
            if (M, K) not in _k5()['table']:
                raise RuntimeError(f'DIRECT_W8A16 K5 dispatch table has no entry for {layer._w8_name} {M}x{K}')
            if self.kind == 'hc_mixed':
                layer.inject_weight.data = layer.inject_weight.data.to(w.device).contiguous()
                n_inj = layer._w8_parts[1]
                if not (bool(layer.inject_weight[:n_inj].abs().sum() > 0) and bool((layer.inject_weight[n_inj:] == 0).all())):
                    raise RuntimeError('DIRECT_W8A16 hc inject rows not loaded / pad not zero: ' + layer._w8_name)
                # The HCV3 opaque op's fake (direct_nb1._hc_combine_and_mix_fake) derives the injection output stride from
                # merged_w.shape[0] = the merged [down | inject | pad] output width (336).  Zero-storage int8 shape carrier
                # [M + NI, K]: correct fake strides, and int8 fails the op's BF16 guard -> the exact nb1 module path (which
                # calls this method).  Found by w8eval-w8n-01 (inductor assert_size_stride 336 vs 320).
                layer.register_buffer('w8_ops_merged_w', torch.empty((1, K), dtype=torch.int8, device=w.device)
                                      .expand(M + layer.inject_weight.shape[0], K), persistent=False)
            worst = _selfcheck(layer)
            first = (M, K) not in _K5_WARMED; _K5_WARMED.add((M, K))
            _selfcheck_k5(layer, first)
            _STATE['pwal'] += 1; _STATE['worst_rel_l2'] = max(_STATE.get('worst_rel_l2', 0.0), worst)
            exp = _targets()['expected_runtime_layers']
            if _STATE['pwal'] in (1, exp):
                _emit('pwal', n=_STATE['pwal'], expected=exp, name=layer._w8_name, shape=[M, K], cfg=layer._w8_cfg,
                      matched=_STATE['matched'], selfcheck_worst_rel_l2=_STATE['worst_rel_l2'],
                      k5={k: _STATE[k] for k in ('k5_selfcheck', 'k5_warm_shapes', 'k5_fdq_checks', 'k5_gemm_checks', 'k5_worst_rel_l2')})

        def apply(self, layer, x, bias=None):
            assert bias is None, layer._w8_name
            if self.kind == 'hc_mixed':
                out = torch.ops.vllm.direct_w8a16_hc_mixed(x, layer.weight, layer.weight_scale, layer.inject_weight, layer._w8_cfg)
            else:
                out = torch.ops.vllm.direct_w8a16_linear(x, layer.weight, layer.weight_scale, layer._w8_cfg, layer._w8_nn)
            return out

    def method_for(layer, prefix):
        if not isinstance(layer, (LinearBase, ParallelLMHead)):
            return None
        c = canon(prefix); ent = _targets()['layers'].get(c)
        if ent is None:
            return None
        kind = ent['kind']
        if kind not in ('int8', 'hc_mixed'):
            raise RuntimeError('DIRECT_W8A16 unknown kind ' + kind)
        if isinstance(layer, ParallelLMHead) != (c == 'lm_head'):
            raise RuntimeError('DIRECT_W8A16 lm_head type mismatch at ' + prefix)
        _STATE['matched'][kind] = _STATE['matched'].get(kind, 0) + 1
        return W8A16LinearMethod(c, kind)

    class W8Config(QuantizationConfig):
        """Explicit config for layers the model builds without a quant config (HC projections, lm_head)."""

        def get_name(self):
            return 'direct_w8a16'

        def get_supported_act_dtypes(self):
            return [torch.bfloat16]

        @classmethod
        def get_min_capability(cls):
            return 0

        @staticmethod
        def get_config_filenames():
            return []

        @classmethod
        def from_config(cls, config):
            return cls()

        def get_quant_method(self, layer, prefix):
            m = method_for(layer, prefix)
            if m is not None:
                return m
            return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None

    _W8CFG = W8Config()

    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    if not getattr(QuarkConfig.get_quant_method, '_direct_w8a16', False):
        _orig_gqm = QuarkConfig.get_quant_method

        def _gqm(self, layer, prefix):
            m = method_for(layer, prefix)
            return m if m is not None else _orig_gqm(self, layer, prefix)
        _gqm._direct_w8a16 = True
        QuarkConfig.get_quant_method = _gqm
    _emit('installed', targets=os.environ.get('DIRECT_W8A16_TARGETS'))


def config_or_none():
    """W8Config for layers the model builds without a quant config, or None when the W8A16 path is off."""
    return _W8CFG if ENABLE else None


def state():
    return dict(_STATE)
