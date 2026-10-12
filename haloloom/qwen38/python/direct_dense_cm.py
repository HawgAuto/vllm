# SPDX-License-Identifier: Apache-2.0
"""Dense column-major weight relayout for K=6144 -> N=2560 projections (GDN out_proj, QSA o_proj).  NON-BITWISE (quality-gated).

Evidence (synthetic operands, real vLLM op torch.ops.vllm.rocm_unquantized_gemm):
  attempts/dense-formulation-01: x @ W^T(contiguous) -> hipBLASLt MT128x96x64 vs F.linear NT MT128x64x64: M4096 4.737 -> 3.639 ms.
  attempts/oproj-colmajor-01   : W.t().contiguous().t() through the vLLM op: faster at every M 1..12288 (M4096 4.822 -> 3.631 ms,
                                 decode M1-4 skips wvSplitK -> MT64x16x64 0.164 -> 0.063 ms warm); rel-L2 vs fp64 ratio 1.0000.
Mechanism: the weight keeps shape [N, K] and values; only storage becomes column-major.  rocm_unquantized_gemm_impl then fails
skinny_operands_compatible (weight.is_contiguous() False) and falls to F.linear -> mm(x, W^T contiguous) (NN hipBLASLt).
Installed by wrapping the layer's own quant_method.process_weights_after_loading (original runs first).  Off unless DIRECT_DENSE_CM=1.
The traced model files that call install() changed, so the vLLM compile cache key changes and graphs recompile with the new strides
(seed artifacts assert (6144, 1) weight strides).
"""
import json, os
import torch

ENABLE = os.environ.get('DIRECT_DENSE_CM', '0') == '1'
SHAPES = {(2560, 6144)}
_STATE = {'installed': 0, 'converted': 0, 'skipped': []}


def _report(tag, **kw):
    print('DIRECT_DENSE_CM ' + json.dumps(dict(tag=tag, **kw), sort_keys=True, default=str), flush=True)


def install(linear, name):
    if not ENABLE:
        return
    qm = getattr(linear, 'quant_method', None)
    if qm is None or not hasattr(qm, 'process_weights_after_loading'):
        _STATE['skipped'].append((name, 'no quant_method')); _report('skip', name=name, why='no quant_method'); return
    orig = qm.process_weights_after_loading

    def pwal(layer):
        orig(layer)
        w = getattr(layer, 'weight', None)
        ok = (layer is linear and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2
              and tuple(w.shape) in SHAPES and w.is_contiguous())
        if not ok:
            why = dict(same=layer is linear, dtype=str(getattr(w, 'dtype', None)), shape=tuple(getattr(w, 'shape', ())),
                       contig=bool(getattr(w, 'is_contiguous', lambda: None)()), qm=type(qm).__name__)
            _STATE['skipped'].append((name, why)); _report('skip', name=name, **why); return
        w.data = w.data.t().contiguous().t()
        _STATE['converted'] += 1
        if _STATE['converted'] in (1, 12, 36, 48):
            _report('converted', n=_STATE['converted'], name=name, shape=tuple(w.shape), stride=tuple(w.stride()), qm=type(qm).__name__)

    qm.process_weights_after_loading = pwal
    _STATE['installed'] += 1
