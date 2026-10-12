#!/usr/bin/env python3
"""CPU control-flow regression of source-bound QSA entry/adapter, not arithmetic.

The single-request HIP ABI has no request count or table-row stride. Multi-request
metadata must reach the existing request-aware Triton path without calling that
ABI. Preserve the strict adapter guard for unexpected singleton geometry.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import types


class Tensor:
    def __init__(self, shape, dtype='bf16', strides=None):
        shape = (shape,) if isinstance(shape, int) else shape
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.is_cuda = True
        self.device = 'cuda:0'
        if strides is None:
            strides = []
            running = 1
            for n in reversed(shape):
                strides.insert(0, running)
                running *= n
        self.strides = tuple(strides)

    def dim(self):
        return self.ndim

    def stride(self, n=None):
        return self.strides if n is None else self.strides[n]

    def is_contiguous(self):
        return self.strides == Tensor(self.shape).strides

    def data_ptr(self):
        return 16


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return types.SimpleNamespace(metadata=types.SimpleNamespace(name='CPU_DOUBLE'))
        return launch


def exercise(source, adapter_path):
    torch = types.ModuleType('torch')
    torch.Tensor = Tensor
    torch.bfloat16, torch.float32, torch.int32, torch.int64 = 'bf16', 'fp32', 'i32', 'i64'
    torch.empty = lambda shape, dtype, device: Tensor(shape, dtype)
    torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: False)
    sys.modules['torch'] = torch
    nb = types.ModuleType('direct_nb1')
    nb.nb_level = lambda: 9
    sys.modules['direct_nb1'] = nb
    adapter = types.ModuleType('source_bound_mqa_adapter')
    exec(compile(adapter_path.read_text(), str(adapter_path), 'exec'), adapter.__dict__)
    adapter._case = lambda: 'cpu-dispatch-regression'
    native_calls = []
    adapter.run = lambda *a: native_calls.append(a)  # HIP launch only is a double.
    kernel = Kernel()
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_validate_mqa', 'qsa_mqa_paged')]
    ns = dict(torch=torch, math=math, HAS_TRITON=True, _MQA_NATIVE=adapter,
              triton=types.SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b,
                                           next_power_of_2=lambda n: 1 << (n-1).bit_length()),
              _qsa_mqa_paged_kernel=kernel)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), ns)
    rows = []
    for requests in (1, 2, 4, 8, 16, 32):
        args = (Tensor((1025, 4, 128)), Tensor((96, 392, 1, 128)),
                Tensor((requests, 3), 'i32', (84, 1)),
                Tensor((1025,), 'i32'), Tensor((1025,), 'i64'),
                Tensor((requests,), 'i32'), 4)
        before = (len(native_calls), len(kernel.calls))
        logits, visible = ns['qsa_mqa_paged'](*args)
        delta = (len(native_calls)-before[0], len(kernel.calls)-before[1])
        assert delta == ((1, 0) if requests == 1 else (0, 1)), (requests, delta)
        assert logits.shape == (1025, 1176) and visible.shape == (1025,)
        if requests > 1:
            _, positional, _ = kernel.calls[-1]
            declaration = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                               and n.name == '_qsa_mqa_paged_kernel')
            bound = dict(zip((a.arg for a in declaration.args.args), positional))
            assert bound['stride_table_req'] == 84
            assert bound['num_requests'] == requests
        rows.append({'requests': requests, 'native_calls': delta[0], 'triton_calls': delta[1]})
    # The native singleton guard is intentionally still fail-closed.
    bad = list(args)
    bad[0] = Tensor((1025, 4, 128), strides=(513, 128, 1))
    bad[2] = Tensor((1, 3), 'i32')
    bad[5] = Tensor((1,), 'i32')
    try:
        ns['qsa_mqa_paged'](*bad)
    except RuntimeError as exc:
        assert str(exc) == 'unexpected successor prefill fallback'
    else:
        raise AssertionError('singleton layout guard was weakened')
    # Shape validation must precede either kernel choice.
    bad = list(args)
    bad[5] = Tensor((1,), 'i32')
    before = (len(native_calls), len(kernel.calls))
    try:
        ns['qsa_mqa_paged'](*bad)
    except ValueError as exc:
        assert 'sequence lengths' in str(exc)
    else:
        raise AssertionError('malformed request metadata admitted')
    assert before == (len(native_calls), len(kernel.calls))
    return {'status': 'PASS_CPU_DISPATCH_ONLY', 'rows': rows,
            'strict_singleton_guard_preserved': True,
            'malformed_metadata_rejected': True,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'adapter_sha256': hashlib.sha256(adapter_path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--adapter', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = exercise(args.source, args.adapter)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
