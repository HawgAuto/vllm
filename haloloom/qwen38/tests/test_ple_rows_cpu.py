#!/usr/bin/env python3
"""Source-bound PLE n-gram index regression; no embedding or GPU claims."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DEVICE = 'cpu'
CANDIDATE = ROOT / 'ple_layer.py'
OUTPUT = ROOT / 'CPU-RESULT.json'


def load_prefix(path, name):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpNGramEmbedding')
    methods = [copy.deepcopy(n) for n in owner.body if isinstance(n, ast.FunctionDef)
               and n.name in ('forward', '_shift_precompute', '_shift_apply')]
    forward = next(n for n in methods if n.name == 'forward')
    end = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'ngram_ids' for t in n.targets))
    forward.body = forward.body[:end + 1] + [ast.Return(value=ast.Name(id='ngram_ids', ctx=ast.Load()))]
    cls = ast.ClassDef(name=name, bases=[ast.parse('torch.nn.Module', mode='eval').body],
                       keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {'torch': torch}
    exec(compile(module, str(path), 'exec'), scope)
    return scope[name]


def instance(cls, capacity):
    obj = cls()
    obj.eos_token_id = 0
    obj.ngram_size = 3
    obj.heads_per_ngram = 2
    obj.register_buffer('positions_buffer', torch.arange(512, dtype=torch.int64))
    obj.register_buffer('padded_buffer', torch.zeros((capacity, 512), dtype=torch.int64))
    obj.register_buffer('layer_multipliers', torch.tensor([1009, 9176, 65537], dtype=torch.int64))
    obj.register_buffer('ngram_heads_vocab_sizes', torch.tensor([101, 103, 107, 109], dtype=torch.int64))
    obj.register_buffer('ngram_heads_offsets', torch.tensor([0, 101, 204, 311], dtype=torch.int64))
    return obj.to(DEVICE)


def fixture(n, reverse=False):
    sequences = []
    contexts = []
    for i in range(n):
        seq = torch.arange(1, 2 + i % 7, dtype=torch.int64) + 13 * i
        if i % 3 == 0:
            seq[len(seq) // 2] = 0  # Segment boundary, not merely positive IDs.
        sequences.append(seq)
        contexts.append([0 if i % 2 == 0 else i + 3, i + 17])
    if reverse:
        sequences.reverse()
        contexts.reverse()
    starts = [0]
    for seq in sequences:
        starts.append(starts[-1] + len(seq))
    return tuple(x.to(DEVICE) for x in (torch.cat(sequences), torch.tensor(starts), torch.tensor(contexts)))


def main():
    old = load_prefix(ROOT / 'original_ple_layer.py', 'OriginalPrefix')
    new = load_prefix(CANDIDATE, 'CandidatePrefix')
    try:
        instance(old, 32)(*fixture(1))
    except RuntimeError as exc:
        assert 'Sizes of tensors must match' in str(exc)
        print('RED_REPRODUCED: capacity32/active1 n-gram concatenation', flush=True)
    else:
        raise AssertionError('original mismatch not reproduced')
    cases = 0
    candidate = instance(new, 32)
    for reverse in (False, True):
        for n in range(1, 33):
            inputs = fixture(n, reverse)
            reference = instance(old, n)(*inputs)
            output = candidate(*inputs)
            assert torch.equal(reference, output), (n, reverse)
            cases += 1
    for n in (1, 2, 4, 8, 16, 32):
        assert torch.equal(instance(old, n)(*fixture(n)), instance(new, n)(*fixture(n)))
    # Source-prefix tracing checks the repaired dynamic request axis. This is
    # the production n-gram arithmetic up to ngram_ids, not a copied oracle.
    n = torch.export.Dim('requests', min=1, max=32)
    t = torch.export.Dim('tokens', min=1, max=512)
    exported = torch.export.export(instance(new, 32), fixture(3), dynamic_shapes={
        'input_ids': {0: t}, 'query_start_loc': {0: n + 1}, 'ngram_context': {0: n}})
    graph = exported.module()
    for count in (1, 2, 4, 8, 16, 32):
        args = fixture(count)
        assert torch.equal(graph(*args), instance(old, count)(*args)), count
    receipt = {'status': 'PASS_SOURCE_PREFIX_CPU_AND_DYNAMIC_EXPORT',
               'source_sha256': hashlib.sha256((ROOT/'original_ple_layer.py').read_bytes()).hexdigest(),
               'candidate_sha256': hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
               'device': DEVICE,
               'old_failure_reproduced': True, 'eager_exact_cases': cases + 6,
               'exported_dynamic_cases': 6, 'active_request_counts': list(range(1,33)),
               'scope': 'exact integer n-gram indices through production forward prefix; no embedding/GPU/full-model claim'}
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
        receipt['status'] = 'PASS_SOURCE_PREFIX_GPU_AND_DYNAMIC_EXPORT'
        receipt['scope'] = 'exact integer n-gram indices on GPU plus exported graph; no embedding/full-model claim'
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    DEVICE, CANDIDATE, OUTPUT = args.device, args.candidate, args.out
    main()
