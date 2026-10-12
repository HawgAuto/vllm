#!/usr/bin/env python3
"""Launch the source-bound Qwen3.8 HaloLoom runtime inside its container."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('/opt/haloloom/qwen38')
MODEL = Path('/model')
WORK = Path('/workspace')


def configure(variant, mtp):
    env = json.loads((ROOT / 'data/runtime-env.json').read_text())
    env['DIRECT_W8A16'] = '1' if variant == 'int8' else '0'
    for key, value in env.items():
        os.environ[key] = str(value)
    for subdir in ['cache', 'tmp', 'config', 'profiles']:
        (WORK / subdir).mkdir(parents=True, exist_ok=True)
    for filename, value in [
        ('moe-ab-phase.json', {'arm': 'wmma5', 'case': 'public-serving'}),
        ('ple-ab-phase.json', {'arm': 'native', 'case': 'public-serving'}),
        ('nb-phase.json', {'nb': 9, 'case': 'public-serving'}),
    ]:
        (WORK / filename).write_text(json.dumps(value) + '\n')


def command(args):
    argv = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', str(MODEL), '--tokenizer', str(MODEL),
            '--served-model-name', 'haloloom-qwen38',
            '--host', args.host, '--port', str(args.port),
            '--trust-remote-code', '--dtype', 'bfloat16', '--quantization', 'quark',
            '--moe-backend', 'triton', '--load-format', 'safetensors',
            '--safetensors-load-strategy', 'lazy', '--pipeline-parallel-size', '1',
            '--max-num-batched-tokens', str(args.max_num_batched_tokens), '--max-num-seqs', str(args.max_num_seqs),
            '--gpu-memory-utilization', '0.7', '--kv-cache-memory-bytes', '5905580032',
            '--limit-mm-per-prompt', '{"image":{"count":1,"height":224,"width":224},"video":0}',
            '--enable-chunked-prefill',
            '--enable-prefix-caching' if args.max_num_seqs == 1 else '--no-enable-prefix-caching',
            '--mamba-cache-mode', 'align', '--generation-config', 'vllm',
            '--reasoning-parser', 'qwen3', '--enable-prompt-tokens-details',
            '--enable-per-request-metrics', '--max-model-len', str(args.context)]
    if args.eager:
        argv += ['--enforce-eager']
    else:
        argv += ['--compilation-config', json.dumps({
            'cudagraph_capture_sizes': [1, 2, 3, 4], 'cudagraph_mode': 'PIECEWISE',
            'max_cudagraph_capture_size': 4})]
    if args.mtp:
        argv += ['--speculative-config', json.dumps({
            'method': 'mtp', 'model': str(MODEL / 'draft'),
            'num_speculative_tokens': args.mtp})]
    return argv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=['bf16', 'int8'], required=True)
    ap.add_argument('--mtp', type=int, choices=[0, 3], default=3)
    ap.add_argument('--context', type=int, default=131072)
    ap.add_argument('--max-num-seqs', type=int, default=1)
    ap.add_argument('--max-num-batched-tokens', type=int, default=12288)
    ap.add_argument('--eager', action='store_true', help='Disable compiled graphs for diagnosis')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--cpu-check', action='store_true', help='Validate installed source and model indices without GPU inference')
    ap.add_argument('--print-command', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.context <= 131072:
        ap.error('context must be between 1 and 131072')
    if not 1 <= args.max_num_seqs <= 32:
        ap.error('max-num-seqs must be between 1 and 32')
    if args.mtp and args.max_num_seqs != 1:
        ap.error('MTP is currently qualified only with --max-num-seqs 1; use --mtp 0 for batching')
    if args.max_num_batched_tokens < 1:
        ap.error('max-num-batched-tokens must be positive')
    configure(args.variant, args.mtp)
    (WORK / 'MEMORY-POLICY.json').write_text(json.dumps({
        'schema': 'haloloom.explicit-kv-policy.v1', 'kv_cache_memory_bytes': 5905580032,
        'gpu_memory_utilization': .7, 'utilization_is_startup_check_not_kv_sizing': True,
        'max_model_len': args.context, 'max_num_seqs': args.max_num_seqs,
        'max_num_batched_tokens': args.max_num_batched_tokens, 'host_memavailable_reserve_bytes': 17179869184,
        'kv_precision': 'BF16', 'scope': 'explicit fixed pool; no automatic memory-budget promotion'}, indent=2) + '\n')
    argv = command(args)
    if args.print_command:
        print(json.dumps({'argv': argv, 'variant': args.variant, 'mtp': args.mtp}, indent=2))
        return
    from validate import verify
    report = verify(MODEL, args.variant, args.mtp)
    print(json.dumps(report), flush=True)
    if args.cpu_check:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm device unavailable: pass /dev/kfd and the gfx1151 render node')
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    if arch != 'gfx1151':
        raise RuntimeError(f'This native kernel bundle is qualified only for gfx1151, not {arch}')
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
