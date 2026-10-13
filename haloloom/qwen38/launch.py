#!/usr/bin/env python3
"""Launch the source-bound Qwen3.8 HaloLoom runtime inside its container.

Profiles are explicit: context length, KV pool size, MTP and sampling defaults are
printed by --print-command so every server argument is visible before launch.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/opt/haloloom/qwen38')
MODEL = Path('/model')
WORK = Path('/workspace')
SEEDS = ROOT / 'seeds'

NATIVE_CONTEXT = 262144
MAX_CONTEXT = 1048576
DEFAULT_CONTEXT = NATIVE_CONTEXT         # the shipped default seeds are built at this length
# 'max' profile: largest context proven on gfx1151/128 GB per variant x MTP mode, with the
# explicit KV pool used in that proof. Filled from measured long-context runs only.
MAX_PROFILE = {
    ('int8', 0): {'context': 1048576, 'kv_gib': 25.5},
    ('int8', 3): {'context': 917504, 'kv_gib': 24.75},
    ('bf16', 0): {'context': 851968, 'kv_gib': 21.0},
    ('bf16', 3): {'context': 720896, 'kv_gib': 19.75},
}
MAX_YARN_FACTOR = 4.0
DEFAULT_KV_BYTES = 5905580032          # 5.5 GiB: the qualified default pool
# BF16 KV pool model: fixed bytes + marginal bytes per token of single-request capacity,
# from two measured pools per mode on this runtime:
#   ordinary 5.5 GiB -> 208,587 tokens, 28.5 GiB -> 1,190,860 tokens
#   MTP3     5.5 GiB -> 170,520 tokens, 27.0 GiB -> 1,015,750 tokens
_ORD_MARGINAL = (28.5 - 5.5) * (1 << 30) / (1190860 - 208587)
_MTP_MARGINAL = (27.0 - 5.5) * (1 << 30) / (1015750 - 170520)
KV_MODEL = {0: (DEFAULT_KV_BYTES - 208587 * _ORD_MARGINAL, _ORD_MARGINAL),
            3: (DEFAULT_KV_BYTES - 170520 * _MTP_MARGINAL, _MTP_MARGINAL)}
KV_HEADROOM = 1.01                       # block rounding / mamba page alignment margin

# Qwen's published sampling recommendations for Qwen3.8-Flash-Next.
SAMPLING = {
    'thinking': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
                 'presence_penalty': 0.0, 'repetition_penalty': 1.0},
    'instruct': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0,
                 'presence_penalty': 1.5, 'repetition_penalty': 1.0},
}


def yarn_overrides(factor):
    """Qwen's static YaRN rope override (text_config.rope_parameters)."""
    return {'text_config': {'rope_parameters': {
        'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'rope_type': 'yarn',
        'rope_theta': 10000000, 'partial_rotary_factor': 0.25, 'factor': float(factor),
        'original_max_position_embeddings': NATIVE_CONTEXT}}}


def kv_bytes_for(context, mtp):
    fixed, per_token = KV_MODEL[mtp]
    need = math.ceil((fixed + context * per_token) * KV_HEADROOM)
    gib = 1 << 30
    return max(DEFAULT_KV_BYTES, math.ceil(need / (gib // 4)) * (gib // 4))


def seed_cache(variant, cache_name, cache_root, seed_arg):
    """Copy the image's prebuilt compile/autotune cache once per cache root.

    vLLM's compile-cache key includes max_model_len but NOT the dense-core variant and
    NOT --hf-overrides (YaRN). So every variant gets its own cache root, and YaRN runs use
    a separate root/seed per YaRN factor. Shipped seeds:
      seeds/<variant>/           default-rope graphs at 262,144 (ordinary + MTP3)
      seeds/<variant>-yarn4/     YaRN x4 graphs at each 'max' profile context
    When no exact seed exists for a cache root, only kernel/autotune files from the
    variant's default seed are copied (never compiled graphs), so the first start
    compiles its graph once but reuses every tuned Triton kernel.
    """
    marker = cache_root / '.haloloom-seed.json'
    if seed_arg == 'none':
        return {'seed': 'disabled'}
    if marker.exists():
        return json.loads(marker.read_text())
    kernels_only = False
    if seed_arg not in (None, 'auto'):
        src = Path(seed_arg)
    elif (SEEDS / cache_name).is_dir():
        src = SEEDS / cache_name
    else:
        src, kernels_only = SEEDS / variant, True
    if not src.is_dir():
        return {'seed': 'unavailable', 'path': str(src)}
    copied = 0
    for path in src.rglob('*'):
        rel = path.relative_to(src)
        if kernels_only and 'torch_aot_compile' in rel.parts:
            continue
        dst = cache_root / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied += 1
    info = {'seed': str(src), 'variant': variant, 'cache': cache_name,
            'kernels_only': kernels_only, 'files_copied': copied}
    meta = src / 'SEED.json'
    if meta.exists():
        info['seed_meta'] = json.loads(meta.read_text())
    marker.write_text(json.dumps(info, indent=2) + '\n')
    return info


def configure(args):
    env = json.loads((ROOT / 'data/runtime-env.json').read_text())
    env['DIRECT_W8A16'] = '1' if args.variant == 'int8' else '0'
    # vLLM's compile-cache key ignores --hf-overrides, so YaRN runs get their own
    # cache root and never reuse (or seed from) default-rope compiled graphs.
    cache_name = args.variant + (f'-yarn{args.yarn_factor:g}' if args.yarn_factor else '')
    cache_root = WORK / 'cache' / cache_name
    for key in ['XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                'AITER_JIT_DIR', 'TVM_FFI_CACHE_DIR']:
        if key in env and str(env[key]).startswith('/workspace/cache'):
            env[key] = str(cache_root) + str(env[key])[len('/workspace/cache'):]
    if args.context > NATIVE_CONTEXT:
        env['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    for key, value in env.items():
        os.environ[key] = str(value)
    for subdir in ['tmp', 'config', 'profiles']:
        (WORK / subdir).mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    for filename, value in [
        ('moe-ab-phase.json', {'arm': 'wmma5', 'case': 'public-serving'}),
        ('ple-ab-phase.json', {'arm': 'native', 'case': 'public-serving'}),
        ('nb-phase.json', {'nb': 9, 'case': 'public-serving'}),
    ]:
        (WORK / filename).write_text(json.dumps(value) + '\n')
    return cache_name, cache_root


def command(args):
    # Optional attributes default to the shipped profile so older callers that build
    # a partial namespace (tests/test_launch_contract.py) keep their exact contract.
    args.served_model_name = getattr(args, 'served_model_name', 'haloloom-qwen38')
    args.gpu_memory_utilization = getattr(args, 'gpu_memory_utilization', 0.7)
    args.kv_cache_memory_bytes = getattr(args, 'kv_cache_memory_bytes', DEFAULT_KV_BYTES)
    args.sampling = getattr(args, 'sampling', 'thinking')
    args.yarn_factor = getattr(args, 'yarn_factor', None)
    args.tool_call_parser = getattr(args, 'tool_call_parser', 'qwen3_coder')
    argv = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', str(MODEL), '--tokenizer', str(MODEL),
            '--served-model-name', args.served_model_name,
            '--host', args.host, '--port', str(args.port),
            '--trust-remote-code', '--dtype', 'bfloat16', '--quantization', 'quark',
            '--moe-backend', 'triton', '--load-format', 'safetensors',
            '--safetensors-load-strategy', 'lazy', '--pipeline-parallel-size', '1',
            '--max-num-batched-tokens', str(args.max_num_batched_tokens), '--max-num-seqs', str(args.max_num_seqs),
            '--gpu-memory-utilization', str(args.gpu_memory_utilization),
            '--kv-cache-memory-bytes', str(args.kv_cache_memory_bytes),
            '--limit-mm-per-prompt', '{"image":{"count":1,"height":224,"width":224},"video":0}',
            '--enable-chunked-prefill',
            '--enable-prefix-caching' if args.max_num_seqs == 1 else '--no-enable-prefix-caching',
            '--mamba-cache-mode', 'align',
            '--reasoning-parser', 'qwen3', '--enable-prompt-tokens-details',
            '--enable-per-request-metrics', '--max-model-len', str(args.context)]
    # The chat template emits Qwen's XML tool calls (<tool_call><function=...><parameter=...>),
    # parsed by vLLM's qwen3_coder parser. API-server option only: not in the compile-cache key.
    if args.tool_call_parser != 'none':
        argv += ['--enable-auto-tool-choice', '--tool-call-parser', args.tool_call_parser]
    # The model's generation_config.json already carries Qwen's thinking-mode values
    # (temperature 1.0, top_p 0.95, top_k 20; min_p/presence/repetition stay at neutral
    # defaults), so 'thinking' needs no override. Generation-config flags are part of
    # vLLM's compile-cache key: the shipped seeds match the default 'thinking' profile.
    if args.sampling == 'vllm':
        argv += ['--generation-config', 'vllm']
    elif args.sampling == 'thinking':
        argv += ['--generation-config', 'auto']
    else:
        argv += ['--generation-config', 'auto',
                 '--override-generation-config', json.dumps(SAMPLING[args.sampling])]
    if args.yarn_factor:
        argv += ['--hf-overrides', json.dumps(yarn_overrides(args.yarn_factor))]
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
    ap.add_argument('--profile', choices=['default', 'max'], default='default',
                    help='default: 262,144 native context (seeded quick start); '
                         'max: largest proven YaRN context for this variant/MTP mode (seeded)')
    ap.add_argument('--context', type=int, default=None,
                    help=f'override max model length; above {NATIVE_CONTEXT} enables YaRN (max {MAX_CONTEXT}); '
                         'contexts other than the profile values start with a one-time graph compile')
    ap.add_argument('--yarn-factor', type=float, default=None,
                    help='static YaRN factor; default when --context exceeds native is context/262144 rounded up to 2 or 4')
    ap.add_argument('--kv-cache-gib', type=float, default=None,
                    help='explicit BF16 KV pool; default 5.5 GiB, or sized to one full --context request when larger')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.7,
                    help='startup free-memory check only; KV is sized by the explicit pool')
    ap.add_argument('--sampling', choices=['thinking', 'instruct', 'vllm'], default='thinking',
                    help='default request sampling: Qwen thinking/instruct recommendation, or vLLM neutral defaults')
    ap.add_argument('--seed', default='auto',
                    help="compile/autotune cache seed: 'auto' (image seed for the variant), 'none', or a directory")
    ap.add_argument('--max-num-seqs', type=int, default=1)
    ap.add_argument('--max-num-batched-tokens', type=int, default=12288)
    ap.add_argument('--served-model-name', default='haloloom-qwen38')
    ap.add_argument('--tool-call-parser', choices=['qwen3_coder', 'none'], default='qwen3_coder',
                    help="tool calling for the OpenAI API: qwen3_coder (matches the chat template) or 'none'")
    ap.add_argument('--eager', action='store_true', help='Disable compiled graphs for diagnosis')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--cpu-check', action='store_true', help='Validate installed source and model indices without GPU inference')
    ap.add_argument('--print-command', action='store_true')
    args = ap.parse_args()
    if args.profile == 'max':
        prof = MAX_PROFILE[(args.variant, args.mtp)]
        if prof is None:
            ap.error(f'no qualified max-context profile for {args.variant} mtp={args.mtp}')
        if args.context is None:
            args.context = prof['context']
        if args.kv_cache_gib is None:
            args.kv_cache_gib = prof['kv_gib']
        if args.yarn_factor is None:
            args.yarn_factor = MAX_YARN_FACTOR
    if args.context is None:
        args.context = DEFAULT_CONTEXT
    if not 1 <= args.context <= MAX_CONTEXT:
        ap.error(f'context must be between 1 and {MAX_CONTEXT}')
    if args.context > NATIVE_CONTEXT and args.yarn_factor is None:
        args.yarn_factor = 2.0 if args.context <= 2 * NATIVE_CONTEXT else 4.0
    if args.yarn_factor is not None:
        if not 1.0 < args.yarn_factor <= 4.0:
            ap.error('yarn-factor must be in (1, 4]')
        if args.context > args.yarn_factor * NATIVE_CONTEXT:
            ap.error('context exceeds yarn-factor * 262144')
    if not 1 <= args.max_num_seqs <= 32:
        ap.error('max-num-seqs must be between 1 and 32')
    if args.mtp and args.max_num_seqs != 1:
        ap.error('MTP is currently qualified only with --max-num-seqs 1; use --mtp 0 for batching')
    if args.max_num_batched_tokens < 1:
        ap.error('max-num-batched-tokens must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        ap.error('gpu-memory-utilization must be in (0, 1)')
    args.kv_cache_memory_bytes = (int(args.kv_cache_gib * (1 << 30)) if args.kv_cache_gib
                                  else kv_bytes_for(args.context, args.mtp))
    cache_name, cache_root = configure(args)
    (WORK / 'MEMORY-POLICY.json').write_text(json.dumps({
        'schema': 'haloloom.explicit-kv-policy.v1', 'kv_cache_memory_bytes': args.kv_cache_memory_bytes,
        'gpu_memory_utilization': args.gpu_memory_utilization, 'utilization_is_startup_check_not_kv_sizing': True,
        'max_model_len': args.context, 'max_num_seqs': args.max_num_seqs, 'yarn_factor': args.yarn_factor,
        'max_num_batched_tokens': args.max_num_batched_tokens, 'host_memavailable_reserve_bytes': 17179869184,
        'kv_precision': 'BF16', 'scope': 'explicit fixed pool; no automatic memory-budget promotion'}, indent=2) + '\n')
    argv = command(args)
    if args.print_command:
        print(json.dumps({'argv': argv, 'variant': args.variant, 'mtp': args.mtp,
                          'kv_cache_gib': round(args.kv_cache_memory_bytes / (1 << 30), 3),
                          'context': args.context, 'yarn_factor': args.yarn_factor,
                          'cache_root': str(cache_root)}, indent=2))
        return
    from validate import verify
    report = verify(MODEL, args.variant, args.mtp)
    print(json.dumps(report), flush=True)
    if args.cpu_check:
        return
    print(json.dumps({'cache_seed': seed_cache(args.variant, cache_name, cache_root, args.seed)}), flush=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm device unavailable: pass /dev/kfd and the gfx1151 render node')
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    if arch != 'gfx1151':
        raise RuntimeError(f'This native kernel bundle is qualified only for gfx1151, not {arch}')
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
