#!/usr/bin/env python3
"""Static installed-runtime and public-model layout verification (not inference)."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct

ROOT = Path('/opt/haloloom/qwen38')
SITE = Path('/opt/venv/lib/python3.14/site-packages')
RECIPE_SHA = 'aabce8b815655175f7608a356fba2a2ec1c1396c7bbd622737c7eb54e3240ffd'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_installed():
    manifest = json.loads((ROOT / 'data/installed-files.json').read_text())
    for row in manifest['files']:
        path = Path(row['installed_path'])
        if not path.is_file() or sha(path) != row['sha256']:
            raise ValueError('installed runtime file mismatch: ' + str(path))
    versions = {k: importlib.metadata.version(k) for k in
                ['torch', 'triton', 'vllm', 'transformers', 'amd-aiter', 'flash_attn']}
    if versions['torch'] != '2.13.0+rocm10.0.0' or versions['triton'] != '3.8.0+git4cff872c.rocm10.0.0':
        raise ValueError('unsupported Torch/Triton ABI: ' + str(versions))
    return {'installed_files_verified': len(manifest['files']), 'versions': versions}


def check_index(folder):
    idx = json.loads((folder / 'model.safetensors.index.json').read_text())
    weights = idx['weight_map']
    seen = set()
    payload_bytes = 0
    for filename in sorted(set(weights.values())):
        path = folder / filename
        if path.resolve().parent != folder.resolve():
            raise ValueError('index requires flat, self-contained shards: ' + filename)
        with path.open('rb') as f:
            raw_size = f.read(8)
            if len(raw_size) != 8:
                raise ValueError('truncated safetensors prefix: ' + filename)
            length = struct.unpack('<Q', raw_size)[0]
            if not 2 <= length <= 100_000_000:
                raise ValueError('invalid safetensors header length: ' + filename)
            header = json.loads(f.read(length))
        for name, info in header.items():
            if name == '__metadata__':
                continue
            # Some source shards can retain non-indexed protected payloads. They
            # are never allowed to shadow an indexed tensor in another shard.
            if name not in weights:
                continue
            if weights[name] != filename or name in seen:
                raise ValueError('duplicate or misbound indexed tensor: ' + name)
            a, b = info['data_offsets']
            if not 0 <= a <= b <= path.stat().st_size - 8 - length:
                raise ValueError('invalid tensor data range: ' + name)
            seen.add(name)
            payload_bytes += b - a
    if seen != set(weights):
        raise ValueError('index tensor coverage mismatch')
    return {'indexed_tensors': len(seen), 'shards': len(set(weights.values())),
            'indexed_payload_bytes': payload_bytes}


def check_metadata(model):
    cfg = json.loads((model / 'config.json').read_text())
    for obj in [cfg, cfg.get('text_config', {})]:
        quant = obj.get('quantization_config', {})
        for key in ['mixed_moe', 'mtp_dense_bridge']:
            request = quant.get(key)
            if not request:
                continue
            if sha(Path(request['recipe_path'])) != RECIPE_SHA:
                raise ValueError('recipe digest mismatch')
            if key == 'mtp_dense_bridge':
                if request['manifest']['recipe_sha256'] != RECIPE_SHA:
                    raise ValueError('inline MTP recipe binding mismatch')
                binding = request.get('native_checkpoint')
                if binding:
                    checkpoint = Path(binding['path'])
                    if sha(checkpoint) != binding['sha256']:
                        raise ValueError('native MTP checkpoint digest mismatch')
                    data = json.loads(checkpoint.read_text())
                    if data['recipe_sha256'] != RECIPE_SHA or len(data['tensors']) != 3122:
                        raise ValueError('native MTP checkpoint completeness mismatch')


def verify(model, variant, mtp):
    report = verify_installed()
    identity = json.loads((model / 'haloloom-variant.json').read_text())
    if identity['runtime_variant'] != variant:
        raise ValueError('model folder and --variant disagree')
    check_metadata(model)
    report['target'] = check_index(model)
    if mtp:
        check_metadata(model / 'draft')
        report['draft'] = check_index(model / 'draft')
    report.update(status='STATIC_RUNTIME_AND_INDEX_CHECK_PASS', variant=variant,
                  inference_verified_by_this_check=False)
    return report


if __name__ == '__main__':
    print(json.dumps(verify_installed(), indent=2))
