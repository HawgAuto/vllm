#!/usr/bin/env python3
"""Exercise the portable launch contract without importing GPU model code."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_launch', ROOT / 'launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def main():
    for streams in (1, 2, 4, 8, 16, 32):
        args = SimpleNamespace(host='127.0.0.1', port=8000, max_num_seqs=streams,
                               max_num_batched_tokens=12288, context=4096,
                               eager=False, mtp=0)
        argv = launch.command(args)
        assert '--enforce-eager' not in argv
        assert '--compilation-config' in argv
        assert argv[argv.index('--max-num-seqs') + 1] == str(streams)
        assert ('--enable-prefix-caching' in argv) == (streams == 1)
        assert ('--no-enable-prefix-caching' in argv) == (streams > 1)
        assert '--speculative-config' not in argv
    args.max_num_seqs = 1
    args.mtp = 3
    argv = launch.command(args)
    assert json.loads(argv[argv.index('--speculative-config') + 1]) == {
        'method': 'mtp', 'model': '/model/draft', 'num_speculative_tokens': 3}
    args.mtp = 0
    args.max_num_batched_tokens = 2048
    assert launch.command(args)[launch.command(args).index('--max-num-batched-tokens') + 1] == '2048'
    print('PASS: six ordinary launch geometries, singleton MTP3, explicit chunk size')


if __name__ == '__main__':
    main()
