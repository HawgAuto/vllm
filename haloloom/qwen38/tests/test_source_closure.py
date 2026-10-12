#!/usr/bin/env python3
"""Fail closed if Git ignores or omits a source-bound runtime member."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]

def main():
    manifest=json.loads((ROOT/'haloloom/qwen38/data/installed-files.json').read_text())
    tracked=set(subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0'))
    checked=0;errors=[]
    for row in manifest['files']:
        source=row['source']
        if source.startswith('native-wheel:'):continue  # verified separately by installed-image gate
        path=ROOT/source
        if source not in tracked:errors.append('untracked runtime member: '+source)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            errors.append('missing or mismatched runtime member: '+source)
        checked+=1
    print(json.dumps({'status':'FAIL' if errors else 'PASS_SOURCE_CLOSURE','source_members_checked':checked,'errors':errors},indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
