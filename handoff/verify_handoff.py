"""Verify frozen handoff inputs using only Python's standard library.

Run this BEFORE regenerating the inventory on the receiving server.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'handoff/输入文件校验_20261102.json')
    args=parser.parse_args()
    inventory=json.loads(args.manifest.read_text(encoding='utf-8'))
    failures=[]
    checked=0
    for entry in inventory['files']:
        path=(ROOT/entry['path']).resolve()
        if ROOT not in path.parents:
            failures.append((entry['path'],'path outside project')); continue
        if not path.is_file():
            failures.append((entry['path'],'missing')); continue
        if path.stat().st_size!=entry['bytes']:
            failures.append((entry['path'],'size mismatch')); continue
        if digest(path)!=entry['sha256']:
            failures.append((entry['path'],'SHA256 mismatch')); continue
        checked+=1
    print(json.dumps(dict(status='failed' if failures else 'passed',files_verified=checked,
        expected_files=len(inventory['files']),snapshot_time=inventory['created_at'],failures=failures),
        ensure_ascii=False,indent=2))
    if failures:
        sys.exit(1)


if __name__=='__main__':
    main()
