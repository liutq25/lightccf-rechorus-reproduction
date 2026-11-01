"""Lock this workspace's completed baseline records before a follow-up run."""
from pathlib import Path
from collections import Counter
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/followup_20261101'
SOURCES=('research/followup_plan_20261101.md','scripts/run_followup_matrix.py',
         'scripts/run_experiment.py','extensions/LightCCF.py',
         'extensions/LightCCFMasked.py','extensions/ReproductionRunner.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock(path,value):
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        if previous!=value:
            raise RuntimeError(f'Locked inputs changed: {path}; audit the change before continuing.')
    else:
        path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    records={}
    counts=Counter()
    for group in ('core','ablation','official_anchor','extension'):
        for path in (ROOT/'runs'/group).rglob('result.json'):
            value=json.loads(path.read_text(encoding='utf-8'))
            if value['status']!='complete':
                raise RuntimeError(f'Incomplete baseline record: {path}')
            counts['validation' if value.get('pilot') else group]+=1
            records[path.relative_to(ROOT).as_posix()]=sha(path)
    if counts!=Counter(core=18,ablation=6,official_anchor=5,extension=12,validation=10):
        raise RuntimeError('Complete the original core/ablation, author anchor, and extension matrices first. '+str(dict(counts)))
    OUT.mkdir(parents=True,exist_ok=True)
    lock(OUT/'source_hashes.json',{p:sha(ROOT/p) for p in SOURCES})
    lock(OUT/'previous_result_sha256.json',records)
    print(f'Locked {len(records)} prior records and {len(SOURCES)} source files.')


if __name__=='__main__':
    main()
