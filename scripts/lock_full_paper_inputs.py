"""Preserve the complete previous study before original-paper expansion."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/paper_expansion'


def main():
    previous={}
    for group in ('core','ablation','official_anchor','extension','followup_20261101'):
        for path in (ROOT/'runs'/group).rglob('result.json'):
            value=json.loads(path.read_text(encoding='utf-8'))
            assert value['status']=='complete'
            previous[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(previous)==96
    OUT.mkdir(exist_ok=True)
    path=OUT/'previous_result_sha256.json'
    if path.exists():
        assert json.loads(path.read_text())==previous
    else:
        path.write_text(json.dumps(previous,indent=2),encoding='utf-8')
    plan=ROOT/'research/full_paper_plan_20261102.md'
    locked=OUT/'plan_sha256.json'
    value={plan.relative_to(ROOT).as_posix():hashlib.sha256(plan.read_bytes()).hexdigest()}
    if locked.exists():
        assert json.loads(locked.read_text())==value
    else:
        locked.write_text(json.dumps(value,indent=2),encoding='utf-8')
    print('Locked 96 previous result records and the expansion plan.')


if __name__=='__main__':
    main()
