"""Compact read-only progress for the locked follow-up matrix."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'runs/followup_20261101'
    counts={'formal':0,'validation_only':0}
    for path in base.rglob('result.json'):
        result=json.loads(path.read_text(encoding='utf-8'))
        if result.get('status')=='complete':
            counts['validation_only' if result['pilot'] else 'formal']+=1
    active=[]
    for path in base.rglob('console.log'):
        if path.with_name('result.json').exists():
            continue
        item={'run':path.parent.parent.name,'seed':path.parent.name}
        history=path.with_name('history.jsonl')
        if history.exists():
            points=[]
            for line in history.read_text(encoding='utf-8').splitlines():
                try:
                    points.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A writer may be between writing and closing a line.
            if points:
                item['epoch']=points[-1]['epoch']
                valid=[x for x in points if 'dev' in x]
                if valid:
                    best=max(valid,key=lambda x:x['dev']['NDCG@20'])
                    item['best_dev_epoch']=best['epoch']
                    item['best_dev_ndcg20']=best['dev']['NDCG@20']
        active.append(item)
    print(json.dumps({'complete':counts,'active':active},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
