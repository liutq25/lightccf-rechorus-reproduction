"""Compare a complete fixed-seed trajectory across the old/recommended environments."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    old=ROOT/'runs/official_anchor/douban-book/seed_2023'
    new=ROOT/'runs/paper_expansion/main/douban-book/seed_2023'
    log=(old/'author_train.log').read_text(encoding='utf-8')
    result=json.loads((new/'result.json').read_text(encoding='utf-8'))
    previous=json.loads((old/'result.json').read_text(encoding='utf-8'))
    rows=[json.loads(s) for s in (new/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    recalls=re.findall(r'Recall:\[([^\]]+)\]',log)
    ndcgs=re.findall(r'NDCG:\s*\[([^\]]+)\]',log)
    loss=re.findall(r'train loss:\s*[0-9.e+-]+=([^\r\n]+)',log)
    assert len(rows)==len(recalls)==len(ndcgs)==54
    metric_differences=[];loss_differences=[]
    for idx,row in enumerate(rows):
        for metric,values in [('Recall',recalls),('NDCG',ndcgs)]:
            expected=[float(s) for s in values[idx].split()]
            metric_differences.extend(abs(round(row['test'][metric+'@'+str(k)],8)-x)
                                      for k,x in zip((10,20),expected))
        loss_differences.extend(abs(round(x,6)-float(y)) for x,y in zip(row['loss'],loss[idx].split('+')))
    same_best=(previous['best_epoch_by_test_recall20']==result['best_epoch'] and
               previous['test_recall20_at_best']==result['test']['Recall@20'] and
               previous['test_ndcg20_at_best']==result['test']['NDCG@20'])
    evidence=[old/'result.json',old/'author_train.log',new/'result.json',new/'history.jsonl',new/'config.json']
    output=dict(dataset='douban-book',seed=2023,epochs_compared=54,
        old=dict(python='3.11',torch=previous['torch'],numpy=previous['numpy']),
        new=dict(python='3.8.18',torch='2.1.0+cu121',numpy='1.24.3'),
        max_absolute_metric_difference_at_native_8dp=max(metric_differences),
        max_absolute_loss_component_difference_at_native_6dp=max(loss_differences),
        full_precision_best_epoch_recall20_ndcg20_equal=same_best,
        interpretation='On this full fixed-seed MF/Douban trajectory, changing to the recommended environment did not change the best result. This is not evidence of environment equivalence for all models/datasets.',
        evidence_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence})
    (ROOT/'research/original_environment_comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
