"""Check native-loop prefix agreement and exact continuation after interruption."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def equal(a,b):
    if torch.is_tensor(a):
        return torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    full=ROOT/'runs/paper_expansion/acceptance/author_uninterrupted'
    resumed=ROOT/'runs/paper_expansion/acceptance/author_resumed'
    read=lambda folder:[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
    a,b=read(full),read(resumed)
    assert len(a)==len(b)==4
    for x,y in zip(a,b):
        for key in ('epoch','loss','test','best_epoch','count'):
            assert equal(x[key],y[key]),(key,x,y)
    aa=torch.load(full/'latest.pt',map_location='cpu')
    bb=torch.load(resumed/'latest.pt',map_location='cpu')
    for key in ('model_state','optimizer_state','rng','epoch','best_recall','best_epoch','best_metrics','count'):
        assert equal(aa[key],bb[key]),key
    log=(ROOT/'runs/official_anchor/douban-book/seed_2023/author_train.log').read_text(encoding='utf-8')
    recalls=re.findall(r'Recall:\[([^\]]+)\]',log)
    ndcgs=re.findall(r'NDCG:\s*\[([^\]]+)\]',log)
    parts=re.findall(r'train loss:\s*[0-9.e+-]+=([^\r\n]+)',log)
    for idx,x in enumerate(a):
        for metric,values in [('Recall',recalls),('NDCG',ndcgs)]:
            expected=[float(s) for s in values[idx].split()]
            for k,value in zip((10,20),expected):
                assert round(x['test'][metric+'@'+str(k)],8)==value
        expected=[float(s) for s in parts[idx].split('+')]
        assert [round(s,6) for s in x['loss']]==expected
    paths=[ROOT/'scripts/run_original_lightccf.py',ROOT/'scripts/validate_original_runner.py',
           ROOT/'runs/official_anchor/douban-book/seed_2023/author_train.log']
    paths+=list(full.glob('*.json*'))+list(resumed.glob('*.json*'))
    evidence={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output=dict(status='passed',epochs_compared=4,native_metric_decimal_places=8,
                native_loss_decimal_places=6,checkpoint_continuation='bitwise equal tensors, optimizer and RNG',
                torch=torch.__version__,numpy=np.__version__,evidence_sha256=evidence,
                caveat='Prefix identity validates the recording/resume wrapper; it does not establish full-paper reproduction.')
    (ROOT/'research/original_runner_acceptance.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
