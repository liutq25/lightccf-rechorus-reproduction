"""Summarize only completed original-paper runs and the author's sparsity groups."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import hashlib
import io
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/LightCCF-main'))
from utility.data_loader import Data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paper=json.loads((ROOT/'research/paper_table5.json').read_text())
    output=dict(status='partial',scope='original-paper expansion, separate from course experiment',datasets={})
    names=dict(zip(('douban-book','tmall','amazon-book'),paper['datasets']))
    for dataset,title in names.items():
        with redirect_stdout(io.StringIO()):
            data=Data(SimpleNamespace(dataset_path=str(ROOT/'vendor/LightCCF-main/dataset')+'/',
                                      dataset=dataset,dataset_type='.txt',sparsity_test=1))
        groups=data.split_test_dict
        assert len(groups)==3 and all(groups)
        flat=[int(u) for g in groups for u in g]
        assert len(flat)==len(set(flat))==len(data.test_dict) and set(flat)==set(data.test_dict)
        record=dict(paper=dict(zip(paper['metric_order'],paper['rows']['LightCCF'][paper['datasets'].index(title)])),
                    completed_seeds=[],runs=[],sparsity_groups=[],selection_uses_test=True)
        grouped=[]
        for idx,g in enumerate(groups):
            degree=[len(data.all_positive[u])+len(data.test_dict[u]) for u in g]
            record['sparsity_groups'].append(dict(index=idx,label=('sparse','normal','popular')[idx],
                users=len(g),total_degree_min=min(degree),total_degree_max=max(degree),
                membership_sha256=hashlib.sha256(json.dumps([int(u) for u in g]).encode()).hexdigest()))
        for folder in sorted((ROOT/'runs/paper_expansion/main'/dataset).glob('seed_*')):
            path=folder/'result.json'
            if not path.exists():
                continue
            result=json.loads(path.read_text())
            if result['status']!='complete':
                continue
            assert not result['diagnostic'] and result['na_mode']=='full'
            assert sha(folder/'config.json')==result['config_sha256']
            history=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in history]==list(range(1,result['epochs']+1)),'Incomplete/duplicate epoch log: audit interruption before using this run'
            best=max(history,key=lambda x:x['test']['Recall@20'])
            assert best['epoch']==result['best_epoch'] and best['test']==result['test']
            users=json.loads((folder/'test_users.json').read_text())
            lookup={r['user_id']:r for r in users}
            assert set(lookup)==set(data.test_dict)
            for metric,value in result['test'].items():
                assert abs(np.mean([r[metric] for r in users])-value)<1e-10
            grouped.append([{metric:float(np.mean([lookup[int(u)][metric] for u in g]))
                             for metric in paper['metric_order']} for g in groups])
            record['completed_seeds'].append(result['seed'])
            record['runs'].append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),
                test=result['test'],epochs=result['epochs'],best_epoch=result['best_epoch'],
                wall_seconds=result['wall_seconds'],stopped_by_patience=result['stopped_by_patience']))
        n=len(record['runs'])
        record['five_seed_main_complete']=n==5
        if n:
            record['metrics']={}
            for metric in paper['metric_order']:
                values=[r['test'][metric] for r in record['runs']]
                mean=float(np.mean(values)); reference=record['paper'][metric]
                record['metrics'][metric]=dict(mean=mean,sample_std=float(np.std(values,ddof=1)) if n>1 else None,
                    seeds=n,paper=reference,absolute_difference=mean-reference,
                    relative_difference_percent=100*(mean/reference-1))
            for idx in range(3):
                record['sparsity_groups'][idx]['metrics']={metric:dict(
                    mean=float(np.mean([g[idx][metric] for g in grouped])),
                    sample_std=float(np.std([g[idx][metric] for g in grouped],ddof=1)) if n>1 else None)
                    for metric in paper['metric_order']}
        output['datasets'][dataset]=record
    if all(v['five_seed_main_complete'] for v in output['datasets'].values()):
        output['status']='main_lightccf_complete_other_paper_experiments_still_pending'
    output['limitations']=['Test Recall@20 selects epochs to reproduce the published entry point; these scores are not an unbiased held-out evaluation.',
        'Unknown original random seeds and unreported original standard deviations prevent claiming statistical equivalence.',
        'Amazon clean graph corrects four released-cache test edges; it cannot be assumed that the paper used that released cache.',
        'Sparsity groups use total train+test user degree, exactly as the dedicated author code; this uses test information for descriptive grouping only.']
    target=ROOT/'analysis/original_paper_expansion.json'
    target.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:dict(completed=v['completed_seeds'],metrics=v.get('metrics')) for k,v in output['datasets'].items()},indent=2))


if __name__=='__main__':
    main()
