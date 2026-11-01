"""Audit the locked follow-up and export fresh-seed and masking comparisons."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import statistics as stats
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/followup_20261101'
ANALYSIS=ROOT/'analysis'
METRICS=('Recall@10','NDCG@10','Recall@20','NDCG@20')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def result(path, pilot=False):
    value=read(path)
    assert value['status']=='complete' and value['pilot']==pilot, str(path)
    if pilot:
        assert value['test'] is None
    else:
        history=[json.loads(s) for s in path.with_name('history.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(history)==value['epochs']
        best=next(x for x in history if x['epoch']==value['best_epoch'])
        assert best['dev']==value['dev']
        assert best['dev']['NDCG@20']==max(x['dev']['NDCG@20'] for x in history if 'dev' in x)
        for phase in ('test','dev'):
            users=read(path.with_name(phase+'_users.json'))
            for key,score in value[phase].items():
                assert abs(stats.mean(row[key] for row in users)-score)<1e-12
    return value


def summarize(paths):
    rows=[result(p) for p in paths]
    return dict(seeds=[r['seed'] for r in rows],best_epochs=[r['best_epoch'] for r in rows],
                epochs=[r['epochs'] for r in rows],
                test={key:dict(mean=stats.mean(r['test'][key] for r in rows),
                               sample_sd=stats.stdev(r['test'][key] for r in rows)) for key in METRICS},
                mean_epoch_seconds=stats.mean(r['mean_epoch_seconds'] for r in rows),
                hit_cap=sum(not r['stopped_by_patience'] for r in rows),
                sources=[p.relative_to(ROOT).as_posix() for p in paths])


def paired(left,right):
    a,b=[result(p) for p in left],[result(p) for p in right]
    assert [x['seed'] for x in a]==[x['seed'] for x in b]
    changes=[x['test']['NDCG@20']-y['test']['NDCG@20'] for x,y in zip(a,b)]
    left_users=[{r['user_id']:r['NDCG@20'] for r in read(p.with_name('test_users.json'))} for p in left]
    right_users=[{r['user_id']:r['NDCG@20'] for r in read(p.with_name('test_users.json'))} for p in right]
    ids=sorted(left_users[0])
    assert all(set(row)==set(ids) for row in left_users+right_users)
    delta=np.array([stats.mean(x[u]-y[u] for x,y in zip(left_users,right_users)) for u in ids])
    assert abs(float(delta.mean())-stats.mean(changes))<1e-12
    rng=np.random.default_rng(9117)
    boot=[]
    for _ in range(10000//50):
        indices=rng.integers(0,len(ids),size=(50,len(ids)))
        boot.extend(delta[indices].mean(1))
    return dict(mean=stats.mean(changes),sample_sd=stats.stdev(changes),per_seed=changes,
                conditional_user_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
                bootstrap_resamples=10000,bootstrap_seed=9117,bootstrap_percentiles=[2.5,97.5],
                bootstrap_scope='resample observed test users jointly, conditional on three fixed training seeds; not population or training uncertainty',
                observed_test_users=len(ids))


def main():
    upstream=ROOT/'framework/ReChorus'
    head=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    assert head=='c164ec4303cc20ddcfbd1b57de366a481811d1e5'
    dirty=subprocess.check_output(['git','-C',str(upstream),'diff','--name-only','--','src'],text=True).strip()
    assert not dirty, f'Upstream training sources changed: {dirty}'
    previous_manifest=OUT/'previous_result_sha256.json'
    if not previous_manifest.exists():
        raise RuntimeError('Missing local input lock. Use scripts/reproduce_followup.py to run the full workflow.')
    for relative,digest in read(previous_manifest).items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
    for relative,digest in read(OUT/'source_hashes.json').items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
    selections=read(OUT/'validation_selection.json')
    assert len(selections)==6
    independent=OUT/'independent_selection.json'
    if independent.exists():
        parallel=read(independent)
        expected=next(s['selected'] for s in selections if s['dataset_prefix']=='ml' and s['model']=='LightCCF')
        assert not parallel['selection_uses_test']
        for key in ('id','config','dev','source'):
            assert parallel['selected'][key]==expected[key]
    rows={}
    fresh={}
    reused=0
    search_cap_hits=0
    for selection in selections:
        prefix,model=selection['dataset_prefix'],selection['model']
        assert not selection['selection_uses_test'] and len(selection['candidates'])==4
        chosen=sorted(selection['candidates'],key=lambda x:(-x['dev'],x['id']))[0]
        assert chosen==selection['selected']
        for candidate in selection['candidates']:
            source=ROOT/candidate['source']
            value=read(source)
            assert candidate['dev']==value['dev']['NDCG@20']
            search_cap_hits+=int(value['epochs']==read(source.with_name('config.json'))['epoch'])
            if candidate['reused']:
                reused+=1
                assert value['epochs']<read(source.with_name('config.json'))['epoch']
            else:
                result(source,pilot=True)
        paths=[OUT/f'confirm_{prefix}_{model.lower()}'/f'seed_{seed}'/'result.json' for seed in (2026,2027,2028)]
        for path in paths:
            config=read(path.with_name('config.json'))
            assert all(config[k]==v for k,v in chosen['config'].items())
        fresh[prefix,model]=paths
        rows[prefix+'_'+model]=dict(selected_id=chosen['id'],config=chosen['config'],**summarize(paths))
    comparisons={}
    for prefix in ('grocery','ml'):
        comparisons[prefix]=paired(fresh[prefix,'LightCCF'],fresh[prefix,'LightGCN'])
        comparisons[prefix]['relative_ndcg20_percent']=100*(rows[prefix+'_LightCCF']['test']['NDCG@20']['mean']/rows[prefix+'_LightGCN']['test']['NDCG@20']['mean']-1)
    masks={}
    for prefix in ('grocery','ml'):
        base=ROOT/('runs/extension/grocery_lightccf_long' if prefix=='grocery' else 'runs/core/ml_lightccf')
        paths={'full':[base/f'seed_{seed}'/'result.json' for seed in (2023,2024,2025)]}
        for path in paths['full']:
            original=result(path)
            config=read(path.with_name('config.json'))
            assert original['stopped_by_patience'] and original['epochs']<config['epoch']
        for policy in ('same_user','random_matched'):
            paths[policy]=[OUT/f'mask_{prefix}_{policy}'/f'seed_{seed}'/'result.json' for seed in (2023,2024,2025)]
            for source,reference in zip(paths[policy],paths['full']):
                current,original=read(source.with_name('config.json')),read(reference.with_name('config.json'))
                assert current['model']=='LightCCFMasked' and current['mask_policy']==policy
                assert current['epoch']==1000
                metadata={'model','mask_policy','epoch','log_file','model_path','output','environment'}
                assert all(current[k]==v for k,v in original.items() if k not in metadata)
        masks[prefix]={policy:summarize(values) for policy,values in paths.items()}
        masks[prefix]['same_minus_full']=paired(paths['same_user'],paths['full'])
        masks[prefix]['same_minus_random']=paired(paths['same_user'],paths['random_matched'])
    counts=Counter()
    for path in OUT.rglob('result.json'):
        value=read(path)
        assert value['status']=='complete'
        counts['validation_only' if value['pilot'] else 'formal']+=1
    assert counts==Counter(formal=30,validation_only=24-reused),counts
    summary=dict(equal_trial_count_per_model=4,search_seed=2023,confirmation_seeds=[2026,2027,2028],
                 reused_candidates=reused,new_runs=dict(counts),total_formal=41+counts['formal'],
                 total_validation_only=10+counts['validation_only'],selections=selections,
                 confirmation=rows,paired_confirmations=comparisons,masking=masks,
                 previous_result_records_unchanged=51,locked_source_hashes_verified=True,
                 previous_result_manifest=previous_manifest.relative_to(ROOT).as_posix(),
                 search_cap_hits=search_cap_hits,confirmation_cap_hits=sum(r['hit_cap'] for r in rows.values()),
                 upstream_sources_match_pinned_commit=True,
                 shared_gpu_scheduling=independent.exists())
    (ANALYSIS/'followup_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    plots(summary)
    print(json.dumps(dict(new_runs=dict(counts),total_formal=summary['total_formal'],paired=comparisons,
                          mask_deltas={k:v['same_minus_random'] for k,v in masks.items()}),ensure_ascii=False,indent=2))


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import use_chinese_font
    use_chinese_font()
    colors=['#4D789A','#C17A49','#27856F']
    for kind in ('tuning','masking'):
        fig,axes=plt.subplots(1,2,figsize=(11,4.1))
        for ax,prefix,title in zip(axes,('grocery','ml'),('亚马逊食品数据','电影评分数据')):
            if kind=='tuning':
                labels=['BPRMF','LightGCN','LightCCF']
                values=[summary['confirmation'][prefix+'_'+label]['test']['NDCG@20'] for label in labels]
            else:
                labels=['原始NA','同用户屏蔽','等数量随机屏蔽']
                values=[summary['masking'][prefix][policy]['test']['NDCG@20'] for policy in ('full','same_user','random_matched')]
            ax.bar(range(3),[v['mean'] for v in values],yerr=[v['sample_sd'] for v in values],color=colors,capsize=3)
            ax.set_xticks(range(3),labels)
            ax.set_ylim(bottom=0)
            ax.set_ylabel('测试 NDCG@20')
            ax.set_title(title)
            ax.grid(axis='y',alpha=.2)
            ax.set_axisbelow(True)
        fig.tight_layout()
        for suffix in ('png','pdf'):
            fig.savefig(ANALYSIS/f'followup_{kind}.{suffix}',dpi=200)
        plt.close(fig)


if __name__=='__main__':
    main()
