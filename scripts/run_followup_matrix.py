"""Execute the locked equal-trial search, fresh-seed confirmation and masks."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/followup_20261101'
CONFIG = ROOT / 'configs/followup_20261101'
SEEDS = (2026,2027,2028)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(name, config, seed, pilot):
    CONFIG.mkdir(exist_ok=True)
    path = CONFIG / (name + '.json')
    if path.exists() and read(path) != config:
        raise RuntimeError(f'Locked configuration changed: {path}')
    path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    directory = OUT / name / f'seed_{seed}'
    result_path = directory / 'result.json'
    if result_path.exists():
        result = read(result_path)
        assert result['status']=='complete' and result['pilot']==pilot
        print(f'SKIP {name} {seed}',flush=True)
        return result_path
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f'Incomplete output needs audit: {directory}')
    directory.mkdir(parents=True)
    command = [sys.executable,str(ROOT/'scripts/run_experiment.py'),'--config',str(path),
               '--output',str(directory),'--seed',str(seed)]
    if pilot:
        command.append('--pilot')
    print(f'RUN {name} seed={seed} validation_only={pilot}',flush=True)
    started=time.perf_counter()
    with (directory/'console.log').open('w',encoding='utf-8') as stream:
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=read(result_path)
    print(f"DONE {name} seed={seed} epochs={result['epochs']} dev={result['dev']['NDCG@20']:.6f} seconds={time.perf_counter()-started:.1f}",flush=True)
    return result_path


def existing_for(prefix, model, config):
    """Reuse only an identical validation protocol whose old cap did not bind."""
    paths=[]
    if model=='LightCCF':
        alpha=config['ssl_weight']
        if alpha==1:
            paths.append(ROOT/f'runs/core/{prefix}_lightccf/seed_2023/result.json')
            paths.append(ROOT/f'runs/extension/{prefix}_lightccf_long/seed_2023/result.json')
        elif prefix=='ml':
            key=str(float(alpha)).replace('.','p')
            paths.append(ROOT/f'runs/extension/ml_alpha_{key}/seed_2023/result.json')
    else:
        label=model.lower()
        paths.extend([ROOT/f'runs/core/{prefix}_{label}/seed_2023/result.json',
                      ROOT/f'runs/extension/{prefix}_{label}_long/seed_2023/result.json'])
    for path in paths:
        if not path.exists():
            continue
        result=read(path)
        actual=read(path.with_name('config.json'))
        keys=set(config)-{'epoch'}
        if result['epochs'] < actual['epoch'] and all(actual.get(k)==config[k] for k in keys):
            return path


def candidates(prefix, model):
    base=read(ROOT/f'configs/{prefix}_{model.lower()}.json')
    base['epoch']=1000
    if model=='BPRMF':
        for lr in (.0005,.001):
            for l2 in (1e-7,1e-6):
                yield f'lr{lr:g}_l2{l2:g}',dict(base,lr=lr,l2=l2)
    elif model=='LightGCN':
        for layer in (1,3):
            for l2 in (1e-7,1e-6):
                yield f'layers{layer}_l2{l2:g}',dict(base,n_layers=layer,l2=l2,lr=.001)
    else:
        for alpha in (.1,.5,1.,2.5):
            yield f'alpha{alpha:g}',dict(base,ssl_weight=alpha,lr=.001)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selection_file=OUT/'validation_selection.json'
    selections=[]
    for prefix in ('grocery','ml'):
        for model in ('BPRMF','LightGCN','LightCCF'):
            rows=[]
            for label,config in candidates(prefix,model):
                reused=existing_for(prefix,model,config)
                path=reused or run(f'search_{prefix}_{model.lower()}_{label}',config,2023,True)
                result=read(path)
                rows.append(dict(id=label,config=config,dev=result['dev']['NDCG@20'],
                                 source=path.relative_to(ROOT).as_posix(),reused=reused is not None))
            assert len(rows)==4
            chosen=sorted(rows,key=lambda row:(-row['dev'],row['id']))[0]
            selection=dict(dataset_prefix=prefix,model=model,candidates=rows,selected=chosen,
                           selection_uses_test=False,search_seed=2023,confirmation_seeds=SEEDS,
                           rule='max validation NDCG@20; exact tie lexicographic candidate ID')
            selections.append(selection)
            selection_file.write_text(json.dumps(selections,indent=2),encoding='utf-8')
            print(f"SELECT {prefix} {model} {chosen['id']} dev={chosen['dev']:.6f}",flush=True)
            for seed in SEEDS:
                run(f'confirm_{prefix}_{model.lower()}',chosen['config'],seed,False)
    for prefix in ('grocery','ml'):
        base=read(ROOT/f'configs/{prefix}_lightccf.json')
        for policy in ('same_user','random_matched'):
            config=dict(base,model='LightCCFMasked',mask_policy=policy,epoch=1000)
            for seed in (2023,2024,2025):
                run(f'mask_{prefix}_{policy}',config,seed,False)
    print('FOLLOWUP MATRIX COMPLETE',flush=True)


if __name__=='__main__':
    main()
