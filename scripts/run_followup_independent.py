"""Execute independent MF confirmations/masks using the locked matrix functions."""
from pathlib import Path
import json
from run_followup_matrix import ROOT, OUT, run, read, candidates, existing_for, SEEDS


def main():
    options=[]
    for label,config in candidates('ml','LightCCF'):
        path=existing_for('ml','LightCCF',config)
        if path is None:
            raise RuntimeError('All four CCF validation candidates must already be complete.')
        result=read(path)
        options.append(dict(id=label,config=config,dev=result['dev']['NDCG@20'],source=path.relative_to(ROOT).as_posix()))
    chosen=sorted(options,key=lambda row:(-row['dev'],row['id']))[0]
    (OUT/'independent_selection.json').write_text(json.dumps(dict(candidates=options,selected=chosen,selection_uses_test=False),indent=2),encoding='utf-8')
    print(f"INDEPENDENT SELECT ml CCF {chosen['id']} dev={chosen['dev']:.6f}",flush=True)
    for seed in SEEDS:
        run('confirm_ml_lightccf',chosen['config'],seed,False)
    for prefix in ('grocery','ml'):
        base=read(ROOT/f'configs/{prefix}_lightccf.json')
        for policy in ('same_user','random_matched'):
            config=dict(base,model='LightCCFMasked',mask_policy=policy,epoch=1000)
            for seed in (2023,2024,2025):
                run(f'mask_{prefix}_{policy}',config,seed,False)
    print('INDEPENDENT TASKS COMPLETE',flush=True)


if __name__=='__main__':
    main()
