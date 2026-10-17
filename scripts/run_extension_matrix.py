"""Run the locked budget and validation-only sensitivity extensions serially."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/extension'
CONFIGS = ROOT / 'configs/extension'
SEEDS = (2023, 2024, 2025)


def run(name, config, seed, pilot=False):
    CONFIGS.mkdir(exist_ok=True)
    config_path = CONFIGS / f'{name}.json'
    serialized = json.dumps(config, ensure_ascii=False, indent=2)
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise RuntimeError(f'Changed locked configuration: {config_path}')
    config_path.write_text(serialized, encoding='utf-8')
    output = OUT / name / f'seed_{seed}'
    result_path = output / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result['status'] != 'complete' or result['pilot'] != pilot:
            raise RuntimeError(f'Invalid existing result: {result_path}')
        print(f'SKIP {name} seed={seed}', flush=True)
        return result
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Incomplete run needs audit: {output}')
    output.mkdir(parents=True)
    cmd = [sys.executable, str(ROOT / 'scripts/run_experiment.py'),
           '--config', str(config_path), '--output', str(output), '--seed', str(seed)]
    if pilot:
        cmd.append('--pilot')
    print(f'RUN {name} seed={seed} validation_only={pilot}', flush=True)
    started = time.perf_counter()
    with (output / 'console.log').open('w', encoding='utf-8') as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = json.loads(result_path.read_text(encoding='utf-8'))
    print(f"DONE {name} seed={seed} epochs={result['epochs']} "
          f"dev_N20={result['dev']['NDCG@20']:.6f} seconds={time.perf_counter()-started:.1f}", flush=True)
    return result


def base(name):
    return json.loads((ROOT / 'configs' / f'{name}.json').read_text(encoding='utf-8'))


def main():
    candidates = []
    # Resolve the fixed-parameter limitation first; these runs never evaluate test.
    for alpha in (.1, .5, 2.5, 5., 10.):
        config = base('ml_lightccf')
        config['ssl_weight'] = alpha
        name = 'ml_alpha_' + str(alpha).replace('.', 'p')
        result = run(name, config, 2023, pilot=True)
        candidates.append(dict(name=name, alpha=alpha, tau=.28, dev=result['dev']['NDCG@20'],
                               source=f'runs/extension/{name}/seed_2023/result.json', config=config))
    for tau in (.20, .22, .24, .26, .30):
        config = base('ml_lightccf')
        config['tau'] = tau
        name = 'ml_tau_' + f'{tau:.2f}'.replace('.', 'p')
        result = run(name, config, 2023, pilot=True)
        candidates.append(dict(name=name, alpha=1., tau=tau, dev=result['dev']['NDCG@20'],
                               source=f'runs/extension/{name}/seed_2023/result.json', config=config))
    for name, group, config_name, alpha in [
        ('ml_alpha0_existing', 'ablation', 'ml_lightccf_alpha0', 0.),
        ('ml_full_existing', 'core', 'ml_lightccf', 1.),
    ]:
        source = f'runs/{group}/{config_name}/seed_2023/result.json'
        result = json.loads((ROOT / source).read_text(encoding='utf-8'))
        candidates.append(dict(name=name, alpha=alpha, tau=.28, dev=result['dev']['NDCG@20'],
                               source=source, config=base(config_name)))
    selected = sorted(candidates, key=lambda row: (-row['dev'], row['alpha'], row['tau']))[0]
    selection = dict(rule='max seed-2023 validation NDCG@20; tie: smaller alpha then tau',
                     selection_uses_test=False, candidates=candidates, selected=selected,
                     post_hoc_exploration=True, grid_type='two one-dimensional slices, not a joint grid')
    (OUT / 'parameter_selection.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"SELECT alpha={selected['alpha']} tau={selected['tau']} dev={selected['dev']:.6f}", flush=True)
    if selected['name'].endswith('_existing'):
        print('REUSE the three existing formal runs of the validation-selected configuration', flush=True)
    else:
        for seed in SEEDS:
            run('ml_validation_selected', selected['config'], seed)
    for seed in SEEDS:
        for name in ('grocery_bprmf', 'grocery_lightgcn', 'grocery_lightccf'):
            config = base(name)
            config['epoch'] = 1000
            run(name + '_long', config, seed)
    print('EXTENSION MATRIX COMPLETE', flush=True)


if __name__ == '__main__':
    main()
