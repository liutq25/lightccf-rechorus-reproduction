"""Run the fixed course comparison matrix with resumable result directories.

All models use the same split, seed, batch size, learning rate, validation
selection rule, and full-catalog ranking evaluator. Each output directory
contains the exact configuration, per-epoch history, selected checkpoint,
and final validation/test result.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = (
    'grocery_bprmf', 'grocery_lightgcn', 'grocery_lightccf',
    'ml_bprmf', 'ml_lightgcn', 'ml_lightccf',
)
ABLATION = ('grocery_lightccf_alpha0', 'ml_lightccf_alpha0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('core', 'ablation'), default='core')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2023, 2024, 2025])
    parser.add_argument('--epochs', type=int, default=None,
                        help='Optional common epoch cap, recorded in each run config')
    args = parser.parse_args()
    names = CORE if args.group == 'core' else ABLATION
    for seed in args.seeds:
        for name in names:
            config = ROOT / 'configs' / f'{name}.json'
            output = ROOT / 'runs' / args.group / name / f'seed_{seed}'
            result = output / 'result.json'
            if result.exists():
                payload = json.loads(result.read_text(encoding='utf-8'))
                if payload.get('status') == 'complete' and not payload.get('pilot'):
                    print(f'SKIP {name} seed={seed}: complete', flush=True)
                    continue
                raise RuntimeError(f'Invalid completed result: {result}')
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f'Incomplete run requires manual audit: {output}')
            cmd = [sys.executable, str(ROOT / 'scripts' / 'run_experiment.py'),
                   '--config', str(config), '--output', str(output), '--seed', str(seed)]
            if args.epochs is not None:
                cmd += ['--epochs', str(args.epochs)]
            print(f'RUN {name} seed={seed}', flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
    print(f'{args.group.upper()} MATRIX COMPLETE', flush=True)


if __name__ == '__main__':
    main()
