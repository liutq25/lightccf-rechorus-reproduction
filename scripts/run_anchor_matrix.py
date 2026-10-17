"""Run the fixed five-seed author-protocol Douban numerical comparison."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2023, 2024, 2025, 2026, 2027])
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    for seed in args.seeds:
        output = ROOT / 'runs' / 'official_anchor' / 'douban-book' / f'seed_{seed}'
        result = output / 'result.json'
        if result.exists():
            data = json.loads(result.read_text(encoding='utf-8'))
            if data['status'] != 'complete' or data['seed_set_before_model'] != seed:
                raise RuntimeError(f'Unexpected completed result: {result}')
            print(f'SKIP author Douban seed={seed}: complete', flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Incomplete author run requires manual audit: {output}')
        print(f'RUN author Douban seed={seed}', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_official_anchor.py'),
                        '--dataset', 'douban-book', '--seed', str(seed),
                        '--epochs', str(args.epochs)], cwd=ROOT, check=True)
    print('AUTHOR DOUBAN MATRIX COMPLETE', flush=True)


if __name__ == '__main__':
    main()
