"""Compare five fixed author-protocol runs with LightCCF paper Table 5."""
from pathlib import Path
import json
import re
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2023, 2024, 2025, 2026, 2027)
PAPER = {'Recall@20': .1877, 'NDCG@20': .1857}


def main():
    runs = []
    for seed in SEEDS:
        path = ROOT / 'runs/official_anchor/douban-book' / f'seed_{seed}/result.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['status'] != 'complete' or data['seed_set_before_model'] != seed:
            raise ValueError(path)
        if not data['selection_uses_test']:
            raise ValueError('The original-protocol comparison must be labelled as test-selected')
        log = (path.parent / 'author_train.log').read_text(encoding='utf-8')
        epochs = re.findall(r'Epoch:\s+(\d+)\| train time ([0-9.]+)\|', log)
        if not epochs:
            raise ValueError(f'No epoch records in {path.parent}')
        data['observed_epochs'] = int(epochs[-1][0])
        data['training_seconds_from_author_log'] = sum(float(value) for _, value in epochs)
        data['mean_epoch_train_seconds'] = data['training_seconds_from_author_log'] / len(epochs)
        runs.append(data)
    for key in ('train_sha256', 'test_sha256', 'unmodified_author_main_sha256', 'max_epochs'):
        if len({str(run[key]) for run in runs}) != 1:
            raise ValueError(f'Inconsistent author protocol: {key}')
    summary = {'dataset': 'douban-book', 'seeds': SEEDS, 'n_seeds': len(runs),
               'selection_uses_test': True, 'paper_table': 5,
               'paper_aggregation': 'mean of five runs; original seed list not reported',
               'wall_seconds_mean': stats.mean(r['wall_seconds'] for r in runs),
               'best_epoch_by_test_recall20': [r['best_epoch_by_test_recall20'] for r in runs],
               'observed_epochs': [r['observed_epochs'] for r in runs],
               'mean_epoch_train_seconds': stats.mean(r['mean_epoch_train_seconds'] for r in runs),
               'metrics': {}, 'runs': runs}
    for metric, key in [('Recall@20', 'test_recall20_at_best'),
                        ('NDCG@20', 'test_ndcg20_at_best')]:
        values = [run[key] for run in runs]
        mean = stats.mean(values)
        summary['metrics'][metric] = {
            'mean': mean, 'sample_sd': stats.stdev(values), 'paper': PAPER[metric],
            'absolute_difference': mean - PAPER[metric],
            'relative_difference_percent': (mean / PAPER[metric] - 1) * 100,
        }
    output = ROOT / 'analysis/official_douban_summary.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
