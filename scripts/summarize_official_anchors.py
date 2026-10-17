"""Compare five fixed author-protocol runs with LightCCF paper Table 5."""
from pathlib import Path
import csv
import json
import re
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2023, 2024, 2025, 2026, 2027)
METRICS = ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')


def parse_history(log):
    """Read all four printed metrics, preserving their common evaluation epoch."""
    blocks = re.split(r'Epoch:\s+(\d+)\| train time ([0-9.]+)\|', log)
    rows = []
    for index in range(1, len(blocks), 3):
        epoch, seconds, body = blocks[index:index + 3]
        match = re.search(r'Recall:\s*\[([^\]]+)\]\s+NDCG:\s*\[([^\]]+)\]', body)
        if not match:
            raise ValueError(f'Missing evaluation after epoch {epoch}')
        recall, ndcg = ([float(x) for x in values.split()] for values in match.groups())
        if len(recall) != 2 or len(ndcg) != 2:
            raise ValueError('Expected author top_K=[10,20]')
        rows.append(dict(epoch=int(epoch), train_seconds=float(seconds),
                         **dict(zip(METRICS, (recall[0], ndcg[0], recall[1], ndcg[1])))))
    if not rows or [r['epoch'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Incomplete or repeated author epoch log')
    return rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    paper_table = json.loads((ROOT / 'research/paper_table5.json').read_text(encoding='utf-8'))
    paper = dict(zip(paper_table['metric_order'], paper_table['rows']['LightCCF'][0]))
    runs = []
    histories, flat = [], []
    for seed in SEEDS:
        path = ROOT / 'runs/official_anchor/douban-book' / f'seed_{seed}/result.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['status'] != 'complete' or data['seed_set_before_model'] != seed:
            raise ValueError(path)
        if not data['selection_uses_test']:
            raise ValueError('The original-protocol comparison must be labelled as test-selected')
        log = (path.parent / 'author_train.log').read_text(encoding='utf-8')
        history = parse_history(log)
        selected = next(r for r in history if r['epoch'] == data['best_epoch_by_test_recall20'])
        for metric, key in [('Recall@20', 'test_recall20_at_best'),
                            ('NDCG@20', 'test_ndcg20_at_best')]:
            if abs(selected[metric] - data[key]) > 5.1e-9:
                raise ValueError(f'Printed metric disagrees with captured result: {seed} {metric}')
            selected[metric] = data[key]
        data['all_metrics_at_selected_epoch'] = {metric: selected[metric] for metric in METRICS}
        data['metric_precision'] = '@10 read from eight-decimal author log; @20 captured at full precision'
        data['observed_epochs'] = history[-1]['epoch']
        data['training_seconds_from_author_log'] = sum(row['train_seconds'] for row in history)
        data['mean_epoch_train_seconds'] = data['training_seconds_from_author_log'] / len(history)
        histories.extend(dict(seed=seed, **row) for row in history)
        flat.append(dict(seed=seed, selected_epoch=selected['epoch'],
                         **data['all_metrics_at_selected_epoch']))
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
    for metric in METRICS:
        values = [run['all_metrics_at_selected_epoch'][metric] for run in runs]
        mean = stats.mean(values)
        summary['metrics'][metric] = {
            'mean': mean, 'sample_sd': stats.stdev(values), 'paper': paper[metric],
            'minimum': min(values), 'maximum': max(values),
            'absolute_difference': mean - paper[metric],
            'relative_difference_percent': (mean / paper[metric] - 1) * 100,
        }
    output = ROOT / 'analysis/official_douban_summary.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(ROOT / 'analysis/official_douban_runs.csv', flat)
    write_csv(ROOT / 'analysis/official_douban_history.csv', histories)
    write_csv(ROOT / 'analysis/paper_agreement.csv',
              [dict(metric=metric, **summary['metrics'][metric]) for metric in METRICS])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import use_chinese_font
    use_chinese_font()
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    for x, metric in enumerate(METRICS):
        item = summary['metrics'][metric]
        ax.errorbar(x, item['mean'] / item['paper'],
                    yerr=item['sample_sd'] / item['paper'], fmt='o', capsize=5,
                    color='#27856F', markersize=7)
        ax.text(x, item['mean'] / item['paper'] - .011,
                f"{item['relative_difference_percent']:+.2f}%", ha='center', fontsize=9)
    ax.axhline(1, color='#A45435', linestyle='--', label='原文表5 = 1.0')
    ax.set_xticks(range(4), ['召回@10', 'NDCG@10', '召回@20', 'NDCG@20'])
    ax.set_ylabel('本机均值 / 原文')
    ax.set_title('豆瓣图书：五次原始协议实验与原文对照')
    ax.set_ylim(min(v['mean'] / v['paper'] for v in summary['metrics'].values()) - .028, 1.012)
    ax.grid(axis='y', alpha=.22)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(ROOT / f'analysis/paper_agreement.{suffix}', dpi=200)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
