"""Summarize only completed, non-pilot course runs; never select by test score."""
from pathlib import Path
import csv
import json
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis'
SEEDS = (2023, 2024, 2025)
NAMES = ('grocery_bprmf', 'grocery_lightgcn', 'grocery_lightccf',
         'ml_bprmf', 'ml_lightgcn', 'ml_lightccf')
ABLATIONS = ('grocery_lightccf_alpha0', 'ml_lightccf_alpha0')
METRICS = ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')


def load(group, name, seed):
    path = ROOT / 'runs' / group / name / f'seed_{seed}' / 'result.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['status'] != 'complete' or data['pilot'] or data['seed'] != seed or data['test'] is None:
        raise ValueError(f'Invalid formal result: {path}')
    config = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
    expected = {'random_seed': seed, 'dataset': data['dataset'], 'model': data['model'],
                'pilot': False, 'emb_size': 64, 'num_neg': 1, 'lr': .001,
                'batch_size': 2048, 'epoch': 200, 'early_stop': 25,
                'eval_every': 5, 'main_metric': 'NDCG@20',
                'framework_commit': 'c164ec4303cc20ddcfbd1b57de366a481811d1e5'}
    if data['model'] == 'LightCCF':
        expected.update(n_layers=0, tau=.28, reg_weight=1e-4, l2=0.,
                        ssl_weight=0. if name.endswith('_alpha0') else 1.)
    else:
        expected['l2'] = 1e-6
        if data['model'] == 'LightGCN':
            expected['n_layers'] = 3
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f'{path.parent}: {key}={config[key]!r}, expected {value!r}')
    data['config_data_sha256'] = config['data_sha256']
    return data


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    all_runs = {}
    flat = []
    dataset_hashes = {}
    for group, names in [('core', NAMES), ('ablation', ABLATIONS)]:
        for name in names:
            for seed in SEEDS:
                result = load(group, name, seed)
                previous = dataset_hashes.setdefault(result['dataset'], result['config_data_sha256'])
                if result['config_data_sha256'] != previous:
                    raise ValueError(f'Inconsistent split files for {name}, seed={seed}')
                all_runs[(name, seed)] = result
                row = {'group': group, 'name': name, 'dataset': result['dataset'],
                       'model': result['model'], 'seed': seed,
                       'best_epoch': result['best_epoch'], 'epochs': result['epochs'],
                       'training_seconds': result['training_seconds'],
                       'mean_epoch_seconds': result['mean_epoch_seconds'],
                       'wall_seconds': result['wall_seconds'],
                       'peak_cuda_mb': result['peak_cuda_mb'],
                       'stopped_by_patience': result['stopped_by_patience']}
                for split in ('dev', 'test'):
                    row.update({f'{split}_{m}': result[split][m] for m in METRICS})
                flat.append(row)
    write_csv(OUT / 'runs.csv', flat)
    summary = []
    for name in NAMES + ABLATIONS:
        rows = [row for row in flat if row['name'] == name]
        row = {'name': name, 'dataset': rows[0]['dataset'], 'model': rows[0]['model'],
               'n_seeds': len(rows)}
        for key in ('best_epoch', 'epochs', 'training_seconds', 'mean_epoch_seconds', 'wall_seconds',
                    'peak_cuda_mb') + tuple(f'{s}_{m}' for s in ('dev', 'test') for m in METRICS):
            values = [float(r[key]) for r in rows]
            row[key + '_mean'] = stats.mean(values)
            row[key + '_sd'] = stats.stdev(values)
        row['hit_epoch_cap'] = sum(not r['stopped_by_patience'] for r in rows)
        summary.append(row)
    write_csv(OUT / 'summary.csv', summary)
    pairs = []
    for prefix in ('grocery', 'ml'):
        for left, right in [('lightccf', 'bprmf'), ('lightccf', 'lightgcn'),
                            ('lightccf', 'lightccf_alpha0')]:
            for metric in ('Recall@20', 'NDCG@20'):
                deltas = [all_runs[(f'{prefix}_{left}', seed)]['test'][metric]
                          - all_runs[(f'{prefix}_{right}', seed)]['test'][metric]
                          for seed in SEEDS]
                pairs.append({'dataset': prefix, 'comparison': f'{left} - {right}',
                              'metric': metric, 'delta_mean': stats.mean(deltas),
                              'delta_sd': stats.stdev(deltas),
                              'delta_by_seed': json.dumps(deltas)})
    write_csv(OUT / 'paired_deltas.csv', pairs)
    (OUT / 'summary.json').write_text(json.dumps({'seeds': SEEDS,
                                                  'summary': summary,
                                                  'paired_deltas': pairs},
                                                 ensure_ascii=False, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from plot_style import use_chinese_font
    use_chinese_font()
    colors = {'BPRMF': '#4D789A', 'LightGCN': '#D58A55', 'LightCCF': '#3A9A85'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, prefix, title in zip(axes, ('grocery', 'ml'),
                                 ('亚马逊食品数据', '电影评分数据')):
        names = [f'{prefix}_{m}' for m in ('bprmf', 'lightgcn', 'lightccf')]
        s = [next(r for r in summary if r['name'] == n) for n in names]
        x = np.arange(3)
        ax.bar(x, [r['test_NDCG@20_mean'] for r in s],
               yerr=[r['test_NDCG@20_sd'] for r in s], capsize=4,
               color=[colors[r['model']] for r in s], edgecolor='white')
        ax.set_xticks(x, [r['model'] for r in s])
        ax.set_ylabel('测试 NDCG@20（用户宏平均）')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.text(.5, .01, '柱高：三个训练种子的均值；误差线：样本标准差',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUT / 'model_comparison.png', dpi=190)
    fig.savefig(OUT / 'model_comparison.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, prefix, title in zip(axes, ('grocery', 'ml'),
                                 ('亚马逊食品数据', '电影评分数据')):
        full = [all_runs[(f'{prefix}_lightccf', seed)]['test']['NDCG@20'] for seed in SEEDS]
        zero = [all_runs[(f'{prefix}_lightccf_alpha0', seed)]['test']['NDCG@20'] for seed in SEEDS]
        for s, a, b in zip(SEEDS, zero, full):
            ax.plot([0, 1], [a, b], marker='o', alpha=.7, label=f'种子{s}')
        ax.set_xticks([0, 1], ['关闭 NA', '开启 NA'])
        ax.set_ylabel('测试 NDCG@20（用户宏平均）')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'na_ablation.png', dpi=190)
    fig.savefig(OUT / 'na_ablation.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    learning_rows = []
    for ax, prefix, title in zip(axes, ('grocery', 'ml'),
                                 ('亚马逊食品数据', '电影评分数据')):
        for model_name in ('bprmf', 'lightgcn', 'lightccf'):
            name = f'{prefix}_{model_name}'
            history = ROOT / 'runs' / 'core' / name / 'seed_2023' / 'history.jsonl'
            points = [json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()]
            points = [point for point in points if 'dev' in point]
            label = all_runs[(name, 2023)]['model']
            learning_rows.extend(dict(name=name, seed=2023, model=label, epoch=point['epoch'],
                                      dev_ndcg20=point['dev']['NDCG@20']) for point in points)
            ax.plot([point['epoch'] for point in points],
                    [point['dev']['NDCG@20'] for point in points],
                    label=label, color=colors[label], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel('训练轮数')
        ax.set_ylabel('验证 NDCG@20（用户宏平均）')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'learning_curves_seed2023.png', dpi=190)
    fig.savefig(OUT / 'learning_curves_seed2023.pdf')
    plt.close(fig)
    write_csv(OUT / 'learning_curves_seed2023.csv', learning_rows)
    print(f'Wrote {len(flat)} formal runs and {len(pairs)} paired comparisons to {OUT}')


if __name__ == '__main__':
    main()
