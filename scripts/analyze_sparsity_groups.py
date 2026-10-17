"""Adapt paper Figure 2 using saved user metrics; grouping uses training counts only.

This is a post-hoc descriptive analysis of fixed checkpoints, not another model
selection procedure and not an exact reproduction of the original datasets.
"""
from pathlib import Path
import csv
import json
import statistics as stats

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2023, 2024, 2025)
MODELS = ('bprmf', 'lightgcn', 'lightccf')
LABELS = ('BPRMF', 'LightGCN', 'LightCCF')
GROUPS = ('sparse', 'normal', 'popular')
METRICS = ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    profiles, rows, summary = {}, [], []
    for prefix, dataset in [('grocery', 'Grocery_and_Gourmet_Food'), ('ml', 'ML_1MTOPK')]:
        folder = ROOT / 'framework/ReChorus/data' / dataset
        train = pd.read_csv(folder / 'train.csv', sep='\t', usecols=['user_id'])
        degree = train.groupby('user_id').size()
        # Discrete cut points; equal histories always stay in the same group.
        q1, q2 = (int(x) for x in np.quantile(degree, [1/3, 2/3], method='inverted_cdf'))
        user_group = {int(u): GROUPS[0 if d <= q1 else 1 if d <= q2 else 2]
                      for u, d in degree.items()}
        test = pd.read_csv(folder / 'test.csv', sep='\t', usecols=['user_id'])
        test_ids = set(int(u) for u in test.user_id)
        profile = dict(dataset=dataset, thresholds=[q1, q2],
                       rule=f'train degree <= {q1}; {q1} < degree <= {q2}; degree > {q2}',
                       groups={})
        for group in GROUPS:
            train_users = sum(g == group for g in user_group.values())
            test_users = sum(user_group[u] == group for u in test_ids)
            profile['groups'][group] = dict(train_users=train_users, test_users=test_users,
                                           test_coverage_percent=100 * test_users / train_users)
        profiles[prefix] = profile
        for model, label in zip(MODELS, LABELS):
            for seed in SEEDS:
                path = ROOT / 'runs/core' / f'{prefix}_{model}' / f'seed_{seed}'
                users = json.loads((path / 'test_users.json').read_text(encoding='utf-8'))
                result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
                ids = [int(u['user_id']) for u in users]
                if len(ids) != len(set(ids)) or set(ids) != test_ids:
                    raise ValueError(f'User coverage disagrees with split: {path}')
                for metric in METRICS:
                    if abs(stats.mean(u[metric] for u in users) - result['test'][metric]) > 1e-12:
                        raise ValueError(f'User metrics do not reconstruct aggregate: {path} {metric}')
                for group in GROUPS:
                    selected = [u for u in users if user_group[int(u['user_id'])] == group]
                    row = dict(dataset=dataset, prefix=prefix, model=label, seed=seed,
                               group=group, n_test_users=len(selected))
                    row.update({metric: stats.mean(u[metric] for u in selected) if selected else None
                                for metric in METRICS})
                    rows.append(row)
            for group in GROUPS:
                selected = [r for r in rows if r['dataset'] == dataset
                            and r['model'] == label and r['group'] == group]
                row = dict(dataset=dataset, prefix=prefix, model=label, group=group,
                           n_seeds=len(selected), n_test_users=selected[0]['n_test_users'])
                for metric in METRICS:
                    values = [r[metric] for r in selected if r[metric] is not None]
                    row[metric + '_mean'] = stats.mean(values) if values else None
                    row[metric + '_sd'] = stats.stdev(values) if len(values) > 1 else None
                summary.append(row)
    out = ROOT / 'analysis'
    write_csv(out / 'sparsity_group_runs.csv', rows)
    write_csv(out / 'sparsity_group_summary.csv', summary)
    (out / 'sparsity_groups.json').write_text(json.dumps(
        dict(grouping='training-count empirical terciles; ties kept together',
             analysis_type='post-hoc descriptive; checkpoints unchanged',
             original_thresholds='paper does not specify exact cut points',
             profiles=profiles, summary=summary), ensure_ascii=False, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#4D789A', '#D58A55', '#3A9A85')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, prefix, title in zip(axes, ('grocery', 'ml'), ('Amazon Grocery', 'MovieLens 1M TopK')):
        for model, color in zip(LABELS, colors):
            selected = [next(r for r in summary if r['prefix'] == prefix
                             and r['model'] == model and r['group'] == group) for group in GROUPS]
            ax.errorbar(range(3), [r['NDCG@20_mean'] for r in selected],
                        yerr=[r['NDCG@20_sd'] for r in selected], capsize=3,
                        marker='o', color=color, label=model)
        ax.set_xticks(range(3), [f"{g}\n(n={profiles[prefix]['groups'][g]['test_users']})" for g in GROUPS])
        ax.set_ylabel('Test user-macro NDCG@20')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'sparsity_groups.{suffix}', dpi=200)
    plt.close(fig)
    print(json.dumps(profiles, ensure_ascii=False, indent=2))
    for prefix in ('grocery', 'ml'):
        print(prefix, [(r['model'], r['group'], round(r['NDCG@20_mean'], 6))
                       for r in summary if r['prefix'] == prefix])


if __name__ == '__main__':
    main()
