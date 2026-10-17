"""Audit the locked extension runs and generate Chinese scientific figures."""
from pathlib import Path
import argparse
import csv
import json
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2023, 2024, 2025)
METRICS = ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')
MODELS = ('bprmf', 'lightgcn', 'lightccf')
LABELS = ('BPRMF', 'LightGCN', 'LightCCF')
OUTPUT_KEYS = {'log_file', 'model_path', 'output'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(path, expected_pilot):
    result, config = read(path / 'result.json'), read(path / 'config.json')
    if result['status'] != 'complete' or result['pilot'] != expected_pilot:
        raise ValueError(f'Incomplete or misclassified run: {path}')
    if expected_pilot and result['test'] is not None:
        raise ValueError('Search run has opened test metrics')
    for result_key, config_key in [('model', 'model'), ('dataset', 'dataset'), ('seed', 'random_seed')]:
        if result[result_key] != config[config_key]:
            raise ValueError(f'Result/config identity differs: {path}')
    if config['pilot'] != expected_pilot or (not expected_pilot and not isinstance(result['test'], dict)):
        raise ValueError(f'Invalid evaluation mode: {path}')
    return result, config


def same_config(reference, candidate, allowed):
    for key in reference.keys() | candidate.keys():
        if key not in OUTPUT_KEYS | set(allowed) and reference.get(key) != candidate.get(key):
            raise ValueError(f'Unexpected configuration difference: {key}')


def history(path):
    return [json.loads(line) for line in (path / 'history.jsonl').read_text(encoding='utf-8').splitlines()]


def summarize(runs):
    summary = {'n_seeds': len(runs), 'seeds': [r['seed'] for r in runs],
               'best_epochs': [r['best_epoch'] for r in runs],
               'epochs': [r['epochs'] for r in runs], 'test': {}, 'dev': {}}
    for phase in ('dev', 'test'):
        for metric in METRICS:
            values = [r[phase][metric] for r in runs]
            summary[phase][metric] = dict(mean=stats.mean(values), sample_sd=stats.stdev(values))
    for key in ('mean_epoch_seconds', 'wall_seconds', 'training_seconds', 'peak_cuda_mb'):
        summary[key + '_mean'] = stats.mean(r[key] for r in runs)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters-only', action='store_true')
    args = parser.parse_args()
    selection = read(ROOT / 'runs/extension/parameter_selection.json')
    if selection['selection_uses_test'] or len(selection['candidates']) != 12:
        raise ValueError('Incorrect selection protocol')
    candidates = []
    reference_config = read(ROOT / 'runs/core/ml_lightccf/seed_2023/config.json')
    for row in selection['candidates']:
        path = ROOT / row['source']
        pilot = row['source'].startswith('runs/extension/')
        result, config = load_run(path.parent, pilot)
        same_config(reference_config, config, {'tau', 'ssl_weight', 'pilot'})
        expected = dict(model='LightCCF', dataset='ML_1MTOPK', n_layers=0,
                        emb_size=64, batch_size=2048, epoch=200, early_stop=25,
                        eval_every=5, tau=row['tau'], ssl_weight=row['alpha'],
                        reg_weight=1e-4, l2=0., random_seed=2023)
        for key, value in expected.items():
            if config[key] != value:
                raise ValueError(f'{path}: {key}')
        if result['dev']['NDCG@20'] != row['dev']:
            raise ValueError('Selection record differs from saved validation score')
        candidates.append(dict(alpha=row['alpha'], tau=row['tau'], dev_ndcg20=row['dev'],
                               best_epoch=result['best_epoch'], epochs=result['epochs'],
                               is_new_validation_only_run=pilot, source=row['source']))
    selected = sorted(candidates, key=lambda x: (-x['dev_ndcg20'], x['alpha'], x['tau']))[0]
    if (selected['alpha'], selected['tau']) != (selection['selected']['alpha'], selection['selected']['tau']):
        raise ValueError('Selected candidate violates the recorded rule')
    parameters = dict(selection=selection, candidates=candidates, selected=selected,
                      new_validation_only_runs=sum(r['is_new_validation_only_run'] for r in candidates))
    out = ROOT / 'analysis'
    csv_write(out / 'parameter_search.csv', candidates)
    (out / 'parameter_search.json').write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_style import use_chinese_font
    use_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    rows = sorted((r for r in candidates if r['tau'] == .28), key=lambda r: r['alpha'])
    axes[0].plot(range(len(rows)), [r['dev_ndcg20'] for r in rows], marker='o', color='#27856F')
    axes[0].set_xticks(range(len(rows)), [f"{r['alpha']:g}" for r in rows])
    axes[0].set_xlabel('邻域聚合权重 α（等间距列示网格）')
    axes[0].set_title('电影评分数据：权重敏感性（温度0.28）')
    rows = sorted((r for r in candidates if r['alpha'] == 1.), key=lambda r: r['tau'])
    axes[1].plot([r['tau'] for r in rows], [r['dev_ndcg20'] for r in rows],
                 marker='o', color='#C17A49')
    axes[1].set_xticks([r['tau'] for r in rows])
    axes[1].set_xlabel('温度 τ')
    axes[1].set_title('电影评分数据：温度敏感性（权重1）')
    for ax in axes:
        ax.set_ylabel('验证 NDCG@20（种子2023）')
        ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'parameter_sensitivity.{suffix}', dpi=200)
    plt.close(fig)
    print(json.dumps({'parameters_complete': True, 'selected': selected}, ensure_ascii=False, indent=2))
    if args.parameters_only:
        return

    selected_name = selection['selected']['name']
    if selected_name.endswith('_existing'):
        group, config_name = ('ablation', 'ml_lightccf_alpha0') if selected['alpha'] == 0 else ('core', 'ml_lightccf')
        selected_paths = [ROOT / 'runs' / group / config_name / f'seed_{seed}' for seed in SEEDS]
        reused = True
    else:
        selected_paths = [ROOT / 'runs/extension/ml_validation_selected' / f'seed_{seed}' for seed in SEEDS]
        reused = False
    selected_runs = []
    selected_config = read((ROOT / selected['source']).parent / 'config.json')
    for path, seed in zip(selected_paths, SEEDS):
        result, config = load_run(path, False)
        same_config(selected_config, config, {'random_seed', 'pilot'})
        if (config['random_seed'], config['tau'], config['ssl_weight']) != (seed, selected['tau'], selected['alpha']):
            raise ValueError('Selected confirmation config differs from selected candidate')
        selected_runs.append(result)

    budget, trajectory_rows, prefix_checks = {}, [], []
    for model, label in zip(MODELS, LABELS):
        original, extended = [], []
        for seed in SEEDS:
            old_path = ROOT / 'runs/core' / f'grocery_{model}' / f'seed_{seed}'
            new_path = ROOT / 'runs/extension' / f'grocery_{model}_long' / f'seed_{seed}'
            old, old_config = load_run(old_path, False)
            new, new_config = load_run(new_path, False)
            if old_config['epoch'] != 200 or new_config['epoch'] != 1000:
                raise ValueError('Wrong budget limits')
            same_config(old_config, new_config, {'epoch'})
            old_history, new_history = history(old_path), history(new_path)
            common = min(len(old_history), len(new_history))
            if any(old_history[i]['epoch'] != new_history[i]['epoch']
                   or ('dev' in old_history[i]) != ('dev' in new_history[i]) for i in range(common)):
                raise ValueError(f'Inconsistent evaluation schedule: {new_path}')
            max_loss_diff = max(abs(old_history[i]['loss'] - new_history[i]['loss']) for i in range(common))
            max_dev_diff = max((abs(old_history[i]['dev'][m] - new_history[i]['dev'][m])
                                for i in range(common) if 'dev' in old_history[i]
                                for m in METRICS), default=0.)
            prefix_checks.append(dict(model=label, seed=seed, common_epochs=common,
                                      only_epoch_limit_and_output_paths_differ=True,
                                      max_loss_absolute_difference=max_loss_diff,
                                      max_validation_absolute_difference=max_dev_diff,
                                      within_small_roundoff=(max_loss_diff <= 1e-5 and max_dev_diff <= 1e-6)))
            original.append(old)
            extended.append(new)
            if seed == 2023:
                trajectory_rows.extend(dict(model=label, seed=seed, epoch=r['epoch'],
                                            dev_ndcg20=r['dev']['NDCG@20'])
                                       for r in new_history if 'dev' in r)
        budget[label] = {'original': summarize(original), 'long': summarize(extended)}
        budget[label]['paired_delta_ndcg20'] = {
            'values': [b['test']['NDCG@20'] - a['test']['NDCG@20'] for a, b in zip(original, extended)]}
        values = budget[label]['paired_delta_ndcg20']['values']
        budget[label]['paired_delta_ndcg20'].update(mean=stats.mean(values), sample_sd=stats.stdev(values))
        budget[label]['long']['hit_cap'] = sum(not r['stopped_by_patience'] for r in extended)
    csv_write(out / 'budget_learning_curves_seed2023.csv', trajectory_rows)
    original_ml = [read(ROOT / 'runs/core/ml_lightccf' / f'seed_{s}/result.json') for s in SEEDS]
    alpha0_ml = [read(ROOT / 'runs/ablation/ml_lightccf_alpha0' / f'seed_{s}/result.json') for s in SEEDS]
    gcn_ml = [read(ROOT / 'runs/core/ml_lightgcn' / f'seed_{s}/result.json') for s in SEEDS]
    report = dict(budget=budget, prefix_checks=prefix_checks, parameters=parameters,
                  selected_test=summarize(selected_runs), selected_reuses_existing_formal_runs=reused,
                  selected_result_paths=[p.relative_to(ROOT).as_posix() for p in selected_paths],
                  original_ml=summarize(original_ml), alpha0_ml=summarize(alpha0_ml),
                  lightgcn_ml=summarize(gcn_ml),
                  new_budget_formal_runs=9, new_selected_formal_runs=0 if reused else 3,
                  new_validation_only_runs=10)
    report['total_formal_runs'] = 29 + 9 + report['new_selected_formal_runs']
    report['long_ccf_vs_gcn_ndcg20_percent'] = (
        budget['LightCCF']['long']['test']['NDCG@20']['mean'] /
        budget['LightGCN']['long']['test']['NDCG@20']['mean'] - 1) * 100
    report['selected_vs_original_ml_ndcg20_percent'] = (
        report['selected_test']['test']['NDCG@20']['mean'] /
        report['original_ml']['test']['NDCG@20']['mean'] - 1) * 100
    report['selected_vs_lightgcn_ml_ndcg20_percent'] = (
        report['selected_test']['test']['NDCG@20']['mean'] /
        report['lightgcn_ml']['test']['NDCG@20']['mean'] - 1) * 100
    report['selected_vs_alpha0_ml_ndcg20_percent'] = (
        report['selected_test']['test']['NDCG@20']['mean'] /
        report['alpha0_ml']['test']['NDCG@20']['mean'] - 1) * 100
    (out / 'extension_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    colors = dict(zip(LABELS, ['#4D789A', '#C17A49', '#27856F']))
    for label in LABELS:
        selected_rows = [r for r in trajectory_rows if r['model'] == label]
        axes[0].plot([r['epoch'] for r in selected_rows], [r['dev_ndcg20'] for r in selected_rows],
                     color=colors[label], label=label)
    axes[0].axvline(200, linestyle='--', color='#687581', label='原200轮上限')
    axes[0].set_xlabel('训练轮数')
    axes[0].set_ylabel('验证 NDCG@20')
    axes[0].set_title('食品数据：长预算验证曲线（种子2023）')
    axes[0].legend(fontsize=8)
    for index, (condition, legend, color, shift) in enumerate([
        ('original', '上限200轮', '#4D789A', -.18), ('long', '上限1000轮', '#27856F', .18)]):
        axes[1].bar([x + shift for x in range(3)],
                    [budget[label][condition]['test']['NDCG@20']['mean'] for label in LABELS],
                    yerr=[budget[label][condition]['test']['NDCG@20']['sample_sd'] for label in LABELS],
                    width=.34, color=color, label=legend, capsize=3)
    axes[1].set_xticks(range(3), LABELS)
    axes[1].set_ylabel('测试 NDCG@20')
    axes[1].set_title('食品数据：轮数预算与最终测试')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'budget_extension.{suffix}', dpi=200)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in (
        'total_formal_runs', 'long_ccf_vs_gcn_ndcg20_percent',
        'selected_vs_original_ml_ndcg20_percent', 'selected_vs_lightgcn_ml_ndcg20_percent')}, indent=2))


if __name__ == '__main__':
    main()
