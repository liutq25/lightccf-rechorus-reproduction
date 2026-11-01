"""Generate self-contained LaTeX numerical blocks from audited experiment files.

The report uses inline PGFPlots coordinates, so compiling its single .tex file
does not require an external image, CSV file, bibliography or network access.
Run with --report PATH to update only marked numerical regions of a report.
"""
from pathlib import Path
import argparse
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'analysis'
COLORS = {'BPRMF': 'plotblue', 'LightGCN': 'plotorange', 'LightCCF': 'plotgreen'}
SEED_COLORS = ['plotblue', 'plotorange', 'plotgreen', 'plotpurple', 'plotgray']


def load_csv(name):
    with (ANALYSIS / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def coordinates(rows, x, y):
    return ' '.join(f'({float(row[x]):.8g},{float(row[y]):.12g})' for row in rows)


def axis(body, options=''):
    return (r'\begin{tikzpicture}[baseline=(current bounding box.north)]\begin{axis}[reportaxis,' + options + ']\n'
            + '\n'.join(body) + '\n' + r'\end{axis}\end{tikzpicture}')


def pair(left, right):
    return left + '\n' + r'\hfill' + '\n' + right


def error_coordinates(values):
    return ' '.join(f'({x},{mean:.12g}) +- (0,{sd:.12g})' for x, mean, sd in values)


def extension_blocks(report):
    """All extra text and charts use the locked extension summary."""
    blocks = {}
    budget = report['budget']
    rows = []
    for label in COLORS:
        for condition, limit in [('original', 200), ('long', 1000)]:
            item = budget[label][condition]
            metric = lambda key: (f"$ {item['test'][key]['mean']:.5f}"
                                  f"\\pm{item['test'][key]['sample_sd']:.5f}$")
            epochs = '/'.join(str(x) for x in item['best_epochs'])
            rows.append(f'{label} & {limit} & {metric("Recall@20")} & {metric("NDCG@20")} & {epochs}' + r' \\')
    blocks['EXTENSION_BUDGET_TABLE'] = '\n'.join(rows)
    curves = load_csv('budget_learning_curves_seed2023.csv')
    body, upper, last_epoch = [], 0., 0
    for label in COLORS:
        selected_rows = [row for row in curves if row['model'] == label]
        upper = max(upper, max(float(r['dev_ndcg20']) for r in selected_rows))
        last_epoch = max(last_epoch, max(int(r['epoch']) for r in selected_rows))
        body.extend([r'\addplot[thick,' + COLORS[label] + ',no marks] coordinates {'
                     + coordinates(selected_rows, 'epoch', 'dev_ndcg20') + '};',
                     rf'\addlegendentry{{{label}}}'])
        best_epoch = budget[label]['long']['best_epochs'][0]
        best = next(r for r in selected_rows if int(r['epoch']) == best_epoch)
        body.append(r'\addplot[only marks,mark=*,mark size=2pt,' + COLORS[label]
                    + ',forget plot] coordinates {' + coordinates([best], 'epoch', 'dev_ndcg20') + '};')
    body.extend([r'\addplot[dashed,plotgray,thick] coordinates {(200,0) (200,' + str(upper * 1.1) + ')};',
                 r'\addlegendentry{原200轮上限}'])
    left = axis(body, r'title={（a）延长训练的验证曲线},xlabel={训练轮数},ylabel={验证 NDCG@20},'
                + f'xmin=0,xmax={last_epoch*1.04:.12g},ymin=0,ymax={upper*1.12:.12g}')
    body, upper = [], 0.
    for condition, legend, color, offset in [
        ('original', '上限200轮', 'plotblue', -.35), ('long', '上限1000轮', 'plotgreen', .35)]:
        points = []
        for x, label in zip((0, 3, 6), COLORS):
            metric = budget[label][condition]['test']['NDCG@20']
            points.append((x + offset, metric['mean'], metric['sample_sd']))
            upper = max(upper, metric['mean'] + metric['sample_sd'])
        body.extend([r'\addplot[forget plot,ybar,bar width=9pt,bar shift=0pt,fill=' + color + ',draw=' + color
                     + r',error bars/.cd,y dir=both,y explicit] coordinates {'
                     + error_coordinates(points) + '};',
                     r'\addlegendimage{area legend,fill=' + color + ',draw=' + color + '}',
                     rf'\addlegendentry{{{legend}}}'])
    right = axis(body, r'title={（b）轮数预算与最终测试},ylabel={测试 NDCG@20},'
                 r'xmin=-1,xmax=7,xtick={0,3,6},xticklabels={BPRMF,LightGCN,LightCCF},ymin=0,'
                 + f'ymax={upper*1.12:.12g},legend columns=2')
    blocks['EXTENSION_BUDGET_FIGURE'] = pair(left, right)
    candidates = report['parameters']['candidates']
    panels = []
    for dimension in ('alpha', 'tau'):
        selected = sorted((r for r in candidates if
                           (r['tau'] == .28 if dimension == 'alpha' else r['alpha'] == 1.)),
                          key=lambda r: r[dimension])
        values = [r['dev_ndcg20'] for r in selected]
        span = max(values) - min(values) or .01
        if dimension == 'alpha':
            coords = ' '.join(f"({i},{r['dev_ndcg20']:.12g})" for i, r in enumerate(selected))
            labels = ','.join(f"{r['alpha']:g}" for r in selected)
            options = (r'title={（a）权重敏感性},xlabel={权重 $\alpha$（网格位置）},'
                       r'xtick={0,1,2,3,4,5,6},xticklabels={' + labels + r'},xmin=-0.2,xmax=6.2,')
        else:
            coords = ' '.join(f"({r['tau']},{r['dev_ndcg20']:.12g})" for r in selected)
            options = (r'title={（b）温度敏感性},xlabel={温度 $\tau$},'
                       r'xtick={0.20,0.22,0.24,0.26,0.28,0.30},xmin=0.195,xmax=0.305,'
                       r'yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=4},')
        panels.append(axis([r'\addplot[thick,plotgreen,mark=*] coordinates {' + coords + '};'],
                           options + r'ylabel={验证 NDCG@20},'
                           + f'ymin={max(0.,min(values)-.12*span):.12g},ymax={max(values)+.12*span:.12g}'))
    blocks['EXTENSION_PARAMETER_FIGURE'] = pair(*panels)
    selected = report['parameters']['selected']
    rows = []
    cases = [
        ('LightCCF原配置', report['original_ml'], '1', '0.28'),
        ('关闭NA对照', report['alpha0_ml'], '0', '--'),
    ]
    if selected['alpha'] == 0.:
        cases[1] = ('关闭NA（验证所选）', *cases[1][1:])
    elif selected['alpha'] == 1. and selected['tau'] == .28:
        cases[0] = ('原配置（验证所选）', *cases[0][1:])
    else:
        cases.append(('验证选择配置', report['selected_test'],
                      f"{selected['alpha']:g}", f"{selected['tau']:.2f}"))
    cases.append(('LightGCN固定基线', report['lightgcn_ml'], '--', '--'))
    for label, item, alpha, tau in cases:
        metric = lambda key: (f"$ {item['test'][key]['mean']:.5f}"
                              f"\\pm{item['test'][key]['sample_sd']:.5f}$")
        rows.append(f'{label} & {alpha} & {tau} & {metric("Recall@20")} & {metric("NDCG@20")}' + r' \\')
    blocks['EXTENSION_SELECTED_TABLE'] = '\n'.join(rows)
    ccf, gcn = (budget[m]['long']['test']['NDCG@20']['mean'] for m in ('LightCCF', 'LightGCN'))
    stopping = '/'.join(str(x) for x in budget['LightGCN']['long']['epochs'])
    cap_hits = budget['LightGCN']['long']['hit_cap']
    blocks['EXTENSION_BUDGET_TEXT'] = (
        f"长预算下 LightCCF 的测试 NDCG@20 为{ccf:.5f}，LightGCN 为{gcn:.5f}，"
        f"相对差为$ {report['long_ccf_vs_gcn_ndcg20_percent']:+.2f}\\%$；"
        "应以此补充比较理解原200轮结果，而不能只保留更有利的预算。"
        f"LightGCN 三次实际轮数为{stopping}，"
        + (f"其中{cap_hits}次仍达到1000轮上限，尚不能宣布充分收敛。" if cap_hits
           else "三次均按原早停规则停止，未再达到1000轮上限。"))
    metric = report['selected_test']['test']['NDCG@20']
    reuse = ("复用该配置已完成的三颗种子，不把复用计为新增独立重复"
             if report['selected_reuses_existing_formal_runs']
             else "按所选配置新运行三颗种子，并各自用验证集选轮后测试")
    blocks['EXTENSION_SELECTED_TEXT'] = (
        f"验证搜索选中$\\alpha={selected['alpha']:g},\\tau={selected['tau']:.2f}$，"
        f"三种子测试 NDCG@20 为$ {metric['mean']:.5f}\\pm{metric['sample_sd']:.5f}$；"
        f"相对原 LightCCF 配置变化$ {report['selected_vs_original_ml_ndcg20_percent']:+.2f}\\%$，"
        f"相对关闭NA对照为$ {report['selected_vs_alpha0_ml_ndcg20_percent']:+.2f}\\%$，"
        f"相对固定 LightGCN 基线为$ {report['selected_vs_lightgcn_ml_ndcg20_percent']:+.2f}\\%$。"
        f"这里{reuse}。"
        f"本项目累计{report['total_formal_runs']}次正式运行，另有10次只看验证集的新增搜索；"
        "长预算复跑不增加独立种子的数量。")
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    summary = json.loads((ANALYSIS / 'summary.json').read_text(encoding='utf-8'))
    summaries = {row['name']: row for row in summary['summary']}
    runs = load_csv('runs.csv')
    learning = load_csv('learning_curves_seed2023.csv')
    official = json.loads((ANALYSIS / 'official_douban_summary.json').read_text(encoding='utf-8'))
    histories = load_csv('official_douban_history.csv')
    groups = json.loads((ANALYSIS / 'sparsity_groups.json').read_text(encoding='utf-8'))
    blocks = {}

    rows = []
    for metric, item in official['metrics'].items():
        rows.append(f"{metric} & {item['paper']:.4f} & ${item['mean']:.5f}\\pm{item['sample_sd']:.5f}$"
                    f" & ${item['absolute_difference']:+.5f}$ & ${item['relative_difference_percent']:+.2f}\\%$ \\\\")
    blocks['PAPER_TABLE'] = '\n'.join(rows)

    metrics = list(official['metrics'])
    values = [(i, official['metrics'][metric]['mean'] / official['metrics'][metric]['paper'],
               official['metrics'][metric]['sample_sd'] / official['metrics'][metric]['paper'])
              for i, metric in enumerate(metrics)]
    left = axis([
        r'\addplot[dashed,plotorange,thick,domain=-0.4:3.4]{1};',
        r'\addlegendentry{原文 = 1.0}',
        r'\addplot[only marks,mark=*,plotgreen,thick,error bars/.cd,y dir=both,y explicit] coordinates {'
        + error_coordinates(values) + '};',
        r'\addlegendentry{本机均值$\pm$标准差}',
    ], r'title={（a）豆瓣图书：原文对齐},ylabel={本机均值 / 原文},ymin=0.965,ymax=1.008,'
       r'xmin=-0.4,xmax=3.4,xtick={0,1,2,3},xticklabels={召回@10,NDCG@10,召回@20,NDCG@20},'
       r'ytick={0.97,0.98,0.99,1.00},legend columns=1')
    body = []
    for index, seed in enumerate(official['seeds']):
        selected = [row for row in histories if int(row['seed']) == seed]
        body.extend([r'\addplot[thick,' + SEED_COLORS[index] + ',no marks] coordinates {'
                     + coordinates(selected, 'epoch', 'Recall@20') + '};',
                     rf'\addlegendentry{{种子{seed}}}'])
        best = official['runs'][index]
        body.append(r'\addplot[only marks,mark=*,mark size=1.6pt,' + SEED_COLORS[index]
                    + ',forget plot] coordinates {('
                    + f"{best['best_epoch_by_test_recall20']},{best['test_recall20_at_best']:.12g}" + ')};')
    body.append(r'\addplot[dashed,plotorange,domain=1:56,forget plot]{0.1877};')
    right = axis(body, r'title={（b）作者协议下的收敛},xlabel={训练轮数},ylabel={测试召回率@20},'
                 r'xmin=0,xmax=58,ymin=0,ymax=0.205,legend columns=3')
    blocks['PAPER_FIGURE'] = pair(left, right)

    panels = []
    for prefix, title in [('grocery', '亚马逊食品数据'), ('ml', '电影评分数据')]:
        body = []
        upper = 0.
        for x, suffix in enumerate(('bprmf', 'lightgcn', 'lightccf')):
            row = summaries[f'{prefix}_{suffix}']
            upper = max(upper, row['test_NDCG@20_mean'] + row['test_NDCG@20_sd'])
            color = COLORS[row['model']]
            body.append(r'\addplot[ybar,bar width=16pt,bar shift=0pt,fill=' + color
                        + ',draw=' + color + r',error bars/.cd,y dir=both,y explicit] coordinates {'
                        + error_coordinates([(x, row['test_NDCG@20_mean'], row['test_NDCG@20_sd'])]) + '};')
        panels.append(axis(body, r'title={' + title + r'},ylabel={测试 NDCG@20},ymin=0,'
                           r'xmin=-0.55,xmax=2.55,xtick={0,1,2},xticklabels={BPRMF,LightGCN,LightCCF},'
                           + f'ymax={upper * 1.12:.12g}'))
    blocks['CORE_FIGURE'] = pair(*panels)

    panels = []
    for prefix, title in [('grocery', '亚马逊食品数据'), ('ml', '电影评分数据')]:
        body = []
        upper = 0.
        last_epoch = 0
        for suffix, label in zip(('bprmf', 'lightgcn', 'lightccf'), COLORS):
            rows = [row for row in learning if row['name'] == f'{prefix}_{suffix}']
            upper = max(upper, max(float(row['dev_ndcg20']) for row in rows))
            last_epoch = max(last_epoch, max(int(row['epoch']) for row in rows))
            body.extend([r'\addplot[thick,' + COLORS[label] + ',no marks] coordinates {'
                         + coordinates(rows, 'epoch', 'dev_ndcg20') + '};',
                         rf'\addlegendentry{{{label}}}'])
            best_epoch = next(int(r['best_epoch']) for r in runs
                              if r['name'] == f'{prefix}_{suffix}' and int(r['seed']) == 2023)
            best = next(row for row in rows if int(row['epoch']) == best_epoch)
            body.append(r'\addplot[only marks,mark=*,mark size=2pt,' + COLORS[label]
                        + ',forget plot] coordinates {' + coordinates([best], 'epoch', 'dev_ndcg20') + '};')
        panels.append(axis(body, r'title={' + title + r'},xlabel={训练轮数},ylabel={验证 NDCG@20},ymin=0,'
                           + f'ymax={upper * 1.12:.12g},xmin=0,xmax={last_epoch * 1.04:.12g}'))
    blocks['CURVE_FIGURE'] = pair(*panels)

    panels = []
    for prefix, title in [('grocery', '亚马逊食品数据'), ('ml', '电影评分数据')]:
        body = []
        all_values = []
        for index, seed in enumerate(summary['seeds']):
            values = [next(float(r['test_NDCG@20']) for r in runs
                           if r['name'] == f'{prefix}_{suffix}' and int(r['seed']) == seed)
                      for suffix in ('lightccf_alpha0', 'lightccf')]
            all_values.extend(values)
            body.extend([r'\addplot[thick,mark=*,' + SEED_COLORS[index] + '] coordinates {'
                         + f'(0,{values[0]:.12g}) (1,{values[1]:.12g})' + '};',
                         rf'\addlegendentry{{种子{seed}}}'])
        span = max(all_values) - min(all_values)
        panels.append(axis(body, r'title={' + title + r'},ylabel={测试 NDCG@20},xmin=-0.2,xmax=1.2,'
                           r'xtick={0,1},xticklabels={关闭 NA,开启 NA},'
                           + f'ymin={min(all_values)-.12*span:.12g},ymax={max(all_values)+.12*span:.12g}'))
    blocks['ABLATION_FIGURE'] = pair(*panels)

    panels = []
    for prefix, title in [('grocery', '亚马逊食品数据'), ('ml', '电影评分数据')]:
        body = []
        upper = 0.
        for model in COLORS:
            rows = [next(r for r in groups['summary'] if r['prefix'] == prefix
                         and r['model'] == model and r['group'] == group)
                    for group in ('sparse', 'normal', 'popular')]
            upper = max(upper, max(row['NDCG@20_mean'] + row['NDCG@20_sd'] for row in rows))
            body.extend([r'\addplot[thick,mark=*,' + COLORS[model]
                         + r',error bars/.cd,y dir=both,y explicit] coordinates {'
                         + error_coordinates([(x, row['NDCG@20_mean'], row['NDCG@20_sd'])
                                              for x, row in enumerate(rows)]) + '};',
                         rf'\addlegendentry{{{model}}}'])
        counts = [groups['profiles'][prefix]['groups'][g]['test_users'] for g in ('sparse', 'normal', 'popular')]
        labels = ','.join(f'{{{g}（{n:,}）}}' for g, n in zip(('稀疏组', '中等组', '活跃组'), counts))
        panels.append(axis(body, r'title={' + title + r'},ylabel={测试 NDCG@20},xtick={0,1,2},'
                           r'xticklabels={' + labels + r'},xmin=-0.15,xmax=2.15,ymin=0,'
                           + f'ymax={upper * 1.12:.12g}'))
    blocks['GROUP_FIGURE'] = pair(*panels)

    rows, appendix_rows, cost_rows, ablation_rows = [], [], [], []
    for prefix, title in [('grocery', 'Grocery'), ('ml', 'MovieLens')]:
        for suffix, label in zip(('bprmf', 'lightgcn', 'lightccf'), COLORS):
            row = summaries[f'{prefix}_{suffix}']
            best = '/'.join(str(r['best_epoch']) for r in runs if r['name'] == f'{prefix}_{suffix}')
            metric = lambda name: f"${row['test_' + name + '_mean']:.5f}\\pm{row['test_' + name + '_sd']:.5f}$"
            rows.append(f'{title} & {label} & {metric("Recall@20")} & {metric("NDCG@20")} & {best} \\\\')
            appendix_rows.append(f'{title} & {label} & {metric("Recall@10")} & {metric("NDCG@10")} \\\\')
            cost_rows.append(f"{title} & {label} & {row['mean_epoch_seconds_mean']:.3f}"
                             f" & {row['wall_seconds_mean']/60:.2f} & {row['peak_cuda_mb_mean']:.1f} \\\\")
        for alpha, suffix in [(0, 'lightccf_alpha0'), (1, 'lightccf')]:
            row = summaries[f'{prefix}_{suffix}']
            ablation_rows.append(f'{title} & {alpha} & {metric("Recall@20")} & {metric("NDCG@20")}'
                                  f" & {row['mean_epoch_seconds_mean']:.3f} \\\\")
    blocks['CORE_TABLE'] = '\n'.join(rows)
    blocks['APPENDIX_TABLE'] = '\n'.join(appendix_rows)
    blocks['COST_TABLE'] = '\n'.join(cost_rows)
    blocks['ABLATION_TABLE'] = '\n'.join(ablation_rows)

    group_rows = []
    for prefix, title in [('grocery', 'Grocery'), ('ml', 'MovieLens')]:
        profile = groups['profiles'][prefix]
        q1, q2 = profile['thresholds']
        bounds = rf'$d\le {q1}$ / ${q1}<d\le {q2}$ / $d>{q2}$'
        counts = ' / '.join(f"{profile['groups'][g]['test_users']:,}"
                            for g in ('sparse', 'normal', 'popular'))
        group_rows.append(f'{title} & {bounds} & {counts}' + r' \\')
    blocks['GROUP_TABLE'] = '\n'.join(group_rows)
    extension_path = ANALYSIS / 'extension_summary.json'
    if extension_path.exists():
        blocks.update(extension_blocks(json.loads(extension_path.read_text(encoding='utf-8'))))

    (ANALYSIS / 'latex_blocks.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.report:
        source = args.report.read_text(encoding='utf-8')
        for name, block in blocks.items():
            if '% BEGIN AUTO '+name+'\n' not in source:
                continue  # Historical figures may be replaced by later audited evidence.
            pattern = re.compile(r'(% BEGIN AUTO ' + re.escape(name) + r'\n).*?(% END AUTO ' + re.escape(name) + r')', re.S)
            source, count = pattern.subn(lambda match: match[1] + block + '\n' + match[2], source)
            if count != 1:
                raise ValueError(f'Expected exactly one region: {name}; got {count}')
        args.report.write_text(source, encoding='utf-8')
    print(f'Generated {len(blocks)} audited LaTeX blocks')


if __name__ == '__main__':
    main()
