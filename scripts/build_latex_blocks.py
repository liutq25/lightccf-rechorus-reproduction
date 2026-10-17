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
        r'\addlegendentry{Paper = 1.0}',
        r'\addplot[only marks,mark=*,plotgreen,thick,error bars/.cd,y dir=both,y explicit] coordinates {'
        + error_coordinates(values) + '};',
        r'\addlegendentry{Local mean $\pm$ SD}',
    ], r'title={(a) Douban-book: agreement},ylabel={Local / paper},ymin=0.965,ymax=1.008,'
       r'xmin=-0.4,xmax=3.4,xtick={0,1,2,3},xticklabels={R@10,N@10,R@20,N@20},'
       r'ytick={0.97,0.98,0.99,1.00},legend columns=1')
    body = []
    for index, seed in enumerate(official['seeds']):
        selected = [row for row in histories if int(row['seed']) == seed]
        body.extend([r'\addplot[thick,' + SEED_COLORS[index] + ',no marks] coordinates {'
                     + coordinates(selected, 'epoch', 'Recall@20') + '};',
                     rf'\addlegendentry{{{seed}}}'])
        best = official['runs'][index]
        body.append(r'\addplot[only marks,mark=*,mark size=1.6pt,' + SEED_COLORS[index]
                    + ',forget plot] coordinates {('
                    + f"{best['best_epoch_by_test_recall20']},{best['test_recall20_at_best']:.12g}" + ')};')
    body.append(r'\addplot[dashed,plotorange,domain=1:56,forget plot]{0.1877};')
    right = axis(body, r'title={(b) Author-protocol convergence},xlabel={Epoch},ylabel={Test R@20},'
                 r'xmin=0,xmax=58,ymin=0,ymax=0.205,legend columns=5')
    blocks['PAPER_FIGURE'] = pair(left, right)

    panels = []
    for prefix, title in [('grocery', 'Amazon Grocery'), ('ml', 'MovieLens 1M TopK')]:
        body = []
        upper = 0.
        for x, suffix in enumerate(('bprmf', 'lightgcn', 'lightccf')):
            row = summaries[f'{prefix}_{suffix}']
            upper = max(upper, row['test_NDCG@20_mean'] + row['test_NDCG@20_sd'])
            color = COLORS[row['model']]
            body.append(r'\addplot[ybar,bar width=16pt,bar shift=0pt,fill=' + color
                        + ',draw=' + color + r',error bars/.cd,y dir=both,y explicit] coordinates {'
                        + error_coordinates([(x, row['test_NDCG@20_mean'], row['test_NDCG@20_sd'])]) + '};')
        panels.append(axis(body, r'title={' + title + r'},ylabel={Test NDCG@20},ymin=0,'
                           r'xmin=-0.55,xmax=2.55,xtick={0,1,2},xticklabels={BPRMF,LightGCN,LightCCF},'
                           + f'ymax={upper * 1.12:.12g}'))
    blocks['CORE_FIGURE'] = pair(*panels)

    panels = []
    for prefix, title in [('grocery', 'Amazon Grocery'), ('ml', 'MovieLens 1M TopK')]:
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
        panels.append(axis(body, r'title={' + title + r'},xlabel={Epoch},ylabel={Validation NDCG@20},ymin=0,'
                           + f'ymax={upper * 1.12:.12g},xmin=0,xmax={last_epoch * 1.04:.12g}'))
    blocks['CURVE_FIGURE'] = pair(*panels)

    panels = []
    for prefix, title in [('grocery', 'Amazon Grocery'), ('ml', 'MovieLens 1M TopK')]:
        body = []
        all_values = []
        for index, seed in enumerate(summary['seeds']):
            values = [next(float(r['test_NDCG@20']) for r in runs
                           if r['name'] == f'{prefix}_{suffix}' and int(r['seed']) == seed)
                      for suffix in ('lightccf_alpha0', 'lightccf')]
            all_values.extend(values)
            body.extend([r'\addplot[thick,mark=*,' + SEED_COLORS[index] + '] coordinates {'
                         + f'(0,{values[0]:.12g}) (1,{values[1]:.12g})' + '};',
                         rf'\addlegendentry{{{seed}}}'])
        span = max(all_values) - min(all_values)
        panels.append(axis(body, r'title={' + title + r'},ylabel={Test NDCG@20},xmin=-0.2,xmax=1.2,'
                           r'xtick={0,1},xticklabels={Without NA,With NA},'
                           + f'ymin={min(all_values)-.12*span:.12g},ymax={max(all_values)+.12*span:.12g}'))
    blocks['ABLATION_FIGURE'] = pair(*panels)

    panels = []
    for prefix, title in [('grocery', 'Amazon Grocery'), ('ml', 'MovieLens 1M TopK')]:
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
        labels = ','.join(f'{{{g} ({n:,})}}' for g, n in zip(('Sparse', 'Normal', 'Popular'), counts))
        panels.append(axis(body, r'title={' + title + r'},ylabel={Test NDCG@20},xtick={0,1,2},'
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

    (ANALYSIS / 'latex_blocks.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.report:
        source = args.report.read_text(encoding='utf-8')
        for name, block in blocks.items():
            pattern = re.compile(r'(% BEGIN AUTO ' + re.escape(name) + r'\n).*?(% END AUTO ' + re.escape(name) + r')', re.S)
            source, count = pattern.subn(lambda match: match[1] + block + '\n' + match[2], source)
            if count != 1:
                raise ValueError(f'Expected exactly one region: {name}; got {count}')
        args.report.write_text(source, encoding='utf-8')
    print(f'Generated {len(blocks)} audited LaTeX blocks')


if __name__ == '__main__':
    main()
