"""Integrate audited follow-up results into the standalone Chinese report."""
from pathlib import Path
import argparse
import json
import math
import hashlib
import re

ROOT=Path(__file__).resolve().parents[1]


def block(name,value,blocks):
    blocks[name]=value
    return '% BEGIN AUTO '+name+'\n'+value+'\n% END AUTO '+name


def chart(items,labels,title):
    colors=['plotblue','plotorange','plotgreen']
    maximum=max(v['mean']+v['sample_sd'] for v in items)*1.15
    lines=[r'\begin{tikzpicture}[baseline=(current bounding box.north)]\begin{axis}[reportaxis,'
           +'title={'+title+'},ylabel={测试 NDCG@20},ymin=0,ymax='+f'{maximum:.12g}'
           +',xmin=-0.55,xmax=2.55,xtick={0,1,2},xticklabels={'+','.join(labels)+'}]']
    for x,(value,color) in enumerate(zip(items,colors)):
        lines.append(r'\addplot[ybar,bar width=16pt,bar shift=0pt,fill='+color+',draw='+color
                     +r',error bars/.cd,y dir=both,y explicit] coordinates {'
                     +f"({x},{value['mean']:.12g}) +- (0,{value['sample_sd']:.12g})"+'};')
    lines.append(r'\end{axis}\end{tikzpicture}')
    return '\n'.join(lines)


def estimate(value):
    return f"${value['mean']:.5f}\\pm{value['sample_sd']:.5f}$"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'复现报告.tex')
    parser.add_argument('--base',type=Path,help='Optional editable base; use --base equal to --source for an already updated report.')
    args=parser.parse_args()
    summary=json.loads((ROOT/'analysis/followup_summary.json').read_text(encoding='utf-8'))
    destination=args.source.resolve()
    archived=ROOT/'tmp/followup_previous_20261101/复现报告.tex'
    old=args.base.resolve() if args.base else (archived if archived.exists() else destination)
    tex=old.read_text(encoding='utf-8')
    last_generated=ROOT/'analysis/followup_report_source.json'
    current=destination.read_text(encoding='utf-8')
    if current!=tex:
        previous=json.loads(last_generated.read_text(encoding='utf-8')) if last_generated.exists() else {}
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=previous.get('sha256'):
            raise RuntimeError('Report changed during the experiments; merge its edits before regeneration.')
    blocks={}
    formal=summary['total_formal']
    validation=summary['total_validation_only']
    abstract=('在ReChorus的两个数据集、两种协同过滤基线上复现LightCCF。'
              'Douban四指标仍比原文低0.59\\%--1.27\\%。'
              '四候选验证选择与三个新种子确认后，相对LightGCN的NDCG@20在Grocery、MovieLens分别变化'
               f"${summary['paired_confirmations']['grocery']['relative_ndcg20_percent']:+.2f}\\%$和"
               f"${summary['paired_confirmations']['ml']['relative_ndcg20_percent']:+.2f}\\%$。"
               '以随机匹配屏蔽对照检验批内竞争解释。')
    tex=re.sub(r'(?s)(\\begin\{abstract\}).*?(\\end\{abstract\})',lambda m:m[1]+'\n'+block('FOLLOWUP_ABSTRACT',abstract,blocks)+'\n'+m[2],tex)
    tex=tex.replace('实验与报告版本：2026年10月18日','实验与报告版本：2026年11月2日')
    tex=tex.replace('作者协议核对固定种子2023--2027、MF 编码器、作者 train/test 文件和默认损失参数，最多120轮。',
                    '作者协议固定种子2023--2027、MF及原损失，最多120轮。原文件为13,024用户、22,347物品、792,062交互，与表4舍入规模一致；无重复交互或跨集重叠。测试中191位用户无训练交互，沿用作者代码保留。')
    tex=tex.replace('两者均为框架提供的数据来源；训练图只含 train 交互。',
                    'Grocery 按现有顺序留末两条，同日并列不保证细粒度先后。MovieLens 保留评分$\\ge4$的反馈并做5-core，按日历跨度80\\%/90\\%切点划分，去除dev/test中训练未见ID；不是按交互数8:1:1。训练图只含train。')
    tex=tex.replace('每人平均12.72个测试正例。','每人平均12.72个测试正例，与验证集仅重叠116位用户。')
    tex=tex.replace('课程 LightCCF 使用零层 MF，LightGCN 基线使用三层。',
                    '固定配置主实验使用零层LightCCF和三层LightGCN，补充搜索的层数另列。')
    tex=tex.replace('正则也采用代码的批内嵌入定义，而非把论文全矩阵记号直接照搬。',
                    '正则采用代码的批内嵌入定义。简化点积梯度包含正邻居吸引和批内软权重竞争，使交互信息通过优化传递；实际归一化和共享参数的限制见第8节。')
    tex=tex.replace(r'\subsection{忠实性证据及其限度}',r'''\subsection{框架移植与数值核验}
模型继承ReChorus的GeneralModel，复用BaseReader。额外传递positive\_item字段，保证候选列打乱后仍以真值构造NA；统一Runner用GPU拒绝采样并保存验证最优权重。屏蔽变体单独成类，保留原BPR与正则。
''')
    rows=[]
    for prefix,title in (('grocery','Grocery'),('ml','MovieLens')):
        for model in ('BPRMF','LightGCN','LightCCF'):
            item=summary['confirmation'][prefix+'_'+model]
            config=item['config']
            if model=='BPRMF':
                selected=rf'$\eta={config["lr"]:g},\lambda=10^{{{round(math.log10(config["l2"]))}}}$'
            elif model=='LightGCN':
                selected=rf'$L={config["n_layers"]},\lambda=10^{{{round(math.log10(config["l2"]))}}}$'
            else:
                selected=rf'$\alpha={config["ssl_weight"]:g}$'
            rows.append(f"{title} & {model} & {selected} & {estimate(item['test']['Recall@20'])} & {estimate(item['test']['NDCG@20'])}"+r' \\')
    figure=[]
    for prefix,title in (('grocery','亚马逊食品数据'),('ml','电影评分数据')):
        items=[summary['confirmation'][prefix+'_'+model]['test']['NDCG@20'] for model in ('BPRMF','LightGCN','LightCCF')]
        figure.append(chart(items,['BPRMF','LightGCN','LightCCF'],title))
    tuning=r'''\subsection{同量验证候选与新种子确认}
此前仅给 LightCCF 做12点单种子切片，未同时调基线。新增方案在运行前锁定：两数据集各模型均4个候选；BPRMF 的学习率 $\eta\in\{0.0005,0.001\}$、权重衰减 $\lambda\in\{10^{-7},10^{-6}\}$；LightGCN 层数 $L\in\{1,3\}$ 与同一正则网格；LightCCF 的 $\alpha\in\{0.1,0.5,1,2.5\}$，其余保持固定。统一1000轮上限及原早停规则。只按2023种子的验证 NDCG@20选择，再用2026/2027/2028三颗新种子确认，最终测试不参与选择。复用已正常早停的同配置验证记录；新搜索均不计算测试。

\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}
\caption{每模型四候选验证选择后，三颗新种子的测试成绩。选择只使用验证集；窄网格不等于穷尽调参。}\label{tab:tuning}
\begin{tabular}{lllrr}\toprule 数据 & 模型 & 所选参数 & Recall@20 & NDCG@20\\\midrule
'''+block('FOLLOWUP_TUNING_TABLE','\n'.join(rows),blocks)+r'''
\bottomrule\end{tabular}\end{table}
\begin{figure}[htbp]\centering
'''+block('FOLLOWUP_TUNING_FIGURE','\n\\hfill\n'.join(figure),blocks)+r'''
\caption{同量候选选择后的三新种子结果，误差线为训练种子样本标准差。候选数相同，但模型每轮计算量和网格覆盖不同。}\label{fig:tuning}
\end{figure}
'''
    conclusions=[]
    for prefix,title in (('grocery','Grocery'),('ml','MovieLens')):
        value=summary['paired_confirmations'][prefix]
        ci=value['conditional_user_bootstrap_95ci']
        conclusions.append(f"{title} 差 ${value['mean']:+.5f}\\pm{value['sample_sd']:.5f}$，"
                           f"条件区间 $[{ci[0]:+.5f},{ci[1]:+.5f}]$。")
    cap_text=f"本轮搜索/确认触顶次数为{summary['search_cap_hits']}/{summary['confirmation_cap_hits']}。"
    tuning+=block('FOLLOWUP_TUNING_TEXT','LightCCF减LightGCN的NDCG@20：'+''.join(conclusions)+cap_text+'固定三次训练，按测试用户成对重采样10000次，取2.5\\%/97.5\\%分位点；不含训练或跨人群不确定性。本轮候选同量，但LightCCF此前探索更多；单搜索种子、窄网格和后验设计仍限制比较。',blocks)+'\n'
    if summary.get('shared_gpu_scheduling'):
        tuning=tuning.replace('新搜索均不计算测试。','新搜索均不计算测试。部分确认任务尝试共享GPU，争用后恢复串行；新增运行时不用于效率比较。')
    title=(r'\subsection{同量验证候选与新种子确认}' if r'\subsection{同量验证候选与新种子确认}' in tex
           else r'\subsection{电影数据的参数敏感性与确认}')
    start=tex.index(title)
    boundary=(r'\section{证据支持到哪里' if r'\section{证据支持到哪里' in tex
              else r'\section{证据边界与方法局限}')
    end=tex.index(boundary,start)
    tuning+=r'''
\subsection{原文覆盖范围}
课程两数据、两基线已完成；原文 Tmall/Amazon-book 全套结果、LightCCF 的0--3层及抗噪实验仍未覆盖。论文未充分定义对抗攻击生成规则，未以随机加边替代。附录逐项说明，不从零层结果推断所有数据无需GCN。
\FloatBarrier

'''
    tex=tex[:start]+tuning+tex[end:]
    title=(r'\subsection{同用户屏蔽与随机对照的实测}' if r'\subsection{同用户屏蔽与随机对照的实测}' in tex
           else r'\subsection{批内竞争是可检验假设}')
    start=tex.index(title)
    boundary=(r'\subsection{“聚合效应”' if r'\subsection{“聚合效应”' in tex
              else r'\subsection{理论解释的适用条件}')
    end=tex.index(boundary,start)
    mask_rows=[]
    mask_figure=[]
    mask_text=[]
    for prefix,title in (('grocery','Grocery'),('ml','MovieLens')):
        item=summary['masking'][prefix]
        values=[item[policy]['test']['NDCG@20'] for policy in ('full','same_user','random_matched')]
        mask_figure.append(chart(values,['原始NA','同用户屏蔽','随机屏蔽'],'亚马逊食品数据' if prefix=='grocery' else '电影评分数据'))
        for key,label in (('same_minus_full','同用户减原始'),('same_minus_random','同用户减随机')):
            delta=item[key]
            ci=delta['conditional_user_bootstrap_95ci']
            mask_rows.append(f"{title} & {label} & ${delta['mean']:+.5f}\\pm{delta['sample_sd']:.5f}$ & $[{ci[0]:+.5f},{ci[1]:+.5f}]$"+r' \\')
        positive=sum(x>0 for x in item['same_minus_random']['per_seed'])
        lower,upper=item['same_minus_random']['conditional_user_bootstrap_95ci']
        mask_text.append(f'{title} 相对随机对照的三次差值中有{positive}次为正，')
        if lower <= 0 <= upper:
            mask_text.append('条件用户区间跨零。')
        elif lower > 0:
            mask_text.append('条件用户区间高于零。')
        else:
            mask_text.append('条件用户区间低于零。')
    masking=r'''\subsection{同用户屏蔽与随机对照的实测}
同用户其他正交互可能进入NA分母。原批量2048的碰撞概率为24.45\%/45.16\%，但每锚点期望仅0.35/0.74条，概率不能代替损害证据。新增干预只屏蔽与锚点同用户的非对角列，保留对角及平滑；随机对照逐行均匀屏蔽相同数量非对角列，独立随机流不改变初始化、负采样与batch排列。两变体用原$\alpha=1,\tau=0.28$、相同2023--2025种子、1000轮上限和各自验证选轮。原始NA复用正常早停记录，新增实现单独成类并通过手算验收。

\begin{figure}[htbp]\centering
'''+block('FOLLOWUP_MASK_FIGURE','\n\\hfill\n'.join(mask_figure),blocks)+r'''
\caption{具体NA干预的三种子测试结果，误差线为样本标准差。随机对照每行移除数量相同，可能偶然移除同用户列；并非验证选参后的配置。}\label{fig:masking}
\end{figure}
\begin{table}[htbp]\centering\small
\caption{NDCG@20配对差；区间为固定三次训练下的用户重采样95\%区间，未包含训练与时间窗口变化。}\label{tab:masking}
\begin{tabular}{llrr}\toprule 数据 & 对比 & 均值$\pm$种子标准差 & 条件区间\\\midrule
'''+block('FOLLOWUP_MASK_TABLE','\n'.join(mask_rows),blocks)+r'''
\bottomrule\end{tabular}\end{table}
'''
    direction=('两数据均未显示三个种子一致的改善。'
               if all(sum(x>0 for x in summary['masking'][p]['same_minus_random']['per_seed'])<3
                      for p in ('grocery','ml')) else '')
    masking+=block('FOLLOWUP_MASK_TEXT',''.join(mask_text)+direction+'随机对照只匹配数量，未匹配相似度；也未覆盖跨用户正物品、温度权重与采样交互。差值不能全归于假负例，亦不足以解释全部退化。',blocks)+'\n'
    tex=tex[:start]+masking+tex[end:]
    tex=tex.replace('总计10项验收记录保存在','最初10项损失验收记录保存在')
    tex=tex.replace('本实现 LightGCN 每批重算三层完整图传播','原主实验 LightGCN 每批重算三层完整图传播')
    tex=tex.replace('ID 嵌入也不能直接解决完全未见用户或物品的冷启动。',
                    'Grocery的26个物品缺少训练正反馈，保留原样；ID嵌入亦不能直接解决全新用户或物品的冷启动。')
    tex=tex.replace('由对角项还得 $\\ell_b', '对非零归一化嵌入，由对角项还得 $\\ell_b')
    tex=tex.replace('三种子、小规模测试覆盖、仅单种子一维选参及缺少各模型等量搜索，仍限制结论。',
                    '本轮已补同量候选及新种子确认，但小规模测试覆盖、单搜索种子、窄网格和方法特定搜索空间仍限制结论。')
    old_conclusion='综上，Douban 四项指标接近原文，但残差及测试选轮问题仍在；Grocery 延长预算后仍观察到优势，MovieLens 的固定参数迁移失败。较小 NA 权重能缓解退化，却仍落后于关闭 NA，不能称为成功的预测算法改进。后续应给各模型等量验证搜索、用新划分复核，再单独检验批内屏蔽；抗噪实验须先获得可执行攻击协议。'
    conclusion=(r'\FloatBarrier'+'\n'+r'\section{结论}'+'\n'
                'Douban四项指标与原文接近，仍有0.59\\%--1.27\\%的残差及测试选轮边界。'
                '同量候选选择及三个新种子确认后，零层LightCCF相对LightGCN的NDCG@20在Grocery与MovieLens分别变化'
                f"${summary['paired_confirmations']['grocery']['relative_ndcg20_percent']:+.2f}\\%$、"
                f"${summary['paired_confirmations']['ml']['relative_ndcg20_percent']:+.2f}\\%$。"
                '同用户屏蔽相对随机屏蔽的配对差分别为'
                f"${summary['masking']['grocery']['same_minus_random']['mean']:+.5f}$、"
                f"${summary['masking']['ml']['same_minus_random']['mean']:+.5f}$，"
                '结论限于当前数据、选参范围及具体干预，不能推广为普遍保证。后续需新时间划分与更多训练种子；抗噪实验须先明确攻击协议。')
    marked_conclusion=block('FOLLOWUP_CONCLUSION',conclusion,blocks)
    if '% BEGIN AUTO FOLLOWUP_CONCLUSION' in tex:
        tex=re.sub(r'% BEGIN AUTO FOLLOWUP_CONCLUSION\n.*?\n% END AUTO FOLLOWUP_CONCLUSION',lambda m:marked_conclusion,tex,flags=re.S)
    elif old_conclusion in tex:
        tex=tex.replace(old_conclusion,marked_conclusion)
    elif r'\section{结论}' in tex:
        tex=re.sub(r'\\section\{结论\}.*?(?=\\section\{小组分工)',lambda m:conclusion.split('\n',1)[1]+'\n\n',tex,flags=re.S)
    tex=tex.replace('全部41次正式实验和10次仅验证搜索',f'全部{formal}次正式实验和{validation}次仅验证搜索')
    tex=tex.replace('第4、\\ref{sec:extensions}节的范围说明','第4、\\ref{sec:extensions}节的范围说明')
    tex=tex.replace('图\\ref{fig:budget}、\\ref{fig:parameters}','图\\ref{fig:budget}、\\ref{fig:tuning}')
    tex=tex.replace('图\\ref{fig:parameters}；表\\ref{tab:selected}','图\\ref{fig:tuning}；表\\ref{tab:tuning}')
    tex=tex.replace('MovieLens 七个权重、六个温度的一维切片，共12配置；单种子选参、三种子确认，非原三数据数值复刻。',
                    '旧MovieLens12点探索归档；新增两数据、每模型4候选与3新种子确认，仍非原三数据数值复刻。')
    tex=tex.replace('正文共7幅自绘中文实验图','正文共8幅自绘中文实验图')
    tex=tex.replace('MovieLens 参数切片提供局部敏感性证据，不能替代原三数据网格或二维联合最优搜索。','本轮补充同量候选与屏蔽对照，仍不能替代原三数据网格或二维联合最优搜索。')
    tex=tex.replace('编译 PDF、', '编译 PDF、')
    tex=tex.replace(r'\section{是否接近论文原文结果}',r'\section{原始协议的数值核对}')
    tex=tex.replace(r'\section{证据支持到哪里：独立思考与不足}',r'\section{证据边界与方法局限}')
    tex=tex.replace(r'\subsection{“聚合效应”不等于一般严格等价}',r'\subsection{理论解释的适用条件}')
    bibliography=re.search(r'(\\begin\{thebibliography\}\{9\})(.*?)(\\end\{thebibliography\})',tex,re.S)
    if bibliography:
        content=bibliography[2]
        entries=dict(re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)',content,re.S))
        order=[]
        for citation in re.findall(r'\\cite\{([^}]+)\}',tex[:bibliography.start()]):
            for key in citation.split(','):
                if key not in order:
                    order.append(key)
        order.extend(key for key in entries if key not in order)
        prefix=content[:content.index(r'\bibitem')]
        ordered=prefix+''.join(r'\bibitem{'+key+'}'+entries[key] for key in order)
        tex=tex[:bibliography.start()]+bibliography[1]+ordered+bibliography[3]+tex[bibliography.end():]
    tex=tex.replace('五颗种子','五个种子').replace('三颗种子','三个种子').replace('每颗种子','每个种子')
    tex=tex.replace('三新种子','三个新种子').replace('三颗新种子','三个新种子')
    tex=tex.replace('补充检验图\\ref{fig:budget}、\\ref{fig:tuning} &', '补充预算与选择检验 &')
    tex=tex.replace(r'\path{analysis/extension_summary.json}；\path{parameter_search.csv}；\path{budget_learning_curves_seed2023.csv}',
                    r'\path{analysis/extension_summary.json}；\path{analysis/followup_summary.json}；\path{runs/followup_20261101/validation_selection.json}')
    # Keep historical curves and the full coverage ledger in the evidence,
    # while giving the report enough room for the new controlled comparisons.
    tex=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',
               lambda m:'' if r'\label{fig:curves}' in m[0] else m[0],tex,flags=re.S)
    budget_start=(r'\subsection{相同轮数上限下延长三个模型}' if r'\subsection{相同轮数上限下延长三个模型}' in tex
                  else r'\subsection{训练预算复核}')
    start=tex.index(budget_start)
    end=tex.index(r'\subsection{同量验证候选与新种子确认}',start)
    historical=json.loads((ROOT/'analysis/extension_summary.json').read_text(encoding='utf-8'))
    recap=('原200轮实验中Grocery的LightGCN触及上限，故三个模型以原种子配对重跑1000轮上限，共9次，均正常早停；公共训练前缀验证值一致。'
           f"原固定参数下LightCCF的NDCG@20仍比LightGCN高${historical['long_ccf_vs_gcn_ndcg20_percent']:.2f}\\%$。"
           '这是预算配对，不增加独立种子。')
    tex=tex[:start]+r'\subsection{训练预算复核}'+'\n'+block('FOLLOWUP_BUDGET_RECAP',recap,blocks)+'\n\n'+tex[end:]
    coverage_heading=(r'\section{原文图表覆盖核查}' if r'\section{原文图表覆盖核查}' in tex
                      else r'\section{原文图表覆盖范围}')
    coverage=r'''\section{原文图表覆盖范围}\label{sec:coverage}
已完成最终NA的实现核验、Douban原文表5中LightCCF一行的五次核对，以及课程两数据、两基线和控制实验。未覆盖原文其余两数据和16个基线全表、CL-SS/CL-UI、LightCCF的0--3层、原噪声鲁棒性实验。稀疏分组与参数搜索使用课程数据，属于扩展。逐项清单见\path{research/report_verification_zh.md}。

'''
    if coverage_heading in tex:
        start=tex.index(coverage_heading)
        boundary=(r'\section{补充数值与追溯索引}' if r'\section{补充数值与追溯索引}' in tex
                  else r'\section{版本与追溯索引}')
        end=tex.index(boundary,start)
        tex=tex[:start]+coverage+tex[end:]
    tex=tex.replace('覆盖清单见附录表\\ref{tab:coverage}','覆盖范围见附录\\ref{sec:coverage}')
    trace=(r'证据索引：\path{research/rechorus_data_manifest.json}、\path{research/paper_source_manifest.json}记录来源；'
           r'\path{analysis/summary.csv}、\path{analysis/followup_summary.json}给出汇总；'
           r'逐轮与逐用户记录位于提交包\path{evidence/runs/}。')
    tex=re.sub(r'\\begin\{table\}.*?\\end\{table\}',
               lambda m:trace if '数据、结果与图表的追溯位置' in m[0] else m[0],tex,flags=re.S)
    tex=re.sub(r'\\noindent\\textbf\{编译方式：\}.*?(?=\\section\{NA 平滑项)',
               lambda m:r'\noindent\textbf{编译：}以XeLaTeX编译至引用稳定。参考文献、表格与矢量坐标均内嵌；环境要求见README。'+'\n\n',tex,flags=re.S)
    effective_epsilon=1e-5*math.exp(1/.28)
    attenuation=(1/2048)/(1/2048+effective_epsilon)
    epsilon_note=r'''对单位归一化向量，令$p_b=\exp(z_{bb})/\sum_c\exp(z_{bc})$，则$q_b=e^{-1/\tau}p_b$，因此还可写为
\begin{equation}
\ell_b=\frac1\tau-\log\!\left(p_b+\varepsilon e^{1/\tau}\right).
\end{equation}
'''+f'当$\\tau=0.28$时，有效平滑量为${effective_epsilon:.8f}$；若仅作$p_b=1/2048$的均匀概率示例，梯度缩放约为${attenuation:.3f}$。这说明平滑、温度与批量相互关联，不能把平滑项一概视为可忽略；该示例不是实测训练分布，也不是性能退化的因果证据。'
    marked_epsilon=block('FOLLOWUP_EPS_NOTE',epsilon_note,blocks)
    if '% BEGIN AUTO FOLLOWUP_EPS_NOTE' in tex:
        tex=re.sub(r'% BEGIN AUTO FOLLOWUP_EPS_NOTE\n.*?\n% END AUTO FOLLOWUP_EPS_NOTE',lambda m:marked_epsilon,tex,flags=re.S)
    else:
        tex=tex.replace(r'\end{document}',marked_epsilon+'\n'+r'\end{document}')
    # Shorten repeated prose while keeping protocols and numerical evidence.
    replacements=[
        ('本次工作分为两个口径：',
         r'本文分别核对作者Douban-book原始协议和课程统一协议。后者在ReChorus 2.0\cite{rechorus}中比较BPRMF、LightGCN与LightCCF，使用两个框架数据集；其绝对分数不作为原表误差。再以NA消融、预算复核、验证搜索和屏蔽对照检验结论。原文覆盖范围见第7.3节及附录\ref{sec:coverage}。'),
        ('四项均值的相对差为',
         r'表\ref{tab:paper}差距超过四位小数舍入范围，五次四指标均低于原文；数值接近，尚未完全一致。原文未公开种子与逐次方差，版本、硬件影响也未隔离，残差原因仍不确定。最佳轮为43/43/45/43/43，54--56轮早停，未触顶。'),
        (r'\textbf{这一对齐不等于无偏泛化证据。}',
         r'测试集反复选轮存在选择偏差，仅用于原数值核对；课程实验用验证集。原文表6零层Douban的Recall@20为0.1866，与表5不同，本文保留表5参照。Amazon-book主结果对应三层GCN，不能概括为全部零层胜出。'),
        ('原固定配置下，Grocery 三组',
         r'固定配置下Grocery三组的LightCCF均高于基线，MovieLens三组均低于基线。MovieLens稀疏训练用户2,018人中仅14人进入测试，覆盖0.69\%；活跃组为164/1,999，即8.20\%。故不能推广至全部稀疏用户；各组相关集和时间窗口不同，得分差不能证明假正例多少。'),
        ('原主实验 LightGCN 每批重算',
         r'原主实验LightGCN每批重算三层完整图，LightCCF单轮分别快约2.6倍和6.1倍，但慢于BPRMF。相对$\alpha=0$，NA使单轮耗时增加约2.9倍和4.0倍：其时间为$O(B^2d)$、得分矩阵显存为$O(B^2)$，纯BPR约$O(Bd)$。本机4GB可运行，效率收益不能抵消MovieLens准确率损失；硬件与计时口径不同，不直接复用原文秒数。'),
        (r'\textbf{建议分工，提交前按实际参与情况调整：}',
         r'\textbf{分工建议（实际参与待确认）：}刘挺祺负责论文研读、模型移植和损失核验；黄凯伟负责数据评价、基线消融和结果图表；共同分析复核。提交前须按实际参与填写。'),
    ]
    for prefix,replacement in replacements:
        tex=re.sub(r'(?m)^'+re.escape(prefix)+r'[^\n]*$',lambda m:replacement,tex)
    tex=re.sub(r'\\begin\{table\}.*?\\end\{table\}',
               lambda m:'' if r'\label{tab:ten}' in m[0] else m[0],tex,flags=re.S)
    tex=tex.replace('@10补充结果见附录。','@10结果随数值汇总归档。')
    tex=tex.replace(r'\section{补充数值与追溯索引}',r'\section{版本与追溯索引}')
    tex=tex.replace('200轮上限、固定配置的测试 NDCG@20，误差线为三个训练种子的样本标准差。各数据集单独标度，柱形纵轴从零开始；分数来自验证集选出的权重。',
                    '200轮上限、固定配置的测试NDCG@20；误差线为三个训练种子的样本标准差。')
    tex=tex.replace('原始协议核对。(a)四项指标除以原文表5对应值，误差线为五次标准差同比缩放，纵轴局部放大；(b)五个种子的完整测试 Recall@20 曲线，圆点为所选轮，虚线为论文0.1877。该图使用测试指标选轮，仅用于作者协议对照。',
                    '作者协议核对。(a)指标除以原文值，误差线为五次标准差同比缩放，纵轴局部放大；(b)五种子测试Recall@20，圆点为所选轮，虚线为原文0.1877。该协议使用测试选轮。')
    tex=tex.replace('原200轮主实验按训练历史分组的测试 NDCG@20；括号为测试用户数，误差线为三个训练种子的标准差。MovieLens 稀疏组只有14名测试用户，其误差线未包含用户抽样不确定性。',
                    '固定配置按训练历史分组；括号为测试用户数，误差线为种子标准差。MovieLens稀疏组仅14人，误差线不含用户抽样波动。')
    # Keep complete tables, version hashes and derivations in the evidence.
    # The submitted report retains the core comparisons and a short proof note.
    tex=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',
               lambda m:'' if any(r'\label{fig:'+key+'}' in m[0] for key in ('core','ablation')) else m[0],tex,flags=re.S)
    if r'\clearpage\appendix' in tex:
        tex=tex.split(r'\clearpage\appendix')[0]+r'\end{document}'+'\n'
    tex=re.sub(r'% BEGIN AUTO FOLLOWUP_EPS_NOTE\n.*?\n% END AUTO FOLLOWUP_EPS_NOTE\n','',tex,flags=re.S)
    tex=tex.replace(r'及附录\ref{sec:coverage}','')
    tex=tex.replace('固定提交见附录','固定提交见README')
    tex=tex.replace('附录逐项说明，不从零层结果推断所有数据无需GCN。',
                    r'其余16基线全表及CL-SS/CL-UI亦未覆盖；逐项清单见研究记录。不从零层结果推断所有数据无需GCN。')
    if '平滑项也影响优化' not in tex:
        note=r'''平滑项也影响优化。令$\varepsilon=10^{-5}$，则
\begin{equation}
\nabla[-\log(q_b+\varepsilon)]
=\frac{q_b}{q_b+\varepsilon}\nabla[-\log q_b].
\end{equation}
小$q_b$的梯度被衰减。对单位归一化向量，$q_b\le e^{-1/\tau}$，$B=1$时NA为常数；缩小批量会改变目标，而非仅节省显存。这些性质尚不是退化的因果证据。

'''
        tex=tex.replace(r'\subsection{复现障碍与结论的有效范围}',note+r'\subsection{复现障碍与结论的有效范围}')
    concise=[
        ('缺失 sigmoid、无效 seed 参数、',
         r'工程改进包括入口修正、显式种子、验证选模及保存权重，提高可审计性，未宣称预测创新。Grocery的26个物品缺少训练正反馈；纯ID嵌入不能解决冷启动。结论仍受单搜索种子、窄网格、有限测试用户及后验设计限制。'),
        (r'我们利用已保存的逐用户成绩作\textbf{事后描述分析}',
         r'按全体训练用户交互数的经验三分位分组，仅用train定阈值，并列历史不拆分。使用已保存成绩，不重新选模；原文未给精确阈值且数据不同，属于事后描述扩展。'),
    ]
    for prefix,replacement in concise:
        tex=re.sub(r'(?m)^'+re.escape(prefix)+r'[^\n]*$',lambda m:replacement,tex)
    compact=[
        ('隐式反馈只表明用户发生过交互',
         r'隐式反馈中未交互不等于不喜欢。BPR以内积排序\cite{bpr}，LightGCN利用交互图传播\cite{lightgcn}。LightCCF以NA损失的梯度传递邻域信息，研究能否降低显式图卷积依赖\cite{lightccf}，在Douban-book、Tmall和Amazon-book报告优势。'),
        ('Grocery 使用 ReChorus 内置划分；',
         r'Grocery沿用框架划分，按现有顺序留末两条，同日并列不保证细粒度时序。MovieLens来自官方notebook的TopK分支\cite{movielens}：评分$\ge4$、5-core、固定UTC+8，按日历跨度80\%/90\%切点划分，移除dev/test中训练未见ID；不是按交互数8:1:1。图仅由train构建，数据哈希归档。'),
        ('同用户其他正交互可能进入NA分母。',
         r'批量2048时，Grocery/MovieLens锚点遇同用户交互的概率为24.45\%/45.16\%，期望0.35/0.74条，不能据此断言损害。干预屏蔽NA分母同用户非对角列，保留对角和平滑；对照逐行均匀随机移除等数量非对角列，用私有随机流隔离初始化、负采样与批次次序。固定$\alpha=1,\tau=0.28$、2023--2025种子、1000轮上限，各自验证选轮；原NA复用正常早停记录。变体单独实现并手算验收。'),
    ]
    for prefix,replacement in compact:
        tex=re.sub(r'(?m)^'+re.escape(prefix)+r'[^\n]*$',lambda m:replacement,tex)
    tex=tex.replace('固定配置按训练历史分组；括号为测试用户数，误差线为种子标准差。MovieLens稀疏组仅14人，误差线不含用户抽样波动。',
                    '训练历史分组，括号为测试用户数，误差线为三种子标准差，不含用户波动；MovieLens稀疏组仅14人。')
    tex=tex.replace('同量候选选择后的三个新种子结果，误差线为训练种子样本标准差。候选数相同，但模型每轮计算量和网格覆盖不同。',
                    '四候选选择后的测试NDCG@20；误差线为三种子标准差。相同候选数不等于相同计算预算。')
    tex=tex.replace('具体NA干预的三种子测试结果，误差线为样本标准差。随机对照每行移除数量相同，可能偶然移除同用户列；并非验证选参后的配置。',
                    r'固定$\alpha=1$的NA干预，误差线为三种子标准差。随机对照匹配数量，可能移除同用户列。')
    tex=tex.replace('核验采用 CPU float64，训练采用 GPU float32，支持公式和自动微分对应，不能证明跨硬件训练轨迹逐位相同。',
                    '核验用CPU float64，训练用GPU float32，不代表跨硬件逐位一致。')
    tex=tex.replace('200轮上限、固定配置的课程主实验测试结果；均值$\\pm$样本标准差，最佳轮按种子2023/2024/2025排列。@10结果随数值汇总归档。',
                    '固定配置、200轮上限；测试均值$\\pm$样本标准差，最佳轮按2023/2024/2025排列。')
    tex=tex.replace('NDCG@20配对差；区间为固定三次训练下的用户重采样95\\%区间，未包含训练与时间窗口变化。',
                    'NDCG@20配对差：均值$\\pm$种子标准差及条件用户重采样95\\%区间。')
    tex=tex.replace('Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme.','Steffen Rendle et al.')
    tex=tex.replace('Xiangnan He, Kuan Deng, Xiang Wang, et al.','Xiangnan He et al.')
    tex=tex.replace('Yu Zhang, Yiwen Zhang, Yi Zhang, Lei Sang, Yun Yang.','Yu Zhang et al.')
    tex=tex.replace('Jiayu Li, Hanyu Li, Zhiyu He, et al.','Jiayu Li et al.')
    tex=tex.replace('屏蔽变体单独成类，保留原BPR与正则。','')
    tex=tex.replace('原文覆盖范围见第7.3节。','')
    tex=tex.replace(r'\setlength{\itemsep}{3pt}',r'\setlength{\itemsep}{1pt}')
    material=(r'代码仓库：\url{https://github.com/liutq25/lightccf-rechorus-reproduction}。'
              f'提交包含单文件LaTeX与PDF、{formal}次正式和{validation}次仅验证运行的配置、日志、逐用户成绩及SHA256；数据由固定版本脚本获取。')
    tex=re.sub(r'(?m)^代码仓库：[^\n]*$',lambda m:material,tex)
    # Additional interventions are audited separately from the original 17 blocks.
    (ROOT/'analysis/followup_latex_blocks.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf-8')
    destination.write_text(tex,encoding='utf-8')
    last_generated.write_text(json.dumps({'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    body=tex.split('% MAIN BODY START')[1].split('% MAIN BODY END')[0]
    prose=re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}','',body,flags=re.S)
    count=lambda s:len(re.findall('[\u4e00-\u9fff]',s))
    print(dict(body_prose=count(prose),body_including_tables_and_figures=count(body),figures=tex.count(r'\begin{figure}')))


if __name__=='__main__':
    main()
