"""Write concise documentation only after the complete study and report audit."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def metric(value):
    return f"{value['mean']:.5f}±{value['sample_sd']:.5f}"


def main():
    study=read('analysis/followup_summary.json')
    audit=read('analysis/final_delivery_audit.json')
    visual=read('analysis/report_stats.json')
    assert audit['formal_runs_total']==study['total_formal']
    assert audit['pdf_sha256']==visual['pdf_sha256']
    assert visual['visual_review_pages']==list(range(1,audit['pages']+1))
    reference=read('analysis/official_douban_summary.json')
    rows=[]
    for prefix,title in (('grocery','Grocery'),('ml','MovieLens 1M TopK')):
        for model in ('BPRMF','LightGCN','LightCCF'):
            result=study['confirmation'][prefix+'_'+model]
            config=result['config']
            parameters=(f"lr={config['lr']:g}, l2={config['l2']:g}" if model=='BPRMF' else
                        f"层数={config['n_layers']}, l2={config['l2']:g}" if model=='LightGCN' else
                        f"MF, α={config['ssl_weight']:g}, τ={config['tau']:g}")
            rows.append(f"| {title} | {model} | {parameters} | {metric(result['test']['Recall@20'])} | {metric(result['test']['NDCG@20'])} |")
    comparisons=[]
    controls=[]
    for prefix,title in (('grocery','Grocery'),('ml','MovieLens')):
        comparison=study['paired_confirmations'][prefix]
        comparisons.append(f"{title} 上 LightCCF 相对 LightGCN 的 NDCG@20 为 {comparison['relative_ndcg20_percent']:+.2f}%。")
        control=study['masking'][prefix]['same_minus_random']
        interval=control['conditional_user_bootstrap_95ci']
        controls.append(f"| {title} | {metric(control)} | [{interval[0]:+.5f}, {interval[1]:+.5f}] |")
    intro=f'''# LightCCF 的 ReChorus 课程复现

本项目复现 SIGIR 2025 论文 *Unveiling Contrastive Learning's Capability of Neighborhood Aggregation for Collaborative Filtering*，在 ReChorus 2.0 的两个数据集上，与 BPRMF 和 LightGCN 比较。当前累计 **{study['total_formal']} 次正式运行、{study['total_validation_only']} 次仅验证搜索**。短跑和软件验收不计入这些测试均值。

[论文](https://arxiv.org/abs/2504.10113v1) · [DOI](https://doi.org/10.1145/3726302.3730111) · [作者实现](https://github.com/ZzYUuuu/LightCCF) · [ReChorus](https://github.com/THUwangcy/ReChorus)

## 主要结果

以下使用每模型本轮四个验证候选、搜索种子2023，随后在2026/2027/2028三个新训练种子上确认。每个种子只用验证 NDCG@20 选轮，表中为用户宏平均测试成绩的均值±样本标准差。全部使用64维、batch2048、单负例、最多1000轮和25轮早停。

| 数据 | 模型 | 所选参数 | Recall@20 | NDCG@20 |
|---|---|---|---|---|
'''+ '\n'.join(rows)+'\n\n'+''.join(comparisons)+'''

相同候选数仅指本轮；LightCCF此前接受过额外搜索，累计预算不同。单搜索种子、窄网格、MovieLens仅226个测试用户，以及已知主结果后的补充探索均限制结论。新训练种子不等于新测试集。保留原固定参数结果于 `analysis/summary.csv`，不将不同配置或预算复跑合并增加独立重复数。

## 同用户屏蔽与等数量随机对照

固定零层LightCCF的α=1、τ=0.28，分别屏蔽NA分母中同用户的非对角交互，或逐行随机屏蔽同样数量的非对角列。保留对角与原1e-5平滑。随机对照使用私有生成器，不改变初始化、训练负采样和batch次序。

| 数据 | 同用户屏蔽减随机屏蔽的 NDCG@20 | 条件用户重采样95%区间 |
|---|---|---|
'''+ '\n'.join(controls)+'''

区间固定三个训练结果，按观察到的测试用户成对重采样10000次；不代表训练或人群总体不确定性。随机对照只匹配数量，未匹配相似度，不能把差值全部归因于语义假负例。完整单种子、配对差、范围和选参证据见 `analysis/followup_summary.json`。

## 原文数值核对

作者Douban-book协议五种子2023–2027：

| 指标 | 本机均值±样本标准差 | 原文表5 | 相对差 |
|---|---|---|---|
'''
    for name in ('Recall@10','NDCG@10','Recall@20','NDCG@20'):
        value=reference['metrics'][name]
        intro+=f"| {name} | {metric(value)} | {value['paper']:.4f} | {value['relative_difference_percent']:+.2f}% |\n"
    intro+='''
这些数值接近但仍有残差，未宣布严格等价。作者代码按测试Recall@20选轮，本核对保留该机制并单独报告；课程实验使用独立验证集。原文未报告种子列表和逐次方差，无法给出两组等价检验。原文件规模792062交互与表4舍入值一致，无重复或跨集重叠；191个测试用户没有训练交互，沿用原代码保留。未验证原文Tmall/Amazon-book整表、全部16基线、LightCCF0–3层及对抗噪声实验；具体覆盖见报告和研究记录。

## 环境、固定版本与数据

- 实测：Windows、Python3.11.15、PyTorch2.5.1+cu121、RTX3050 Laptop4GB。完整版本见 `research/followup_environment.json` 和 `requirements.txt`。
- ReChorus：`c164ec4303cc20ddcfbd1b57de366a481811d1e5`；作者代码：`45645ef15fc08e858ffc81e1b2530b4b1805ad34`。
- Grocery直接来自框架；MovieLens由官方TopK notebook流程重建，固定UTC+8，评分≥4、5-core及日历时间切分。验证/测试移除训练未见ID。序列化显式CRLF，跨系统换行不会改变已核验的输入字节。
- 验证候选只屏蔽train，测试屏蔽train+dev；不利用当前或未来阶段正例剔除竞争物品。训练负采样只拒绝train阳性；ID0排除，同分按物品ID。
- 模型移植见 `extensions/LightCCF.py`；实验变体单独位于 `extensions/LightCCFMasked.py`；共同评价器为 `extensions/ReproductionRunner.py`。

命令中的python须指向CUDA环境，上游LightGCN构造器需要NVIDIA CUDA。先安装合适的CUDA版PyTorch，再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
python scripts/setup_upstream.py --with-author
python -m unittest discover -s tests -v
```

## 完整重跑与恢复

先重建原阶段记录，再运行补强矩阵。各矩阵跳过已完成项，遇到未完成目录会要求核查，保护已保存结果。

```powershell
python scripts/audit_rechorus_data.py
python scripts/profile_datasets.py
python scripts/analyze_batch_collisions.py
python scripts/run_matrix.py --group core --seeds 2023 2024 2025
python scripts/run_matrix.py --group ablation --seeds 2023 2024 2025
python scripts/run_anchor_matrix.py
python scripts/summarize_results.py
python scripts/summarize_official_anchors.py
python scripts/analyze_sparsity_groups.py
python scripts/run_extension_matrix.py
python scripts/summarize_extensions.py
python scripts/build_latex_blocks.py
python scripts/reproduce_followup.py
```

补强入口先锁定本工作目录自己的51份前阶段记录及源码，再训练和汇总；不要求另一次重跑的日志具有本机相同的时间/路径字节。默认串行使用GPU。当前研究曾短暂尝试共享GPU，发现争用后恢复串行，见 `research/followup_scheduling_amendment.md`；新时长不作方法速度证据，报告效率表取原串行主实验。

监控进度：`python scripts/followup_status.py`。方案、选择规则及全部来源见 `research/followup_plan_20261101.md`、`runs/followup_20261101/validation_selection.json`。

## 报告、复核与提交包

LaTeX参考文献、表格和矢量坐标全部内嵌。报告源码随本地提交包交付；提供源码时可更新指定数值区：

```powershell
python scripts/build_latex_blocks.py --report 复现报告.tex
python scripts/update_followup_report.py --source 复现报告.tex --base 复现报告.tex
python scripts/compile_report.py --source 复现报告.tex
```

更新数值后仍需检查正文解释是否符合重跑结果。编译需XeLaTeX、ctex、pgfplots与中文字体。独立科学图使用真实CJK字体；其他平台可用LIGHTCCF_CJK_FONT指定字体路径。

完整交付工作目录可安装pypdf后运行 `python scripts/verify_submission.py`，核对数据、日志、逐用户宏均值、模型哈希、数值区与报告。从公开仓库重建审计所需的11份论文/作者快照，可运行 `python scripts/fetch_audit_sources.py`；该命令逐份验证固定哈希，拒绝覆盖不一致的文件。完整审计还需要重跑得到的权重和本地报告。仅校验压缩包完整性可用普通Python，无需CUDA或PDF依赖：

```powershell
python scripts/verify_submission.py --zip 完整提交包路径.zip --zip-only
```

该命令检查所有条目都被MANIFEST_SHA256覆盖、逐文件哈希、压缩CRC和重复条目。原始数据通过固定版本脚本获取；本地提交包提供报告、代码、分析、原始运行证据与校验清单。
'''
    (ROOT/'README.md').write_text(intro,encoding='utf-8')
    note=f'''# LaTeX报告与交付说明

正式报告为单文件复现报告.tex及同源PDF，{audit['pages']}页、{audit['figures']}幅实验图、{audit['tables']}张表。引用、表格与全部矢量坐标内嵌，可独立编译。使用XeLaTeX连续编译至引用稳定；环境要求和命令见code/README.md。

字数核查覆盖全文，包括摘要、图表与参考文献：去掉重复页眉后的全文可见汉字为{audit['full_visible_chinese_characters_excluding_repeated_headers']}，正文叙述汉字为{audit['body_prose_chinese_characters']}。英数混合标记另有保守估计{audit['full_visible_mixed_token_estimate']}；它由正则表达式统计，不等同于Word字数，具体口径见analysis/final_delivery_audit.json。

累计{study['total_formal']}次正式运行、{study['total_validation_only']}次仅验证搜索，23项模型/梯度/评估器/掩码验收通过。原51份记录保持不变。新增同量候选、新种子确认及随机匹配屏蔽对照，结果与限制均在报告中；未覆盖的原文实验已明确列出。

本次编译及{audit['pages']}页渲染完成后逐页检查；数值区、仓库链接和SHA256核验通过。小组分工仍为建议稿，提交前按实际参与情况调整。

提交包：根目录报告与本说明；code/可复现代码与文档；analysis/汇总及图表；evidence/runs/逐轮和逐用户证据；MANIFEST_SHA256.txt逐文件校验。公开仓库：https://github.com/liutq25/lightccf-rechorus-reproduction。
'''
    (ROOT/'LaTeX报告说明.md').write_text(note,encoding='utf-8')
    verification=f'''# 最终报告与实证复核

## 运行和代码

当前{study['total_formal']}正式、{study['total_validation_only']}验证模式结果均complete；原51份结果哈希不变。本轮新增30正式（18个所选配置新种子确认、12个屏蔽对照）和15验证搜索；24个候选有9个复用已有正常早停验证记录。选择只使用dev NDCG@20，精确并列按配置ID；确认种子2026/2027/2028。各模型本轮候选数相同，累计探索预算仍不同。

固定ReChorus提交及跟踪源码未改；训练源码锁定、输入六份课程划分、作者两份原始数据、11份论文与作者来源文件均复核。23项验收详见research/followup_acceptance.json；原10项损失验收最大误差8.88e-16，模型文件仍匹配原记录。

同用户屏蔽保留对角，随机对照逐行匹配数量、独立随机流。只匹配数量没有匹配相似度；差值不是语义假负例的完整因果证明。共享GPU尝试降低吞吐，记录后恢复串行；新的混合时长不参与方法效率比较。

## 数值与统计

''' + '\n'.join(comparisons)+'''

报告保留原固定参数、作者测试选轮和新验证选轮三种口径。原文Douban四指标仍有残差；原文种子与逐次方差未公开，未作等价宣称。测试有191个无训练交互用户，未擅自筛除。MovieLens仅226测试用户，dev/test重叠116；用户bootstrap只条件于固定三次训练和观察人群，不代表跨时间或全体用户。

## 原文覆盖索引

| 原文位置 | 本次证据与边界 |
|---|---|
| 表1、表2与图1 | 重建最终NA公式及梯度；未运行CL-SS/CL-UI中间目标，未把概念图当作实测图 |
| 表3、表7 | 推导批内二次复杂度，实测课程模型开销；硬件和计时口径不同，非原表复刻 |
| 表4 | 核对作者Douban文件规模、重复及重叠；课程两个数据另行归档 |
| 表5 | Douban的LightCCF一行、四指标、五次；其余数据和16基线全表未运行 |
| 表6 | 使用零层LightCCF；未完成LightCCF的0–3层对照，LightGCN基线不能替代该实验 |
| 图2 | 课程数据上的稀疏分组扩展，非原图同数据数值复刻 |
| 图3 | 未完成；缺少充分定义的对抗攻击协议，未用随机加边替代 |
| 图4 | 旧MovieLens单参数切片及本轮同量候选扩展，非原三数据联合网格 |

## 报告与交付

''' + f'''报告{audit['pages']}页，{audit['figures']}图、{audit['tables']}表；正文汉字{audit['body_prose_chinese_characters']}，含图表、摘要与参考文献的全文可见汉字{audit['full_visible_chinese_characters_excluding_repeated_headers']}。混合标记估计{audit['full_visible_mixed_token_estimate']}，不冒称Word软件统计。{audit['auto_blocks_verified']}个实际存在的数值区逐字匹配汇总。旧曲线与详表保留在analysis/evidence，正文压缩重复审计说明。独立结论、小组分工、代码地址及方法局限均覆盖。

PDF与LaTeX同源；仓库链接可点击；编译无缺字、未解析引用或越界；全部{audit['pages']}页渲染逐页检查。报告SHA256：{audit['pdf_sha256']}。提交包采用课程指定文件名，旧版本归档；逐文件清单用于完整性复核。小组实际分工待成员填写，不将建议分工当作事实。
'''
    (ROOT/'research/report_verification_zh.md').write_text(verification,encoding='utf-8')
    print('Updated README, LaTeX delivery instructions, and verification ledger from audited results.')


if __name__=='__main__':
    main()
