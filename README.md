# LightCCF 的 ReChorus 课程复现

本项目复现 SIGIR 2025 论文 *Unveiling Contrastive Learning's Capability of Neighborhood Aggregation for Collaborative Filtering*，在 ReChorus 2.0 的两个数据集上，与 BPRMF 和 LightGCN 比较。当前累计 **71 次正式运行、25 次仅验证搜索**。短跑和软件验收不计入这些测试均值。

[论文](https://arxiv.org/abs/2504.10113v1) · [DOI](https://doi.org/10.1145/3726302.3730111) · [作者实现](https://github.com/ZzYUuuu/LightCCF) · [ReChorus](https://github.com/THUwangcy/ReChorus)

## 主要结果

以下使用每模型本轮四个验证候选、搜索种子2023，随后在2026/2027/2028三个新训练种子上确认。每个种子只用验证 NDCG@20 选轮，表中为用户宏平均测试成绩的均值±样本标准差。全部使用64维、batch2048、单负例、最多1000轮和25轮早停。

| 数据 | 模型 | 所选参数 | Recall@20 | NDCG@20 |
|---|---|---|---|---|
| Grocery | BPRMF | lr=0.001, l2=1e-06 | 0.05974±0.00035 | 0.02592±0.00024 |
| Grocery | LightGCN | 层数=3, l2=1e-07 | 0.06923±0.00215 | 0.02909±0.00085 |
| Grocery | LightCCF | MF, α=1, τ=0.28 | 0.07279±0.00290 | 0.03229±0.00092 |
| MovieLens 1M TopK | BPRMF | lr=0.0005, l2=1e-06 | 0.10633±0.00359 | 0.09413±0.00261 |
| MovieLens 1M TopK | LightGCN | 层数=3, l2=1e-06 | 0.11162±0.00232 | 0.09450±0.00131 |
| MovieLens 1M TopK | LightCCF | MF, α=0.1, τ=0.28 | 0.09836±0.01537 | 0.08200±0.00858 |

Grocery 上 LightCCF 相对 LightGCN 的 NDCG@20 为 +11.00%。MovieLens 上 LightCCF 相对 LightGCN 的 NDCG@20 为 -13.22%。

相同候选数仅指本轮；LightCCF此前接受过额外搜索，累计预算不同。单搜索种子、窄网格、MovieLens仅226个测试用户，以及已知主结果后的补充探索均限制结论。新训练种子不等于新测试集。保留原固定参数结果于 `analysis/summary.csv`，不将不同配置或预算复跑合并增加独立重复数。

## 同用户屏蔽与等数量随机对照

固定零层LightCCF的α=1、τ=0.28，分别屏蔽NA分母中同用户的非对角交互，或逐行随机屏蔽同样数量的非对角列。保留对角与原1e-5平滑。随机对照使用私有生成器，不改变初始化、训练负采样和batch次序。

| 数据 | 同用户屏蔽减随机屏蔽的 NDCG@20 | 条件用户重采样95%区间 |
|---|---|---|
| Grocery | -0.00021±0.00020 | [-0.00039, -0.00004] |
| MovieLens | -0.00148±0.00327 | [-0.00365, +0.00065] |

区间固定三个训练结果，按观察到的测试用户成对重采样10000次；不代表训练或人群总体不确定性。随机对照只匹配数量，未匹配相似度，不能把差值全部归因于语义假负例。完整单种子、配对差、范围和选参证据见 `analysis/followup_summary.json`。

## 原文数值核对

作者Douban-book协议五种子2023–2027：

| 指标 | 本机均值±样本标准差 | 原文表5 | 相对差 |
|---|---|---|---|
| Recall@10 | 0.13164±0.00051 | 0.1330 | -1.02% |
| NDCG@10 | 0.18372±0.00015 | 0.1848 | -0.59% |
| Recall@20 | 0.18532±0.00092 | 0.1877 | -1.27% |
| NDCG@20 | 0.18458±0.00050 | 0.1857 | -0.60% |

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
