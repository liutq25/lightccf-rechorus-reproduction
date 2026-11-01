# LightCCF 推荐模型课程复现

2026-11-01交付复核：全部51份结果（41正式、10仅验证）及17个LaTeX数值块重新核验，原始训练结果未改动。补充全候选评价器的独立手算验收，连同既有损失验收共16项通过。正文叙述3863个汉字，含正文图表文字4918个汉字；7幅图、11张表、12页。报告由单一源码重新编译，提交包逐文件校验，避免压缩包内PDF与目录版本不一致。

验收命令：`python -m unittest discover -s tests -v`；PDF环境安装`pypdf`后可运行`python scripts/verify_submission.py`，加`--zip 提交包路径`可同时核对报告一致性和全包SHA256清单。字数排除图表环境时仍保留自动生成的正文段落，具体计数方法见脚本。

## 最新 LaTeX 报告与图表核查

课程提交报告全文使用单文件 LaTeX，表格、参考文献和7幅 PGFPlots 矢量实验图的坐标全部内嵌。全部图的标题、轴名、分组和说明使用中文，标准模型名与指标缩写保留。源码包含小组个人信息，随课程提交包交付；公开仓库保留通用重建脚本与不含个人信息的数值证据。

补核 Douban 全部四项指标：R@10 **0.13164±0.00051**、N@10 **0.18372±0.00015**、R@20 **0.18532±0.00092**、N@20 **0.18458±0.00050**；相对原文表5分别为 **−1.02%、−0.59%、−1.27%、−0.60%**。四项均值及每次成绩均略低于原文，结论是“数值接近，有残差”。@10 读自作者八位小数日志；四列均取同一个最佳测试 R@20 轮。

新增训练历史分组分析（对应原文图2的研究思路）：Grocery 阈值4/7条，MovieLens 为35/94条，保留并列。MovieLens 稀疏组有2018名训练用户，仅14名测试用户，覆盖0.69%，不能外推为全体稀疏用户的普遍效果。分组只用训练计数，复用保存的成绩，不重新选模。

2026-10-18补充完成：Grocery三个模型各三种子的1000轮上限对照，九次均按原规则早停；LightCCF相对LightGCN的NDCG@20仍高10.88%。MovieLens完成12个不同配置的一维验证切片（其中10个新增点只计算验证成绩），选中α=0.1、τ=0.28后作三种子测试确认。确认结果比原α=1改善23.05%，但仍比α=0低9.66%、比LightGCN低14.91%。新增结果没有被包装为算法改进。

总计41次正式运行，另有10次仅验证搜索。覆盖审计见 research/report_verification_zh.md，补充方案见 research/extension_plan_20261017.md。原始0–3层消融、Tmall/Amazon-book全套结果和对抗噪声实验仍未覆盖；MovieLens切片不是原三数据参数图的数值复刻。原文全部16基线×3数据×5次即至少240次基线训练，不作为本地课程项目的必需范围；对抗实验还需取得明确定义的攻击协议。

重建数值与图表：

    python scripts/summarize_results.py
    python scripts/summarize_official_anchors.py
    python scripts/analyze_sparsity_groups.py
    python scripts/summarize_extensions.py
    python scripts/build_latex_blocks.py

若有报告源码，可用下面命令更新数值标记块（不改正文）：

    python scripts/build_latex_blocks.py --report report.tex

编译需 XeLaTeX、ctex、pgfplots 及可用中文字体，由ctex按平台选择；本机使用宋体/黑体并嵌入Unicode映射。提供源码时，可运行 python scripts/compile_report.py --source report.tex 自动编译至引用稳定。独立PNG/PDF图使用 scripts/plot_style.py 加载真实中文字体；其他平台可用 LIGHTCCF_CJK_FONT 指定字体文件，缺少字体时明确报错。

## 项目概况

本项目复现 SIGIR 2025 论文 *Unveiling Contrastive Learning's Capability of Neighborhood Aggregation for Collaborative Filtering* 的 LightCCF 方法，并按课程要求在 ReChorus 2.0 中与 BPRMF、LightGCN 比较。课程数据为 ReChorus 的 `Grocery_and_Gourmet_Food` 和 `ML_1MTOPK`。此外保留作者原始代码与 Douban-book 数据，用于独立的论文原始协议核对。

公开代码仓库不重新分发上游框架、论文作者代码或 MovieLens 原始数据。首次使用时运行下文的固定版本设置脚本获取它们；本地课程提交目录保留已运行数据与原始日志。

论文：[arXiv 2504.10113](https://arxiv.org/abs/2504.10113) · [DOI 10.1145/3726302.3730111](https://doi.org/10.1145/3726302.3730111) · [作者代码](https://github.com/ZzYUuuu/LightCCF) · [ReChorus](https://github.com/THUwangcy/ReChorus)

## 已完成的实测结果

核心比较18次、NA消融6次，共24次课程主实验；随机种子为2023/2024/2025。下表固定200轮上限，为用户宏平均测试NDCG@20的均值±样本标准差；后续预算与选参结果另列。

| 数据集 | BPRMF | LightGCN | LightCCF (MF) |
|---|---|---|---|
| Amazon Grocery | 0.02532±0.00078 | 0.02930±0.00045 | 0.03274±0.00067 |
| MovieLens 1M TopK | 0.09148±0.00312 | 0.09635±0.00173 | 0.06663±0.00255 |

LightCCF 相对 LightGCN 的 NDCG@20 在 Grocery 上提高约 11.71%，MovieLens 上下降约 30.85%，三颗种子各自方向一致。关闭 NA 后，Grocery 为 0.02347±0.00033，MovieLens 为 0.09075±0.00375；加入 NA 的配对差值分别为 +0.00926±0.00098 和 −0.02412±0.00382。该结果表明固定的 Douban 参数不能在两个课程数据集上稳定占优；本项目没有用测试集调参。

完整 @10/@20 指标、最佳轮数、训练时间与显存见 `analysis/runs.csv`、`analysis/summary.csv`；每种子配对差值见 `analysis/paired_deltas.csv`。原Grocery LightGCN三次均达到200轮上限，因此已将三个模型统一延长至1000轮上限，结果如下；仍不能解释成无限预算下的排序。课程数据与原论文数据不同，绝对分数不作为原表数值复现误差。

| Grocery预算条件 | BPRMF | LightGCN | LightCCF |
|---|---|---|---|
| 200轮上限 | 0.02532±0.00078 | 0.02930±0.00045 | 0.03274±0.00067 |
| 1000轮上限 | 0.02560±0.00122 | 0.02952±0.00059 | 0.03274±0.00067 |

三个模型分别在60–250、230–250、185–210轮早停，选模、负采样、损失和其余配置与原实验一致。相同种子公共训练前缀的验证指标完全一致，损失差最大4.51e-9；长预算是配对复跑，不能把种子数算作六个。

MovieLens搜索为α的7点和τ的6点，共享原配置，共12个不同配置；新增10点的test均为null。按种子2023验证NDCG@20选中的α=0.1、τ=0.28，在三个种子上得到NDCG@20 **0.08198±0.00340**。该结果低于α=0的0.09075±0.00375及固定LightGCN的0.09635±0.00173。搜索发生在已知主实验结果之后，且未给基线相同搜索预算，只能视作补充探索；不能声称盲测或等量调参后的性能排名。完整四指标、选择记录和预算前缀审计见analysis/extension_summary.json。

作者原始 Douban-book 协议的五个种子（2023–2027）也已全部完成：Recall@20 为 **0.18532±0.00092**、NDCG@20 为 **0.18458±0.00050**；相对论文表5的 0.1877/0.1857，分别低约 1.27%/0.60%。最佳轮数为 43/43/45/43/43，均在54–56轮早停。该核对使用作者的测试集选轮，须与课程验证集选轮结果分开解释；全部数值和数据哈希见 `analysis/official_douban_summary.json`。原文其他数据集及全部基线/消融未重跑。

## 目录

- `framework/ReChorus/`：固定在提交 `c164ec4303cc20ddcfbd1b57de366a481811d1e5` 的课程框架；新增 `src/models/general/LightCCF.py` 与 `src/helpers/ReproductionRunner.py`。
- `extensions/`：上述两个新增文件的可发布副本，由 `scripts/setup_upstream.py` 安装到框架目录。
- `vendor/LightCCF-main/`：作者代码与原始论文数据快照，不能将其输出与课程实验混为一谈。
- `scripts/`：数据准备、数据审计、单次实验、实验矩阵、结果汇总、作者代码锚点运行。
- `configs/`：训练前锁定的配置。`*_alpha0.json` 是邻域聚合损失消融。
- `configs/extension/`：补充预算、验证敏感性与所选配置确认；方案在补充训练前锁定。
- `research/`：论文、原始表 5、来源哈希、实现差异、课程要求与数据审计。
- `runs/`：每次实验的配置、日志、逐轮指标、最佳模型与用户级结果。`pilot/` 仅用于运行检查，不计入正式均值。
- `analysis/`：正式结果的汇总 CSV、配对差值和图表；由脚本生成。

## 环境与运行

已运行环境为 Windows、Python 3.11.15、PyTorch 2.5.1+cu121、NumPy 2.4.6、NVIDIA RTX 3050 4GB。依赖 ReChorus 源码所用的 `pandas`、`scipy`、`scikit-learn`、`tqdm`、`matplotlib` 等 Python 包。现有 LightGCN 上游编码器直接调用 `.cuda()`，矩阵实验需要可用的 NVIDIA CUDA 环境。

先安装与本机驱动匹配的 CUDA 版 PyTorch，再运行 `python -m pip install -r requirements.txt`。运行时可用 `python -c "import torch; print(torch.cuda.is_available())"` 确认所选解释器能看到 GPU。

在项目根目录运行以下命令（把 `python` 替换为可用的 CUDA Python 解释器）：

```powershell
python scripts/setup_upstream.py
python scripts/audit_rechorus_data.py
python scripts/profile_datasets.py
python scripts/analyze_batch_collisions.py
python scripts/run_matrix.py --group core --seeds 2023 2024 2025
python scripts/run_matrix.py --group ablation --seeds 2023 2024 2025
python scripts/summarize_results.py
```

MovieLens 处理后的 `ML_1MTOPK` 如需重新生成，应将 GroupLens 官方 `ml-1m.zip` 放于 `data/raw/`，再运行 `python scripts/prepare_rechorus_data.py`。处理脚本固定 UTC+8，以匹配框架原 notebook 保存的划分规模。数据文件的 SHA256、计数和排除规则见 `research/rechorus_data_manifest.json`。

论文原始 Douban-book 数据的五种子核对单独运行：

```powershell
python scripts/setup_upstream.py --with-author
python scripts/run_anchor_matrix.py
python scripts/summarize_official_anchors.py
```

完成原课程矩阵后运行两项补充检验：

```powershell
python scripts/run_extension_matrix.py
python scripts/summarize_extensions.py
python scripts/build_latex_blocks.py
```

扩展矩阵串行使用GPU；已完成且状态匹配的运行会跳过，不覆盖原始课程结果。参数搜索使用验证模式，选择规则及候选来源保存在runs/extension/parameter_selection.json。原始日志较大且含本机路径，随本地提交包交付；公开仓库提供汇总数据和重跑脚本。

该核对固定种子 2023–2027，沿用作者代码用测试集 Recall@20 选轮的机制，因此其数字不能与课程矩阵中验证集选轮的结果合并平均。作者代码自身的 `--seed` 参数未在入口中实际设置种子；包装脚本在模型创建之前显式设置 Python、NumPy 和 PyTorch 种子。原文未公开五个种子；本机仅核对 Douban-book，未覆盖 Tmall/Amazon-book 的整套原表。
固定的作者提交另有一处入口调用笔误：`load_models()` 无参数，却以 `load_models(args)` 调用。包装脚本只在内存中修正该调用，不修改作者源码；结果 JSON 会记录修正与原文件哈希。

## 实验口径

课程主实验使用同一数据划分、64维、batch2048、一个训练负例、学习率0.001、最多200轮，每5轮用验证NDCG@20选模型；补充Grocery对照只将轮数上限改为1000。最终在测试集做用户级全物品排序。验证屏蔽train历史，测试屏蔽train+dev历史。三个种子给出均值及样本标准差。课程数据与论文三套原始数据不同，不拿绝对分数做原表“复现误差”。详细方案见 `实验方案锁定.md` 和 research/extension_plan_20261017.md。

## 复核要点

`research/loss_validation_results.json` 记录了本项目 NA/BPR/正则损失与作者代码在固定小批次上的值及嵌入梯度核对。`research/rechorus_audit.md` 解释了框架默认 `test_all` 的未来标签屏蔽问题，以及本项目的全候选评价器为何单独实现。每次正式运行只保留验证集选择的最佳 checkpoint；原始日志和配置保存在对应 `runs/` 目录。
