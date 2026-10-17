# 结果文件说明

## LaTeX 版本新增证据

- official_douban_runs.csv、official_douban_history.csv、paper_agreement.csv：五次作者协议的四指标、全部逐轮记录和原表误差。所有指标取同一个最佳测试 R@20 轮；@10 从八位小数日志读取，@20 从完整精度结果读取。
- learning_curves_seed2023.csv：两课程数据、三个模型、种子2023的逐评价点验证曲线。
- sparsity_groups.json、sparsity_group_runs.csv、sparsity_group_summary.csv：训练历史分组的逐种子/汇总成绩。分组只用 train，复用固定 checkpoint；不同于原文数据，非原图数值复刻。
- latex_blocks.json：从汇总生成的17个表格/矢量图/数值说明块，由 scripts/build_latex_blocks.py 生成。报告内嵌同一份数值和坐标，可单文件编译。
- 本地 PNG/PDF 图由结果汇总及分组脚本重建。正式课程报告全文使用 LaTeX，包含7幅 PGFPlots 矢量实验图。中文标签使用实际中文字体；Matplotlib图由scripts/plot_style.py设置字体，报告由ctex选择系统中文字体。

## 2026-10-18补充实验

- extension_summary.json：9次Grocery长预算与原9次运行的配对比较、公共训练前缀核对、3次MovieLens所选配置确认，以及相对原配置、α=0和LightGCN的差值。总正式运行数41；三种子仍为2023/2024/2025，配对复跑不增加独立种子数量。
- parameter_search.csv、parameter_search.json：12个不同配置的验证成绩及来源；其中10点是新增的仅验证搜索，test为null，原α=0/1两点复用已保存验证值。选择规则为最高种子2023验证NDCG@20，精确并列时选较小α、再较小τ。
- budget_learning_curves_seed2023.csv：三个模型1000轮上限的验证学习曲线。实际九次均正常早停，LightGCN为250/250/230轮。
- budget_extension.png/pdf、parameter_sensitivity.png/pdf：长预算对照与两条一维参数曲线。参数曲线只显示验证成绩，没有单种子误差线；权重轴按网格位置等间距显示。
- 主要发现：长预算LightCCF相对LightGCN的NDCG@20高10.88%；验证选中α=0.1/τ=0.28，其测试为0.08198±0.00340，仍低于α=0及LightGCN。方案与范围见research/extension_plan_20261017.md。

## 既有结果文件

- `runs.csv`：24 次正式课程运行，每行一个配置/种子，包含验证和测试的 @10/@20 指标、最佳轮数、实际轮数、训练时间与 CUDA 已分配显存峰值。
- `summary.csv`、`summary.json`：按模型/数据集汇总三个种子的均值和样本标准差；`hit_epoch_cap` 记录到达 200 轮上限的次数。
- `paired_deltas.csv`：以相同种子配对计算 LightCCF 减去各对照的测试差值；不据此宣称总体统计显著性。
- `dataset_profile.json`：训练历史长度、测试用户覆盖及训练未见物品的描述统计。
- `batch_collisions.json`：随机无放回 2048 条交互批次中，锚点遇到同用户其他正交互的概率与期望；这是数据诊断，不是性能变化的因果证明。
- `official_douban_summary.json`：固定种子2023–2027的作者协议结果、与论文表5的差值、每次最佳轮数与数据哈希；该实验使用测试集选轮。
- 主结果、学习曲线及 NA 消融的 PNG/PDF 由 `scripts/summarize_results.py` 重建；原文误差和分组图分别由作者协议汇总脚本与分组脚本重建。

课程结果统一使用验证 NDCG@20 选择 checkpoint，再评价测试集；图中及汇总表中的误差为训练种子间的样本标准差。原作者 Douban-book 核对沿用测试 Recall@20 选轮，单独保存，不能与课程结果合并。

历史字段peak_cuda_mb实际除以2**20，单位为MiB，报告已使用正确单位；字段名保留以兼容既有原始记录。

数据 SHA256 和固定版本见 `research/rechorus_data_manifest.json`，实验方案见项目根目录 `实验方案锁定.md`。
