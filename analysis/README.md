# 结果文件说明

- `runs.csv`：24 次正式课程运行，每行一个配置/种子，包含验证和测试的 @10/@20 指标、最佳轮数、实际轮数、训练时间与 CUDA 已分配显存峰值。
- `summary.csv`、`summary.json`：按模型/数据集汇总三个种子的均值和样本标准差；`hit_epoch_cap` 记录到达 200 轮上限的次数。
- `paired_deltas.csv`：以相同种子配对计算 LightCCF 减去各对照的测试差值；不据此宣称总体统计显著性。
- `dataset_profile.json`：训练历史长度、测试用户覆盖及训练未见物品的描述统计。
- `batch_collisions.json`：随机无放回 2048 条交互批次中，锚点遇到同用户其他正交互的概率与期望；这是数据诊断，不是性能变化的因果证明。
- `official_douban_summary.json`：固定种子2023–2027的作者协议结果、与论文表5的差值、每次最佳轮数与数据哈希；该实验使用测试集选轮。
- 本地生成的 PNG 图由 `scripts/summarize_results.py` 重建，用于课程报告。

课程结果统一使用验证 NDCG@20 选择 checkpoint，再评价测试集；图中及汇总表中的误差为训练种子间的样本标准差。原作者 Douban-book 核对沿用测试 Recall@20 选轮，单独保存，不能与课程结果合并。

数据 SHA256 和固定版本见 `research/rechorus_data_manifest.json`，实验方案见项目根目录 `实验方案锁定.md`。
