# ReChorus 官方框架、数据和评估审计

审计日期：2026-10-16；本文件记录代码阅读和数据检查，不包含任何训练结果。

## 1. 固定的框架来源

- 官方仓库：<https://github.com/THUwangcy/ReChorus>，默认分支 `master`，ReChorus 2.0。
- 本地位置：`framework/ReChorus`。
- 固定提交：`c164ec4303cc20ddcfbd1b57de366a481811d1e5`，提交时间 2026-02-11 12:35:22 +0800，标题 `numpy version minor fix`。
- 上游 MIT License 应随源代码保留；论文引用可使用 Li et al., *ReChorus2.0: A Modular and Task-Flexible Recommendation Library*, RecSys 2024, 454–464。
- 此轮未改动上游源码；只增加了数据处理与审计脚本、处理后的第二数据集和研究记录。
- 课程 PDF 第 4–5 页要求 ReChorus、至少两个框架提供的数据集、至少两个同类别算法；`general/BPRMF.py` 和 `general/LightGCN.py` 均属于一般协同过滤，满足这一项。

官方资料：

- [框架 README](https://github.com/THUwangcy/ReChorus/blob/c164ec4303cc20ddcfbd1b57de366a481811d1e5/README.md)
- [内置数据集说明](https://github.com/THUwangcy/ReChorus/blob/c164ec4303cc20ddcfbd1b57de366a481811d1e5/data/README.md)
- [官方实验设置与演示结果](https://github.com/THUwangcy/ReChorus/blob/c164ec4303cc20ddcfbd1b57de366a481811d1e5/docs/demo_scripts_results/README.md)

## 2. 已准备好的两个内置数据集

`data/README.md` 明确列出 Amazon Grocery、MovieLens-1M、MIND-Large 为内置数据。4GB 显存机器选择前两者即可，不必使用更大的 MIND。

| 数据集 | 用户 | 物品 | 训练行 | 验证行 / 用户 | 测试行 / 用户 | 全体交互密度 |
|---|---:|---:|---:|---:|---:|---:|
| Grocery_and_Gourmet_Food | 14,681 | 8,713 | 121,892 | 14,681 / 14,681 | 14,681 / 14,681 | 0.118245% |
| ML_1MTOPK | 6,032 | 3,125 | 568,761 | 2,562 / 214 | 2,874 / 226 | 3.046138% |

以上是实际文件统计。完整哈希与来源位于 `research/rechorus_data_manifest.json`。

### Grocery

直接使用固定上游提交附带的三个 TSV（扩展名 `.csv`），不重分样本、不抽小子集。上游 Amazon notebook 采用 5-core 数据，将用户最后一次交互作测试、倒数第二次作验证，其余训练。它按 `time,user_id` 稳定排序；时间同一天的交互可能出现相同时间戳，故应称为按现有顺序的留二划分，不能声称严格细粒度的未来预测。

有 26 个物品未出现在训练正交互中；训练正交互覆盖物品为 8,687。这些物品仍可能被均匀负采样抽到，因此不能声称其嵌入完全没有参数更新，但它们缺少训练正反馈支持。报告应保留并披露这一原始条件，不能未经说明剔除难样本。上游提供每个评估目标的 99 个负例，允许负例重复、排除用户所有阶段阳性，适用于复核上游 sampled ranking；主实验采用后文统一全候选评估，不使用这些预采样候选。

### MovieLens

原始包来自 [GroupLens 官方下载](https://files.grouplens.org/datasets/movielens/ml-1m.zip)，已保存 `data/raw/ml-1m.zip`，SHA256：

`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`。

处理脚本 `scripts/prepare_rechorus_data.py` 实现固定提交内 `data/MovieLens_1M/MovieLens-1M.ipynb` 的 TopK 分支：评分 >= 4 为阳性；按阳性做用户和物品 5-core 过滤；沿日历时间跨度 80%、90% 两个切点划分；去除验证和测试中训练未见的用户/物品；ID 从 1 起连续编号。运行命令：

```powershell
python scripts/prepare_rechorus_data.py
```

必须固定 UTC+8：原 notebook 使用依赖宿主机时区的 `datetime.fromtimestamp`。UTC+8 精确匹配 notebook 保存的 568761/2562/2874 行、6032 个训练用户、3125 个物品、214/226 个验证/测试用户；UTC 则产生 568731/2589/2877，存在不同划分。脚本显式 `Asia/Shanghai`，两切点为距首日第 831、935 天。过滤前验证/测试还分别有 89、90 行冷启动记录，按官方规则移除。

实现修正只涉及 notebook 的路径/未定义变量拼写错误、固定时区、跳过未使用 CTR/元数据分支、显式迭代至 5-core 收敛。对当前输入，5-core 一轮即收敛，成员与 notebook 完全一致。保留原始字符串 ID 的字典序编号、官方候选采样的 NumPy RandomState 与 seed=1/2，以使输入来源可追溯。生成的 `neg_items` 仍采用官方排除所有阳性的方式；它们仅为框架默认输入兼容，不用于无泄漏全候选主实验。

**数据局限：**ML 的验证/测试只覆盖长期活跃的小部分用户，且人均目标数不同；不能把“交互行数 2,874”当作“测试用户数 2,874”，也不能将其随机视为独立用户做显著性检验。用户层宏平均和交互层平均回答不同问题。

## 3. 上游训练机制与基线差别

`src/helpers/BaseReader.py` 读取 train/dev/test，`n_users`、`n_items` 由三个阶段 ID 最大值确定，训练点击集合仅来自 train；LightGCN 图只由 `train_clicked_set` 创建。已知 ID 空间本身是常见的传导式设定，必须避免将 held-out 边加入图或损失。

`src/models/BaseModel.py` 的 `GeneralModel.Dataset.actions_before_epoch` 每个 epoch 为每条训练阳性交互均匀采样负物品，仅拒绝训练阳性；未来验证/测试阳性可能被作为训练负例，这符合“训练过程不查看未来标签”，不应以排除未来标签来美化效果。

`GeneralModel.loss` 是多个负例 softmax 加权 BPR 变体；`num_neg=1` 时等价于单负例 BPR（含数值截断）。BPRMF 为内积矩阵分解，Embedding 初始化 N(0,0.01)；LightGCN 用训练二部图对称归一化、无自环，每层 sparse-mm，平均第 0 至 L 层表示，其直接 Parameter 初始化为 Xavier uniform。两者上游默认正则为 Adam weight_decay，作用于所有非 bias 参数；这与 LightCCF 官方实现“采样原始嵌入的平方范数 / batch_size”正则不等价。应保留并记录基线定义，分别在验证集调节正则，不能拿同一数字宣称正则强度相同。

上游 `BaseRunner.fit` 每 batch 打乱 `item_id` 的正负次序，再只还原 `prediction`。LightCCF 若在 `forward` 里按 `item_id[:,0]` 构造邻居聚合损失，会把随机负例当正例。新增模型/Runner 应显式传递 `positive_item_id`，或使用不重排正负语义的专用 Runner，且测试 batch 内正例身份。这一风险不是论文特性，而是接入错误。

上游 epoch 保存模型条件是 `max(history)==history[-1]`，平分时保存最后一个 epoch；日志 `best_epoch=history.index(max)` 却记录第一个最高点。建议统一“严格提高才保存”的条件，保存真实 `best_epoch` 到 JSON，明确 patience。默认 early_stop 的连续不升触发逻辑也不是常规固定 patience，应明确新 Runner 行为。

## 4. 全候选评估风险与建议协议

### 已确认的上游行为

1. `BaseModel.py:194–198` 的 test_all 候选是 `[target]+[1,...,n_items-1]`，目标有两份；预测矩阵第 0 列是目标，其余列刚好以 item ID 对应。
2. `BaseRunner.py:244–251` 将每个用户 `train_clicked_set | residual_clicked_set` 的列全部置负无穷。`residual_clicked_set` 同时含 dev 与 test。因此验证评分利用测试阳性来排除竞争物品；ML 每条测试样本还会排除同一用户其他测试阳性。这属于利用 held-out 标签调整候选集合，不能直接用于本文要求的无泄漏主实验。
3. 评估采用每行唯一目标的 HR/NDCG，再按行平均；ML 多目标用户会重复计权，不能把这个结果标作标准用户宏平均 Recall@K/NDCG@K。
4. `predict` 没有 `torch.no_grad`；LightGCN 每个评估 batch 重算图传播，BPR/LightGCN forward 还显式生成 B×I×d 的物品嵌入和重复用户嵌入，全候选可能浪费显存。
5. `evaluate_method` 以 `>=` 计算目标排名，同分时全体同分物品排在目标之前，是悲观同分策略。若自定义 topk 使用不同 tie-break，必须记录规则；对 POP 或冷物品尤其不能忽略。

### 推荐的共同主协议

- 保持两个内置数据集原始 train/dev/test 划分，所有模型使用同一 Reader、数据 SHA256、负采样逻辑、候选物品全集和评价器。
- 一个用户只评价一次；同一阶段所有阳性组成相关物品集合 `R_u`。验证候选排除 train 已知交互；测试候选排除 train+dev 已知交互。训练图、参数更新仍只使用 train。
- 测试候选不得利用其他 test 阳性来剔除竞争物品。验证候选也不读取 test 阳性；目标集合只用于命中判定，不用于候选掩码。
- 对 ML 使用真实的多阳性 `Recall@K=|TopK∩R_u|/|R_u|` 和二元相关性 NDCG，IDCG 为 `sum_{r=1}^{min(K,|R_u|)} 1/log2(r+1)`，最后按用户宏平均；Grocery 每用户一个目标，此时 Recall=HR。
- 明确额外给出 HR 时的定义：多阳性用户命中率 `1[|TopK∩R_u|>0]`，与 Recall 不同。课程正文可主报 Recall@20/NDCG@20，辅报 @10。
- 对 score 同分采用确定性 item-ID 次序或期望同分评价；固定并测试，不可随运行变化。
- 只有验证集主指标负责超参和早停，测试仅在配置选定后评估保存的最佳验证 checkpoint；使用固定 seed 列表如 2024/2025/2026。调参 seed 与最终 seed 的关系应写清楚，不以测试选择最佳种子。
- 也可把原框架 sampled HR/NDCG 作为独立诊断，复核 BPRMF 和 LightGCN 是否达到数量级；它与论文原始数据/划分和全候选主实验均不可直接数值对照。

**关于历史掩码选择：**测试屏蔽 train+dev 是时间上已知历史；若为了严格模拟未更新历史的静态训练模型选择只屏蔽 train，也可形成一致协议，但必须所有方法一致并在报告写清楚。对于本任务，更建议前一种。

### 最小有意义的评价器验收

主实现完成后，使用可手算的 2–3 用户/5–8 物品样例验收：train 阳性被屏蔽、dev 的掩码不受 test 标签影响、同一用户两个 test 阳性共同计入召回且均保留候选、重复目标去重、item0 不参与候选、同分顺序确定、批量不同不改变排名、冷物品处理一致、模型训练后 reload 保持分数。随后抽取少量真实用户以简单 Python 全排序复核向量化 evaluator；不必反复做与实现完全同构的测试。

## 5. 适合 RTX3050 4GB 的运行建议

- Windows 使用 `num_workers=0`，先确认 CUDA PyTorch 和设备正确；上游 requirements 混入 conda 的 `python/cudatoolkit` 以及标准库 pickle，不能盲目逐行 pip 安装。固定实际可用版本并记录。
- LightGCN 的邻接张量上游直接 `.cuda()` 且不是 buffer，CPU 测试/设备切换会失败；可小范围改为 register_buffer 或显式 device，保存修复 diff。
- 初始嵌入维度 64、一个训练负例、Adam；Grocery batch_size=1024/2048、ML batch_size=2048 作为显存探测起点。模型比较固定相同 batch，大 batch 会改变 LightCCF 的批内对比负例规模，应在报告中明确，不能仅称为无影响的加速。
- 训练前先跑极短 smoke test，检查每项损失有限、各参数梯度存在、loss/梯度与官方小样本实现一致、评估指标范围正确。记录峰值 CUDA memory_allocated 和每 epoch 时间后再估计完整预算。
- 评价使用 `inference_mode/no_grad`；每次评估先计算一次全部用户/物品表示，用户批 B×d 与物品矩阵 d×I 相乘，评分只需 B×I。Grocery B=512 时 float32 分数矩阵约 17.8 MB；ML 同批约 6.4 MB。不要显式展开 B×I×d。
- LightGCN 每训练 batch 重算完整图传播是上游机制。ML 有约 113.8 万有向边，batch=2048 时约 278 batch/epoch；不要沿用 256 造成约 2222 次图传播。先衡量实际时长，按验证趋势与预设 patience 运行，不以“凑齐论文表数字”为停止或延长依据。
- LightCCF 对比项产生 B×B 矩阵，B=2048 时单张 float32 16.8 MB；需要同时考虑中间张量与反传。若显存不足，优先采用精确分块 logsumexp，验证梯度等价；不能把 batch 减半而仍当成同一个超参配置。
- 建议最小完整矩阵为 2 数据集 × (BPRMF, LightGCN, LightCCF) × 3 种子；再加 LightCCF 的 λ=0 消融至少覆盖两数据集。若 LightCCF 使用 MF 编码器，λ=0 最接近同初始化/同正则的 MF 控制组，能分离损失贡献；若使用 GCN，则对照应明确。
- 调参用共同学习率候选 {1e-3,5e-4}，基线正则按各自语义小范围搜索；LightCCF τ 与 λ 以论文/官方配置为中心。先看 smoke/短诊断确认实现，再选定有限网格和停止规则。不要在测试集上寻找“更像论文”的参数。

## 6. 如何解释不一致

课程两内置数据集与 LightCCF 原文数据不一定相同，数值不应直接相等；报告应区分“原方法实现验证”“课程数据再实验”“论文原表复核”。先排查正例排列、损失与梯度、图只含训练边、输入划分/哈希、padding 0、训练负采样、候选掩码、多阳性指标、初始化/正则/批量、早停记录这些内部因素。发现实现错误则修复后重新跑受影响实验，不继续沿用旧结果。确认协议、环境和模型都正确后，即使方法没有超过基线也应如实报告；这可以反映稀疏度、数据时间偏移、批内假负例、超参数敏感性及计算成本上的局限，不能仅由一两个结果推出机制普遍失效。
