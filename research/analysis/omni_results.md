# Qwen3-Omni 六套 Benchmark 结果

推理队列已结束。成功 5,839 / 6,231 条，未完成 392 条（6.29%）。以下只统计完整成功音频，排除其余音频及其中已成功的窗口；不是完整测试集成绩。

| Benchmark | 成功 / 总数 | 排除 | 区间 F1 / 秒段 F1 | Clip 召回率 | Clip 阴性 FPR |
|---|---:|---:|---:|---:|---:|
| desed-public | 681 / 692 | 11 | 42.93% | 76.76% | 1.92% |
| realdesed-test | 1000 / 1007 | 7 | 24.00% | 64.69% | 11.79% |
| urbansed-test | 1993 / 2000 | 7 | 22.10% | 36.31% | 6.77% |
| tag-test | 441 / 483 | 42 | 66.67% | 95.32% | 不可测 |
| tacos-test | 1676 / 2000 | 324 | 42.78% | 93.52% | 不可测 |
| maestro-development | 48 / 49 | 1 | 62.09% | 89.53% | 30.90% |

## 统计口径

- 模型：本地 Qwen3-Omni-30B-A3B-Instruct；10 秒窗口、5 秒步长；temperature=0，seed=42。提示给出类别/原查询，不提供 GT；同查询重叠预测区间合并。
- 前五项为生成区间的 pooled intersection F1。DTC=GTC=0.7，TAG 为 0.5；TACOS 沿用当前近义 GT 合并。MAESTRO 为十一类、一秒段 micro F1，软 GT >0.5。
- Clip 召回率 = 检出的阳性音频–类别/查询对 / 阳性对总数，同一类多次事件合为一次，不检查起止时间。FPR = FP / 阴性对。
- TAG/TACOS 现有查询均有阳性 GT，故不能测 Clip 误报能力；高召回不等于能正确拒绝不存在的声音。
- 这些是 F1，不是 PSDS/mpAUC，不能和既有论文主表数值直接排名；若比较，应对 OVSED 在相同成功音频子集上计算同一指标。

## 运行与完整性

- 队列墙钟耗时约 29.92 小时，包含重试与计分等开销。
- 原自动计分遇到未完成音频时断言失败，原 metrics.json 保留；本次新增 metrics.complete-subset.json，明确保存覆盖范围与排除 ID。
- 失败中存在解析器拒绝裸 JSON 数组的情况：内容可能可恢复，不能归为声学识别错误。还有截断、无效时间等错误；尚未逐条恢复。
- 仅凭 Clip 与区间指标差距不能归因于纯时间定位：事件次数、漏检、误报与匹配要求也参与其中。

结果目录：/home/zhanghc/projects/ov-sed/benchmark/runs-extra/qwen3-omni-30b/
复算：~/venvs/finelap/bin/python scripts/run_omni_benchmark.py --score-complete-only
日志：log/20261023/omni_complete_subset_score.log

## Clip 准确率与二值 mAP

| Benchmark | Precision | 类别判断准确率 | 整条音频全对率 | 二值 mAP |
|---|---:|---:|---:|---:|
| desed-public | 88.41% | 94.65% | 63.00% | 70.78% |
| realdesed-test | 57.17% | 83.61% | 10.60% | 42.02% |
| urbansed-test | 77.04% | 71.32% | 9.73% | 53.09% |
| tag-test | 100.00% | 95.32% | 92.29% | 不报告 |
| tacos-test | 100.00% | 93.52% | 89.86% | 不报告 |
| maestro-development | 58.33% | 75.76% | 4.17% | 51.09% |

- 类别判断准确率 = (TP+TN)/(TP+TN+FP+FN)，统计音频–类别对；阴性类别多时可能偏高。
- 整条音频全对率：一条音频所有被评测类别/查询均判断正确才计正确；忽略时间和事件次数。TAG/TACOS 仅覆盖给定阳性查询，不代表没有额外事件误报。
- 二值 mAP：有生成事件记1、未生成记0，每类使用 sklearn average_precision_score 非插值AP再宏平均；同分统一作为并列处理，无阳性类不计宏平均。它只反映二值输出的粗粒度排序，不是模型原生连续分数mAP，也不是定位mAP。
- TAG/TACOS没有阴性查询，其形式AP会退化到1、无法反映漏检，因此不报告。Precision形式为100%也只是查询集没有阴性导致。
- AP定义参考：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

## 与五个 OVSED 模型比较：Clip mAP

| 模型 | DESED Public | RealDESED | UrbanSED | MAESTRO Dev |
|---|---:|---:|---:|---:|
| dasm | 87.62% | 70.22% | 73.85% | 84.27% |
| finelap | 88.33% | 72.02% | 90.45% | 80.65% |
| flexsed | 66.03% | 61.22% | 70.45% | 72.31% |
| mga-clap | 90.99% | 74.64% | 73.83% | 88.49% |
| openflam | 69.59% | 72.54% | 88.35% | 89.09% |
| qwen3-omni-30b | 70.78% | 42.02% | 53.09% | 51.09% |

- 四套数据均只使用 Omni 完整成功音频，分别为 desed-public: 681/692, realdesed-test: 1000/1007, urbansed-test: 1993/2000, maestro-development: 48/49。同音频、同类别、同 GT、同 AP 实现。
- OVSED 取各类逐帧最高分；MAESTRO 先一秒加权再取最高分。Omni 只有生成类别的0/1输出。两种分数表示的信息量不同，不能将差值完全解释为模型能力差距；该表不是统一输出条件的受控优劣结论。
- 这里是 Clip 级类别存在性 mAP，不检查起止时间，不是 PSDS、MAESTRO 原 mpAUC 或定位 mAP。AP 不需选固定阈值。
- TAG/TACOS 原查询均为阳性，无有效 Clip mAP 对比，不加入表中。
- 本次不为 OVSED 临时指定0.5或按测试GT选择最佳阈值，因此没有伪造对应的Precision/准确率比较；这些固定操作点指标需要另设验证集选阈值协议。
- 逐类AP、音频ID与分数来源：runs-extra/qwen3-omni-30b/clip-comparison.json。
复算：~/venvs/finelap/bin/python scripts/run_omni_benchmark.py --compare-clip
