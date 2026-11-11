# SED Benchmark 实验总结

更新日期：2026-10-19

## 1. 实验范围

本轮对 DASM、FineLAP、FlexSED、MGA-CLAP、OpenFLAM 五个模型，在 DESED、RealDESED、UrbanSED、TAG、TACOS、MAESTRO 六个 benchmark 上进行评测。使用各模型发布权重与既有推理配置，没有训练或微调。

共完成 40 项“模型 × 数据划分”的推理，其中 35 项已获得当前口径下的指标；其余 5 项对应缺少真值的 MAESTRO evaluation。目标是分析现有模型的能力与瓶颈，不宣称完整复现 COSED。

## 2. 总体结果

| 模型 | DESED Public | DESED Validation | RealDESED | UrbanSED | TAG | TACOS | MAESTRO Dev |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DASM | **0.4686** | **0.4194** | 0.1016 | 0.0800 | 0.6205 | 0.1133 | 0.5282 |
| FineLAP | 0.3238 | 0.3244 | 0.1784 | **0.4212** | 0.6479 | 0.1239 | 0.5173 |
| FlexSED | 0.1605 | 0.1453 | 0.0789 | 0.0795 | **0.6485** | 0.1268 | 0.4020 |
| MGA-CLAP | 0.2637 | 0.2807 | **0.1833** | 0.0807 | 0.5076 | **0.1326** | **0.5488** |
| OpenFLAM | 0.0241 | 0.0538 | 0.0643 | 0.2875 | 0.4098 | 0.0878 | 0.5386 |

前四列及 TACOS 为 **PSDS1**，TAG 为 **Grounding PSDS**，MAESTRO 为 **mpAUC**。均越高越好，但不同列不能直接横比或平均。加粗仅表示本列数值最高，不表示统计显著优势。

## 3. 数据范围与计分定义

| 数据划分 | 每个模型的音频数量 | 指标与说明 |
|:---:|:---:|:---|
| DESED Public | 692 | 标准 PSDS1；公开评测集 |
| DESED Validation | 1,068 / 1,168 | 标准 PSDS1；固定使用现有样本进行辅助诊断 |
| RealDESED Test | 1,007 | 标准 PSDS1 |
| UrbanSED Test | 2,000 | 标准 PSDS1 |
| TAG Test | 483 | 995 个文件—短语配对，Grounding PSDS |
| TACOS Test | 2,000 | 5,151 个文件—caption 配对，近义真值合并 PSDS1 |
| MAESTRO Development | 49 | 官方 11 类，1 秒片段 mpAUC |
| MAESTRO Evaluation | 26 | 仅完成推理，没有真值，不能计分 |

### PSDS1

DTC=0.7、GTC=0.7、alpha_CT=0、alpha_ST=1，最大 eFPR=100/小时。使用 `sed_scores_eval` 实现，不包含此前已删除的自定义 0.3/0.3 对照或 PSDS2。PSDS1 不是识别准确率。

### TAG

各文件—短语配对作为评测样本，合并为单一事件类别。DTC=GTC=0.5、alpha_ST=0、最大 eFPR=800/小时。使用本地 v2.2 清单；FineLAP 使用未滤波帧分数，其他模型保留既有分数后处理。不能将此分数与 PSDS1 直接比较。FineLAP 旧三帧滤波结果为 0.6601616055，保留在 legacy 配置；未滤波结果为 0.6479156424，只确认修正了这一处后处理，不代表完整官方复现。

### TACOS

全部文件—caption 配对作为单一事件类别池化评分。主表采用 MiniLM-L6-v2、同一文件内余弦相似度 >0.8 的直接邻居真值合并；不做传递闭包，查询和预测不变。原始 exact-caption 协议也保留，两种配置均可通过统一入口复算。编码器及其他实现尚未获作者确认，不能宣称与 COSED 完全等价。

### MAESTRO

使用全部 49 条 development 录音，而非官方挑战的特定 validation 划分或隐藏 evaluation。选择官方 11 个评测类别，将软标签严格大于 0.5 的区间视为正标签；预测分数按帧与一秒片段的重叠时长加权平均。计算 FPR≤0.1 的阶梯 ROC 曲线面积，除以 0.1 后对类别取平均。共 11,418 个片段，包含末尾不足一秒的片段。

已使用独立逐秒真值与 sklearn ROC 核对每类曲线，面积按 `sed_scores_eval` 规则计算；不是直接使用 sklearn 默认的标准化部分 AUC。

## 4. 主要发现

1. **没有一个模型在所有任务上领先。** DASM 在 DESED 两个划分最高；FineLAP 在 UrbanSED 最高；FlexSED 的 TAG 数值略高于修正后的 FineLAP（0.6485 vs 0.6479），差距很小；MGA-CLAP 在 RealDESED、近义合并 TACOS 与 MAESTRO development 最高。旧 exact-caption TACOS 下 FineLAP 最高，仅适用于旧口径。
2. **模型表现具有明显的任务差异。** FlexSED 的 TAG 分数接近 FineLAP，但在固定类别任务上较弱；OpenFLAM 在 UrbanSED、MAESTRO 相对有竞争力，在 DESED 较弱。不能仅凭这种差异归因于某一种架构设计。
3. **DESED 已出现可追查的类别弱项。** 餐具声在五个模型中普遍偏弱；FlexSED、OpenFLAM 的说话声分数在两个划分中都很低。仍需区分误报、漏报与时间定位的贡献。
4. **TACOS 当前口径下整体分数偏低。** 现有统计不能分离描述理解、时间定位和标注歧义各自的影响，也不能用当前分数推断与 COSED 的性能差距。

## 5. 解释范围与限制

- 各模型在同一数据划分上使用相同样本，但模型训练数据暴露未控制，不作为纯零样本能力排名。
- DESED Validation 缺失 100 条；不再把补齐作为前置条件。后续对照固定使用同一份 1,068 条清单，不假定缺失对结果完全没有影响。
- 本轮未计算置信区间或统计显著性，小幅分差不能直接解释为可靠优势。
- 原始模型分数、类别分布与事件难度不同；跨数据集分差不能直接归因于单一因素。
- MAESTRO evaluation 没有真值，不填零分、不借用 development 标注替代。
- 误报候选需要人工复核，不能把候选数量当作实际误报数量或误报率。

## 6. 下一步分析

按 benchmark 逐个推进，首先分析 DESED Public Test 上的 Speech 与 Dishes，Validation 作为辅助：

1. 对照真值检查事件级误报和漏报，并人工复核代表性案例。
2. 按事件时长、重叠情况和预测边界偏差分组，提出可验证的瓶颈假设。
3. 在 Validation 上选择 query、阈值或后处理配置，再在 Public 上统一评价。

Public 用作固定配置后的主评测。与论文对比时，需确认使用相同划分与指标。暂不引入居家项目的类别优先级和部署要求。

## 7. 文件位置与复算

以下路径相对于 rabbit05 的 `/home/zhanghc/projects/ov-sed/benchmark/`：

| 内容 | 路径 |
|:---|:---|
| 本总结 | `analysis/experiment_summary.md` |
| 可阅读的结果网页 | `analysis/comparison.html` |
| DESED 总体及逐类别结果 | `analysis/desed/official_psds.json` |
| DESED、RealDESED、UrbanSED 详细结果 | `runs-source-audit-v2/模型/数据集/metrics.json` |
| TAG、TACOS、MAESTRO 详细结果 | `runs-extra/模型/数据集/metrics.json` |
| 统一计分入口 | `scripts/score_benchmarks.py` |
| 统一推理入口 | `scripts/run_benchmark.py` |
| 误报候选试听页 | `negative-review/review.html` |

```bash
cd /home/zhanghc/projects/ov-sed/benchmark

# DESED 标准 PSDS1
~/venvs/finelap/bin/python scripts/score_benchmarks.py

# TACOS 与 MAESTRO development
~/venvs/finelap/bin/python scripts/score_benchmarks.py --datasets tacos-test maestro-development
```

## 8. 方法参考

- [DCASE 2023 Task 4：PSDS 配置](https://dcase.community/challenge2023/task-sound-event-detection-with-weak-labels-and-synthetic-soundscapes)
- [DCASE 2024 Task 4：MAESTRO 评测说明](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels)
- [DCASE MAESTRO 计分实现](https://github.com/DCASE-REPO/DESED_task/blob/master/recipes/dcase2024_task4_baseline/local/sed_trainer_pretrained.py)
- [sed_scores_eval](https://github.com/fgnt/sed_scores_eval)
- [TACOS 数据集](https://github.com/OptimusPrimus/tacos)

本报告数值来自本地保存的评测结果，以上链接用于说明指标及数据定义，不作为本地分数的外部验证。


## 逐类统计与训练条件补充（2026-10-19）

[逐类报告](class_results.md) · [简洁三线表](comparison.html)。覆盖全部六个 benchmark、七个有真值的 split；明确区分目标数据集训练暴露、跨数据集直接迁移与尚未核实的同源录音重叠。五个发布权重本轮均未继续微调。

## 8. 协议复算与保存

在服务器 benchmark 根目录执行（不重新推理）：

```bash
~/venvs/finelap/bin/python scripts/score_benchmarks.py --datasets tacos-test --tacos-policy both
~/venvs/finelap/bin/python scripts/score_benchmarks.py --datasets tag-test --models finelap --tag-policy both
```

TACOS 分别写入 `runs-extra/模型/tacos-test/metrics.exact.json`、`metrics.semantic.json`；TAG 分别写入 `metrics.legacy.json`、`metrics.official.json`。`official` 只表示 FineLAP 使用官方 TAG PSDS 的未滤波分数分支。历史 `metrics.json` 保留，不代表新版主表；原始 NPZ 不变。统一启动入口 `scripts/run_benchmark.py` 接受相同配置参数，默认两种都算。

案例页已移除任意阈值及其预测区间，撤回依赖它们的边界、误报与漏报横向判断。既有标准 PSDS、片段 AP 与连续响应曲线仍可用于分析。正式操作点应在验证集按一致规则确定后固定到测试集。
