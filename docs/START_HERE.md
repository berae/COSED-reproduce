# 从这里开始：实验与研究上下文

更新：2026-11-11。目的是让协作者理解我们做过什么、证据在哪里、哪些问题值得继续检验。此仓库为 COSED 的独立重建及相关 OVSED 实验归档，不是官方实现。

## 建议阅读顺序

1. [当前基线及六项结果](../README.md)，[完整数值与指标口径](../results/benchmark_summary.json)。
2. [结构、训练及已知差异](reproduction.md)，[环境与外部依赖](environment.md)。
3. 本文实验索引，再按问题进入配置、日志、源码快照。
4. [完整实验记录](../research/analysis/experiment_summary.md)：按时间追加，早期结论可能已被后续条目修正；不能只读开头的旧汇总。
5. [源码核查记录](../research/SOURCE_AUDIT.md)、[维护记录](../research/analysis/maintenance.log)。

不必一次读完全部文件。优先看一轮实验的配置、summary、日志，再看对应源码。网页是本地分析快照，GitHub 不会直接运行其交互功能；音频和 localhost/SSH 服务依赖也未随仓库部署。

## 我们要研究什么

关注开放词表声音事件检测：给定音频和文本查询，检测目标事件及其时间范围。当前是科研分析，不预设居家项目的最终标签集。先明确失败现象和可区分的机制假设，再选择架构或训练改动。

六个 benchmark 不等于六个严格“未见类别”测试。必须结合模型训练数据、预训练数据和标签语义重叠解释结果；没有使用目标数据集训练，不代表目标声音概念从未见过。

## 实验索引

| 实验 | 配方 / 目的 | 结果及状态 | 可核查入口 |
|---|---|---|---|
| 五个发布模型 | DASM、FineLAP、FlexSED、MGA-CLAP、OpenFLAM；统一核查推理与计分 | 有效核心预测保留；早期采样率、TAG 后处理、TACOS 口径已修正 | [runner 说明](../evaluation/rabbit05/runners/README.md)、[计分入口](../evaluation/rabbit05/scripts/README.md)、[类别结果](../research/analysis/class_results.md) |
| FineLAP 五源 BF16 | AS Strong / FineLAP-100K / AudioCaps / TACOS / Clotho，总 batch 96，10 epochs | 与大配方相比，部分分类数据集下降，TACOS 上升；不是单因素对照 | [训练及源码](../experiments/finelap/runs/finelap_cosed_mix_20261029/)、[rabbit05 最终评测](../experiments/finelap/rabbit05/finelap_cosed_mix_20261029/evaluation/) |
| FineLAP 五源 FP32 | 同五源、batch 96、10 epochs，FP32 加激活重计算 | 未消除主要差距；MAESTRO 有变化，不能说精度完全无影响 | [配置、源码、结果](../experiments/finelap/runs/finelap_cosed_fp32_20261102/) |
| FineLAP 公开代码配方重建 | 十源混合，2,322,976 条记录，batch 512，FP32，10 epochs | DESED / UrbanSED 较好；含相应训练集，不能归入严格 zero-shot | [配置、loss、结果](../experiments/finelap/runs/finelap_original_20261031/) |
| COSED 初次重建 | `seed31415926` | GRU 梯度/更新故障，不能用来判断架构能力 | [历史源码与日志](../experiments/cosed/runs/seed31415926/) |
| COSED 修正 GRU、scale 0.1 | `seed31415926_grufp32` | 检测训练失败；在线与 EMA 表征均有退化证据 | [实验](../experiments/cosed/runs/seed31415926_grufp32/)、[诊断](../experiments/cosed/diagnostics/) |
| COSED scale 10 | 按作者澄清从初始化重新训练，保留并微调 MGA-CLAP codebook；总 batch 1024 | 第 21 轮早停，选第 11 轮最佳 EMA；六项结果见下表 | [完整实验](../experiments/cosed/runs/seed31415926_grufp32_scale10/)、[评测汇总](../results/benchmark_summary.json) |
| Qwen3-Omni 对照 | 音频输入、文本事件输出 | 与逐帧模型输出形式不同；报告的 F1/AP 不自动等价于连续帧分数 PSDS | [结果与口径](../research/analysis/omni_results.md)、[运行代码](../evaluation/rabbit05/scripts/run_omni_benchmark.py) |
| 误报、漏报和案例分析 | 同时保留事件级和 clip 级视角 | 部分历史页面只是候选筛查，不是已确认失败样本 | [分析代码](../research/analysis/error_rates.py)、[审阅页面](../research/analysis/error_rates/review.html)、[历史负例审阅](../research/negative-review/) |

`pilot_1024`、`*_smoke`、`official_reproduction_20261029/preflight.json` 是准备或短测试，不要当作完整训练。

## 当前可以使用的结果

| 本地评测 | 当前 COSED 重建 | COSED 论文表值 |
|---|---:|---:|
| DESED Public / PSDS1 | 0.505185 | 0.437 |
| RealDESED / PSDS1 | 0.189551 | 0.224 |
| MAESTRO development / mpAUC | 0.601733 | 0.643 |
| UrbanSED / PSDS1 | 0.190154 | 0.234 |
| TAG / grounding PSDS | 0.661590 | 0.648 |
| TACOS / semantic PSDS1 | 0.171354 | 0.187 |

论文列来自此前核对的论文主表，是参照而非完全同条件复现证明。本地单种子；MAESTRO 划分、TACOS 编码器/改写等未完全对齐。TACOS exact 为 0.102147；DESED Validation 为 0.440192（1,068 条，缺 100 条），不与 Public 的 692 条混算。

当前权重身份：`best.pt` SHA256 `7d401db9e827acec282c1fe52a1d51462dfdd48d15019bbe0dc379ad8ac91365`。实际评测目录：[benchmark_7d401db9e827](../experiments/cosed/runs/seed31415926_grufp32_scale10/benchmark_7d401db9e827/)。

## 已修正的判断与尚未解决的问题

- **旧任意阈值不能支持横向误报/漏报判断。** `[0.5, 0.5, 0.2, 0, 0.5]` 的展示阈值会把低幅但排序正确的曲线判成无事件。它不改变已保存 PSDS，却会误导案例分析。“未触发筛查”也不等于正确或漏检。
- **FineLAP TAG：** 官方 PSDS 分支使用未滤波分数，本地旧中值滤波值约 0.660162，修正后约 0.647916。保留 `metrics.official.json` 与历史结果；不要直接读取旧 `metrics.json` 当主成绩。
- **TACOS：** exact 与 semantic 都保留；近义合并改变 GT 口径，当前使用 MiniLM-L6-v2、余弦阈值 0.8、直接邻居合并，不作传递闭包。应报告所用策略，不能将两种成绩拼在同一排行。
- **梯度累积：** 固定同一 Mel 输入的小批审计中，整批/分块 loss 和梯度基本一致（相对差低于约 2e-6）；未支持“分块自动丢失负例”这一猜测。不能据此证明所有输入规模都等价。
- **预处理仍有微批依赖：** AmplitudeToDB 的截断参考、mixup 和随机层会随分组变化；原始预处理条件下曾观察到约 22% 的 BiGRU 梯度差。没有证据单独证明它导致训练失败。
- **表征退化证据对应旧 scale 0.1 实验。** 不仅 EMA query，在线表征也有退化；现有诊断不能直接当作 scale 10 模型的诊断结果。
- **scale 10 新训练有效，但不是所有因素的完整因果消融。** 不要把全部改善归因于架构，也不要把 codebook 说成必然在这轮才首次加入。作者确认其应保留并微调。
- **配方与数据规模存在混杂。** 五源与十源的样本总量、目标任务训练集、batch、监督配比都不同；结果不能单独证明 captions 无用、AudioSet 必需或某个编码器更优。

## 值得讨论的三个方向（尚未验证）

| 研究问题 | 最小可判别实验 | 可能推翻假设的结果 |
|---|---|---|
| 同一声音换成类别名、同义表达或具体描述，时序证据是否稳定？ | 固定模型与音频，对人工核查的查询组比较帧排序、区间覆盖和验证集操作点；加入确实不存在的近义/近邻事件 | 差异主要由标注不完整或阈值尺度解释，而非查询语义 |
| 混合数据的负例是否含有未标注真事件？ | 抽样核查负对的标注覆盖，固定数据量/steps，仅改变有证据的负例 mask/权重 | 噪声比例低，或修正后无稳定收益 |
| 分类监督与描述定位监督是否存在任务间权衡？ | 固定初始化、训练预算与数据暴露，改变一种监督比例；同时看类别与描述任务及 source-wise 梯度统计 | 多任务都随预算提高，原先差异主要是训练不足 |

这些不是已经确定的新方法，也未做新颖性检索。先利用已有模型与少量人工复核建立证据，不必马上训练新架构。

## 文件范围与复算

见 [归档范围、缺项和路径映射](ARCHIVE.md)。原始音频、checkpoint、大型逐帧预测未上传；要独立重算分数或听音，需要服务器访问。仓库可支持代码审查、历史实验比较和研究讨论，但不是无需外部资产的一键复现实验包。
