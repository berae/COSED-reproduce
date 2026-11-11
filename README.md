# COSED-reproduce

COSED 的独立研究复现。不是作者官方实现，不包含 Meta 内部代码。

本仓库保存当前 COSED 重建，以及这条 SED 研究线的历史训练、发布模型评测、诊断代码和实验记录。代码尚保留实验服务器路径；音频、权重、大型预测数组和完整运行环境需要单独准备，因此 clone 后不能直接训练。

**其他助手 / 新协作者请先读：[实验与研究上下文](docs/START_HERE.md)。** 其中区分了当前有效结果、失败实验、已撤回的判断和待验证的研究问题。

## 归档目录

| 目录 | 内容 |
|---|---|
| 根目录 Python 文件 | 当前 COSED 模型、训练、评测及诊断入口 |
| `experiments/cosed/` | 早期失败、GRU 修正、scale 10 各轮配置、源码快照、日志与成绩 |
| `experiments/finelap/` | 五源 BF16、五源 FP32、十源公开代码配方，以及 rabbit05 最终评测 |
| `evaluation/rabbit05/` | 五个发布模型与 Omni 的 benchmark runner、清单、日志及计分结果 |
| `evaluation/frozen/` | 训练实验使用的冻结评测代码与清单 |
| `research/` | 实验总结、类别统计、误报漏报、案例网页及维护记录 |
| `third_party/` | 实际使用的模型与评分源码；保留各自来源与条款 |
| `docs/inventory-*.json` | 原始路径、导出文件 SHA256、排除项目 |

范围和复算限制见 [归档说明](docs/ARCHIVE.md)。多个版本有明确的历史用途，请勿把归档目录中的旧脚本作为当前默认入口。

## 当前基线

- 实验：`seed31415926_grufp32_scale10`，随机种子31415926。
- ATST-F + MGA-CLAP文本塔（保留并微调codebook）+ 2层BiGRU；210,845,956参数。
- 两路乘法scale初始化10，bias初始化−10，均可学习。
- 五源：AudioSet Strong、FineLAP-100K、AudioCaps、TACOS、Clotho；倍率1/1/2/5/4。
- 4×RTX3090，总batch1024，每卡micro8；BF16 autocast，BiGRU及帧归一化FP32。
- 第21轮早停；AS-S验证最佳为第11轮EMA，PSDS1=0.2311966779。
- best.pt SHA256：`7d401db9e827acec282c1fe52a1d51462dfdd48d15019bbe0dc379ad8ac91365`。

| 本地benchmark划分 | 指标 | 本次结果 |
|---|---|---:|
| DESED Public Test | PSDS1 | 0.505185 |
| RealDESED Test | PSDS1 | 0.189551 |
| MAESTRO Development | mpAUC | 0.601733 |
| UrbanSED Test | PSDS1 | 0.190154 |
| TAG Test | Grounding PSDS | 0.661590 |
| TACOS Test，近义真值合并 | PSDS1 | 0.171354 |

TACOS exact口径为0.102147。DESED Validation单独计分为0.440192（1068条，缺100条），不混入Public Test。MAESTRO划分、TACOS编码器及部分处理细节尚未与作者完全确认；本次为单种子，不能直接称为完整官方复现。

完整数字与协议见 [results/benchmark_summary.json](results/benchmark_summary.json)。

## 文件

| 文件 | 用途 |
|---|---|
| `model.py` | 双编码器、codebook、BiGRU、校准及focal项 |
| `data.py` | 五源数据读取、采样和监督目标 |
| `train.py` | 四卡训练、EMA、验证、轮内断点恢复 |
| `evaluate.py` | 连接既有冻结评测组件，运行六benchmark |
| `run.sh` | 当前实验服务器启动入口 |
| `audit_gradients.py` | 整批/分块梯度等价检查 |
| `audit_representations.py` | 初始化/在线/EMA表征分布检查 |
| [docs/reproduction.md](docs/reproduction.md) | 结构图、参数量、训练说明与已知重建差异 |
| [docs/environment.md](docs/environment.md) | 已验证环境、外部依赖与迁移要求 |

历史审计脚本显式使用旧实验scale=0.1及旧checkpoint；它们的结果不能混作当前scale10模型结果。主训练默认scale=10。

## 运行边界

当前已验证工作目录为 `/work/zhanghc/Myllm/COSED`。先对照环境文档准备本地数据、预训练权重、vendor及评测组件，核对 `data.py`、`evaluate.py`、`run.sh` 中的路径，再执行：

```bash
bash run.sh
```

启动器使用锁避免重复启动。首步、每50步以及验证前后原子保存`last.pt`，包含在线权重、EMA、优化器、轮内位置和各卡随机状态。重复启动会从该实验的`last.pt`恢复；改变初始化必须使用独立实验目录。当前实验已经完成，原目录不会重新训练。

## 研究记录与限制

1. 旧版本cuDNN GRU的FP16梯度下溢已修正，BiGRU固定FP32。
2. 根据作者澄清将乘法scale从0.1改为10；本表为重新初始化并训练的结果。
3. 当前Mel的dB截断和mixup存在微批依赖，更换micro不能保证逐步等价。
4. TACOS筛选/改写、ontology调整和增强细节仍有重建差异。
5. 后续优化实验应独立命名，保留本基线及原始预测。

## 来源

- [COSED论文](https://arxiv.org/abs/2609.30083)
- [PretrainedSED / ATST-F](https://github.com/fschmid56/PretrainedSED)
- 文本初始化来自MGA-CLAP公开权重；五源标注复用现有研究数据管线。

仓库包含独立重建代码、第三方源码快照、实验记录、评测清单及部分逐样本结果。第三方代码、数据与权重需遵循各自来源的条款；本次未选择统一开源许可证。未上传模型权重、音频或认证凭据。部分网页仍依赖服务器音频和本地服务；完整原始训练标注与大型预测数组不在 Git 中。
