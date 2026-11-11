# COSED-reproduce

COSED 的独立研究复现。不是作者官方实现，不包含 Meta 内部代码。

第一版保存已完成实验的模型、训练、评测及诊断代码。代码尚保留实验服务器路径；数据、权重、第三方编码器以及冻结的评测组件需要单独准备，因此 clone 后不能直接训练。

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

本仓库仅归档独立重建代码及汇总数字。第三方代码、数据与权重需遵循各自来源的条款；本次未选择统一开源许可证。未上传权重、音频、逐样本标注、密钥或邮件正文。