# 归档范围与来源

2026-11-11 从本地研究目录、rabbit02、rabbit05 采集。原始实验目录未移动、删除或重写。归档覆盖本轮 OVSED / FineLAP / COSED / Omni benchmark 研究线，不宣称包含服务器上所有其他项目（例如早期 NAPE、TTA 生成和 Bilibili 采集全管线）。

## 路径对应

| 仓库位置 | 原始位置 |
|---|---|
| 根目录 COSED 实现 | rabbit02 `/work/zhanghc/Myllm/COSED` |
| `experiments/cosed/runs`、`log` | 上述目录的 `runs`、`log` |
| `experiments/finelap/runs`、`source`、`log` | rabbit02 `/work/zhanghc/Myllm/FineLAP` |
| `experiments/finelap/rabbit05` | rabbit05 `/home/zhanghc/projects/ov-sed/FineLAP/exps` |
| `evaluation/frozen` | rabbit02 FineLAP 的 `dependencies/benchmark` |
| `third_party/scoring` | rabbit02 FineLAP 的 `dependencies/scoring` |
| `third_party/cosed_vendor` | rabbit02 COSED 的 `vendor` |
| `evaluation/rabbit05` | rabbit05 `/home/zhanghc/projects/ov-sed/benchmark` |
| `third_party/published_models` | rabbit05 `/home/zhanghc/projects/ov-sed/{DASM,FineLAP,FlexSED,MGA-CLAP,OpenFLAM}` |
| `research` | 本地 `D:/codex/2026-09-23/wo-d/work/cosed-reproduction` 的分析、评测与审阅文件 |
| `experiments/cosed/diagnostics` | 本地上述目录的 `cosed/{gradient_audit,representation_audit}.json` |

服务器副本与本地工作副本分别保存，没有假定它们始终一致。当前研究入口是 `docs/START_HERE.md`；冻结源码用于解释对应实验，不能随手替换成最新工作副本。

## 收录与排除

- 收录可审阅源码、配置、文字日志、指标 JSON/CSV、清单、已有图表与 HTML。
- 单文件导出上限为 20 MiB。大型 `candidates.json` / `candidates.csv` 等未收录；按 benchmark 的较小候选文件仍保留。详情查 inventory 中的 `omitted`。
- 不收录音频、模型 checkpoint、NPZ/NPY、压缩数据包、缓存、Git 历史或认证文件。因此逐帧预测的数值重算和听音需连接原服务器。
- 已将指定的 `source.tar.gz`、`source_before_step_checkpoints.tar.gz`、`source_snapshot.zip` 展开为可直接阅读的源码目录。其他压缩包不上传。
- 模型源码导出只收录源文件、配置、说明和许可证等，不是完整可安装环境；存在二进制扩展及外部权重依赖。
- `inventory-*.json` 记录导出时文件原始路径、字节数、SHA256 和排除项。后续在仓库维护的文档可能与导出哈希不同，应通过 Git diff 和维护日志追踪；哈希不是后续修改的自动校验声明。
- 采集脚本见 `tools/collect_experiments.py`。它是受限类型导出器，不是通用凭据清洗器；上传前另做了凭据模式检查。

当前主要 checkpoint 原件：

```
rabbit02:/work/zhanghc/Myllm/COSED/runs/seed31415926_grufp32_scale10/best.pt
rabbit02:/work/zhanghc/Myllm/FineLAP/exps/
```

大型逐帧预测位于各实验 `evaluation`、COSED 的 `benchmark_7d401db9e827`，以及 rabbit05 benchmark 的各 `runs*` 目录。具体原始文件位置由配置、identity/provenance 和清单给出，不应根据仓库缺少 NPZ 推断实验未完成。

## 当前与历史计分

- 发布模型日常入口：`evaluation/rabbit05/scripts/run_benchmark.py`、`score_benchmarks.py`。根目录 `evaluate.py` 是 COSED 的实验评测适配器。
- 旧 `runs`、`runs-source-audit`、`runners/archive` 和 `scripts/archive` 保留追溯，不与修正后的 `runs-source-audit-v2` 混用。
- TAG 当前 FineLAP 官方后处理分支看 `metrics.official.json`，旧中值滤波结果另保留。
- TACOS 看明确标注的 `metrics.semantic.json` 或 `metrics.exact.json`，不要只根据 `metrics.json` 文件名决定口径。
- MAESTRO 为 development；DESED Validation 缺项已记录。`status: complete` 只表示对应任务执行完成，不单独证明数据划分、指标或官方等价性。

## 代码来源与使用边界

第三方文件保留上游声明、LICENSE/NOTICE（源目录存在时）。模型地址与来源信息见 [repositories.json](../third_party/published_models/repositories.json) 及各模型 README。FineLAP、PretrainedSED、评分工具的本地适配与训练快照均按实际使用版本保存，不承诺与上游最新版本相同。

`repositories.json` 是早期克隆记录，其中 `weights_downloaded` / `environment_installed` 的 false 值不是当前状态。以实际实验配置和已完成结果为准。归档目录关闭 Git 自动换行转换，以保留源码身份哈希。

本仓库保持私有，不转授第三方代码、标注或权重的许可。如果后续公开，应逐项确认各来源许可、标注再分发和本地路径信息。当前归档只用于已授权的研究协作。
