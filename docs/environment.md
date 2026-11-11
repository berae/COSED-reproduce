# 已验证环境与外部依赖

这是2026-11-11从实验环境读取的版本记录，不是完整可安装的lockfile。

| 依赖 | 版本 |
|---|---|
| Python | 3.9（finelapt环境） |
| torch | 2.4.1+cu121 |
| torchaudio | 2.4.1+cu121 |
| transformers | 4.57.6 |
| numpy | 2.0.2 |
| scipy | 1.13.1 |
| pandas | 2.3.3 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| einops | 0.8.2 |
| timm | 1.0.27 |

## 必须单独准备

- `vendor/PretrainedSED`：上游commit `1aa47e482f7e89904cba2338999345025d8b4e36`，实际本地修改需另外核对；以及`vendor/ontology.json`。
- `weights/ATST-F_weak.pt`。
- `weights/mga/`下的`mga.pt`、BERT配置及tokenizer文件。
- 五个来源的JSONL、对应音频和AudioSet Strong metadata。路径在`data.py`。
- `evaluate.py`依赖服务器FineLAP项目的`dependencies/benchmark`与`dependencies/scoring`，包括冻结清单、runner、scorer、TACOS语义合并映射。本仓库尚未迁移这些组件。
- 当前`sed-scores-eval`在pip显示`0.0.0`，训练/评测实际显式优先加载上述本地scoring目录。不能用同名PyPI包版本替代并假定结果完全一致。
- `run.sh`还检查现有pilot、provenance和text_check记录；新机器上需要重建这些检查记录，不能通过伪造文件跳过。

## 迁移步骤

1. 在独立目录准备上面的依赖、音频及标注。
2. 核对`data.py`、`evaluate.py`、`run.sh`和两个audit脚本中的服务器路径。
3. 核查预训练权重加载、文本输出一致性、真实样本正反向梯度和评分口径。
4. 使用新的实验名称启动；不要覆盖已完成的基线目录。

当前版本优先保留产生报告成绩的代码原貌。可移植配置和完整依赖打包是后续整理工作，不应与基线方法改动混在同一个实验中。