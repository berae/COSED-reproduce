# COSED benchmark 复现

> 人工阅读结果入口：`analysis/results.html`（总体表、DESED 类别表、未计分状态和指标说明）。


> 当前启动方式以 `scripts/README.md` 为准：统一使用 `scripts/run_runners/archive/benchmark.py`，旧队列名称仅供历史追溯。

## 目录与常用命令

- `log/YYYYMMDD/`：运行输出、报错和启动日志；按日期归档。
- `state/queues/`：队列进度 JSON，记录各任务完成、失败或运行状态；不是历史日志。
- `state/pids/`：历史启动时记录的进程号，仅供追溯，不能据此直接停止进程（进程号可能复用）。
- `scripts/`：队列启动、重新计分、模型检查、误报筛选等命令脚本。
- `runner-*.py`、`runners/archive/benchmark.py`：冻结的推理与评测代码，保留原位置和内容，避免影响断点续跑和实验追溯。
- `manifests/`：评测样本清单。
- `runs-source-audit-v2/`：前三项 benchmark 源码核查后的结果；`runs-extra/`：TAG、TACOS、MAESTRO 输出。
- `runs/`、`runs-source-audit/`：历史实验输出，不能与当前结果混用。
- `negative-review/`：疑似误报记录、试听音频与复核页面。

查看队列（无需 GPU）：
```bash
cd /home/zhanghc/projects/ov-sed/benchmark
cat state/queues/audit-v2-queue-gpu5.json
cat state/queues/extra-queue-gpu3.json
```

启动示例见下方日志章节；现在使用 `python3 scripts/archive/run_extra_queue.py 3 openflam`，其中 `3` 是 GPU 编号，`openflam` 是模型名。启动前确认该 GPU 空闲。其他队列脚本也都在 `scripts/`。不要运行 `scripts/` 中所有脚本：`run_queue.py`、`run_audit_queue.py` 为历史阶段；当前额外 benchmark 使用 `run_extra_queue.py`，FlexSED 修正使用 `run_flexsed_fixed.py`。

整理时 GPU 3 的旧队列仍在运行，根目录的 `extra-queue-gpu3.json` 暂为兼容软链接，实际状态保存在 `state/queues/`。旧队列完成后可删除该软链接；新启动的脚本直接写入新目录。状态文件同名启动会更新，应以任务时间和逐实验 status.json 核对，不把历史 running 字样当作当前进程证据。


本轮使用 rabbit05 的 GPU 3、4、5。任务已脱离 SSH 会话，本机关机或 VPN 断开不影响服务器上已启动的进程。

## 当前队列
- GPU 3：DASM。
- GPU 4：OpenFLAM。
- GPU 5：FlexSED，结束后运行 MGA-CLAP。
- 每个模型依次运行 DESED public、DESED validation、RealDESED test、UrbanSED test。
- FineLAP 官方 Hugging Face 版本已通过 5 条推理检查；后台等待 GPU 4 的 OpenFLAM 队列结束后开始。单独进度 state/queues/queue-gpu4-finelap.json。
- TAG、MAESTRO、TACOS 尚未排队，仍需对齐各自评测协议。

## 查看进度
```bash
cat ~/projects/ov-sed/benchmark/state/queues/queue-gpu{3,4,5}.json
cat ~/projects/ov-sed/benchmark/runs/dasm/desed-public/status.json
tail -f ~/projects/ov-sed/benchmark/log/20261016/dasm_desed_public.log
```
结果存于 runs/模型名/数据集/metrics.json；逐帧预测为同目录的 npz 文件。日志中的早期报错保留，当前进度以 queue 和 status JSON 为准。

## 统计范围与限制
DESED public 692 条完整；validation 1068/1168 条，缺失 100 条。保留无事件样本，不因无标签而排除。RealDESED test 1007 条；UrbanSED test 2000 条，排除 AppleDouble 文件。每项实际文件与标注在 manifests 中固定。

各模型独立环境，使用官方权重。长音频用 10 秒窗、5 秒步长，重叠逐帧分数平均。DASM 中值窗 16 帧，MGA-CLAP/FineLAP/OpenFLAM 3 帧，FlexSED 官方 val.py 使用 9 帧中值滤波；首轮误用了不平滑设置，旧分数已失效，修正结果见 flexsed-postprocessing-check.json。结果是待核对的复现候选，不能直接宣称已复现 COSED 数值；尚需确认论文使用的精确划分、权重版本、标签措辞和后处理。Public 和 validation 不混用。

PSDS1：DTC=GTC=0.7，alpha_ct=0，alpha_st=1，max_eFPR=100/h。原标注超出音频边缘的轻微偏差在 manifests 的 boundary_clips 中记录。原始数据未修改。

## 断点续跑
相同 runner 与 manifest 下，已有 npz 会跳过。更改推理代码或清单时会拒绝混用旧预测；应创建新实验输出。队列逐项保存退出码，单项失败会继续下一项。runners/archive/runner-v1.py 是本批次冻结版本。

## 2026-10-16 计分修复
UrbanSED 原始标注含同类重叠事件，新版计分库要求先取同类区间并集。DASM 部分末帧在六位小数精度下长度为零，计分时剔除此类零时长帧。scripts/archive/rescore.py 仅规范化计分输入，不修改原始音频、标注或预测；使用单进程避免子进程异常导致主进程等待。修复日志 log/20261016/rescore.log。DASM 原阻塞计分进程已停止，其队列继续进行 UrbanSED 推理。

## FlexSED 修正
官方 src/val.py 的 median_filter=9，首轮接入只依据 api.py，遗漏了评测平滑。复用相同 raw 预测加 9 帧中值滤波，PSDS1：DESED validation 0.1446533585（原 0.0607200928），RealDESED test 0.0789102681（原 0.0171111262）。这次只改后处理，指标参数、音频及模型输出均不变。旧 runs 下 metrics.json 保留用于审计，不能继续作为正确 FlexSED 结果引用。

## 后三项与误报候选分析
后三项由 scripts/archive/run_extra_queue.py 后台运行，输出 runs-extra。TAG 使用公开 FineLAP grounding PSDS2021 参数（DTC/GTC=.5，max_eFPR=800/h），文件内完全相同的 phrase 合并；原始 v2.2 列表与 COSED 的精确对应未确认。TACOS 先完成推理，待明确论文 Sentence-BERT 近义描述合并实现后计分；MAESTRO evaluation 无真值，development 仅诊断，不冒充论文评测结果。

negative-review/ 为 benchmark 误报候选筛选。按模型/数据集/类别挑 5 个不同文件，各文件选目标标注覆盖为零的最高平均响应 2 秒窗，步长 1 秒。不统一使用 0.5，不以分数跨模型排序，不将候选数量当作误报率。分为靠近目标边界（0.5 秒范围）、与其他已标注事件重叠、无已标注事件背景。目标类在片段内未标注不代表声学上确定不存在，必须人工试听；不将 benchmark 测试数据加入训练。

## 日志目录（2026-10-16 起）
所有实验日志统一放在 `log/YYYYMMDD/`，日期按服务器本地时间。新队列的每个实验使用 `HHMMSS_微秒_实验阶段_模型_数据集.log`，避免多次运行混写。历史日志按文件最后修改日期归档，内容不变，文件名中的连字符改为下划线。

例如：
```text
log/20261016/extra_openflam_tacos_test.log
log/20261016/153000_123456_extra_finelap_tag_test.log
```

查看当天日志：
```bash
cd /home/zhanghc/projects/ov-sed/benchmark
ls -lh log/$(date +%Y%m%d)/
tail -f log/20261016/extra_openflam_tacos_test.log
```

手工启动新队列时，也将队列自身输出放到当天目录：
```bash
cd /home/zhanghc/projects/ov-sed/benchmark
DAY=$(date +%Y%m%d)
mkdir -p "log/$DAY"
# 示例：确认 GPU 可用后再执行；会启动实验。
nohup python3 scripts/archive/run_extra_queue.py 3 openflam > "log/$DAY/$(date +%H%M%S)_extra_queue_gpu3.log" 2>&1 < /dev/null &
```

整理时正在运行的旧队列无需重启，已打开日志继续写入新位置。根目录暂留两个 `extra-openflam-maestro-*.log` 软链接，仅用于旧队列后续任务，实际内容写入 log/20261016；队列结束后可删除这两个软链接。结果、预测和状态 JSON 仍留在原位置。

## 2026-10-16 队列收尾
5 个模型的 TAG test（各 483 条）、TACOS test（各 2000 条）、MAESTRO development（各 49 条）与 evaluation（各 26 条）共 20 项任务均正常退出，预测文件数量齐全。TAG 已完成当前本地协议计分；TACOS 与 MAESTRO 仍有评测协议或真值限制，推理完成不等于论文指标复现完成。旧队列已经退出，前文所述三个临时兼容软链接已删除，所有状态与日志直接使用 state/queues/ 和 log/ 下的路径。

## DESED 当前计分入口
使用 `scripts/score_benchmarks.py`，只执行标准 PSDS1。结果与定义见 `analysis/desed/README.md`，机器可读结果为 `analysis/desed/official_psds.json`。此前该分析中的自定义放宽条件和 PSDS2 对照已删除。

## Runner 目录整理
当前推理代码统一放在 `runners/`，旧版在 `runners/archive/`。使用 `scripts/run_benchmark.py` 启动，具体分工见 `runners/README.md`。2026-10-16 移动后已验证全部 40 组当前预测身份与源码、清单一致，源码内容未修改。

## TACOS / MAESTRO 计分更新
TACOS 原始 caption PSDS1 与 MAESTRO development（49 条、11 类）的 mpAUC 已完成，人工阅读入口为 analysis/results.html 第 3 节。TACOS 未做 COSED 近义合并；MAESTRO evaluation 仍无真值。统一计分入口为 scripts/score_benchmarks.py。上文“尚未计分”描述为此前状态，以本节和结果页为准。
