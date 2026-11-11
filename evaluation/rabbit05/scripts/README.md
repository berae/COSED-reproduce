# Benchmark 脚本

日常只用下面三个入口。先进入：
```bash
cd /home/zhanghc/projects/ov-sed/benchmark
```

| 脚本 | 用途 | GPU |
|---|---|---|
| run_benchmark.py | 指定模型和数据集，运行推理及该 runner 支持的计分 | 需要 |
| score_benchmarks.py | 读取已有预测，计算 DESED PSDS1、TACOS PSDS1、MAESTRO mpAUC | 不需要 |
| negative_review.py | 从已有预测筛选六套 benchmark 的失败候选，含排序和时间形态，不等于确认错误 | 不需要 |

## 1. 运行 benchmark
必须明确指定 GPU、模型、数据集，没有默认整批运行。先预览，不启动任务：
```bash
python3 scripts/run_benchmark.py --gpu 3 --models finelap --datasets desed-validation --dry-run
```
确认 GPU 空闲后，去掉 `--dry-run` 才实际运行：
```bash
python3 scripts/run_benchmark.py --gpu 3 --models finelap --datasets desed-validation
```
多个模型和数据集用空格分隔，依次执行；不自动多卡并行。脚本不自动选择或抢占 GPU。

可选模型：`dasm finelap flexsed mga-clap openflam`。
可选数据集：`desed-public desed-validation realdesed-test urbansed-test tag-test tacos-test maestro-development maestro-evaluation`。

后台运行示例：
```bash
DAY=$(date +%Y%m%d)
mkdir -p "log/$DAY"
nohup python3 scripts/run_benchmark.py --gpu 3 --models finelap --datasets desed-validation > "log/$DAY/$(date +%H%M%S)_queue_gpu3.log" 2>&1 < /dev/null &
```

- 每项实验日志：`log/YYYYMMDD/时间_模型_数据集.log`。
- 队列状态：`state/queues/日期_时间_gpu编号.json`；新队列不会覆盖旧队列状态。
- 固定类别 benchmark 结果：`runs-source-audit-v2/模型/数据集/`。
- 其他结果：`runs-extra/模型/数据集/`。
- 沿用冻结 runner 的断点续跑，已有预测会跳过，完成后仍可能重新计分；配置不兼容会报错。
- 队列 complete 只表示命令正常退出。TACOS 和 MAESTRO development 完成推理后自动计分；MAESTRO evaluation 缺少真值，仅推理。

## 2. DESED 标准计分
```bash
~/venvs/finelap/bin/python scripts/score_benchmarks.py
```
读取五个模型在 Public、Validation 上的预测，只计算标准 PSDS1。结果写入 `analysis/desed/official_psds.json`；定义见该目录 README。重复执行会更新这个结果文件。

## 3. 误报候选筛选
```bash
~/venvs/finelap/bin/python scripts/negative_review.py
```
当前覆盖 DESED Public、RealDESED Test、UrbanSED Test、TAG Test、TACOS Test、MAESTRO Development；五模型全量已有预测。输出到 `negative-review/`，逐数据集 CSV、可筛选 review.html 与代表试听。无手写部署阈值；粗分类定义与限制见该目录 README。加 `--render-only` 可只重建页面和试听。

## 维护规则
不再按日期、GPU 或临时修复新增启动脚本；优先修改已有入口。旧队列、旧重新计分和一次性检查脚本都在 `archive/`，仅供追溯，不作为日常入口。runners/ 中的文件是冻结实验代码，本次未改内容；其内容关联预测身份校验。

当前 runner 的职责划分见 `../runners/README.md`。统一启动脚本自动路由模型，尤其 FlexSED 使用修正版本。

## TACOS、TAG 与 MAESTRO 计分
复用已有预测，无需 GPU：
```bash
~/venvs/finelap/bin/python scripts/score_benchmarks.py --datasets tacos-test --tacos-policy both
~/venvs/finelap/bin/python scripts/score_benchmarks.py --datasets tag-test --tag-policy both
~/venvs/finelap/bin/python scripts/score_benchmarks.py --datasets maestro-development
```
TACOS 可选 exact / semantic / both，默认 both。semantic 固定 MiniLM-L6-v2、同文件余弦 >0.8 直接邻居真值合并，保留原查询及预测，不做传递闭包。分别保存 metrics.exact.json、metrics.semantic.json；主表读取 semantic。

TAG 可选 legacy / official / both，默认 both。legacy 使用保存的 scores；official 仅对 FineLAP 使用 raw，其他模型仍用 scores。它只对齐 FineLAP 官方 TAG PSDS 的未滤波分支，不表示整个评测等价；未复算 th-AUC。分别保存 metrics.legacy.json、metrics.official.json；主表读取 official。

文件均位于 runs-extra/模型/数据集/。历史 metrics.json 不覆盖，不能当作新版主表入口。统一 run_benchmark.py 接受相同两个参数，默认自动算两种配置；冻结 runner 未改，其内置旧计分不能替代新入口。

MAESTRO 仍写 metrics.json；development 全部 49 条、11 类、软标签 >0.5、一秒加权平均、最大 FPR=0.1。26 条 evaluation 没有真值，不计分。

完整性核查仍用 analysis/tacos_audit.py；不加 --semantic 时保留上次语义成绩；加参数时全部成功后才原子更新 audit.json，失败保留上次完整结果。


## Qwen3-Omni 六套 benchmark

2026-10-23 启动本地 Qwen3-Omni-30B-A3B-Instruct，服务复用 qwen-omni-service/start.sh，GPU 3、4，端口 8901。输出 runs-extra/qwen3-omni-30b/；config.json 保存完整提示词与身份，status.json 保存进度，各数据集每条音频一个 JSON，包含原始回复、语法修复记录、窗口与合并事件。

```bash
# 服务器端运行；不依赖本机 VPN 持续在线
OMP_NUM_THREADS=4 ~/venvs/finelap/bin/python scripts/run_omni_benchmark.py
# 仅每套一条完整音频（包括该音频全部窗口）
OMP_NUM_THREADS=4 ~/venvs/finelap/bin/python scripts/run_omni_benchmark.py --smoke-only
```

先验证六套各一条，任一失败则停止；通过后继续全量 6,231 条。固定类别提供本清单完整类别；TAG/TACOS 提供该音频原始 queries，均不把 GT 事件或时间发给模型；MAESTRO 使用十一评测类。10 秒窗、5 秒 hop，重叠窗同查询事件按区间并集合并；temperature=0。失败音频记录后继续，重启会跳过完成文件并续跑未完成窗口。

仅修复可明确识别的末尾缺失一个外层 JSON 右括号，原始文本与修复标记都保存；不补造事件或时间。不合法事件/截断输出保留错误并重试。

每套完成后使用官方 sed_scores_eval intersection fscore 对生成区间计算池化文件–查询 F1（TAG DTC/GTC=.5，其他事件任务 .7）；MAESTRO 使用一秒生成区间 micro F1。Omni 没有原生逐帧概率，这些不是 PSDS/mpAUC，不填入原表同一列。失败或缺失的音频不当作无事件，计分缺失会明确报错。

日志 log/20261023/omni_benchmark.log；服务日志同目录 omni_server.log。当前只运行一个实例，两张卡共同承载一个模型。

提示词修订：初次试跑含具体时间数字作为 JSON 示例，观察到不同音频重复该数值。为消除格式示例的时间暗示，正式批次改为只描述字段类型，不提供具体事件时间示例。原始试跑保存在 runs-extra/qwen3-omni-30b/archive-initial-smoke/，不与正式结果混合；不是按测试集成绩择优选择提示词。
