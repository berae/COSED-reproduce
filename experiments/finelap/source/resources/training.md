# FineLAP Training & Fine-tuning

Before training, make sure that all files from [here](https://huggingface.co/AndreasXi/FineLAP_Pytorch) have been downloaded to `./weights/`. 

## Environmental Setup
```bash
conda create -n finelap python=3.9 

git clone https://github.com/facebookresearch/fairseq.git
pip install "pip<24.1" -U; cd fairseq; pip install -e ./

pip install -r requirements_train.txt
```

## Data Setup
To train FineLAP, we format the data in a JSONL structure as follows:

```json
{
  "audio_id": "Ycq6bqC_AsO4.flac",
  "audio_path": "path/to/audio.wav",
  "caption": "Birds are chirping with background noise.",
  "phrases": [
    {
      "phrase": "Background noise",
      "segments": [
        [0.498, 10.0]
      ]
    },
    {
      "phrase": "Bird vocalization, bird call, bird song",
      "segments": [
        [0.629, 4.114],
        [4.313, 10.0]
      ]
    }
  ]
}
```

Each entry contains:

- audio_id: Unique identifier of the audio sample.
- audio_path: Path to the audio file.
- caption: A clip-level description of the audio content.
- phrases (optional): A list of sound events, where each includes:
  - phrase: Textual phrase of the event
  - segments: Time intervals (in seconds) indicating when the event occurs

For data without frame-level annotations, the `phrases` field can be omitted. The dataset will automatically detect this and skip the frame-level loss for such samples.
An example training metadata file with 10 samples is provided at `data/training_metadata_example.jsonl`.

The current training pipeline uses the phrase bank `data/phrase_bank_new_with_FSDLabel_UrbanSED.jsonl`.

Once the dataset metadata JSONL is ready, include it in the `train_data_args.metadata_files` list defined in `config/data_config/data_eat.yaml` or `config/data_config/data_htsat.yaml`.

## Start Training
Run
```bash
bash scripts/train.sh
```
to start training. This will use the config `config/finelap_eat_config.yaml`. The output will be saved in `exps/${exp_name}`. 

## Fine-tuning From a FineLAP Checkpoint
The training code now supports loading an existing FineLAP checkpoint before training starts. This is useful when you want to finetune from a previously trained model such as `weights/finelap_fixed.pt`.

In `config/finelap_eat_config.yaml`, set:

```yaml
model_args:
  ckpt_path: './weights/finelap_fixed.pt'
```

If `ckpt_path` is an empty string:

```yaml
model_args:
  ckpt_path: ''
```

then no FineLAP checkpoint will be loaded, and training will start from the encoder initialization defined by `audio_encoder_ckpt` and `text_encoder_ckpt`.

This finetuning path loads model weights only. It does not restore the optimizer state or resume the previous epoch count.

## rabbit02 数据准备核查（2026-10-29）

本节记录数据核查阶段；当前试训状态见下方“COSED 数据组合试训”。源码核查固定在 `ad82bc1e242de4f56b80cf36f6d9f3009bb96c7b`；使用 `git show HEAD:文件` 区分上游代码与本机修改，不覆盖已有 train.py / mix_dataset.py 改动。

### AudioSet Strong：标注已下载，能与现有数据对接

目录：`/work/zhanghc/datasets/AudioSet/metadata/strong/`。

| 文件 | 音频数 | 标注行数 | 用途 |
|---|---:|---:|---|
| audioset_train_strong.tsv | 103,463 | 934,821 | 训练事件起止时间、MID |
| audioset_eval_strong.tsv | 16,996 | 139,538 | 独立测试标注 |
| audioset_eval_strong_framed_posneg.tsv | 14,203 | 958,528 | 960 ms 网格正负标签；不能混同普通事件 TSV |
| mid_to_display_name.tsv | — | 456 | Strong 专用类别名称映射；无表头 |

四个文件由 Google 官方地址下载，经本机中转到 rabbit02，SHA256 两端一致。来源、哈希、数量和映射核查保存在同目录 `download_manifest.json`。原始 TSV 保持不变。

用 WavCaps 的 `as_final.json` 与真实 FLAC 文件名匹配：Strong train 有 **92,450** 条同时具备音频和描述，eval 有 **15,867** 条。103,463 条官方 train 的片段开始时间全部能对上原 AudioSet CSV；这是元数据对齐，不等于逐条验证音频波形截取正确。尚未对共享原始 WAV 库完成 Strong 的逐文件覆盖统计。

现有 WavCaps 合并清单 `wavcaps_merged_sl_ac_train.jsonl` 共 400,812 条，其中包含 **15,710 条 Strong eval**。该清单原本用于其他任务，不能直接视为 AudioSet Strong 官方训练划分。训练构建时必须按官方 train 筛选，并对本次评测集做 ID 排除；不要根据文件名中的 train 或 sl 推断划分/强标注。

全部 456 个 Strong 类别名都在当前 FineLAP 短语库 alternatives 中。官方 train 有一条零时长标注：`vqj-pMnJ9Zg_18000`，9.260–9.260，`/m/07rbp7_`。原文件保留，生成训练标签时排除该事件并记录；不因此丢弃整条音频。

### FineLAP 各数据源的作用

| 数据源 | 如何进入训练 | 当前需要做什么 |
|---|---|---|
| AudioSet Strong + WavCaps AudioSet_SL | 同一音频的 caption + MID 映射名称 + onset/offset | 已具备至少 92,450 条的配对条件；整理 JSONL 并落实评测排除 |
| WavCaps FreeSound / BBC / SoundBible | caption，通常仅全局损失 | 现有文本、音频可用；转换字段、筛选重叠样本 |
| 普通 AudioSet + AudioSetCaps | 使用 AudioSetCaps 提供的描述，仅全局损失 | 原 WAV 已有；还需取得并匹配 AudioSetCaps caption，不用原类别 CSV 冒充描述 |
| AudioCaps / Clotho | 原始人工 caption；Clotho 多参考描述 | 按官方训练划分盘点，不能把本地 AudioCaps test 目录当训练来源 |
| DESED Strong / UrbanSED | 事件名称与区间；需配套非空 clip caption | 对接训练音频和时间标注；作者精确 caption/过滤清单尚未确认 |
| FineLAP-100K | FSD50K 事件前景与背景混合，合成时间标签和 caption | 已有 23/29 分片、79,312 行；补齐发布数据可避免重新生成随机混音 |

依据：`config/data_config/data_eat.yaml` 的十份元数据路径，`utils/make_finelap_data.py` 的 AudioSet_SL 分支，以及 `utils/make_caption_data.py`。前者明确将 Strong TSV 与 WavCaps caption 合并；配置中的 FSD50K 路径指向合成后的 `dev_simulated_sed_data/sed_metadata.jsonl`，并非原 FSD50K 弱标注直接作为强标注。

### FSD50K 合成流程与作者补充

作者 issue #1 已补充：先选只有一个标签的音频，再做能量裁剪，最后保留裁剪时长严格满足 `1 < duration < 7.5` 秒的样本。`filter_energy.py` 的默认阈值为 -50 dB，而论文配方对应的用例显式传入 -20 dB，不能直接运行默认参数。

`simulate_sed_data.py` 接收已经筛选的前景 CSV：每个 10 秒混音抽 1–5 个前景，短于 3 秒的前景尝试重复 1–3 次；同一前景的重复不互相重叠，不同前景允许重叠；SNR 12–20 dB。caption 由事件名填入模板，时间标签来自放置位置。示例使用 Adobe Ambience 背景，但参数默认背景指向 FSD50K 长音频，必须显式指定配方。函数名 `extract_single_events` 并没有实现单标签筛选，而是随机选取已有标签之一，因此前置筛选不能省略。

公开仓库没有包含作者完整的前景/背景清单及最终随机混音配置。复用发布的 FineLAP-100K 与自行重建新混音应分别记录，不能声称音频逐条相同。

### 加载器中影响复现的细节

- `MixDataset` 直接拼接元数据；训练使用普通 DistributedSampler，没有固定数据源比例或强弱监督均衡采样。
- 有 caption 无 phrases 的样本参与全局损失；有有效时间区间的样本额外参与帧级损失。帧级 loss 在有效强标签样本上求均值。
- caption 为列表时随机选一条；模型没有针对空 caption 的全局损失屏蔽。不要依赖转换脚本默认空字符串作为正式联合训练数据。
- 原版读取前 10 秒、取第一声道并重采样至 16 kHz。不是随机裁剪，也不是多声道均值。
- 标签落到 64 帧、每帧 0.15625 秒；onset/offset 都用整除截断，短区间可能落在同一帧边界而没有正帧，需要统计。
- 每条强标签样本最多处理前 20 个正短语，再从非正簇补负短语至 20 个；随机采样同簇替代表述。现有 phrase bank 为 494 行，加载为字典后有 491 个唯一簇、673 个唯一替代表述，Strong 名称已完全覆盖。
- 数据读取失败时原版会回退为零波形，故训练前需剔除缺失/损坏音频，不能仅以训练不报错判断数据有效。
- 公开 train.sh 是 4 卡、每卡 batch 128，实际全局 batch 512；论文写的是 1024。脚本覆盖 YAML 的 epochs，运行 10 轮。后续训练需明确采用哪套；普通梯度累积也不自动等价于增大全局对比学习负样本集合。

### 尚未证实、不能靠文件名猜测的部分

配置中 `_em`、`_filtered` 对应的完整处理流程和最终样本清单未随仓库提供。不能把 `_em` 直接认定为伪强标注或已缓存 embedding；当前加载器仍现场编码文本。论文总体规模只能作为数量级参考，不能替代逐源 train/val/test ID 清单。下一步优先构建并验收 AudioSet Strong + WavCaps 配对，再补 AudioSetCaps 文本和其余训练集，而非重新下载已有音频。

参考：
- [AudioSet Strong 官方文本](https://research.google.com/audioset/download_strong.html)
- [FineLAP 论文 §5.2–5.3、附录 D](https://arxiv.org/html/2604.01155v1)
- [固定版本数据配置](https://github.com/xiquan-li/FineLAP/blob/ad82bc1e242de4f56b80cf36f6d9f3009bb96c7b/config/data_config/data_eat.yaml)
- [AudioSet Strong 转换脚本](https://github.com/xiquan-li/FineLAP/blob/ad82bc1e242de4f56b80cf36f6d9f3009bb96c7b/utils/make_finelap_data.py)
- [FSD50K 预处理作者答复](https://github.com/xiquan-li/FineLAP/issues/1#issuecomment-4260280522)
- [AudioSetCaps 来源及评测排除说明](https://github.com/JishengBai/AudioSetCaps)

## COSED 数据组合试训（2026-10-29）

目的：保留 FineLAP 的 EAT + RoBERTa + Transformer 时序头和原损失，改用 COSED 的五源数据及重复采样倍数。此次是第一轮工作基线试验，不是 COSED 模型或其完整训练配方的复现。

状态更新（2026-10-30）：下载和转换全部完成。10月29日首次完整训练在第一轮内因坏 FLAC 退出，没有完整 epoch checkpoint；现先逐文件解码审计、记录并排除不可用条目，再从原初始化重启。实时状态以 `exps/finelap_cosed_mix_20261029/status.json` 为准，不能把 smoke 权重当正式实验结果。

### 数据与采样

| 数据源 | 本轮音频数 | 每轮重复次数 | 每轮样本数 |
|---|---:|---:|---:|
| AudioSet Strong train | 102,830 | 1 | 102,830 |
| FineLAP-100K | 100,000（补齐后） | 1 | 100,000 |
| AudioCaps train | 49,063 | 2 | 98,126 |
| TACOS development | 10,358（音频传输完成后） | 5 | 51,790 |
| Clotho development | 3,839 | 4 | 15,356 |
| 合计 | 266,090 个源内条目 | — | 368,102 |

合计不是跨数据源去重后的独立音频数量，AudioCaps 和 AudioSet 可能共享音频。重复次数通过重复 JSONL 路径、拼接后统一打乱实现；每个 batch 不保证固定源占比。来源为 [COSED §3.2](https://arxiv.org/html/2609.30083v1#S3.SS2)。

- AudioSet 使用官方 Strong train；253 条未匹配，380 条因 TAG 测试或 AudioCaps 验证/测试重叠被排除。时间范围零长度的 1 个事件排除，保留原始 TSV。
- AudioCaps 使用官方 train CSV，按音频合并描述；缺失和排除明细保存在 `data/cosed/source_audit.json`。AudioSet 与 AudioCaps 均排除 TAG test 和 AudioCaps val/test 的视频 ID。
- TACOS 只用 development，test 不进入训练。**论文额外排除了 1,401 条含语音训练音频，精确名单未取得；本轮使用完整 10,358 条公开 development。** 不用元数据中的 Speech 字段冒充那份排除名单。
- Clotho 只用 development，5 条参考描述中每次随机选择一条。
- 超过 10 秒的音频随机裁剪，强标签与裁剪区间相交并平移。被裁掉全部正事件的强标注音频仍保留全负帧监督。弱描述仍是原整段 caption，可能不完全适配某次裁剪。
- AudioSet 全局 caption 优先取 WavCaps；10,752 条无该描述的样本使用事件名模板。这是适配 FineLAP 全局损失的处理，不等同 COSED 的 class-query 监督。
- TACOS 自由文本短语按原文保留，未知短语沿用 FineLAP 原有回退逻辑；负采样仍使用原 phrase bank，没有加入 COSED 的跨数据源监督掩码或其固定 447 类查询机制。
- 不直接加入 DESED、UrbanSED、其余 WavCaps、普通 AudioSetCaps。FSD50K 在本轮通过发布的 FineLAP-100K 合成数据参与。

### 本轮训练设置

rabbit02 使用 GPU 2、3、4；0、1 号卡已有任务。每卡 batch 32，全局 96；BF16；10 epochs；学习率 5e-5；seed 31415926。保留 FineLAP 原优化器与调度实现，clip/frame loss 权重为 1/5；本配置未启用蒸馏。固定训练 10 轮、逐轮保存，不根据六个 benchmark 测试分数挑选 checkpoint。

从 `weights/EAT-base_epoch30_ft.pt` 和 `weights/roberta-base` 初始化，`ckpt_path` 为空，不从 `finelap_fixed.pt` 继续训练。全局 batch 与 COSED 论文的 1024、40 epochs 不同；架构、初始化、损失等也保留 FineLAP。不能把后续与发布模型的涨跌直接归因于单一数据源。

环境为 `/work/zhanghc/miniconda3/envs/finelapt/bin/python`。本项目独立使用 `dependencies/fairseq`，上游 commit 为 `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`，通过训练进程的 PYTHONPATH 指定；未修改共享 `/work/zhanghc/fairseq`。读取器指定 soundfile 后端以支持 TACOS MP3；读取失败立即报错，不默默替换成静音。

已完成：已知区间的裁剪/平移检查、全负裁剪监督检查、帧级 BCE 反向检查、三卡 batch 32 的 3 步联合训练、有限 loss 和 checkpoint 保存。短测试只是可运行性检查，不是收敛或效果结论。

### 查看与运行

所有下列命令在 rabbit02 执行。入口只有数据准备和后台串行流程两个；仍调用原 `train.py`。

```bash
cd /work/zhanghc/Myllm/FineLAP
cat exps/finelap_cosed_mix_20261029/status.json
tail -n 30 exps/finelap_cosed_mix_20261029/pipeline.log
# stage 为 training 后查看训练进度：
tail -f exps/finelap_cosed_mix_20261029/train.log
```

后台流程 `scripts/train_cosed.py` 当前已运行，不要重复启动。它依次补齐下载、验证 TACOS 官方 MD5、生成五源清单、核对 100,000/10,358 数量、检查短测试记录和空闲 GPU，再启动完整训练。断开本机或 VPN 不会主动结束服务器上的后台进程；下载失败则记录 failed，不会用残缺数据直接训练。

如流程失败，先查看 `status.json` 指向的日志。正式训练未开始时，可修复问题后用下面命令继续下载和准备；已有分片及断点会保留：

```bash
cd /work/zhanghc/Myllm/FineLAP
nohup /work/zhanghc/miniconda3/envs/finelapt/bin/python -u scripts/train_cosed.py >> exps/finelap_cosed_mix_20261029/pipeline.log 2>&1 < /dev/null &
```

训练中断后，不要直接重跑此下载入口来假装断点训练。原 `train.py --resume_from` 可从最后完整 epoch 的 `models/last_epoch.pt` 恢复；须使用已保存的同一配置和 GPU/batch 设置，未完成 epoch 会重跑。完整训练日志、配置、逐轮 checkpoint、loss CSV/JSON/曲线均在 `exps/finelap_cosed_mix_20261029/`，短测试单独放在同级 `_smoke/`。

数据转换入口为 `utils/prepare_cosed_data.py`；正式清单为 `data/cosed/*.jsonl`，数量及差异说明在 `mixture.json`。`mixture_smoke.json` 仅用于短测试，不代表正式训练覆盖。原数据、原模型、既有 benchmark 结果保留。

### 2026-10-30 重启与音频完整性检查

前次完整训练于 10月29日 17:44 启动、17:54 退出。出错的 WavCaps 文件 `Y3sSblRfEG2o.flac` 头信息异常，soundfile 解码失败，sox 解码为零帧；不能以换解码器或填静音解决。

`utils/prepare_cosed_data.py --validate-existing` 对五源清单中去重后的 257,177 个音频逐一解码，检查时长、解码帧数和有限数值。异常文件不删除，异常路径、原因、各源排除 ID 和排除前数量写入 `data/cosed/audio_validation.json`；仅重写本轮训练清单并更新 `mixture.json`，五源重复倍数保持不变。表格中原数量为排除前数量，最终数量以 mixture.json 为准。

启动入口增加 `--prepared`，复用已准备且通过审计的清单，不再重复提取音频。已有正式 checkpoint 时拒绝从头覆盖，需显式选择恢复；此次没有完整 checkpoint，因此从 EAT/RoBERTa 初始化开始。当前后台使用 `--prepared --wait-validation` 等待本次审计成功后启动，审计失败则停止，不会绕过检查。

```bash
cd /work/zhanghc/Myllm/FineLAP
cat exps/finelap_cosed_mix_20261029/status.json
tail -n 10 exps/finelap_cosed_mix_20261029/prepare_full.log
# stage=training 后：
tail -f exps/finelap_cosed_mix_20261029/train.log
```

原失败日志保留，新日志追加到同一个实验目录。启动参数显式指定 `--alpha 1 --beta 5`，其余维持 GPU 2/3/4、batch32/卡、BF16、10 epochs。

本次审计结果：257,177 个独立文件中 5 个 FLAC 时长头异常；排除 AS Strong 5 条、AudioCaps 1 条（其中一个坏文件被两源共用）。最终 AS Strong 102,825、FineLAP100K 100,000、AudioCaps 49,062、TACOS 10,358、Clotho 3,839；每 epoch 重复采样合计 368,095。完整训练于 2026-10-30 01:03 在服务器重新启动，实时进度仍以 status.json/train.log 为准。

### 训练完成（2026-10-30）

完整运行于 01:03 启动、15:08 完成，约 14 小时 5 分钟；10 epochs、每轮 3,835 步，共 38,350 次优化器更新。status.json 为 completed，GPU 2/3/4 已释放。epoch_001.pt 至 epoch_010.pt 和 last_epoch.pt 均保存；last_epoch.pt 已在 CPU 读取核查。

日志记录的 epoch 平均 loss（主进程日志，不另称全卡聚合均值）：

| Epoch | Total | Clip | Frame BCE（未乘5） |
|---|---:|---:|---:|
| 1 | 2.9798 | 2.3432 | 0.1273 |
| 5 | 1.0504 | 0.7443 | 0.0612 |
| 10 | 0.6701 | 0.4296 | 0.0481 |

训练损失逐轮下降且有限；不能仅凭训练 loss 判断泛化能力或胜过发布模型。本轮尚未计算六个 benchmark 的新权重成绩。后续固定使用第 10 轮 checkpoint，沿用既有官方推理/计分协议对比，避免按测试集挑 epoch。

最终权重：`/work/zhanghc/Myllm/FineLAP/exps/finelap_cosed_mix_20261029/models/epoch_010.pt`。逐轮数值与曲线仍使用原目录的 `metrics/train_loss.json`、`metrics/train_loss.csv`、`plots/train_loss_curve.png`。

### 第 10 轮权重测评（2026-10-30）

入口：`scripts/evaluate_cosed.py`，GPU 2，在 rabbit02 运行。依次评估 DESED Public、DESED Validation、TAG、TACOS、UrbanSED、RealDESED、MAESTRO development；Public/Validation 分开，MAESTRO development 不称公开测试集。复用 rabbit05 冻结的 core/grounding runner 和 score_benchmarks.py，其文件哈希与 checkpoint SHA256 记录在实验的 evaluation/provenance.json。原版 FineLAP 结果保留，没有覆盖。

直接使用原生 FineLAP 构造器，严格加载本次 epoch_010.pt；沿用既有查询、16kHz/第一声道预处理、窗口、后处理和计分。TAG 主结果使用未滤波 raw；TACOS 同时报告 exact/semantic，semantic 复用相同 MiniLM 编码器和原查询生成的近义区间。计分库 sed_scores_eval 0.0.4 单独安装在 dependencies/scoring，不替换训练环境依赖。音频通过 soundfile 解码。

rabbit05 GPU 均被其他任务占用，因此通过 VPN 内网只读传输公开 benchmark 音频，在 rabbit02 `/work/zhanghc/datasets/cosed-benchmarks/audio` 缓存，TACOS 已有文件优先复用。首轮包含传输时间；每份音频传输完整后原子保存，预测可断点续跑。

结果位于 `exps/finelap_cosed_mix_20261029/evaluation/`：status.json 为队列状态，evaluate.log 为日志，summary.json 为已完成数据集成绩，runs-source-audit-v2/runs-extra 内保存逐帧预测与指标。训练 loss 图仍在 plots/train_loss_curve.png，现拆分加权贡献和未加权 Frame BCE；当前没有验证 loss 曲线，也不能从训练曲线断言过拟合与否。

2026-10-30 测评顺序调整：rabbit02 的 TACOS audio.zip 已含完整 2,000 条测试音频，现从本地已校验压缩包解出测试集，核对与本轮 10,358 条训练清单音频文件名交集为 0。优先跑 tacos-test，之后恢复 DESED 等；之前已生成的 DESED 预测保留。该模型使用过 TACOS 训练集，后续与发布 FineLAP 比较时需同时标注训练数据差异。

### 首个测评结果：TACOS test（2026-10-30）

已完成全部 2,000 条音频、5,151 个音频–文本查询配对。与发布 FineLAP 的旧结果核对，manifest SHA256、PSDS1 参数和两种标注合并口径一致。训练清单与该测试集音频文件名交集为 0。分数越高越好。

| TACOS PSDS1 口径 | 发布 FineLAP | 本轮 epoch 10 | 差值 |
|---|---:|---:|---:|
| Exact caption | 0.0766 | 0.2143 | +0.1377 |
| Semantic caption（主对比口径） | 0.1239 | 0.2444 | +0.1205 |

本轮包含 TACOS development 训练监督，且数据组合、batch、训练过程不同于发布权重；不能把提升单独归因于结构或某个采样倍率，也不是未见数据集上的零样本成绩。与 COSED 论文的精确测试划分及语义编码器等仍未完全确认，不以本结果直接宣称超过 COSED。

完整指标：`evaluation/runs-extra/finelap/tacos-test/metrics.exact.json` 与 `metrics.semantic.json`；原始预测保留。队列继续 DESED Public，再跑独立 Validation 与其他数据集。


### 六套 benchmark 完整复制（2026-10-30）

将 rabbit05 `/home/zhanghc/datasets/cosed-benchmarks/` 全树复制到 rabbit02 `/work/zhanghc/datasets/cosed-benchmarks/`，包含 DESED、RealDESED、MAESTRO、UrbanSED、TAG、TACOS 的现有音频、标注、划分和下载归档。完整展开约 85.5 GB，但不再整批重复传输：先以 SHA256 核对 rabbit02 其他目录及 audio/ 缓存，内容一致的文件通过本地软链接复用；目录内部链接改为 rabbit02 相对链接。仅 benchmark_missing.files 中的缺失文件走网络，保留已有数据。原 JSON 中若包含 rabbit05 绝对路径，需要按上述根目录映射；此复制不改动冻结的测评清单。

- 后台入口：`/work/zhanghc/Myllm/FineLAP/scripts/sync_benchmarks.py`；同一入口可重新执行，使用已审计的缺失文件清单，部分文件续传，锁防止重复启动。首次等待源端 SHA256 完整清单，通过内网只读 transfermeta 模块取得 benchmark_source_hashes.json.gz；读取 benchmark_inventory.json.gz、benchmark_reuse_plan.json.gz，核对本地内容后保存 benchmark_reuse_verified.json.gz 与缺失清单；这些审计记录都集中在同一天的日志目录。已有音频目录需要保留，本地引用依赖这些目录。
- 状态：`/work/zhanghc/Myllm/FineLAP/log/20261030/benchmark_transfer.status.json`。
- 日志：`/work/zhanghc/Myllm/FineLAP/log/20261030/benchmark_transfer.log`。
- 失败自动最多尝试 3 次；完成后用 rsync dry-run 核对大小/修改时间，传输过程使用 rsync 的文件完整性校验。只有状态 completed 才表示整批传输完成。
- 源端临时只读 rsync 服务为 10.160.4.167:40919，仅允许 rabbit02 内网地址，72 小时后自动退出；需更晚重启传输时先恢复源服务。相关配置和 PID 位于 rabbit05 benchmark/log/20261030/benchmark_transfer_rsync.*。
- 本机断网不影响服务器之间传输；当前 evaluation 继续运行，40918 音频服务暂时保留。


### 测评迁移到 rabbit05（2026-10-30）

为避免把全部 benchmark 音频搬到02后才测评，将本轮 epoch010 权重传到数据所在的05。02完整训练检查点保留；inference_export/epoch_010_inference.pt 仅移除 optimizer，model/config/epoch 保持原值。SHA256 与原检查点来源见 inference_export/export.json。02测评已暂停并打包已有预测与 summary，供05续跑。

rabbit05 项目：/home/zhanghc/projects/ov-sed/FineLAP；入口 scripts/evaluate_cosed.py --bootstrap --gpu 5。结果 exps/finelap_cosed_mix_20261029/evaluation/，日志统一在 /home/zhanghc/projects/ov-sed/benchmark/log/20261030/finelap_cosed_evaluation.log。自动流程：完成传输→SHA256→迁移已有结果→与02三条DESED原始/滤波帧分数对照（rtol=1e-4, atol=2e-5，标签与时间戳完全一致）→通过后优先RealDESED，再继续UrbanSED、TAG、MAESTRO；已完成项跳过。失败会写status并停止，避免混入不一致结果。

05保留原venv，新增项目内fairseq/runtime依赖。原生模型CPU初始化389参数键成功；同结构初始化权重的GPU5显存预检（非精度结果），10s/10类/FP32：峰值allocated 0.9767 GiB、reserved 1.0332 GiB，不含所有CUDA上下文开销。训练权重输出一致性尚需传完后验证。02 benchmark全量补传继续独立运行。


2026-10-30 17:26：迁移后的六benchmark及独立DESED Validation已全部在05完成，最终summary和新增预测保存在05 evaluation目录。RealDESED PSDS1=0.0811718140，原版本地复测0.1784253678；其他完整成绩见05 summary及本地experiment_summary.md。已结束临时40918音频HTTP、40920权重导出服务；40919数据补传不受影响。


### AudioSetCaps 补充（2026-10-30）

官方 baijs/AudioSetCaps 的 Dataset/AudioSetCaps_caption.csv 已下载至 /work/zhanghc/datasets/AudioSetCaps/，版本 683e2487c9c9435fd0a8dcb04874cb377c6169d6，374,495,515 字节；SHA256 3ec45c9f20da7fa01d5e7463f027b3ea52841ae230e9a00b29f54460c6b382a7 与官方一致。复用现有 AudioSet WAV，未重复下载音频。

维护现有 utils/prepare_cosed_data.py，新增 --prepare-audiosetcaps。使用 finelapt 环境运行，输出 data/finelap_original/audiosetcaps.jsonl（弱监督音频–文本清单）、audiosetcaps_excluded.jsonl（排除记录）、audiosetcaps_audit.json（核对统计）。原 COSED 清单及训练结果未修改。1,910,920 条源描述中保留 1,908,730 条；排除与 AudioSet eval/Strong eval、AudioCaps val/test、TAG test、DESED Public/Validation 保留 ID 重叠的 2,190 条（0.115%）。没有因音频未匹配而剔除的记录。用户确认无需补回排除项。随机 100 条完整解码通过；此前 5 条损坏 FLAC 对应 WAV 也是空音频，已加入排除规则，但它们没有额外减少本次清单条数。未完成全量解码审计。

文本不含起止时间：仅要求 AudioSet unbalanced 元数据起点唯一，尚未独立核对全部波形裁剪。caption 用于片段级文本监督，不直接提供事件时间标签。本清单是公开文本重建的输入，作者未发布的 unbal_train_em_filtered.jsonl 精确过滤仍未知；其余混合数据和训练配置仍待接齐，本次未启动新训练。


### 公开代码配方重训：finelap_original_20261031

本轮独立实验目录为 /work/zhanghc/Myllm/FineLAP/exps/finelap_original_20261031，保留上一轮 COSED mix。用户已选择对齐公开 scripts/train.sh 的总 batch 512，而不是论文的 1024。计划使用 rabbit02 GPU 0、1、2、3，每卡 128，10 epochs；EAT/RoBERTa 初始化，无 FineLAP 权重续训；FP32、Adam、学习率 5e-5、1000 步 warmup、cosine、seed 31415926，clip/frame 损失权重 1:1，原版 gather 实现、20 个短语、原短语库、SpecAugment、首 10 秒/16kHz。不使用 TACOS，不使用上一轮的重复采样倍率。

维护现有入口 scripts/train_cosed.py --original 与 utils/prepare_cosed_data.py --prepare-original，配置为 config/finelap_original.yaml、config/data_config/data_original.yaml。十份来源清单统一 data/finelap_original/；mixture.json 保存实际计数、来源哈希和重建差异，excluded.jsonl 保存排除原因。

作者最终 *_em_filtered JSONL 未发布，本轮公开来源重建：AudioSet Strong 只保留有 WavCaps 自然描述的数据；DESED 训练标签配 WavCaps/AudioSetCaps 描述；UrbanSED 按论文约 8k 数量采用 train+validate，排除 test，片段描述由事件列表模板生成。AudioSet 弱监督清单去除已纳入强标注或人工 AudioCaps 的重复音频，并应用公开 WavCaps 评测黑名单；这是本地明确的重建规则，不能宣称与作者私有过滤完全相同。旧独立解码过的数据复用，新数据先检查音频头、路径和时长，未宣称全量完整解码。

直接 FP32 每卡 128 在 24GB 上溢出，因此只增加可关闭的 Transformer 激活重计算，保留实际 batch 和跨卡负样本，不用梯度累积。原实现/重计算小批次相同随机状态下损失一致，所有有效梯度最大误差 0；单卡 128、16 条强标注的两次完整更新成功，峰值 12.742 GiB。证据在 log/20261031/original_memory_probe.json。四卡试跑会额外检查 DDP、全部十源、保存 checkpoint，成功后才从编码器初始化启动正式训练。

查看进度：
```bash
cat /work/zhanghc/Myllm/FineLAP/exps/finelap_original_20261031/status.json
tail -f /work/zhanghc/Myllm/FineLAP/log/20261031/original_train.log
```
准备日志 original_prepare.log、管理日志 original_job.log、四卡测试 original_smoke.log 统一放在 log/20261031/。作业独立于 SSH/本机连接，准备或试跑失败会停止并写 failed，不悄悄减少 batch 或缺省某个数据源。每轮保存模型/优化器，曲线和训练损失保存在本实验 metrics/、plots/；source_snapshot.zip 与 provenance.json 固定实际源码和训练数据哈希。

2026-10-31 更新：音频检查期间已提前完成十源代表样本的四卡联合预检（2048 条、3 次更新，25% 强标注），损失有限、checkpoint 和曲线保存成功，进程正常结束。平均 total 10.5675、clip 9.9091、frame 0.6584；这是启动验证，不代表模型收敛。完整音频头检查结束后管理进程会对最终清单再执行门禁，随后从编码器初始化训练 10 轮。当前正式训练尚未开始；排队状态见 status.json。


2026-11-02：finelap_original_20261031 完整 10 轮训练已结束，epoch_001–010.pt 与 last_epoch.pt 均已保存。第10轮 total=2.1960779846、clip=2.1431843933、frame=0.0528935918；第9→10轮总loss下降约1.18%，仅说明训练目标趋稳，尚无验证结果支持性能收敛。最终模型 models/epoch_010.pt。

维护既有 scripts/evaluate_cosed.py，增加 --experiment 与 --gpu，直接映射02已复制的本地音频。全部7个划分共7299音频条目路径检查通过；新实验不再依赖05临时音频HTTP。命令：finelapt环境下 python scripts/evaluate_cosed.py --experiment finelap_original_20261031 --gpu 0。输出本实验 evaluation/，日志 log/20261102/original_evaluation.log。依次DESED Public、RealDESED、UrbanSED、TAG、TACOS、MAESTRO development、DESED Validation。沿用冻结runner/评分/清单，TAG官方raw与legacy、TACOS exact与semantic分别保存；MAESTRO仍是development，DESED Validation仍为既有1068条。旧训练结果和原版指标不覆盖；当前测评刚启动，尚未有完整benchmark成绩。

2026-11-02 测评完成：全部7划分预测计数与清单一致，runner哈希及清单SHA256与上一轮一致。最终PSDS1：DESED Public 0.3400774702、RealDESED 0.1903688110、UrbanSED 0.4277146532、TACOS semantic 0.1209163918（exact 0.0817727013）；TAG raw PSDS 0.6566147279（legacy 0.6682903868）；MAESTRO development mpAUC 0.5397745427；独立DESED Validation 0.3324533567。原有Validation缺100条/MAESTRO development/TACOS论文精确口径未确认的限制不变。最终成绩见本实验evaluation/summary.json，status已completed；没有改动原版权重或上一轮结果。


### 五源配方精度对照：finelap_cosed_fp32_20261102

用户要求补一轮数值精度对齐实验。本轮复制 finelap_cosed_mix_20261029 实际保存的有效配置，仅将 BF16 混合精度改为 FP32，并启用已验证与普通 FP32 小批次损失/梯度一致的激活重计算以节省显存。保留同一五份 JSONL、重复倍率 1/1/2/5/4、随机裁剪、原 phrase bank/负采样、EAT+RoBERTa 初始化、seed31415926、Adam/5e-5/1000步warmup/cosine、Clip:Frame=1:5、GPU2/3/4 每卡32（总96）、10 epochs。每轮368095记录、3835更新；不使用上一轮或本轮成品FineLAP权重续训。

入口仍是 scripts/train_cosed.py，新增 --fp32-cosed；启动命令（finelapt环境）：python scripts/train_cosed.py --fp32-cosed。新实验目录 /work/zhanghc/Myllm/FineLAP/exps/finelap_cosed_fp32_20261102，输入配置在该目录 input_config.yaml/input_data.yaml。启动时逐项验证有效配置只有上述变化、数据条目数与旧mixture一致，保存数据SHA256、源码快照及provenance。旧实验与数据未修改。正式训练前用同一完整数据清单做三卡三步预检，成功保存checkpoint后再重新初始化正式训练。

状态查看：
```bash
cat /work/zhanghc/Myllm/FineLAP/exps/finelap_cosed_fp32_20261102/status.json
tail -f /work/zhanghc/Myllm/FineLAP/log/20261102/cosed_fp32_train.log
```
管理/预检日志同目录 cosed_fp32_job.log、cosed_fp32_smoke.log。任务独立于本机/SSH连接；每轮保存checkpoint、metrics/train_loss.json/csv及plots/train_loss_curve.png。重复启动若发现正式checkpoint会拒绝覆盖，续训需明确指定checkpoint。该实验用于检验上一轮的精度影响，不是COSED模型的完整复现；没有同时改成总batch512或loss权重1:1。当前状态以status.json为准。

2026-11-02 启动确认：三卡预检通过，正式第1轮已完成12次更新；GPU2/3/4各约8.3GB且利用率100%。读取正式保存的config.yaml与旧实验逐字段比较，仅precision bf16→fp32及activation_checkpointing新增为true两处差异。初始稳定步时约2.3秒，10轮粗估约24–28小时，需以实际后续速度更新；不是已完成。
