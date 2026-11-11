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


## 本轮 FineLAP COSED mix 测评（2026-10-30）

权重来自rabbit02完成的10轮训练，exps/finelap_cosed_mix_20261029/inference_export/epoch_010_inference.pt只移除optimizer，参数值不变。原版模型权重和原benchmark成绩不覆盖。

启动/断点继续：`/home/zhanghc/venvs/finelap/bin/python -u scripts/evaluate_cosed.py --bootstrap --gpu 5`（在本项目目录执行）。这会补齐导出包、校验SHA256、导入02已有结果，并通过3条DESED预测对照后才继续完整测评。GPU可手工指定，启动前检查至少8GB可用显存。首次已有独立权重传输时可指定 --transfer-pid，避免并发写同一文件。

结果与状态：exps/finelap_cosed_mix_20261029/evaluation/summary.json、status.json；一致性结果handoff_check.json；来源provenance.json。默认优先RealDESED，再UrbanSED、TAG、MAESTRO，已有完整成绩跳过，中间npz断点续跑。日志统一 /home/zhanghc/projects/ov-sed/benchmark/log/20261030/finelap_cosed_evaluation.log。依赖隔离在dependencies/runtime和dependencies/fairseq，不更改原benchmark的HF加载方式。

显存预检：同结构初始化权重、10秒音频/10类查询/FP32，PyTorch allocated峰值0.9767GiB、reserved峰值1.0332GiB，CUDA上下文另计；该预检不代表本轮训练权重的精度检查。


### 2026-10-30 测评完成

17:26全部完成：Public PSDS1=0.2323945227；RealDESED=0.0811718140；UrbanSED=0.2253738922；TAG raw grounding PSDS=0.6268471959；TACOS semantic PSDS1=0.2444138296；MAESTRO development mpAUC=0.4188379804；DESED Validation=0.2234121207（1068/1168条）。7项清单与原benchmark一致、计分参数与原版复测一致，预测计数全部匹配输入。跨机3条DESED raw/scores误差均0，检查显存allocated峰值0.97725GiB。原版指标及原始预测保留。
