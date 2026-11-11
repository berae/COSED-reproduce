# DESED 标准 PSDS1

仅使用标准 PSDS1，作为 DESED 当前唯一主指标。自定义 0.3/0.3 与 PSDS2 对照结果、对应旧日志已经删除。

## 计分配置
- DTC=0.7，GTC=0.7。
- alpha_CT=0（不启用交叉触发惩罚），alpha_ST=1。
- max_eFPR=100/小时。
- 库：sed_scores_eval 0.0.4。
- 参数依据：https://dcase.community/challenge2023/task-sound-event-detection-with-weak-labels-and-synthetic-soundscapes

## 总体结果
| 模型 | Public（692条） | Validation（1068条） |
|---|---:|---:|
| dasm | 0.4686 | 0.4194 |
| finelap | 0.3238 | 0.3244 |
| flexsed | 0.1605 | 0.1453 |
| mga-clap | 0.2637 | 0.2807 |
| openflam | 0.0241 | 0.0538 |

## 范围与复算
Validation 缺失 100/1168 条；两个 split 分开计分。这是标准指标在当前本地样本上的结果，不宣称完整官方挑战成绩，也不宣称完全复现 COSED。模型训练数据暴露未控制，不作纯零样本排名。
复用 runs-source-audit-v2 的预测及各模型已有后处理；同类重叠真值取并集，真值截断至预测时长，剔除时间戳舍入后的零长度帧。原始数据和预测未修改。10 项 PSDS1 均与此前标准分数相差小于 1e-9。

```bash
cd /home/zhanghc/projects/ov-sed/benchmark
~/venvs/finelap/bin/python scripts/score_benchmarks.py
```

- scripts/score_benchmarks.py：直接列出标准参数，唯一计分入口。
- analysis/desed/official_psds.json：总体与逐类别 PSDS1。
- log/20261016/desed_standard_psds1.log：本次运行日志。
