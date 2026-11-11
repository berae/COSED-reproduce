# 源码核查记录（2026-10-16）

本次检查的是已接入的 DESED、RealDESED、UrbanSED。TAG、MAESTRO、TACOS 尚未接入，不能视为已核查完成。与官方单模型代码一致，不等于与 COSED 未公开的全部配置一致。没有根据测试分数挑选提示词、阈值或滤波长度。

| 模型 | 对照源码 | 已核查与处理 | 未确定事项 |
|---|---|---|---|
| FlexSED | api.py、src/val.py、src/prepare_clap.py、src/dataset/tsed_val.py、src/utils/sed.py | 16 kHz、峰值归一化、CLAP 查询、sigmoid、25 fps 一致。修正遗漏的 9 帧中值滤波。 | api.py 将事件词首字母大写，而 prepare_clap.py 保留原大小写；当前遵循公开推理 API。COSED 的精确查询列表未获得。官方 AS 验证 alpha_st=0；本项目跨数据集 PSDS1 按 COSED 与其他官方 SED 脚本用 alpha_st=1，不能套用 AS 指标。 |
| DASM | 官方 detect_any_sound.ipynb、src/models/detect_any_sound/detect_any_sound_htast.py、src/preprocess/feats_extraction.py、src/codec/encoder.py | text_query/as_full 权重、sound of 小写查询、407 个 base queries 与 attention mask、temp_w=0.5、片段级门控、32 kHz、50 fps、16 帧滤波一致。 | 官方 notebook 是演示入口，没有 RealDESED 专用标签映射或 COSED 的窗口边界细则。分数差异原因未确定。 |
| FineLAP | evaluate.py::validate_sed、data_handling/sed_dataset.py、utils/eval_utils_sed.py、models/feature_extractor.py、HF modeling_finelap.py | 修正为原始类别字符串，不再统一替换下划线/小写；修正为第一声道与 torchaudio 重采样；短音频交由 mel 补齐，不先补 waveform。3 帧滤波、局部温度与偏置一致。新版结果独立写到 runs-source-audit。 | 当前用官方 HF 移植权重，strict 加载无缺失/多余键；与原 fairseq 路径尚无逐元素前向等价验证。 |
| MGA-CLAP | zero_shot_sed.py、models/ase_model.py | 类别下划线换空格并小写、不加模板；MSC 文本向量与归一化帧余弦；32 kHz、短音频重复到 10 秒、3 帧滤波一致。重采样改为官方 torchaudio 后重跑。 | COSED 精确 split 与长音频尾窗约定未获得。 |
| OpenFLAM | test/local_example.py、src/openflam/hook.py、src/eval/eval_local.txt | 48 kHz librosa mono、unbiased 相似度、3 帧滤波一致。执行官方 sanity_data/sanity_gt 特征对照，详细误差在 openflam-official-sanity.json。 | 仓库只提供局部推理示例与结果文本，没有完整 DESED/RealDESED benchmark 标签模板及计分入口。不能证明 COSED 使用同一权重/查询/划分；不为贴近分数试选模板。 |

## 共用计分修复
- UrbanSED 同类重叠/相接标注先合并成并集，满足新版 sed_scores_eval 输入要求；原始标注文件未修改。
- 时间戳按计分库的 6 位小数精度处理，剔除舍入后为零长度的最后一帧；原始预测保留。
- 使用单进程计分，避免工作进程异常后父进程等待。
- PSDS1 DTC=GTC=0.7，alpha_ct=0，alpha_st=1，max_eFPR=100/h。没有改成准确率，也没有用已知真实类别过滤预测。
- DESED validation 1068/1168，public 692/692，两个 split 分开报告；RealDESED test 1007/1007，UrbanSED test 2000/2000。
- 保留无事件样本。它们是误报统计的一部分；不能照搬某些官方脚本删除无事件样本而不说明。
- 10 秒窗/5 秒步长和重叠平均来自 COSED 论文；尾窗、不同原生帧率下的对齐细则是本地实现，尚未得到作者实现核对。

## 结果位置
首轮预测 runs；FlexSED 修正结果为各目录 metrics-source-audit.json（优先于旧 metrics.json）；FineLAP/MGA-CLAP 修正版在 runs-source-audit。原版不覆盖。audit-queue*.json 为重跑进度。

## 固定源码版本
- DASM: c3e883d0fbeaf7031b467d45a3c46a88a76c00b6
- FineLAP: ad82bc1e242de4f56b80cf36f6d9f3009bb96c7b
- FlexSED: 89aac3351e807de57c4fe3148d982922a3318e46
- MGA-CLAP: 48ca5a5cd22cd34427e118bd8cf332090ec54770
- OpenFLAM: 855d9e2133788a17360a4a9ad6feb0f4a95a804b

## 固定长度片段修正
源码核查发现，DESED/UrbanSED 官方入口均限制为前 10 秒；首轮把部分多出几毫秒的音频误当长音频额外滑窗。最终 runner-source-audit-v2.py 对三个固定长度 split 截至 10 秒，计分真值同时截到实际评测时长（原标注不改），RealDESED 保留完整时长与 10/5 滑窗。该修正对全部五个模型统一应用，后台重跑至 runs-source-audit-v2。之前 runs-source-audit 是中间检查，不能混入最终汇总。数据头部统计见 source-audit-data.json。

## FlexSED 补跑修复
修正 median=9 的字符串替换误将 FlexSED 采样率 16000 改成 96000，导致 v2 四项在首条推理前报错、未生成预测。已用 runner-flexsed-fixed.py 恢复 16000，保留失败身份记录并单独补跑。其他模型采样率未被修改。
