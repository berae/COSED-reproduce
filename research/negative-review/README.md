# 逐条音频：真值与五模型实际输出

当前主入口：[打开查看页](http://127.0.0.1:18765/)。原 review.html 也已更新。

选择 Benchmark、音频和类别，页面显示完整音频、全部同类 GT、五模型完整时间轴分数，以及每一段 GT 内的加权平均分数和峰值。没有“未触发规则”状态，也不把 GT 辅助最佳区间当作实际输出。每条音频一道题，覆盖六套现有清单的 6,231 条音频。

猫叫例子 -0H7vrzA80U_4_14 的五段真值和全部逐帧分数已嵌入页面，可直接打开查看。其他音频从 rabbit05 已保存预测按需读取，完整音频按需转换为 16 kHz 单声道 WAV 播放，需要 VPN 与 SSH 连接。五个模型纵轴独立，分数值不直接跨模型比较。人工笔记沿用按音频保存及导出。

服务在服务器仅监听 127.0.0.1:18765，本机通过 SSH 转发访问，不占 GPU。

```bash
# rabbit05：读取已有预测与音频
OMP_NUM_THREADS=4 ~/venvs/finelap/bin/python ~/projects/ov-sed/benchmark/scripts/negative_review.py --serve
# 本机：建立通道
ssh -N -L 127.0.0.1:18765:127.0.0.1:18765 rabbit05-vpn
```

重建此主页面使用 `--render-output-view`。旧 candidates.json / CSV 与分类定义保留为辅助诊断资料，不再代替实际输出。下面是历史筛查方法说明。

---

# 六个 Benchmark 失败候选复核

更新：2026-10-22。[可筛选列表、试听与曲线](review.html) · [全量 CSV](candidates.csv)。只读取既有预测，未改权重、推理、主表或正式评分。

## 范围与使用

扫描五模型六套评测数据，DESED 仅 Public，MAESTRO 使用有真值的 Development，并非测试集。FineLAP TAG 用 raw；TACOS 使用当前 MiniLM-L6-v2、同文件余弦 >0.8 直接邻居真值合并，未宣称作者完全等价。

网页可按数据集、模型、类型和查询筛选。每条有稳定 ID、音频路径、时间位置与真值。代表样本附 16 kHz 单声道试听和连续响应曲线，最长 16 秒；其余保留完整服务器路径。笔记可在浏览器保存并导出。尚未人工听音确认。

## 定义

主列表每条音频只出现一次，点开后合并展示全部模型/查询/事件的诊断细项。旧 92,954 是修正前的细项数，不是题目数；已排除跨 GT 边界帧触发的背景峰值标记。全量扫描并列出符合规则的候选，不等于穷尽官方失败集合。PSDS 跨阈值计分，没有固定操作点就没有唯一的 FP/FN 清单。候选数不用于模型排名；一条可有多个类型，同音频可有多模型、多事件记录。

| 类型 | 筛选依据 | 限制 |
|---|---|---|
| 目标/背景排序倒置 | 单 GT 事件与同音频非目标时间的时长加权 AUC <0.5 | 排除其他同类事件；不是漏检率 |
| 背景峰值超过目标 | 完全处于全部同类 GT 之外的帧，其最高分 > 当前事件最高帧分数 | 跨 GT 边界的帧不参与背景峰值；高峰仍只是复核线索，不等于误报 |
| 连续区间覆盖不足 | 没有单连续区间同时满足覆盖与纯度要求；最佳 IoU 区间覆盖不足 | 是形态诊断，不是官方漏检 |
| 区间外扩/拖尾 | 同上，最佳 IoU 区间纯度不足 | 包括提前启动、拖尾、桥接相邻事件 |
| 区间覆盖与纯度均不足 | 同上，两项都不足 | 可能错位，暂不归因 |
| 断续响应 | 单区间不可行且最佳 IoU 所在分数水平有多个区间与 GT 相交 | 官方匹配可累积多个检测区间，不一定失败 |
| 无标注目标高响应 | 整条无该标签，最高分 ≥ 同模型同类阳性音频最高分中位数 | 测试集排序诊断，不是部署阈值；无标注不等于声音不存在 |
| 低响应阳性秒段 | MAESTRO 正秒段分数 ≤ 同模型同类负秒段分数中位数 | 沿用软标签 >0.5、一秒加权；相邻候选秒段合并 |
| 高响应阴性秒段 | MAESTRO 负秒段分数 ≥ 同模型同类正秒段分数中位数 | 不评价亚秒边界，不是误报率 |

连续区间枚举全部唯一分数对应的超水平连通区间。覆盖=与单个 GT 的交集/GT 时长；纯度=交集/连续区间时长。TAG 参考 0.5，其余事件任务参考 0.7；只借用比例作诊断，未修改官方计分。最佳 IoU 区间由测试 GT 辅助选择，是逐事件乐观诊断，不是统一阈值。AUC 用原始网格精确时长，部分边界帧可以同时有正负权重，不等同此前 100 ms 网格的片内 AUC。

TAG/TACOS 只扫描官方清单已有音频–查询配对，未另造 absent-query 负例。全阳性样本无背景，AUC 留空。所有结论待听辨，不预先标注具体原因。

## 分数据集完整列表

| Benchmark | 总音频题数 | 诊断细项（辅助） | 有候选的音频题数 | 逐音频清单 |
|---|---:|---:|---:|---|
| DESED Public | 692 | 6979 | 634 | [CSV](desed-public.csv) |
| RealDESED Test | 1007 | 14188 | 995 | [CSV](realdesed-test.csv) |
| UrbanSED Test | 2000 | 19667 | 1998 | [CSV](urbansed-test.csv) |
| TAG Test | 483 | 2505 | 349 | [CSV](tag-test.csv) |
| TACOS Test | 2000 | 31373 | 1813 | [CSV](tacos-test.csv) |
| MAESTRO Development | 49 | 10429 | 49 | [CSV](maestro-development.csv) |

## 按音频分类数量（同一题可有多种问题）

| Benchmark | 类型 | 音频题数 |
|---|---|---:|
| DESED Public | 目标/背景排序倒置 | 363 |
| DESED Public | 背景峰值超过目标 | 458 |
| DESED Public | 连续区间覆盖不足 | 271 |
| DESED Public | 区间外扩/拖尾 | 437 |
| DESED Public | 区间覆盖与纯度均不足 | 229 |
| DESED Public | 断续响应 | 161 |
| DESED Public | 无标注目标高响应 | 370 |
| RealDESED Test | 目标/背景排序倒置 | 538 |
| RealDESED Test | 背景峰值超过目标 | 889 |
| RealDESED Test | 连续区间覆盖不足 | 637 |
| RealDESED Test | 区间外扩/拖尾 | 760 |
| RealDESED Test | 区间覆盖与纯度均不足 | 316 |
| RealDESED Test | 断续响应 | 489 |
| RealDESED Test | 无标注目标高响应 | 909 |
| UrbanSED Test | 目标/背景排序倒置 | 1101 |
| UrbanSED Test | 背景峰值超过目标 | 1606 |
| UrbanSED Test | 连续区间覆盖不足 | 1179 |
| UrbanSED Test | 区间外扩/拖尾 | 1476 |
| UrbanSED Test | 区间覆盖与纯度均不足 | 583 |
| UrbanSED Test | 断续响应 | 954 |
| UrbanSED Test | 无标注目标高响应 | 1748 |
| TAG Test | 目标/背景排序倒置 | 245 |
| TAG Test | 背景峰值超过目标 | 313 |
| TAG Test | 连续区间覆盖不足 | 89 |
| TAG Test | 区间外扩/拖尾 | 214 |
| TAG Test | 区间覆盖与纯度均不足 | 37 |
| TAG Test | 断续响应 | 65 |
| TACOS Test | 目标/背景排序倒置 | 1706 |
| TACOS Test | 背景峰值超过目标 | 1075 |
| TACOS Test | 连续区间覆盖不足 | 842 |
| TACOS Test | 区间外扩/拖尾 | 908 |
| TACOS Test | 区间覆盖与纯度均不足 | 486 |
| TACOS Test | 断续响应 | 750 |
| MAESTRO Development | 低响应阳性秒段 | 47 |
| MAESTRO Development | 高响应阴性秒段 | 49 |

## 试听起点

代表样本优先选严重现象并尽量覆盖不同模型与录音；非随机抽样，不估算错误率。

### DESED Public

| 案例 ID | 模型 | 查询 | 时间（秒） | 类型 | 试听 |
|---|---|---|---|---|---|
| 695e4fd3d188f85c | dasm | Dog | 8.382–8.681 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/695e4fd3d188f85c.wav) |
| d055b41e9df73e12 | finelap | Dishes | 7.515–7.765 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/d055b41e9df73e12.wav) |
| f9c6168e6dfd9a7d | flexsed | Dog | 3.029–3.318 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/f9c6168e6dfd9a7d.wav) |
| e2ea6d81c4bd663e | mga-clap | Dishes | 9.118–9.368 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/e2ea6d81c4bd663e.wav) |
| 5421528b5cf8e435 | openflam | Alarm_bell_ringing | 8.323–9.995 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/5421528b5cf8e435.wav) |
| 574fb07ad4c2f48d | dasm | Alarm_bell_ringing | 0.000–0.255 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/574fb07ad4c2f48d.wav) |
| 5c068299d3fc2bcd | mga-clap | Speech | 9.153–9.403 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/5c068299d3fc2bcd.wav) |
| 98d634bec0f44b5c | openflam | Alarm_bell_ringing | 5.201–5.451 | 目标/背景排序倒置；区间覆盖与纯度均不足 | [wav](listen/98d634bec0f44b5c.wav) |
| 589393b75ba97553 | finelap | Speech | 5.692–7.445 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/589393b75ba97553.wav) |
| 129a79bb9a188083 | flexsed | Frying | 8.480–8.520 | 无标注目标高响应 | [wav](listen/129a79bb9a188083.wav) |
| dca5fcb16a6357fd | mga-clap | Alarm_bell_ringing | 3.125–3.438 | 无标注目标高响应 | [wav](listen/dca5fcb16a6357fd.wav) |

### RealDESED Test

| 案例 ID | 模型 | 查询 | 时间（秒） | 类型 | 试听 |
|---|---|---|---|---|---|
| ba2c9b5be74771a8 | dasm | light_switch | 34.864–35.234 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/ba2c9b5be74771a8.wav) |
| c40afed9c2bcb201 | finelap | light_switch | 10.702–11.220 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/c40afed9c2bcb201.wav) |
| 36bf11c392ca089d | flexsed | light_switch | 8.584–9.207 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/36bf11c392ca089d.wav) |
| e611f088d5a9610a | mga-clap | light_switch | 3.212–3.495 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/e611f088d5a9610a.wav) |
| 35f7fa7b3c40c31e | openflam | phone_ringing | 22.341–24.798 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/35f7fa7b3c40c31e.wav) |
| 5f87bf0ff0458f51 | dasm | phone_ringing | 6.553–6.802 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/5f87bf0ff0458f51.wav) |
| 0c7c0c0f8d0e4920 | finelap | phone_ringing | 13.071–13.258 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/0c7c0c0f8d0e4920.wav) |
| 9d13731507f62db2 | flexsed | phone_ringing | 6.553–6.802 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/9d13731507f62db2.wav) |
| e12f22620116611e | mga-clap | phone_ringing | 12.748–13.488 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/e12f22620116611e.wav) |
| bd5f057b688744ae | openflam | cutlery_dishes | 2.899–3.232 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/bd5f057b688744ae.wav) |
| 21302b9ee6af8509 | finelap | keychain | 28.350–31.281 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足；断续响应 | [wav](listen/21302b9ee6af8509.wav) |
| 5ced88ff99d576a6 | flexsed | window_open_close | 9.720–9.760 | 无标注目标高响应 | [wav](listen/5ced88ff99d576a6.wav) |
| 622dbbf0d0e6d6b4 | mga-clap | light_switch | 25.625–25.938 | 无标注目标高响应 | [wav](listen/622dbbf0d0e6d6b4.wav) |

### UrbanSED Test

| 案例 ID | 模型 | 查询 | 时间（秒） | 类型 | 试听 |
|---|---|---|---|---|---|
| c4db0c5456c18a31 | dasm | engine_idling | 4.289–4.830 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足；断续响应 | [wav](listen/c4db0c5456c18a31.wav) |
| 6e22e0fab808cfe9 | finelap | dog_bark | 2.090–3.041 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/6e22e0fab808cfe9.wav) |
| f1228ec91bff6760 | flexsed | engine_idling | 5.038–5.638 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足；断续响应 | [wav](listen/f1228ec91bff6760.wav) |
| f482e1a6be18440b | mga-clap | air_conditioner | 2.598–3.084 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/f482e1a6be18440b.wav) |
| d1939dcd1a54f0bb | openflam | air_conditioner | 8.152–9.657 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足 | [wav](listen/d1939dcd1a54f0bb.wav) |
| 757125478889a755 | dasm | engine_idling | 6.754–7.873 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/757125478889a755.wav) |
| 17aa5b6d09f9d376 | flexsed | engine_idling | 2.359–3.084 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/17aa5b6d09f9d376.wav) |
| 82a63932d0568d3d | mga-clap | dog_bark | 3.184–3.735 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/82a63932d0568d3d.wav) |
| 4ea7dc27f275e903 | openflam | drilling | 3.798–4.340 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/4ea7dc27f275e903.wav) |
| 62dfad6a1a4296e8 | finelap | siren | 4.969–5.910 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/62dfad6a1a4296e8.wav) |
| 621ab9c3880abc70 | flexsed | drilling | 6.880–6.920 | 无标注目标高响应 | [wav](listen/621ab9c3880abc70.wav) |
| e0ac45716d2f368c | mga-clap | air_conditioner | 0.000–0.312 | 无标注目标高响应 | [wav](listen/e0ac45716d2f368c.wav) |

### TAG Test

| 案例 ID | 模型 | 查询 | 时间（秒） | 类型 | 试听 |
|---|---|---|---|---|---|
| 0603dd77875385ea | dasm | becomes louder | 3.321–6.019 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/0603dd77875385ea.wav) |
| cd3bf320bad2a881 | finelap | growling | 2.877–3.144 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/cd3bf320bad2a881.wav) |
| abd672fece39859a | flexsed | the dog 's nails scratch a hard surface | 0.953–1.461 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/abd672fece39859a.wav) |
| 1084f17592e05081 | mga-clap | a man speaking | 4.378–4.700 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/1084f17592e05081.wav) |
| 2027aaf805549e10 | openflam | people speak in the background | 7.563–9.999 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足 | [wav](listen/2027aaf805549e10.wav) |
| 390b86c37b6de9fc | dasm | sirens | 0.018–1.269 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/390b86c37b6de9fc.wav) |
| df99e6e2bbf3208f | mga-clap | a person speaking | 5.281–5.534 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/df99e6e2bbf3208f.wav) |
| 254a44fb74f6efab | openflam | a kid talking | 3.650–4.550 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/254a44fb74f6efab.wav) |
| 9eb65bbb23e36be3 | finelap | growling | 5.554–6.317 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足；断续响应 | [wav](listen/9eb65bbb23e36be3.wav) |

### TACOS Test

| 案例 ID | 模型 | 查询 | 时间（秒） | 类型 | 试听 |
|---|---|---|---|---|---|
| 9e1f4460746f3b92 | dasm | A hollow cylindrical metallic object is spinning. | 27.213–27.693 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/9e1f4460746f3b92.wav) |
| a03d6367b8ee27d0 | finelap | Pulsed, wave-like sound with pauses in between. | 21.595–21.765 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/a03d6367b8ee27d0.wav) |
| c467d1e225f7b7a4 | flexsed | A person is breathing softly. | 16.782–17.153 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足；断续响应 | [wav](listen/c467d1e225f7b7a4.wav) |
| 516a580329f260f1 | mga-clap | A short light tapping on wood. | 19.456–19.586 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/516a580329f260f1.wav) |
| 3c159f596e6e5c1f | openflam | A car passes by. | 23.959–26.534 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/3c159f596e6e5c1f.wav) |
| c7356b290ed7135d | dasm | Pulsed, wave-like sound with pauses in between. | 13.595–14.164 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/c7356b290ed7135d.wav) |
| 225724629e63e7ca | flexsed | A person breathes and exhales softly several times. | 16.782–17.153 | 目标/背景排序倒置；背景峰值超过目标；区间外扩/拖尾 | [wav](listen/225724629e63e7ca.wav) |
| 3b5bf41c4e47f4c6 | mga-clap | Echoing, resonant background noise. | 6.519–7.154 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/3b5bf41c4e47f4c6.wav) |
| 38c8a04d59157474 | openflam | A bass drum is kicked. | 14.418–14.829 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足 | [wav](listen/38c8a04d59157474.wav) |
| 7af7e42eea16a15d | dasm | A noisy drill operates, fading at the end. | 2.599–3.768 | 目标/背景排序倒置；背景峰值超过目标；区间覆盖与纯度均不足；断续响应 | [wav](listen/7af7e42eea16a15d.wav) |
| 57ad27d7229cb93d | finelap | A drum kit is being played with heavy synthesizer effects. | 0.025–3.174 | 目标/背景排序倒置；背景峰值超过目标；连续区间覆盖不足；断续响应 | [wav](listen/57ad27d7229cb93d.wav) |

### MAESTRO Development

| 案例 ID | 模型 | 查询 | 时间（秒） | 类型 | 试听 |
|---|---|---|---|---|---|
| 24bff18f064682e7 | dasm | footsteps | 0.000–1.000 | 低响应阳性秒段 | [wav](listen/24bff18f064682e7.wav) |
| 0c56c94914dbfe18 | finelap | footsteps | 90.000–91.000 | 低响应阳性秒段 | [wav](listen/0c56c94914dbfe18.wav) |
| 895d0c139a8b9c79 | flexsed | large_vehicle | 0.000–54.000 | 高响应阴性秒段 | [wav](listen/895d0c139a8b9c79.wav) |
| d15107d64a25588e | mga-clap | brakes_squeaking | 4.000–5.000 | 高响应阴性秒段 | [wav](listen/d15107d64a25588e.wav) |

## 复算

在 rabbit05 benchmark 根目录运行：

```bash
OMP_NUM_THREADS=4 ~/venvs/finelap/bin/python scripts/negative_review.py
# 仅更新页面与试听
~/venvs/finelap/bin/python scripts/negative_review.py --render-only
```

覆盖见 summary.json；完整候选见 candidates.json / CSV。旧三个划分每类五条的筛查由本轮替代。原始预测和旧协议成绩未改。
