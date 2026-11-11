# 当前评测 Runner

日常从 `scripts/run_benchmark.py` 启动，不需要手动挑选这里的文件。

| 文件 | 使用范围 | 结果目录 |
|---|---|---|
| core.py | DASM、FineLAP、MGA-CLAP、OpenFLAM；DESED、RealDESED、UrbanSED | runs-source-audit-v2 |
| core_flexsed.py | FlexSED；DESED、RealDESED、UrbanSED | runs-source-audit-v2 |
| grounding.py | 五个模型；TAG、TACOS、MAESTRO | runs-extra |
| archive/ | 四个早期版本，保留追溯 | 不作为当前入口 |

## 为什么保留两个 core 文件
两个文件仅 FlexSED 采样率不同。早期 core.py 中 FlexSED 的采样率存在错误，因此 FlexSED 必须路由到 core_flexsed.py；统一启动脚本已处理。其他四个模型使用 core.py，与已有预测一致。请不要直接用 core.py 跑 FlexSED。

源码全文和样本清单共同构成预测身份校验。当前只移动位置，保留源码字节不变，避免已有预测失效；没有为了合并文件而跳过校验或修改旧 identity.json。40 组模型/数据集的身份校验已通过。

## 后续修改
目录和启动方式调整不改变模型计算。若修改模型推理、预处理或后处理，应建立明确的新实验版本与输出目录，不覆盖本轮预测，也不修改旧身份校验来强行续跑。

这些 runner 的 complete 代表命令执行结果；grounding.py 对 TACOS、MAESTRO 目前主要保存预测，实际计分状态以 metrics.json 为准。DESED 的标准 PSDS1 汇总入口为 scripts/score_benchmarks.py。

当前统一启动脚本会在 grounding.py 完成 TACOS / MAESTRO development 推理后，继续调用 scripts/score_benchmarks.py 计分；冻结 runner 本身未改动。直接运行 grounding.py 仍会写入旧的待计分状态，因此日常请使用统一启动入口。

2026-10-19 修订：冻结 runner 不修改。日常仅用 scripts/run_benchmark.py；TAG 和 TACOS 的当前计分分别查看 metrics.official.json、metrics.semantic.json，旧 metrics.json 仅供历史追溯。统一入口同时保留 legacy / exact 配置。
