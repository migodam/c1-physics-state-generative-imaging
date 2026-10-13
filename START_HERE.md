# 最新：B1 × C1 Joint P2

[共同图库](https://migodam.github.io/b1-probabilistic-maxwell-imaging/research/joint_p2/) · [C1独立报告](p2/C1_P2_RESEARCH_REPORT_ZH.md) · [裁决](p2/C1_P2_DECISION.md) · [权重与原始记录](https://github.com/migodam/c1-physics-state-generative-imaging/releases/tag/joint-p2-20261014)

共同四维vector-Mie posterior上，小型蒸馏MDN的事件MAE约.0132；完整协方差并未稳定胜出事件质量，连续条件CDF约.142，完整posterior目标未达到。下面P1等原结果保留。

# C1 阅读入口

历史：[C1-P1 密度修复报告](p1/C1_P1_RESEARCH_REPORT_ZH.md)，[决定](p1/C1_P1_DECISION.md)，[结果](p1/C1_P1_RESULTS.csv)。旧C1报告与失败证据保留。

这是 2026-10-13 完成的研究与最小实验公开记录。无需预定义 topology 类别，直接生成连续材料分布；当前 physics-state/memory 未显示稳定增益。

建议阅读顺序：

1. [中文研究报告](C1_RESEARCH_REPORT_ZH.md)：物理含义、连续概率定义、公平对照、结果及失败。
2. [方向修订](C1_DIRECTION_V2.md)：保留、修改、终止的原方案及竞争性架构。
3. [架构与实验规范](C1_ARCHITECTURE_EXPERIMENT_SPEC_V2.md)：统一材料支持、候选状态、训练与评价。
4. [下一阶段判断](C1_NEXT_DECISION.md)：负结果、值得继续的简单密度和参考修复方向。
5. [计算账本](C1_RESOURCE_LEDGER.json)、[逐 seed](results/SUMMARY_METRICS.csv)、[逐 parent](results/PARENT_METRICS.csv) 与 [原始记录](results/)：完整费用、失败和数值限制。

核心边界：约 5 个百分点事件概率误差，尚未校准；三 seed 未建立 state/memory 稳定优势。U0 是二维连续原型，P0 是固定 13-cell vector-DDA、两材料系数原型；一般未知拓扑与真实成像没有运行。不同条件状态不被当作独立测量，不重复 likelihood，不将 residual softmax 当作 posterior。

原研究包在 [inputs/source_package/C1_Research_Package](inputs/source_package/C1_Research_Package/) 中保留作来源。其 topology-stratified 推荐由用户指定的 V2 方向覆盖。发布不触发原包中的执行任务。

发布是已有结果快照，没有新增 gate、SHA256 检查、smoke test 或实验重跑。历史命名 C1_GATE_DECISION.json 和 DELIVERY_QA.json 仅是已完成记录；文件存在不构成新的审批要求。详见 [发布说明](PUBLICATION_NOTES.md)。
