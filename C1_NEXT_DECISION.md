# C1 下一阶段裁决

2026-10-13。**保留统一材料概率生成；NO-GO扩大当前physics-state/GRU模型；先修复简单density的概率质量与tail稳定性。**

无需拓扑标签的共同材料support已恢复多个参数区域，P0简单MDN/state/memory的mass-weighted region recall约98–99%。但事件MAE约5%，P0 90%区间覆盖约83–85%，所有primary配置未达预设概率目标。这不是已校准的真实posterior，也不是未知任意拓扑成像。

## 真实state收益

同数据、近似同参数、同MDN family的memory相对measurement-MDN，三个seed的reference KL为：

|seed|直接MDN|MDN+memory|差(memory-direct)|parent paired95%CI|
|---|---:|---:|---:|---|
|2610131|1.080618|1.003493|-0.077125|[-0.127676,-0.022480]|
|2610132|1.017235|0.872362|-0.144873|[-0.179239,-0.114427]|
|2610133|0.892477|0.996573|+0.104096|[+0.014620,+0.194517]|

有局部收益，但第三seed有显著反向结果，平均改善约3.94%。单state的KL差为-.00336/-.02044/+.08255，亦不稳定。观测truth-log-score的memory差CI均跨0。不能把更不稳定的direct flow tail失败当成支持state路线的强证据；原validation-selected baseline与same-family控制均保留。

## 最值得继续的统一架构

下一小阶段优先**support-aware统一条件密度**：保留未标注MDN与有界conditional flow作为认真对手，先解决within-region相关形状、边界tail与校准。可以测试full-covariance components或显式有密度的prior-defensive混合，但两者都要用proper log score及精度控制；prior混合会改变fast q，不能假称免费正确posterior。下一representation用统一多分辨率被动材料场及非零fine-scale随机尾，不给generator输入孔洞数。

state只作为可选计算condition，优先测试少量任意candidate响应对比/gradient/probes。当前数据没有理由优先Transformer、Diffusion或更多GRU history。只有简单模型概率达到要求后，才值得比较每个production hypothesis自己的physics refresh与memory，并为完整Jacobian或有效路径概率/target-preserving correction付费。当前frozen shared pilot不等于已经实现这种particle-specific过程。

## 最小下一实验与停止

先在同类二维连续参考上，用一个冻结的tail-sensitive开发panel比较原MDN、改进density、以及它们的state消融；全部同数据、近似同容量，物理成本和reference误差一起报告。此时只能开展新的已记录小预算阶段，不能把已暴露test重新称blind。若进入几何/材料场扩展，先给与新prior一致的参考；B1三模板参考不能直接充当一般field真值。

推进条件是：至少三个seed上同精度cost/概率质量的稳定改善，或同总成本下proper-score/coverage的明确改善；不只best-of-K、某seed、residual或神经FLOPs。这个判断不等于研究方向永远失败。

**GO_LIMITED_UNIFIED_DENSITY_AND_REFERENCE_REPAIR**：继续低成本统一材料分布与支持研究。

**NO_GO_SCALE_PHYSICS_STATE_MEMORY**：不扩大当前网络、history或GPU训练，不宣称Maxwell状态已有稳定架构增益。

**HOLD_GENERAL_TOPOLOGY_AND_REAL_IMAGING**：未知复杂边界/孔洞、一般vector-Maxwell几何与真实测量尚未测试。表示能画孔洞的单元检查不能代替这些实验。

原始失败、数值细化、全成本及未运行项保留。详细见[中文研究报告](C1_RESEARCH_REPORT_ZH.md)、[机器裁决](C1_GATE_DECISION.json)、[结果表](results/SUMMARY_METRICS.csv)、[账本](C1_RESOURCE_LEDGER.json)。
