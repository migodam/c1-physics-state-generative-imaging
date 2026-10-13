# C1 P2 子模块科研裁决

阶段起始日期：2026-10-13；共同target `joint-p2-radial-v1:C:tensor-Chebyshev-five-per-physical-axis-radial-v1`。本版本以accepted reference为主要事件和连续材料依据。

**保留小型posterior-distilled MDN作为proposal；不将raw模型升级为可信全材料posterior。** 联合最终裁决由B1的物理、reference、downstream与完整成本证据形成。

| 问题 | 实际证据 | 子模块建议 |
| --- | --- | --- |
| 无topology输入能否保留多个材料事件 | accepted reference下三个full-distill seeds count/mass recall均1，event MAE平均.01321508 | 保留测量条件密度接口 |
| full covariance是否在共同B1 target仍有价值 | 旧13阶latent KL平均.21572对.27482；差−.05910，区间[−.06711,−.05094] | 保留数值proposal价值；KL未重算、不等于physical KL |
| event mass是否稳定优于diagonal | accepted差+.00018710，区间[−.00057984,+.00088033] | 不主张full在事件概率上优越；两者均保留 |
| 是否可以删去独立reference或traditional sampling | 所有learned结果仍是raw proposal；teacher物理成本已支付，完整continuous/mode精度未认证 | 不删reference/sampling，不作速度优越性结论 |
| distillation是否纯架构优势 | 训练512posterior teacher draws/obs；joint基线objective仅truth，validation选择仍用teacher | 明确额外监督；架构比较以同teacher为主 |
| false exclusion/confidence是否消除 | Distill此64-parent panel未触发；diag-joint有具体遗漏和过信案例 | 是有限panel结果，不是无遗漏保证 |
| Top-K是否足够 | 旧13阶表平均约5.2事件、reference质量.97764；这不是accepted Top-K重算，事件内材料仍连续 | 下游保留条件samples与remainder，不只传代表图 |
| continuous posterior是否达到目标 | accepted grid-CDF mean≈.098906、重要事件内≈.141761；旧latent KL估计≈.216 | grid-CDF目标实测未达，完整posterior/physical-KL未获认证 |
| full是否所有连续指标更好 | accepted conditional grid-CDF差−.020007，边际CDF差区间跨0，SD/prior-width误差更大 | 保留具体指标结果，不作全面优越性结论 |
| full-likelihood correction能否运行 | 一次development observation，三方法各64 full calls；full/diag/Gaussian corrected MAE=.01275/.02598/.01863、ESS36.6/29.5/23.9 | 合法有限样本概率修正可行；不作算法排名或fresh泛化结论 |
| 是否扩展大NN/state/memory | 没有新的瓶颈与独立收益证据 | 停止本阶段扩展GRU/Transformer/大型训练 |
| 是否有独立IEEE TAP机制贡献 | 已知MDN/distillation工具；本子模块没有证明新的Maxwell物理机制 | 不从density结果单独提出原创物理论文主张 |

最终reference已对全部128做event/mean/SD13/17检查，12/39/48初始13阶失败后升级17并通过21交叉检查，unresolved为0。Positive-partition边际和important conditional CDF单调违例均为0，固定65点值不能成为uniform supremum保证；CDF13/17和inverse quantiles仍主要是固定前八个子集，加三项adaptive refinement。旧node、direct、partition和13阶KL文件均保留，参考接受不等于KL或full-physics认证。这些修正没有消除约.1的模型CDF差距。

未来若有新授权，优先做匹配预算、多observation的corrected proposal收益与条件tails；不在同一fresh panel反复调参，不因单次64-draw结果宣布full优越。没有全成本收益时保留compact proposal/benchmark，不无限调大网络。

原始15模型、失败对照、节点CDF、teacher和各数值版本保留；后处理不改变选择、不重置预算。主要连续结果见 [accepted复核](results/continuous_accepted/CONTINUOUS_COMPARISON.md)，B1最终事件结果为`REFINED_FRESH_EVENT_COMPARISON.json`；[历史方法表](results/summary/METHOD_SUMMARY.md) 与 [原概率行](https://github.com/migodam/c1-physics-state-generative-imaging/releases/tag/joint-p2-20261014) 不改称最终reference。有限SNIS零计数/ESS不是无遗漏证明。

