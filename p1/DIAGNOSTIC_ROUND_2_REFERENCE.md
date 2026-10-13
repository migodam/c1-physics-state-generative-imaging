# 第二轮针对性诊断：P0 连续积分精度

记录时点：preparation完成、任何新训练开始之前。preparation在全部 P0 train/validation上发现40/64 CDF最大差.0375、tail .0356、quantile .1424、evidence .0290；事件最大差.00341。不能用这些未稳定的量假称模型误差。

目的：在0新增Maxwell预算内，检验已有 response bank 能否支持可靠的更密连续积分。把两份独立40/64 Gauss银行分别拟合为 degree 8/12 的 tensor Chebyshev response approximation；在未用于拟合的另一银行节点上检查 response error，并在更密96/128节点比较 posterior event/evidence/CDF/矩/quantile/tail。

这不是改变prior、support、noise或目标；它是原forward的额外数值近似。无全域误差证书，银行间误差不等于任意材料上的保证。原40/64参考指标与teacher始终保留；64-node teacher不改。模型评价同时输出原64-node KL和数值refinement KL；若后者与原值冲突，保留冲突，未稳定分数不用于精确排名。所有方法使用相同refinement，不筛案例。

预计CPU≤20秒，wall≤30秒，Maxwell/B1/GPU=0，计入900/1800独立总上限。此项占第二轮针对性诊断名额，另一轮是已注册的固定validation weighted-EM容量诊断；之后不再增开模型诊断。
