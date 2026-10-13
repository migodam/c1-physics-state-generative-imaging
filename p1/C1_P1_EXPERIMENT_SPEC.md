# C1-P1：统一材料 posterior 密度修复

2026-10-13，在任何新数值任务前记录。原 C1 数据、权重与账本只读。本阶段新增 Maxwell/B1 调用与 GPU 均为零；独立上限 900 CPU 秒、1800 受控任务 wall 秒，单 worker、单 BLAS thread、nice=15。具体配置见 configs/protocol_v2.json。

## 已读证据与要区分的问题

P0 原 diagonal MDN 三个 seed 均在最早的 250 更新选择 checkpoint；之后训练 NLL 下降而 validation NLL 急剧恶化。这个证据支持泛化/有限监督信息问题，尚不能排除 covariance 表示不足。精确 logit 与 Jacobian 已实现；float64 仍有少数旧 U0 极端尾部舍入，不能将这项数值问题直接解释为约 5% 的事件误差。

## 最小 2×2 比较

四个主要方法：保留未标注 diagonal MDN；换成 full-covariance MDN；在 diagonal 或 full-covariance 上做 posterior distillation。两项改动分别检验表示能力与监督目标/信息。不是四种新的网络。

所有模型约 30k 真实参数、8 个无类别身份的 mixture component、相同 connected material support，三个固定训练 seed、相同观测与训练更新次数。U0 1500、P0 3000，Adam 1e-3、batch 128、每250更新记 checkpoint。full covariance 使用严格正定相关系数 rho=.995 tanh(raw)，其余 mean/log-scale 与旧 MDN 同定义。

Joint 模型用已保存 material/measurement joint targets；distill 用相同 conditioning observations 的连续 posterior 数值积分，权重是 proper prior density × 单次 likelihood × quadrature volume，经归一化后做无偏于该求积目标的随机 NLL。积分节点是连续积分工具，不是真实离散先验。教师仅训练用，生成器推断不接收参考、sign、truth 或教师标签。

所有 P1 方法以同一 validation posterior-expected material log score 选择 checkpoint；同时完整保留 observed true-material score 的最优 checkpoint 与学习曲线。这是明确记录的 checkpoint 修订，不能把改善全部归于 covariance 或 distillation。旧已发布权重作为历史锚点另外评价。

## 参考、精度与评价

P0 只重用 40/64 Gauss response bank，不调用 forward。U0 likelihood 可直接计算，使用64节点 teacher 与96/128 reference。先核对训练/validation 的 prior measure、quadrature weights、事件/evidence/矩/CDF 的精度。事件概率用 mixture CDF 而非1024样本频率：diagonal 为正态CDF乘积，full covariance 用 Plackett 相关积分并做32/64数值核对。密度归一化由完整 Gaussian mixture 与精确 logit 变换保证，另报告积分精度。

KL 排名同时保留 coarse/fine 差；未收敛项目不用于科学排名。连续参考 CDF 由半区间 Gauss marginal density 的 Legendre 插值积分得到，独立阶数核对并报告非单调/越界，不静默 clip。mean/variance 使用平滑积分；quantile、tail 与 within-region shape 分别记录 reference 与样本误差。

旧 U0/P0 test 只称 exposed regression panel。所有 P1 主要权重选定后，按提前保存 seed 生成新的 U0 256 parents×1 noise；只用于一次最终评估，不用于模型选择。P0 没有新盲参考，结论限于已曝光 panel。

记录三个 seed 独立结果、按 parent 配对的2000次 bootstrap；不能用最优 seed、图像多样性或 best-of-K 代替概率质量。旧 flow KL 已知未收敛，不参与新架构胜负排名。

## 有限诊断与后续动作

固定前8个 validation 观测、两次无 sign 初始化的 weighted EM、60步，比较 diagonal/full covariance 的非摊销密度拟合能力。这仅是 oracle family-capacity 诊断，不是可用的测量生成算法。最多再记录一轮针对性数值/训练诊断；不增加模型族、预算或物理调用。

若 density 修复相对 diagonal joint 在三个 seed 的 KL 与事件误差均改善，再用 P0 已缓存 features 做 measurement/state/GRU 的同 family 小消融。其未来在线准备仍需要原有物理成本，当前 cache 复用不能称免费新测量或 cost 优势。没有稳定收益即停止 state 训练。

目标 event MAE≤.02、KL≤.02 保留为科学标准，不引入新的执行审批或发布流程。必要测试针对新 Gaussian 密度、变换、接口与零物理约束；不做 smoke、SHA256 或旧实验重跑。所有失败和诊断都进新账本。
