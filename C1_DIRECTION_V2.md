# C1 方向 V2：统一材料世界的概率生成

2026-10-13。用户的新方向优先于原研究包。本文在任何 C1 训练与新物理计算之前写入；实验后结论另见 `C1_NEXT_DECISION.md`。

研究对象是条件材料分布，而不是先预测孔洞类别再在类别内画图。问题是：**候选材料的 Maxwell 计算状态，能否让一个小型统一生成器，以完整成本计，比直接从测量生成材料更可靠地保留不同的可行材料世界？**

## 保留、修改与终止

保留原包的 Maxwell 电流方程、真实极化率导数、有限材料变化的 secant、forward–adjoint 灵敏度关系；保留共同噪声模型、proper prior、单次 likelihood、概率区域定义、支持下界、重复样本质量守恒、历史状态不增加独立测量信息及完整成本核算。文献审计的先例与不可访问全文的边界继续成立，不重新开展全面检索。

修改统计坐标：从 `(H,z_H)` 改为统一材料坐标 `u` 与同一个解码器 `chi=D(u)`。生成器、训练批次与初始化都没有 H 输入。拓扑、材料阈值事件及模式划分只进入离线评价。原分支对比改为任意候选材料之间的响应差异，不能先假定世界数量或孔洞数。

终止默认 topology-stratified generation、强制 E0 sign head、强制 18-state 分类 E1、固定的实心/单孔/双孔生成头，以及必须按旧 E0→E1→E2 顺序等待的架构推荐。原有限分类网络仅可作为后续可选对照。停止大型 GPU、3D diffusion、OPM/current-selection 和全局证书 sweep。

## 统一材料表示

长期表示是多分辨率实系数场 `u(s)=sum_l a_l psi_l(s)+tau v(s)`。在有限网格上，保留非零的满秩细尺度随机尾项；不能把所有弱观测方向压成零。用有界 sigmoid 表示实介电常数、非负 loss 表示被动材料；同一个材料同时服务全部源。`e^{-i omega t}` 下绝对材料 loss 非负，增量的符号不作为被动性判断。多频因果性需要额外的共享色散族，不由单频实验证明。

孔洞与复杂边界是阈值材料场的性质，不是生成条件。冻结阈值、连通规则、最小特征尺度及边界处理后，可以统计 0/1/2 孔等事件。连续阈值场不必有恰好为零的 hole-radius atom：有限尺度的低材料区域本身是正测度事件。若目标要求严格真空平台而非阈值近似，需要明确含原子或多相材料的 prior，不能把 sigmoid 的近似称为严格真空。

第一轮采用二维连续坐标：先检验多模态统计，再在 13 个三维电偶极单元的两个空间材料区域内检验状态作用。两个系数都连续变化、均无类别含义。这是受限 prior 的原型，不是任意形状、任意拓扑 posterior。另给一个满网格材料表示的 0/1/2 孔支持单元检查；它仅证明表示可以表达这些图像，不证明从散射恢复它们。

## 物理状态进入随机生成的位置

由测量 y 和固定、与未知真值独立的候选初始化得到两轮 pilot。每轮在当前候选上计算预测、残差、真实材料 Jacobian、梯度与正则化 GN 信息；电磁原型还保留总电场与电极化的空间区域摘要。它们改变条件生成映射的尺度与平移，从而改变整个条件样本分布。不会只在输出末端增加 residual loss。

pilot 是计算上下文，不是第二份证据。所有 pilot 在 production randomness 之前冻结。标准 affine-coupling 密度的条件只依赖冻结的 `c(y)`，所以可用完整三角 Jacobian；若后续把当前 production draw 的物理状态插入 conditioner，就必须重新计算真正的 Jacobian 或改用有明确路径概率的 stochastic process。stop-gradient 不会修复数学密度。

不在生产样本后附加未核算 GN projection，也不把优化轨迹频率称为 posterior。磁流未引入：当前非磁性介质使用电极化/电流，磁流需要合适的等效面定义或磁性 constitutive model。

## 三个竞争架构

1. **直接统一条件密度**：未标注的 8-component MDN 与 6-block conditional flow 都是认真基线。mixture component 没有孔洞或符号类别含义；flow 在同一连接支持内可表达多个密度峰。以验证集 proper score 选择基线，不能故意选择更差者。
2. **冻结物理状态条件 flow**：与直接 flow 使用相同 invertible material map；只加入最后一个候选的状态摘要。测量仍完整可见，不能通过减少基线测量输入人为制造增益。
3. **两轮 recurrent-state conditional flow**：GRU 处理同两轮候选状态，再调制统一 flow。与无记忆状态模型容量近似相同。历史可能帮助有限模型的计算表示，不提供新的独立数据。

Attention、conditional flow matching/diffusion 和在线 stochastic unrolling 都保留为候选路线。第一轮选择 normalized flow 是因为二维密度可直接核对，不是认定它比 diffusion 最好。只有连续参考、模式质量与完整成本显示简单模型不足时，才值得测试更复杂模型。RIM、Flow Matching 与 SBC 已有公开先例，参见 [RIM](https://arxiv.org/abs/1706.04008)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[SBC](https://arxiv.org/abs/1804.06788)。原包全文审计未重新认证；DDA 方程的背景参见 [Yurkin–Hoekstra](https://arxiv.org/abs/0704.0038)，本原型不是该综述的数值复现。

## 统计目标与模式概率

在同一 bounded material support 上，`p(u|y) ∝ p0(u) exp(-0.5||Sigma^{-1/2}(F(u)-y)||²)`，基准测度和变换 Jacobian 明确写入实验 specification。学习输出是 normalized `q_theta(u|y)` 的近似；其条件样本频率估计 **q 的事件质量**，与参考 p 的误差另报。不会把这些频率自动升级为真实实验的已校准 posterior。

预定义的符号/材料区域事件不是 generator branch。连续图像的单点质量为零；参考积分节点和显示代表图只是数值或可视化对象。数据驱动 cluster 若出现，先在开发数据冻结 partition，再用独立样本估计质量。展示 K 与积分 N 分开；没有 residual softmax、没有 best-of-K 主门槛、没有按显示数量分配证据。未建模的 other 不给虚构概率。

采用 reference KL / proper log score、区域概率 MAE 与尾部、Brier、条件坐标覆盖、模式 precision/recall、生成支持与全波 residual，辅以独立 seed 和 parent 配对区间。参考由独立阶数积分与局部自适应积分核对，属于经验收敛，不是全域数学证书。

## 最小决定性实验与停止标准

U0：统一连续二维 prior，平方测量产生不同符号的自然材料模式，弱线性测量提供可控可辨识性。不输入符号，不限制每个 mixture head 的模式身份。先验证参考与概率规则，再训三个 seeds。

P0：独立的小型 vector electric-dipole Maxwell 离散模型；同一 continuous core/shell permittivity support、固定 acquisition、实噪声的 intensity measurements。这是与原包复杂场测量不同的明确观测任务，不能暗称 B1 重现。候选总电场、极化、精确离散 Jacobian 与 GN 信息可在推断时重算。分别检查同数据/容量与成本账；强参考不收敛或物理检查失败时停止科学主张。

采用最高 8192 个新材料–频率评估、1200 CPU 秒、1800 wall 秒、GPU 0 的已记录第一轮上限。模型不超过 100k parameters，每 configuration 三个 seeds，最多两个小模型诊断重试。失败、参考、pilot、warm-up、展示样本 forward 都入账；不从 B1 借资源。B1 可信 reference 只有在其正式交付且任务支持相符时才能作独立评价，不能用于偷偷训练 C1。

只有与验证集选出的最佳简单概率基线相比，参考 excess log score 至少改善 10%，parent-paired 区间排除零，校准/支持不劣，且完整成本没有更差，才支持继续 physics-state 路线；或在同概率容差下总成本降低 25%。若只在 toy、单 seed、oracle states、best-of-K 或神经 FLOPs 上改善，记 HOLD/NO-GO。后续架构调整须记录原因、消耗和开发曝光，不用扩大模型掩盖负结果。
