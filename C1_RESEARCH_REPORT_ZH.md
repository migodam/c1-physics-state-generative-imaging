# C1 统一材料概率生成：第一轮研究与最小实验

2026-10-13。**统一、无拓扑标签的生成器已能保留多个材料参数世界；概率质量未达到目标，物理状态与记忆没有跨三个seed的稳定优势。停止扩大本轮状态/记忆模型，保留统一材料生成方向。**

## 物理问题与定位

散射数据不能总是只对应一张材料图。想要多个世界有用，必须给它们所在材料区域定义概率，而不能仅展示更多图或选择残差最小者。这里先研究一个正确prior/likelihood下的条件材料分布，用小模型检查物理计算状态能否帮它保留多个解释。

原研究包的Maxwell算子、真实constitutive derivative、有限差分响应、概率质量守恒、单次likelihood与信息分析保留；默认solid/one-hole/two-hole生成分支停止。MDN的8个Gaussian component是未标注密度基函数，没有任何孔洞或符号类别身份。生成器接口没有topology/sign输入。详见[C1_DIRECTION_V2](C1_DIRECTION_V2.md)与[更新specification](C1_ARCHITECTURE_EXPERIMENT_SPEC_V2.md)。

文献审计沿用原包的先例/不可访问全文边界。物理unrolling、recurrent inference、conditional generation均已有先例，当前结果不支持强novelty。仅重新核对了[RIM](https://arxiv.org/abs/1706.04008)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[SBC](https://arxiv.org/abs/1804.06788)的primary metadata，以及[DDA背景](https://arxiv.org/abs/0704.0038)；没有全面重做检索或以未见全文判断机制不存在。

## 两个连续目标

U0用同一连接support上的两个实材料坐标，proper exponential-tilt prior、平方测量和可控的弱线性测量，产生不同符号的自然模式。符号区域只用于生成后评价。1024 train parents/128 validation/128 test；noise先在parent内部重复，再按parent配对分析。

P0使用13个三维电偶极的向量离散Green模型，两个连续空间材料系数、两个照明与四个intensity readouts；256/64/64 material parents。epsilon_b=2，绝对epsilon_real∈(0.5,3.5)，loss=.02，mu_r=1，共同材料服务两个源。它是固定几何、受限prior的离散电磁原型，不是任意形状的连续Maxwell验证，不是原包B1的complex-field/Mie实验。单频被动性不证明全频因果色散。

复振幅a=S p保留Maxwell电极化算子关系；观测是|a|²，所以测量Jacobian必须再经过2Re(conj(a) da)链式法则。原包affine-response的Gaussian sufficiency不能直接升级为这个非线性intensity目标的充分统计结论。E与归一化电极化p由当前candidate求解，物理电流与p相差已知常数；没有凭空增加磁流。所有truth fields在生成监督观测后丢弃，推断只计算candidate states。

共同likelihood只使用一次：p(x|y)正比于p0(x) exp(-Phi_y(x))。网络输出normalized q(x|y)，训练用proper density log score；没有最后附加physics loss、GN projection或residual-softmax权重。q的事件频率和参考p的误差分别报告，单个精确连续材料图像点质量为零。四个坐标符号区域不自动等于四个拓扑或四个局部density peaks。

## 物理状态如何进入生成

从固定、与真值无关的candidate(.65,-.45)计算两轮状态：material、prediction、whitened residual、真实离散J、gradient、GN diagonal，以及E/p的区域real/imag摘要。第一轮regularized GN仅更新pilot。最后状态或两步GRU memory调节整个条件density，production randomness之前context冻结；standard flow logdet因此成立。没有使用当前production draw的未记账物理信息。

直接MDN/direct flow完整看到同样测量和metadata。四个模型参数实际约30k，跨模型差<0.3%，同训练集、batch、学习率、checkpoint规则；三个seeds独立保留。U0用1500更新；P0所有模型统一3000。U0 validation MDN胜出后，在P0任何数据/训练/test前把state/memory density family改为MDN，保留direct flow认真对照。原方案、调整、初始失败均在[CHANGELOG](CHANGELOG.md)记录。

所有state是已有测量与候选的计算结果，不是额外独立测量；history可以帮助有限网络表示函数，不能增加条件于完整y的测量信息。

## 结果：覆盖了事件，概率仍不够好

下表是三个seed均值，**没有把seeds合并为置信区间**。流模型KL在尾部随求积细化剧烈变化，表中的大值是数值诊断，不能当作已收敛精确KL去宣布state获胜。完整逐seed结果见[SUMMARY_METRICS](results/SUMMARY_METRICS.csv)。

|实验/模型|参考 KL（nats）|事件质量 MAE|mass-weighted region recall|90%坐标覆盖|
|---|---:|---:|---:|---:|
|U0/mdn|0.315601|0.054973|0.9964|0.8828|
|U0/measurement|1.269990|0.053401|0.9942|0.9232|
|U0/state|1.008300|0.051525|0.9939|0.9108|
|U0/memory|6.135276|0.055987|0.9968|0.9056|
|P0/mdn|0.996777|0.051433|0.9848|0.8333|
|P0/measurement|8.795249|0.054522|0.9844|0.8568|
|P0/mdn_state|1.016358|0.052652|0.9906|0.8438|
|P0/mdn_memory|0.957476|0.050327|0.9886|0.8464|

U0独立adaptive案例与split Gauss最大事件差6.31e-14。全panel原64/96网格差2.34e-6，严格reference目标失败；保留初始回执后只做数值96/128细化，事件差2.92e-12通过。没有重训或重选模型。P0 40/64网格最大事件差.001724、最大log-evidence差.000993，经验参考门槛通过，但没有uniform remainder证书。

U0 MDN region recall约99.6%，P0 MDN/state/memory约98–99%，说明统一模型没有被迫只输出一个材料世界。它们是mass≥.05的预定义区域覆盖，不能称为任意拓扑mode发现。事件MAE仍约.05，明显高于.02目标；90%材料坐标区间的P0覆盖约.83–.85。图像多样性与best-of-K不能消除这个失败。

P0同MDN family比较，state减去measurement-MDN的KL差为-.00336、-.02044、+.08255；memory为-.07713、-.14487、+.10410。memory前两个seed的parent-bootstrap95%区间完全低于0，第三个完全高于0。对应memory观测truth-log-score区间均跨0。三seed平均memory KL只减少约3.94%，不能用两个有利seed替代稳定优势。

P0 validation选择direct flow为三个seed的简单baseline；这个记录不改。其test尾部失稳，40/64 KL均值差约.0616/5.8908/3.1120nats；因此必须同时保留same-family MDN对照，不能专门用这个不稳flow作为弱对手。state zero/permutation会恶化validation score，说明模型实际使用了state，但这个off-distribution reliance测试不是优势证据。

首seed、固定16 observation×8 iid draws/model的512次完整物理检查，四模型的whitened residual中位数约2.49/2.35/2.49/2.29，passivity violation均0。所有gallery24图额外forward，按原样显示，未按残差选择。这个稀疏检查不是全体样本的物理验收。

![材料样本](figures/P0_MATERIAL_GALLERY.png)

每张图只是同一测量下一个iid q材料样本，不分配点概率。独立1024 draw估计的区域质量见[概率图](figures/P0_EVENT_MASSES.png)和[gallery manifest](results/p0/GALLERY_MANIFEST.json)。固定13-cell core/shell显示不是一般三维孔洞成像。

![same-family对照](figures/PAIRED_KL_DIFFERENCES.png)

## 失败原因与边界

直接flow的validation score没有发现极端材料边界附近的tail风险；增大积分节点揭示了其KL不稳定。invertibility/logdet正确不保证tail质量。MDN的diagonal logit-Gaussian也不能充分拟合所有within-region形状，材料区间欠覆盖。有限数据与优化进度、不同seed、候选单一anchor与小状态摘要都是可能原因；当前诊断不能给出唯一因果解释。

原float32 decode发生156个endpoint rounding；改用float64、同一frozen latent RNG诊断后降至7，均在U0 MDN极端tail。没有静默clip/resample，原primary、gallery坐标及其物理residual不改写。这项NUMERICAL_UNRESOLVED进一步排除精确posterior宣称，不是~5%事件误差的完整解释。

单一全局pilot context还没有测试每个随机材料世界自己的在线field/history演化。完整网格decoder能表达0/1/2 enclosed threshold holes的测试仅是表示能力检查，不是从Maxwell测量自然生成未知复杂拓扑的实验。

## 计算、B1与停止

全部local CPU/失败/setup/训练/reference/context/图/计时记在[C1_RESOURCE_LEDGER](C1_RESOURCE_LEDGER.json)及两个append-only JSONL。每个material-frequency forward有2source RHS，derivative还需4RHS；adjoint/JVP另记。内部timer嵌套于process job，不能重复加总。原错误timing包含summary扫描，保留后重测；persistent in-memory query1024样本CPU中位数MDN .292ms、flow1.343ms、state1.359ms、memory1.334ms。state/memory每个新query仍需2material evaluations/12RHS；固定anchor本可cache，这项优化尚未执行，不能把当前代价当作不可改善下界。没有在相同合格概率质量上建立cost优势。

预算1200CPU秒/1800job-wall秒、8192material-frequency evaluations、localGPU0，阶段不借额度。实验/diagnostic已收口，完整测试77通过、decode新增测试1通过、原包12 algebra检查重跑通过；这些是实现证据，不是成像/校准/科研接受。

B1只读核对最终文件，不修改、不执行其代码、不读取其posterior/teacher数组。其prior/solver/readout与本P0不同，状态NOT_USED_TARGET_MISMATCH，见[B1兼容性](B1_REFERENCE_COMPATIBILITY.md)。没有OPM sweep、大GPU训练、3D diffusion、真实仪器、continuum convergence或未知三维拓扑测试。

结论与下一项小实验见[C1_NEXT_DECISION](C1_NEXT_DECISION.md)。本次是本地交付，没有自动公开发布。
