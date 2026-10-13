# C1 统一架构与实验 specification V2

预注册日期 2026-10-13。原包只读保存在 inputs/source_package；本 specification 根据用户 V2 要求替换旧分支执行路线。`configs/protocol_v2.json` 为机器可读预算与 seed。

## U0 连续概率实验

材料 `x=(x1,x2)` ∈ (-1.5,1.5)²。proper prior `p0(x)=prod_j exp(a_j x_j)/Z_j`，a=(0.4,-0.2)，`Z_j=2 sinh(1.5 a_j)/a_j`。同一连接支持内没有 gap 和 discrete sign variable。随机 seed 生成 parent 后才叠加 noise。观测 `F=(x1²,x2²,kappa*(x1+0.5*x2))`，kappa∈{0,0.25,0.6}、共同标准差 sigma∈{0.12,0.25}；metadata 已知。四个开象限仅是离线事件。

参考：分割坐标轴的 Gauss–Legendre 张量积分，以每轴总64/96节点核对（每半轴32/48节点）；代表性冻结观测再用独立 adaptive integration 核对。logsumexp 用 prior、likelihood 和 quadrature cell weights，不把一格的质量称为图像点质量。训练 1024 独立 parents×4 noise；validation 128×2；locked test 128×2，训练不能读取 test。三 seeds 2610131/2/3。

## P0 向量电磁原型

固定 13 个 xyz 电偶极体积单元，背景 epsilon_b=2，mu_r=1，时间约定 exp(-i omega t)，背景波数 k=4 [length^-1]，cell spacing 0.08 [length]，volume=spacing³。连续 epsilon_core=2+x1+i0.02、epsilon_shell=2+x2+i0.02；x 与 U0 同 support 和 prior。材料范围 [0.5,3.5]，同一材料服务两个源/两接收投影。

真实离散极化率 alpha=alpha_CM/(1-i k³ alpha_CM/(6 pi))，alpha_CM=3 V (epsilon-epsilon_b)/(epsilon+2 epsilon_b)。电场 Green dyadic 无自相互作用；辐射自项在 alpha 中。解 `p=alpha(E_inc+G p)`，p 为按epsilon_0 epsilon_b归一化的电偶极矩体积变量：物理电偶极矩为epsilon_0 epsilon_b p，polarization density为epsilon_0 epsilon_b p/V；体积积分电流为-i omega epsilon_0 epsilon_b p，current density还需除V。state中的p/V摘要是归一化极化密度。不得把p误称为额外独立测量。磁流不使用。

以四个远场投影的 |a|² / fixed_scale 为观测均值；Gaussian 实 intensity readout noise sigma∈{0.12,0.25}，允许噪声后的 readout 为负。fixed_scale 由明确的参考材料单次计算得到，非 test-fit。频率固定，两个源相干复振幅分别计算、intensity 不跨源混合。该任务不是 complex-field-noise posterior，也不是 B1 concentric Mie reference。

forward/Jacobian：一次材料–频率评估有两个 forward RHS；两个材料 derivative 每个两个 RHS。精确 `dp=L^{-1} diag(dalpha) E`，intensity 导数 `2 Re(conj(a) da)`。额外 transpose/adjoint 检查、有限差分、失败均记录；不宣称 matrix-free JVP 更便宜，二维 full J 是小原型诊断而非高维架构要求。

训练 256 独立 material parents×2 noise；validation 64×1；test 64×1。参考每轴总40/64节点（每半轴20/32节点），总5696 distinct materials；不得额外占用 B1 solver/cache/ledger。目标经验 event MAE refinement ≤0.01，log-evidence 差 ≤0.05 nats；精确 log-score 随积分误差另检查。数据/节点数在首次运行前按预算算术修订，详见 CHANGELOG。

## 模型与公平性

M-MDN：8 个未标注 diagonal Gaussian components，作用在 z=logit((x+1.5)/3)。M-Y：6-block bounded-scale affine flow；M-S：同 flow、完整 y+最后候选状态；M-R：同 flow、完整 y+两轮 GRU state。所有模型约30k trainable params，容量差≤5%，训练数据、batch128、Adam learning rate1e-3、1500 updates 与 validation checkpoint 规则一致。baseline 是 M-MDN 与 M-Y validation log score 的胜者。

实施更新（P0任何数据/训练/test之前）：U0 validation三seed均选MDN，因此P0的M-S/M-R改为与M-MDN相同的未标注8-component密度head，仅替换为state/GRU encoder；仍约30k params。P0 direct flow保留为第二简单对照。所有primary P0模型在locked评估前冻结，此更新不改变已冻结U0实验。原“改用flow”的计划没有被当作必须胜出的路线。

P0训练前的学习进度修订：P0四模型统一3000 updates，仍以validation选择checkpoint、CPU阶段上限360秒；U0保留1500，不重训。此变更依据U0 validation仍在改善，在CHANGELOG预先记录，未扩大CPU/GPU/物理预算。

frozen pilot x0=(0.65,-0.45)，正则化 GN 后 clip 到 (-1.45,1.45) 得到 x1；固定初始候选不编码类别。状态仅由 (candidate,y,known metadata) 计算，函数接口不接受 truth。M-S 与 M-R 的两个 candidate round 成本相同。状态摘要包括 material、prediction、whitened residual、gradient、J、GN diagonal；P0 加 spatial field/polarization real/imag summary。state normalization 只 fit train。context 在生成 ε~N(0,I) 前冻结；density 包含 logit Jacobian 和所有 flow layer logdet。没有 final projection。

如训练没有满足门槛，可运行最多两个预先记录的小诊断 configuration，使用 validation/development 数据；不能在 test 结果后选新冠军再写 blind gate。额外 measurement-only capacity/training 对照需计费并明确探索状态。概率目标为 normalized learned q，不添加改变目标的 residual penalty。

## 评价

在冻结128(U0，两个noise)/64(P0，一个noise) test parents上，reference quadrature计算 KL(p||q)；生成1024 draw/observation计算四个象限 q event masses、mass MAE/p95、weighted recall(真mass≥0.05)、mode precision(qmass≥0.05)、conditional 90% coordinate coverage、proper Brier、best-of-K(仅辅助)。模型样本无拓扑标签输入。按 parent 先平均noise，再 paired bootstrap2000次、seed26101399；保留三个trainingseed独立结果。

P0仅对固定首16个test observations的8 production samples调用完整solver检查预测误差、被动性、跨源一致性；gallery另24样本全部重新forward入账。all-sample residual NOT_RUN；不能将这个稀疏诊断称为全样本物理验收。representative selection 不按 best residual，不改变 event mass；K与integration_count分开。

## Gates 与成本

U0参考 event difference<1e-8、独立adaptive event check<1e-7；P0经验reference门槛如上。实现 tests 不能替代这些实验。候选生成器质量标准：mass MAE≤.02，p95≤.05，mean reference KL≤.02。physics-state贡献采用 C1_DIRECTION_V2 的10%/25%规则；若固定成本统计不能支持 matching，记 HOLD_COST_MATCH，不认定贡献成立。

1200 CPU 秒分阶段固定：tests60、U0 train300、P0 preparation180、P0 train360、evaluation240、diagnostics/report60；1800 wall秒含启动与失败；GPU0；8192材料–频率评估硬限制。阶段不能相互借CPU额度；进程预算在运行前强制，ledger append-only，重复run ID拒绝，失败费用保留。交互写作、外部语言模型/检索服务未完整计量，不称零研究成本。

B1保持运行独立；不修改或执行它的main、cache、GPU、observations/reference；本阶段B1引用状态为 NOT_USED。原包历史12代数checks可在新结果路径重跑，历史ledger不覆盖。SHA256不检查、不生成。
