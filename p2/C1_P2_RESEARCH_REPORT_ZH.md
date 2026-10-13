# B1 × C1 Joint P2：四维共同材料 posterior 的条件生成与歧义保留

阶段起始日期：2026-10-13。本版本采用最终accepted reference；物理机制、共同reference、完整成本和最终联合裁决以B1 Joint P2报告为准。

## 结果是什么

在同一个连续径向材料prior、同一份实化Maxwell测量与同一nuisance likelihood下，C1已实际训练、冻结并评价15个小型CPU density models。最终accepted reference下，三个full-covariance distillation seeds的事件MAE为 **.013793、.012966、.012887**，平均 **.01321508**；diagonal平均 **.01302798**。每个full/diagonal distillation seed在128个fresh observations上都保留全部预注册重要事件，未触发false exclusion。原13阶评价中的latent KL为.21116/.22458/.21143、平均.21572；它仍是**13阶quadrature expectation的empirical估计，未用accepted reference重算，也不是physical material KL**。

这些结果支持本台架上有限材料事件的可保留性，尚未证明完整连续posterior的准确恢复。同teacher下full的旧13阶latent KL更低，但最终event MAE没有稳定改善。最终accepted连续参考给出full的平均物理grid-CDF误差约 **.098906**、重要事件内约 **.141761**，明显未达到.01目标。旧13阶fresh表完整保留为历史raw proposal诊断；另有一次合法full-Mie修正的development小样本实验，二者分开报告，不构成一般三维概率认证。

最终连续结果见 [accepted comparison](results/continuous_accepted/CONTINUOUS_COMPARISON.md) 与 [JSON](results/continuous_accepted/CONTINUOUS_COMPARISON.json)。最终事件比较由B1 `research/joint_p2/results/REFINED_FRESH_EVENT_COMPARISON.json` 保存；`ACCEPTED_REFERENCE_AUDIT.json`说明实际接受阶数与误差。旧 [METHOD_SUMMARY.md](results/summary/METHOD_SUMMARY.md)、[METHOD_SUMMARY.json](results/summary/METHOD_SUMMARY.json) 和 [EVALUATION_SUMMARY.json](https://github.com/migodam/c1-physics-state-generative-imaging/releases/tag/joint-p2-20261014) 不覆盖。模型点密度不是材料图像概率；非零质量关联到预注册材料事件。

## 共同物理目标与材料空间

模型学习四个连续坐标 `x=(z_c,z_s,Re ε,Im ε)`，prior 为独立 uniform：`[-3,1]²×[2.2,2.8]×[.04,.16]`。`z_c,z_s` 经共同 radial implicit decoder 形成材料；`u<0` 为严格真空，其余区域共享复 ε。拓扑在生成后计算，generator 从未接收 cavity count、topology class、truth feature、physics state 或 memory。

中心空腔半径和环形空腔半宽分别是

\[
r_c=.45\sqrt{1-1/\sqrt{-z_c}},\quad
h_s=.18\sqrt{1-1/\sqrt{-z_s}}
\]

（对应 coefficient `<-1` 时成立，否则为零，单位为外球半径）。两个 coefficients 均在 `[-1,1]` 时没有空腔，因而实心 prior event 有25%质量。inactive coefficient 的不同值产生同一物理材料；它们仍必须按原 prior 正确积分，不能将 coefficient-uniform 偷换成 radius-uniform。

C1 primary 只学习共同 **C target**：`kR=1.6`、八个角度、两种偏振，16个复通道按 Re 后接 Im 实化为32维；另加 log sigma 和 gain prior SD，形成34维 condition。σ 是预先固定的 reference response RMS 的0.12倍，不随候选变化。材料 ε 和共享实 gain 均未知；gain proper truncated Gaussian 的原始均值为1、SD .15、范围 [.7,1.3]，边缘化保留其正确归一化及 determinant 项。C1 未为获得歧义增大噪声。

冻结 identity 为：

`joint-p2-radial-v1:C:tensor-Chebyshev-five-per-physical-axis-radial-v1`

这明确指向共同 response interpolation 所定义的连续 **numerical proxy posterior**。Full-Mie solver 的准备、独立核验与补偿诊断由 B1 单独保存和计费。A/B/D 条件、附加频率的物理机制结果不能被冒称为这个 C1 模型已学习的 target。该支持是同心球、有限径向材料 family；没有偏心孔洞、多物体、一般未知边界或开放三维支持。

## normalized density 的数学含义

对每个有界 latent 使用 `t=log(x-lower)-log(upper-x)`。八个 Gaussian mixture components 是可交换的密度组件，full covariance 用正对角 Cholesky factor，single Gaussian 和 diagonal 是明确对照。Exact logistic Jacobian 将 normalized `dt` density 转为 normalized `dx` density：

\[
q_\theta(x\mid y)=q_\theta(t(x)\mid y)
\prod_j\frac{b_j-a_j}{(x_j-a_j)(b_j-x_j)}.
\]

模型具有声明 open latent support 上的正密度，但全 support 不保证有限 draws 找到全部 modes，也不保证其质量正确。采样保留原始 transformed coordinates；浮点端点舍入被标记，不进行 clipping、projection 或 resampling 来悄悄改变概率 law。六项必要数学检查已由共同计费流程通过；它们验证公式和采样一致性，不等于科学概率校准。

材料 field 的概率是 latent law 的 decoder pushforward。非单射 decoder 使几何零半径有原子质量，却不使单个完整复 ε 图像自动具有非零点质量。输出事件为中心半径四档与环形半宽三档的十二个组合；连续 ridge 可跨多个区间，事件数不能直接叫 topology-mode 数。

## 实际训练与 fresh panel

训练数据是 B1 已支付物理 bank 与共同 reference 产生的 `C1_TRAINING_DATA_REF13.npz`：128 train parents、各两次噪声，共256 observations，每个 observation 提供512个 posterior teacher draws。32 validation parents 同样各两次噪声。Teacher 来自13阶 tensor quadrature 的 randomized stratified resampling；inactive coefficients 另按其正确 conditional uniform prior 抽样。

五方法分别为 full/diagonal posterior distillation、single full Gaussian distillation，以及 full/diagonal joint truth NLL。每个方法使用三个固定 seeds `26101321/22/23`，同为两个96单元 hidden layers、Adam1e-3、batch128、实际3000 updates。Full mixture有24,312参数、diagonal有19,656参数、single Gaussian有14,127参数。计算预算和 hidden encoder 匹配，参数个数并不完全一致，不能把比较解释成参数数目严格匹配的唯一因果证明。

每250步对同一 validation teacher 计算 posterior-expected log score，选择 checkpoint。**Joint-training 对照的训练 objective 只使用 joint truth pairs，但其模型选择也使用共同 validation posterior teacher。** 它不是端到端不依赖 teacher 的严格纯监督基线。Distillation 与 joint 的巨大差异包含额外 posterior objective 监督，不能归结为 full covariance 架构独有优势。Full/diagonal distillation 使用相同 teacher，二者是更直接的 covariance family 对照。

15个模型全部冻结后，B1 才生成64个 fresh parents、每个两次独立噪声，共128 observations；seed `261013213`。这些是本授权内新模拟的 cases，未访问旧 locked truth，未用于调参或选模型。每个模型、每个 observation 的事件质量用原始8192个 iid model draws 估计；这些是 `qθ` 的样本，不自动成为 target posterior 样本。

## 最终事件比较与保留的13阶历史结果

全128个observations的event/mean/SD均进行了13/17复核。Observation12、39、48的初始13阶结果未满足要求，升级到17阶并用21阶交叉检查后通过；最终没有unresolved rows。Target、模型、weights、teacher及样本RNG均未改变。最终主要事件结果为：

| method | seed | accepted event MAE | important count recall | important mass recall | false exclusion count |
| --- | --- | --- | --- | --- | --- |
| full_distill | 26101321 | .01379269 | 1 | 1 | 0 |
| full_distill | 26101322 | .01296581 | 1 | 1 | 0 |
| full_distill | 26101323 | .01288675 | 1 | 1 | 0 |
| diag_distill | 26101321 | .01314530 | 1 | 1 | 0 |
| diag_distill | 26101322 | .01335598 | 1 | 1 | 0 |
| diag_distill | 26101323 | .01258266 | 1 | 1 | 0 |

Full-minus-diagonal最终event MAE差=.00018710，parent配对95%区间[−.00057984,+.00088033]，仍没有明确event概率优势。

下面的15-model表是**历史13阶raw表**，先合并同parent两次噪声，再等权平均64个parents。其event MAE、reference Top-K、Brier与reference-dependent诊断不代替最终accepted比较。KL仍为旧13阶四维latent expectation估计；模型自身区间与truth coverage不依赖reference细化、保持原值。Top-K列从未重新归一化。

| method | seed | event MAE | latent KL | important mass recall | false exclusion | false confidence | Top-K reference mass | physical 90% marginal coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_distill | 26101321 | .013797 | .211159 | 1.000 | 0 | 0 | .975948 | .925781 |
| full_distill | 26101322 | .012971 | .224577 | 1.000 | 0 | 0 | .980071 | .927734 |
| full_distill | 26101323 | .012892 | .211433 | 1.000 | 0 | 0 | .976911 | .921875 |
| diag_distill | 26101321 | .013149 | .269942 | 1.000 | 0 | 0 | .973868 | .925781 |
| diag_distill | 26101322 | .013360 | .281818 | 1.000 | 0 | 0 | .975482 | .921875 |
| diag_distill | 26101323 | .012587 | .272712 | 1.000 | 0 | 0 | .975089 | .914063 |
| gaussian_distill | 26101321 | .026543 | .671659 | 1.000 | 0 | 0 | .976102 | .939453 |
| gaussian_distill | 26101322 | .028373 | .694111 | 1.000 | 0 | 0 | .972331 | .927734 |
| gaussian_distill | 26101323 | .027749 | .649544 | 1.000 | 0 | 0 | .976273 | .925781 |
| full_joint | 26101321 | .046013 | 15.602214 | 1.000 | 0 | 0 | .973209 | .871094 |
| full_joint | 26101322 | .045328 | 19.049445 | 1.000 | 0 | 0 | .972363 | .867188 |
| full_joint | 26101323 | .053359 | 25.435419 | .999556 | 0 | 0 | .970912 | .880859 |
| diag_joint | 26101321 | .039697 | 5.335496 | .986496 | .046875 | .007813 | .938544 | .835938 |
| diag_joint | 26101322 | .038275 | 6.922364 | .993133 | .023438 | .023438 | .951501 | .855469 |
| diag_joint | 26101323 | .037151 | 4.933731 | .995475 | .031250 | .031250 | .951256 | .839844 |

重要事件定义为 reference mass≥.05；mass-aware discovery 要求模型质量≥.01，false exclusion 为重要事件的模型质量<.005；false confidence 要求模型最大质量≥.90且 reference 至少两个重要事件。这些是事前明确的操作性统计，不是对所有歧义类型的完备定义。Full joint 虽然未触发这两个 flag，其 event MAE和 KL仍很差，说明“未触发 flag”不能代替概率准确性。

历史13阶full/diagonal mean event MAE为.013220/.013032，最终为.01321508/.01302798；结论没有改变。旧latent KL为.215723/.274824，仍未重新做latent-KL积分细化。Single Gaussian旧13阶event MAE=.027555、KL=.671771：较宽coverage不能说明概率质量更准确。

## 匹配比较究竟支持什么

2,000次parent配对bootstrap使用seed26101329。先合并同parent两次噪声，再对三个model seeds的差取均值；seeds不被当作额外独立parents。下表保留旧13阶比较；**最终accepted event MAE对应区间为上表所述，而非此处历史区间**。

| metric | difference | paired 95% interval | 可以得出的解释 |
| --- | --- | --- | --- |
| latent KL | −.059101 | [−.067108,−.050940] | 同 teacher 下 full covariance 改善该数值 latent density |
| event MAE | +.000188 | [−.000543,+.000874] | 事件质量没有明确优势 |
| Top-K reference mass | +.002830 | [+.001131,+.004576] | 选择集合略多保留 reference mass；平均K也更大 |
| truth Brier | +.000029 | [−.008042,+.007504] | 无明确差异 |
| continuous truth log score | −.014760 | [−.125645,+.105199] | fresh truth proper score无明确优势 |
| mean physical 90% coverage | +.004557 | [−.001302,+.011719] | 覆盖率差异不确定 |

KL是latent density KL，decoder中inactive coefficients差异不改变实际材料，不能等同于material field KL。这里的KL估计及其bootstrap还条件于旧13阶节点；event/moment reference接受不自动认证KL积分收敛。Full仍有proposal价值，但改善哪个实际材料信息需要事件及physical coordinates单独证明。两种distillation都保留，不能因P1正结果省略diagonal。

## 两个可直接核查的概率案例

历史raw observation0、parent0的reference给出实心事件0为.717920，中心半径(.10,.20]且无环形空腔事件6为.119664，中心半径>.20且无环形空腔事件9为.118734；该observation接受阶数仍为13。Full-distill seed26101321给出.690552、.094849、.151001，没有将三个重要解释压成单一实心答案。历史选出事件`[0,9,6,3]`包含.993713 reference mass；下游应使用B1最终reference而不是把C1模型直接当全波认证概率。其loss ε区间未覆盖truth，多事件保留不代表连续坐标准确。

失败对照保留为原始raw案例：diag-joint seed26101321、observation9、parent4的事件1 reference .053205对模型.000122，触发false exclusion；observation12、parent6的旧13阶reference给出实心.761434、另有.115774/.076819重要空腔事件，模型实心.950806，触发false confidence。Observation12最终reference升级17，其旧概率不能代替accepted数值。Full-distill observation106、parent53的历史MAE=.041081，事件2从.057078推到.219604、事件9从.452290压到.236572，important recall却仍为1。Coverage和mass accuracy必须同时检查。

## 连续材料、参考误差与尚未认证的部分

主要事件质量现在采用B1最终accepted积分：125个observations为13阶，12/39/48为17阶且经21阶检查，全部128的event/mean/SD refinement通过、没有unresolved。CDF13/17独立复核范围仍为固定前八个加三项数值细化，不能说全128的CDF已做uniform精度认证。Model event binomial SE在质量.5时约.00552；零计数不证明零质量，95%上界约.000366。Parent bootstrap条件于保存reference及原始model MC，未覆盖solver误差。

本次独立审阅发现一个重要限制：`reference_samples` 的物理坐标来自13阶tensor节点重采样。其CDF是staircase，即使平滑连续density正确，模型CDF与该node staircase的差仍可很大。原始 `physical_cdf_max_abs` 与旧参考 quantiles 保留为 **NODE_CDF_DIAGNOSTIC / NODE_QUANTILE_DIAGNOSTIC**；不能把约.2的诊断值直接归因为生成器连续CDF误差，也不能用node quantiles宣称连续收敛。

B1最终CDF改为同一frozen density的positive-partition积分，全部128的边际及important conditional CDF单调性违例均为0，末端归一化最大误差约4.33e−15。旧截断CDF、nodeCDF和partition文件保留。不同积分partition对同一exact evidence的数值结果存在可见差异，最大旧新CDF变化.014763、partition event mass差.005506；这些记录在`ACCEPTED_REFERENCE_AUDIT.json`，不能称积分误差为零，也不是更改prior/target或给模型节点重新归一化。

C1后处理从保存参数和原始RNG复现同一8192个draws、核对原event counts，未训练或重选模型。最终结果在`results/continuous_accepted/`，此前`results/continuous/`保留为历史13阶直接CDF结果。比较全部128的physical四坐标固定65点CDF、重要事件内CDF及accepted mean/SD；完整direct inverse quantiles仍只在前八个观测提供。

| method | seed | physical direct grid-CDF error | important-event conditional grid-CDF | mean error/prior width | SD error/prior width |
| --- | --- | --- | --- | --- | --- |
| full_distill | 26101321 | .098970 | .139602 | .051078 | .051113 |
| full_distill | 26101322 | .102582 | .148769 | .048644 | .054657 |
| full_distill | 26101323 | .095165 | .136912 | .049634 | .052243 |
| diag_distill | 26101321 | .100561 | .155315 | .049325 | .048163 |
| diag_distill | 26101322 | .107056 | .171087 | .052938 | .049809 |
| diag_distill | 26101323 | .097226 | .158901 | .049520 | .042705 |

每个observation先取四坐标最大误差，再按parent合并两个噪声、等权平均parents。最终full/diagonal三seed mean direct grid-CDF=.098906/.101614；重要事件内=.141761/.161768。相比node staircase诊断误差降低，仍远高于.01；accepted数值细化没有改变这个负结论。

最终配对parent bootstrap的full-minus-diagonal边际grid-CDF差为−.002709，95%区间[−.007287,+.001952]，没有明确边际优势；重要事件内差为−.020007，区间[−.025564,−.014829]，支持受限conditional grid-CDF改善。Full的SD/prior-width误差差为+.005778，区间[+.003442,+.008204]，并非所有连续指标都更好。

固定前八个positive-partition 13/17 CDF差最大.00097375、important conditional CDF差最大.00414377；原直接inverse quantile差/prior width最大.00025561、coherence均通过。全128 event/mean/SD接受与这个CDF/quantile子集范围分别记录；65点最大值不等于连续supremum，不能升级full-Mie/continuum bound。其余120个旧node quantiles仍为`NODE_QUANTILE_DIAGNOSTIC`，原模型自身区间和truth coverage不改变。最终数值见 [accepted comparison](results/continuous_accepted/CONTINUOUS_COMPARISON.md) 与 [JSON](results/continuous_accepted/CONTINUOUS_COMPARISON.json)。**完整连续posterior精度仍未达到：grid-CDF≈.1；latent KL≈.216仍是旧13阶估计，不能当作accepted physical-KL精度保证。**

90% coverage对应模型自身等尾marginal intervals；四维同时落入这些区间并非一个90%联合credible region。几何零半径原子使几何坐标覆盖率可超过.90，不能据此证明calibration。连续ε与几何的conditional tails、prior sensitivity和mode遗漏仍需各自证据。

## 一次合法full-Mie修正：可运行，但不足以排名

B1另在预注册的`large_cavity/C` development observation上，使用seed26101321的三种已冻结distillation模型，分别做64次实际full-vector-Mie评估，共192次。Proposal是`0.9 qθ + 0.1 proper uniform latent prior`；使用完整matching-n14 likelihood和正确proposal density作self-normalized importance correction。其target identity与训练的proxy分开，gain保持同一个bounded proper prior。

| corrected proposal | new material-frequency calls | ESS | corrected event MAE | largest event error |
| --- | --- | --- | --- | --- |
| full_distill | 64 | 36.628 | .012752 | .062072 |
| diag_distill | 64 | 29.463 | .025983 | .155896 |
| gaussian_distill | 64 | 23.873 | .018627 | .090630 |

这说明C1模型可以进入合法physical probability correction接口，不能说明full稳定优于另两种方法或传统Bayesian sampling。只有一个已暴露development observation、每种一次64-draw实现，重要事件的MC误差仍可约.03–.10；SNIS在有限样本下有bias，ESS和defensive support都不保证没有漏mode。例子中full的某个event empirical mass为0，但对应独立reference约.04126，零bootstrap区间不能当作排除证书。模型外质量仍为null/unquantified，未访问的modeled区域没有遗漏质量上界。

原始proposal、修正weights、事件区间、ESS、失败、ledger IDs和成本都在B1的`research/joint_p2/results/CORRECTED_PROPOSALS.json`与配套arrays保存。此诊断不是fresh panel，不能用于重新选模型。它目标为声明数值full-Mie设置，仍不等于continuum Maxwell或现实model adequacy认证。

## 成本与对后续的建议

实际15模型训练wall=38.814s、CPU=37.969s；fresh评价wall=42.004s、CPU=41.440s；历史连续后处理wall=8.705s、CPU=8.524s，最终accepted后处理另wall=8.839s、CPU=8.562s。每次都记录、不能删去旧成本重置。这些C1纯统计阶段新增Maxwell/GPU为0；observations、bank、teacher、reference与fresh truths由B1唯一物理账本支付。Full-Mie correction另实付192次、wall18.865s、CPU17.797s，不能仅拿它宣称端到端优势。Parent输出处理中曾有string-parent summary产生NaN及metadata NameError，修复后的文件与原失败成本均由B1保留，二者新增物理为0；它们不是posterior normalization的科学结论。

当前建议保留 full/diagonal distillation作为小型、measurement-only的proposal；不要扩展GRU/Transformer，也不要声称raw MDN优于传统Bayesian inference或能够替代独立physical reference。KL≈.216远大于远期.02目标，event mean MAE达.02研究目标并不补足这一差距；直接CDF复核也确认材料内部概率仍不够准确。连续reference已经纠正，下一项真正有价值的工作应是同预算、多observation的corrected proposal收益与重要conditional tails，而不是再次在同fresh panel调参数。没有新的实际授权和明确全成本收益时，停止raw MDN完整posterior优越性主张，保留其benchmark与downstream proposal作用。

这里的normalized MDN、logistic transform与distillation本身属于已知方法。多样采样、低残差候选或这份受限bench的结果，都不足以单独支持IEEE TAP原创物理机制。补偿等价性、context更新及联合最终方向由B1独立物理与decision实验给出。

