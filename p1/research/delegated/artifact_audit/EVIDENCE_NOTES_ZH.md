# 旧 C1 证据可支持的诊断范围

以下仅作静态抽取和解释边界，不对 C1-P1 新模型作结果预判。

## 数据、优化与checkpoint

`src/c1/data.py`先按 connected exponential tilt prior 抽材料 parent，再重用材料生成不同独立噪声。P0共256个train parent、512个joint training pair；validation和test各64个parent/observation。U0共1024个train parent、4096个pair，validation/test各128个parent、256个observation。这是从连续 prior×measurement likelihood 联合律获得的有限监督样本，既不是按sign/topology均匀分组，也不是由reference nodes建立的离散材料先验。

`src/c1/train.py`使用float32 normalized inputs/states和float32 logit targets；Adam lr=0.001、batch=128、grad norm cap=5。每步从训练行有放回抽mini-batch。在250倍数checkpoint用整个validation的平均 `-log q_z`选最小值；所有模型共同该准则。

记录里的 `validation_log_score_z` 和 `truth_log_score` 都是负对数密度，越低越好；它们不是“分数越大越好”的log概率。material变换Jacobian固定且不依赖模型参数，因此使用同一validation的z-space NLL与x-space NLL选择checkpoint排序相同；不能误把两种数字差别叫目标错配。

每个checkpoint的 `train_loss` 只是当步、参数更新前的随机mini-batch loss；validation score是更新后的full validation值。它们不是严格同一时刻和同一数据范围的full training-vs-validation gap，不能据精确差值估泛化误差。但是P0所有MDN运行首尾训练minibatch NLL下降、验证NLL持续大幅升高，且最优checkpoint极早，足够构成“继续训练并不改善泛化”的实证记录。

| P0模型 | seed | 最优step | first validation NLL_z | last validation NLL_z |
|---|---:|---:|---:|---:|
| mdn |2610131|250|3.306212|32.959797|
| mdn |2610132|250|3.200454|8.602470|
| mdn |2610133|250|3.251872|10.683739|
| mdn_state |2610131|250|3.298168|9.948760|
| mdn_state |2610132|250|3.214672|18.098507|
| mdn_state |2610133|250|3.183898|20.556381|
| mdn_memory |2610131|250|3.197141|9.008072|
| mdn_memory |2610132|250|3.074471|8.640055|
| mdn_memory |2610133|500|3.169583|12.246708|
| measurement |2610131|750|3.483997|8.448617|
| measurement |2610132|1000|3.494770|80.506699|
| measurement |2610133|750|3.488793|55.739975|

表的较长小数以 `TRAINING_HISTORY_COMPLETE.csv` 原始JSON浮点值为准。250步前没有验证记录，无法排除最佳checkpoint早于250；增加总update也不能自动解决该信息缺口。U0多数模型最后一步最优则留下优化未完成的可能性，不能把P0现象推广成所有问题的必然规律。

## normalization

`data.py`仅在train split计算measurement mean/std（std floor=1e-5）和state mean/std（按全部train观察和两个pilot rounds合并，std floor=1e-4）。后续split使用同一保存数组。

U0 measurement std = [0.71901135,0.70573557,0.39552347,0.24674136,0.36684242]；P0 = [0.56505061,0.57567079,0.56332424,0.55824801,0.36688378]。当前记录不显示measurement除以零或test参与normalization。U0 state维18，std最小floor1e-4、最大88.99482；P0 state维38，std最小0.00055950、最大85.01984。不同物理量原始scale跨度已明确standardize，但没有alternative scaling或test extrapolation专项记录，不能排除conditioning问题。

## reference与model density的分离

`data.reference`计算 `prior_logpdf(x)+gaussian_loglike(F(x),y,sigma)`，证据计算加入积分weights w，posterior node probability为密度×w归一化。quad节点仅是连续积分数值工具。

U0初始64/96参考event max差2.3424e-6未达到旧1e-8要求；96/128后变为2.9168e-12，logZ max差1.3979e-11，三个预先定义adaptive case中最大6.3088e-14。旧初始reference和evaluation被保留。此次细化未重新训练或改变模型选择。

P0已有40/64 bank共1600+4096次forward reference，event max差0.0017242、logZ max差0.00099294。没有独立adaptive P0 reference，更没有continuous solver mesh收敛结论。这支持已保存panel上event reference相对5%model error较小，但不能视为绝对无误差。

reference本身的event/evidence稳定并不推出integral `p(log p-log q)`稳定，因为log q在tail可以很陡。现有MDN family P0平均KL refinement差约0.00035–0.00367，个别observation差达到0.03227；direct flow平均差0.06160、5.89085、3.11201，单observation可到94.72199。U0 memory平均变化1.70079/1.02126，单观察变化79.29179/57.86062。所有细化值已收集入 `DENSITY_SCORE_REFINEMENT.csv`，不隐藏失败。

`Density.log_prob_x`包括logit Jacobian，open support外为minus infinity。Diagonal Gaussian mixture在logit space、affine flow由standard normal及可逆triangular coupling构成，数学定义均为normalized conditional density；现有结果中没有完整q密度积分收敛表。数学归一化不保证float32运算或tail积分准确。

## 支持与tail

原float32 sigmoid对相同冻结weights、相同latent RNG的诊断出现156个boundary draw。新float64 decode减少为7，均为U0 mdn seed1/3的极端latent（最大绝对值43.69007/45.38064）。P0在该repeat诊断为0。没有截断、投影、重抽样或重训。旧gallery仍使用其当时验证的材料，不偷改坐标。

这记录了finite precision tail问题，不能单靠156/3932160draws判定其造成5%事件误差，特别是sign事件对近boundary rounding不如logdensity敏感。也不能因为float64主要修复边界，就宣称tail probability已准确。

## 旧指标包含与缺失

primary metrics有reference KL、四sign事件MAE/每观察max误差p95、expected Brier excess、region weighted recall/precision、missing-region count、单个truth的proper NLL、coordinate90%coverage及reference interval content error。每个observation的q event masses基于1024次iid sample；old maximum sampling SE小于等于约0.015625。q参考事件频率不是对数密度积分得出的精确event mass。

旧输出没有系统记录continuous CDF error、mean/variance/quantile误差、within-region conditional形状、tail event probability、数值密度normalization积分。虽然KL理论上对完整形状敏感，但flow KL未收敛，不能以它覆盖所有missing diagnostics。

`CALIBRATION_TABLES.csv`的bin按预测q定义，包含q_mean、reference_mean和truth_event_frequency。truth材料parents有限；同一个parent的不同噪声不是独立材料。旧paired bootstrap先平均parent内噪声，再抽parent，跨seed分开报告。三个training seed共用数据，所以稳定性证据不等于三个新holdout。

## 现有state结论的范围

P0候选state是同一预计算2-round pilot context，generator不对每个production material refresh state。零/置换现有state恶化validation，说明model用了该feature；这是已训练model的reliance诊断，不是新训练同family ablation，也不是增加独立measurement信息。

P0原validation三seed都选择measurement flow，EVALUATION主paired rows依该规则比较；DIAGNOSTICS另存MDN同family比较。memory对MDN KL改善seed1/2但恶化seed3，不能以不稳定flow的巨大KL差升级state价值。原proper truth NLL差跨seed均有CI跨0；所有saved comparison完整保留。

## 账本、曝露及本次动作

JOB_LEDGER共23个result：21 PASS、2 FAILED_CHARGED。JOB_STARTS只存后17次开始，前6条为早期launcher阶段；这些job仍有完成receipt和cost。physics账本含7967 charge/result操作对，其中6个cached action material_frequency=0，总MF仍7961。总RHS =15922 forward+5264 derivative+4 adjoint。

原成本CPU146.839642 measured+10reserve，wall166.736402 measured+12reserve。imports/failures/setup包括在job；nested model/physics timing不能再次叠加。interactive/model-service reasoning、network及人工阅读未全计时，旧账本明确未assert free。

本次只执行stdlib文件读取/JSON和CSV扁平化、已有指标的描述性汇总，以及saved normalization arrays解码。没有new CI/bootstrap、model infer、new test、训练、Maxwell/B1/GPU或digest。所有输出写在本目录；已公开旧test只可标为exposed regression panel。
