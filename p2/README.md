# B1 × C1 Joint P2：共同材料 posterior 的连续密度

这是四维共同径向材料 target 的 CPU 密度实现。P1 的二维模型、原训练结果、资源账本保持原样。本模块不导入物理 solver，不读取旧 locked truth，不训练 topology 专家，也不提供 memory/state 输入。

**已经完成**：15模型×3000共同updates、冻结后的64 fresh parents×两噪声评价与accepted reference复核。最终full/diagonal三seed event MAE=.01321508/.01302798，差.00018710、95%区间[−.00057984,+.00088033]；没有明确event优势。旧13阶latent KL=.21572/.27482，未重算、不等于physical KL。Accepted material grid-CDF≈.098906/.101614、重要事件内≈.141761/.161768，仍未达到.01。Fresh表是raw proposal；192 full-call correction仅是一次development诊断。先读 [研究报告](C1_P2_RESEARCH_REPORT_ZH.md)、[裁决](C1_P2_DECISION.md) 与 [accepted连续复核](results/continuous_accepted/CONTINUOUS_COMPARISON.md)；[旧方法表](results/summary/METHOD_SUMMARY.md)作为历史保存。

`BoundedMDN` 的 encoder 接受32个实化 Maxwell 通道、log sigma 与 gain prior SD，默认两个96单元 hidden layers。四个连续坐标为 `z_c,z_s,Re(epsilon),Im(epsilon)`，界为 `[-3,1]×[-3,1]×[2.2,2.8]×[.04,.16]`。八个 mixture components 仅是可交换的密度组件，不能解释为八类材料。完整 covariance 用严格正对角 Cholesky；diagonal 以及单 Gaussian 是明确对照。

## 概率语义

网络在变换坐标 `t=log(x-lower)-log(upper-x)` 输出 normalized Gaussian mixture，精确 Jacobian 给出相对于四维 `dx` 的 normalized density。其生成材料 field 的 law 是 decoder pushforward：部分 inactive coefficients 映射为同一材料，材料空间可有实心几何原子质量，四维 latent 本身没有离散 topology 输入。

`log_prob(latent, condition)` 返回材料 latent density；`sample(condition,n,generator)` 返回 `MaterialDraws`，保留 `latent`、原始 `transformed`、选中 component 和 `rounded_endpoint`。极端 transformed 坐标在浮点下可能落到界面，接口标记此情况，不裁剪、不重采样。`log_prob_draws` 可直接从保存的 transformed 坐标计算 density。

`export_parameters(condition)`、`log_prob_batch`、`sample_parameters` 为 NumPy 后端提供便携接口。模型密度不在 reference 节点上重归一化。未修正的 generated frequency 是模型事件频率；只有独立 target reference 或合法 full likelihood correction 才能支持 posterior event mass。

## 数据与执行

训练 NPZ 必需字段：`train_y[N,34]`、`train_x[N,4]`、`train_teacher[N,T,4]`、`validation_y[V,34]`、`validation_teacher[V,T,4]`、scalar `target_identity`。可以提供对应 teacher probability weights 与 parent IDs。条件 normalization 只由 train observations 拟合；所有 checkpoint 用同一 posterior-expected validation log score。训练入口不接受 final test 文件。

共同训练 job 包含 full/diagonal distillation、单 Gaussian distillation、full/diagonal joint NLL，每项使用三个 seeds 26101321/22/23。同步训练循环保证实际更新数一致，最多3000步、总训练wall最多600秒；每250步做共同验证。早停只选择相同预算内的 checkpoint，没有额外 physics、GPU 或未计费 teacher。

实际数据由B1共同reference提供，保存位置为：

- `research/joint_p2/results/C1_TRAINING_DATA_REF13.npz`：实际训练的细化teacher，256 train/64 validation observations，512 draws/obs。
- `research/joint_p2/results/C1_FINAL_DATA.npz`：模型冻结后新生成的128 observations、4096 reference-node resamples/obs、精确reference事件积分与新truth。
- `research/joint_p2/results/FRESH_CONTINUOUS_REFERENCE_ACCEPTED.npz`：最终positive-partition固定65点CDF、conditional CDF、accepted mean/SD与各row reference orders。
- `research/joint_p2/results/ACCEPTED_REFERENCE_AUDIT.json`与`REFINED_FRESH_EVENT_COMPARISON.json`：全128复核与三项17/21升级记录、最终主要event比较。
- 原`FRESH_CONTINUOUS_REFERENCE.npz`、node/truncated/partition文件和C1原始结果保留为历史，不覆盖。

从C1仓库根目录设置B1路径。以下为当前工作区的真实路径；从公开包复现时将`B1_P2`改为解包后的B1仓库目录，内部相对文件名保持不变。

```bash
B1_P2="/Volumes/migodam's-external-brain/Research/Inv_SLAM/Gaussian/B1/b1-joint-p2"
export PYTHONPATH=p2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 使用已冻结参数重放评价；新输出目录保留原始结果。
python -m c1p2.evaluate "$B1_P2/research/joint_p2/results/C1_FINAL_DATA.npz" \
  --models p2/results/models/*-2610132[123].pt --output p2/replay/evaluation
python -m c1p2.summarize p2/replay/evaluation/EVALUATION_SUMMARY.json --output p2/replay/summary
python -m c1p2.continuous_audit "$B1_P2/research/joint_p2/results/FRESH_CONTINUOUS_REFERENCE_ACCEPTED.npz" \
  --evaluation p2/replay/evaluation --output p2/replay/continuous_accepted

# 重训只在有对应计算授权时执行；没有追加物理调用，但须计训练wall。
python -m c1p2.train "$B1_P2/research/joint_p2/results/C1_TRAINING_DATA_REF13.npz" \
  --output p2/replay/models --wall-seconds 600 --max-steps 3000
```

环境需要NumPy、SciPy、CPU PyTorch；本实现没有新CUDA依赖。所有受控科研命令由B1统一计费job启动，使用新的job编号，不能将重放称为独立新validation。六项必要密度检查已通过，独立重放入口为`python p2/tests/test_density_probability.py`；无需pytest，也不执行额外smoke、SHA或旧campaign。

final NPZ 使用 `target_identity`、`y`、`parent_ids`，加 `reference_points/reference_weights` 或 `reference_samples`；可选 normalized `reference_log_density`、base-measure `integration_weights`、event IDs、truth。没有 reference log density 时 KL 明确不可用。每个 generator condition 至少8192个 iid draws，报告 binomial sampling误差和零计数上界，不能将这类误差当作 reference/Maxwell 认证。

原final NPZ的`reference_event_mass[O,12]`是13阶版本；它使node-resampling误差另列，但原`evaluate/summarize`重放仍是历史13阶结果。最终event主要指标使用B1 `REFINED_FRESH_EVENT_COMPARISON.json`，不得把原`C1_FINAL_DATA.npz`悄悄换target或覆盖。Actual material坐标为中心半径/R、环形半宽/R、Re ε、Im ε，zero-radius原子保留；模型自身truth区间不因reference细化更改。

重要事件阈值是 reference mass≥.05，mass-aware discovery 为模型质量≥.01，false exclusion 为重要事件模型质量<.005，false confidence 为模型最大质量≥.90且 reference 至少两个重要事件。有限 MC 估计的这些标记是操作性指标，保留其采样误差。Top-K 取覆盖至少95%模型质量的最小事件集合，分别报告该集合的模型和 reference 质量，不重新归一化。

`summarize`不训练，分列seeds、按parent配对两次噪声；2000 bootstrap、seed26101329。它保留旧13阶latent-KL expectation及其数值scope；accepted event/mean/SD并不认证该KL，也不能因decoder非单射而叫physical-material KL。Bootstrap条件于固定数值参考和MC draws，不包含continuum/solver保证。

原`EVALUATION_SUMMARY.json`参考CDF/quantiles保持`NODE_CDF_DIAGNOSTIC`；历史`results/continuous/`也保留。最终`results/continuous_accepted/`复现相同8192模型draws，与accepted positive-partition积分比较；模型、checkpoint及自身truth coverage不变。固定65点grid最大值不是uniform supremum，inverse quantiles只在固定前八个提供。

`ACCEPTED_REFERENCE_AUDIT.json`记录：全部128做event/mean/SD13/17核验，obs12/39/48初始13失败后升级17、经21检查通过，无unresolved；positive边际/important conditional CDF单调违例均0。Fixed前八个partition CDF13/17差最大.00097375，conditional差.00414377；direct inverse quantile/prior-width差.00025561、coherence通过。CDF13/17 scope仍是这个子集加adaptive3，不外推全support/full-Mie校准。旧node quantiles不升级成direct。

三个distillation模型（均model seed26101321）的proposal还在B1 `CORRECTED_PROPOSALS.json`保存了一次development `large_cavity/C` correction：每法64 full calls，`0.9q+0.1prior`和正确importance density；full/diag/Gaussian corrected event MAE=.01275/.02598/.01863、ESS36.6/29.5/23.9。它只是一case、一次实现，SNIS有限样本有bias、零count不证明排除、ESS不证明无漏mode。不得以此重选模型或声称算法优于Bayesian sampling。

实际训练wall38.814s、CPU37.969s；评价wall42.004s、CPU41.440s；历史连续后处理wall8.705s、CPU8.524s，accepted后处理另wall8.839s、CPU8.562s。旧成本全部保留；这些C1纯统计阶段新Maxwell/GPU为0。Correction另192物理调用、wall18.865s，teacher/setup/fresh truths及reference成本由B1唯一账本承担。旧B1 384/384与C1 P1只读、不重置。Parent的string-parent NaN summary、metadata NameError修复与失败记录也保留，新增物理0。联合科学裁决、补偿、context更新与发布由B1负责；weights及大数组复用现有C1/B1 Release，不新建repo。
