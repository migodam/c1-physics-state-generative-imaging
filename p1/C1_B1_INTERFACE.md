# C1–B1 共同 posterior benchmark 的最小接口

本阶段交付的是可运行的 NumPy 数据与函数约定，不是新的物理 benchmark。实现位于 [`c1p1/benchmark.py`](c1p1/benchmark.py)，纯接口测试位于 [`tests/test_benchmark.py`](tests/test_benchmark.py)。接口不导入 B1 或 C1 物理内核，不加载 bank，不在构造或评价时调用 Maxwell。P1 新增 Maxwell/B1 调用的授权均为 0。

## 为什么现在不能直接比较 C1 与 B1

已读 C1 `src/c1/material.py`、`probability.py`、`data.py`、`configs/protocol_v2.json` 与 `B1_REFERENCE_COMPATIBILITY.md`。当前 P0 在共同的连续二维 latent `x ∈ (-1.5,1.5)²` 上使用 exponential-tilt proper prior，再映射到 13 个向量电偶极的两区域材料。观测是四个实 intensity readout，加固定尺度的实 Gaussian 噪声。符号区域仅为生成后的材料参数事件，不是生成输入或孔洞拓扑类别。

只读检查了 B1 正式仓库的 `README.md`、`DECISION.md` 与 `configs/protocol.json`。其已发布模型采用三个固定同心模板，连续复介电常数 uniform prior；16 个复散射通道、3D vector-Mie、实化噪声及可选 gain nuisance。它与 C1 的 prior、材料空间、观测和 solver 均不同。B1 的已有 posterior 不是 C1 P0 的真值。

另读到本地下一阶段包 `Gaussian/B1/b1-e0-posterior-audit/inputs/b1_next_2026_10_13/` 的报告、README 和 Codex 任务书，以及 `research/b1_next/e0/results/PROTOCOL.json`。该包建议统一径向 implicit material field；E0 定义仍是原三个模板、原 uniform prior、冻结 7×7 response interpolation 的连续 target。**在这些已读本地文件中，未找到 B1-2 最终统一径向 benchmark 的完整 prior、likelihood、solver 与 reference 定义。** 因此不宣称 B1-2 实验已成功，也不把研究建议登记成已完成参考。

该包 E1-U 的具体建议为 `u_z(s)=1+z_c B_c(s)+z_s B_s(s)`：两个 compact radial bump、`u<0` 为 vacuum、`u>0` 为共享复 ε，roots 转成 Mie 层边界，共四个实参数。文档仍允许 Gaussian 或有界 coefficient prior，没有给出唯一最终 prior、退化 root/薄层处理规则及配套最终 reference。这个建议可以由接口表达，不能直接当作已完成共同 benchmark。

B1 公共来源：[正式仓库](https://github.com/migodam/b1-probabilistic-maxwell-imaging)、[正式材料/物理配置](https://github.com/migodam/b1-probabilistic-maxwell-imaging/blob/main/configs/protocol.json)。本次工作不读取 B1 `.npz`、不执行其代码、不改写 B1 文件或账本。没有为接口重新实现 Mie 或 13-cell forward。

## 约定中的对象

| 对象 | 表达什么 | 必须由提供方说明 |
|---|---|---|
| `TargetIdentity` | 材料空间、support、base measure、prior、likelihood、noise、nuisance、acquisition、solver、reference target 与 observation 标识 | 每个标识对应的实际定义；不是 hash 或正确性证书 |
| `MaterialSpace` / `MaterialSupport` | 统一 latent、支持与可选材料场 decoder | 坐标/单位、场参数化、分辨率、被动性、decoder 是否降维 |
| `ProperPrior` | 对指定 base measure 的 normalized prior log density 与 sampler | proper normalization 的解析或数值依据 |
| `MeasurementMetadata` / `Observation` | 观测、noise/nuisance/acquisition 元信息、数据曝光状态 | real/complex packing、单位、noise covariance、nuisance prior 与边缘化方式 |
| `MaxwellForwardAdapter` | 可选、惰性的 `predict` 约定 | solver id、实际求解成本与先付费机制 |
| `ContinuousReference` | 可计算的连续 posterior density | 基准测度、normalization 与近似误差 |
| `SampleReference` | posterior 样本及采样方法 | iid/MCMC/其他方法；未知 ESS 不填造出的值 |
| `QuadratureReference` | 连续 posterior 的积分节点、base-measure integration weights 与 log density | 独立 refinement、误差与是否 surrogate target |
| `MaterialEvent` | 生成后判定的可测材料事件 | 事件阈值、空间尺度与规则；不传给 generator |
| `CandidateFeatures` | 候选材料与观测对应的 state/history | 来源、候选坐标、历史状态与准备成本；不得来自真实未知材料的 oracle |
| `CostRecord` / `ProbabilityEvaluation` | 阶段成本和概率评价结果 | 训练、准备、评价、采样、失败与 warm reuse 分别记录 |

`BoxSupport` 只是当前低维有界模型的便利实现。`MaterialSupport` 是轻量 Protocol，未来可接非箱形或无界 latent，不要求 B1 改成 C1 的两区域/13-cell 模型。

`ConditionalMaterialDensity.log_prob(latent, observation)` 与 `sample(rng, count, observation)` 接受测量，不接受 sign/topology 标签。材料场的概率是 latent law 经 decoder 的 pushforward；低维 decoder 的密度不等于全体素 Lebesgue density。连续单张材料图的点概率为 0，事件概率才有非零质量。

## 密度、prior measure 与 quadrature 不混用

若 base measure 是 `dx`，节点 integration weights 为 `w_i ≈ dx`，则

```text
posterior_log_density_i = log_prior(x_i) + log_likelihood(y | x_i) - log_evidence
posterior_quadrature_mass_i = w_i * exp(posterior_log_density_i)
```

`QuadratureReference.integration_weights` 是 `w_i`；`probability_weights` 才是用于 expectation 的归一化节点质量。节点本身不是新的离散 prior，不能给每个节点相同 prior probability 代替原连续先验。C1 已有 `grid['w']` 正是连续积分权重；原 `reference(... )['weights']` 是 posterior 节点质量，两者不得互换。

如果 B1 输出的是相对于 prior measure `μ₀` 的 density，接口同样可以表达，但须把 `base_measure_id` 明确改为该 `μ₀`，integration weights 也必须积分同一测度；与 C1 的 `dx` log density 比较前必须完成正确的测度转换。改变变量时推送同一个 prior，并保留变换 Jacobian；不能在新坐标里重新使用 uniform prior 后仍称为同一 posterior。

`reference_target_id` 描述实际统计/数值目标，例如 `full-Mie@specified-settings` 或 `frozen-poly7-likelihood@specified-settings`。二者必须用不同标识。参考 artifact 的 `reference_id` 可随 refinement 改变，而目标本身未变时 `reference_target_id` 保持相同。误差字典仅记录已测的数字，不凭空填全域保证。

## 最小接入示例

已提供无需占位符的真实只读 C1 接入函数：

```python
from c1p1.adapter import saved_p0_benchmark
benchmark = saved_p0_benchmark(observation_index=0, split_name="validation", order=64)
```

它装载已保存的 P0 观测与 response bank，并绑定正确的连续 prior、likelihood、sigma、观测身份和 quadrature；`forward=None`，同时禁用旧物理backend，不产生新求解。此函数可运行，相关测试由P1独立账本记录。未来B1只需按同一约定提供自身资产，无需迁就C1的物理模型。当前训练好的C1密度head仍是2维，未来径向latent维数及nuisance确定后需要相应扩维和重新训练，不能称即插即用已验证。

从 C1 根目录让 `p1` 可导入即可使用接口。下面只接已有 C1 P0 数据，不触发任何物理调用；`x, w, logp` 必须由调用方从已经保存、且定义匹配的 reference 获得：

```python
import numpy as np
from c1.material import prior_logpdf, prior_sample
from c1p1.benchmark import (
    TargetIdentity, BoxSupport, MaterialSpace, ProperPrior,
    MeasurementMetadata, QuadratureReference, PosteriorBenchmark,
)

# 这些语义标识应与实际冻结配置一致；不要把不同 solver 共用一个 id。
identity = TargetIdentity(
    material_space_id="c1-two-region-latent-v1",
    support_id="open-square-bound-1.5",
    base_measure_id="Lebesgue-dx1-dx2",
    prior_id="connected-exponential-tilt-v1",
    likelihood_id="c1-p0-intensity-gaussian-v1",
    noise_id="fixed-real-sigma-v1",
    nuisance_id="none",
    acquisition_id="c1-p0-two-source-four-readout-v1",
    solver_id="ACTUAL_SAVED_P0_SOLVER_ID",
    reference_target_id="ACTUAL_SAVED_P0_REFERENCE_TARGET_ID",
    observation_id="ACTUAL_SAVED_P0_OBSERVATION_ID",
)
space = MaterialSpace(identity.material_space_id, identity.support_id,
                      identity.base_measure_id,
                      BoxSupport(np.full(2, -1.5), np.full(2, 1.5)),
                      ("x_core", "x_shell"),
                      field_description="restricted two-region material latent")
prior = ProperPrior(identity.prior_id, identity.base_measure_id,
                    prior_logpdf, prior_sample,
                    "analytic product exponential-tilt normalizers")
metadata = MeasurementMetadata(
    identity.likelihood_id, identity.noise_id, identity.nuisance_id,
    identity.acquisition_id, packing="four real intensity readouts",
    units="known-reference-normalized intensity",
    noise={"distribution": "real Gaussian", "sigma": "per observation"},
)
reference = QuadratureReference(identity, "saved-P0-reference", x, w, logp,
                                "continuous quadrature; saved refinement errors")
benchmark = PosteriorBenchmark(identity, space, prior, metadata, reference)
# benchmark.forward is None. No solve is performed.
```

上述 `ACTUAL_SAVED_...` 需要使用实际资产里的定义，示例本身不为尚未确认的 B1-2、未知 solver 或 observation 编造身份。

`PosteriorBenchmark` 在绑定时检查 prior/base measure、材料空间、noise/nuisance/acquisition 元信息、reference target 与可选 solver 标识。`require_same_target` 在模型与参考比较前逐项匹配。标识一致是必要条件；调用方仍需确认函数内容、坐标转换和 likelihood 完全相同。修改配置但复用旧语义标识会违反约定，接口不能仅靠字符串证明实现等价。

`observation_id` 绑定这一次条件 posterior：模型密度 callback 必须对应同一个 y 与它的实际 noise/nuisance 元信息。改变 y 时为目标记录绑定新的 observation id，不能拿同一 prior/solver 下另一个 y 的 reference 进行比较。`PosteriorBenchmark.observation` 可选绑定实际观测；若绑定，id 与 measurement metadata 必须一致。

## 评价与成本边界

`evaluate_quadrature` 仅在已有 reference 节点上调用提供的模型密度，输出 KL、proper conditional log score、事件质量、均值/方差与 reference/model 积分质量。模型节点质量不被偷偷归一化；例如模型只积分到 0.5 时，会保留 `model_integral=0.5`，不能用除以总和的方法掩盖尾部缺失。缺少 support 时 KL 可以为 `inf`；数值 overflow 会明确拒绝该次评价，不能作为已收敛指标排名。

节点 KL 仍是数值近似。reference 积分、模型积分、log score 与材料 CDF/分位数必须分别 refinement；事件质量稳定并不保证 within-region shape 或 tails 已收敛。接口不提供自动正确性门槛，不把软件测试称为科学成功。P1 的完整实验评价可在这些对象之上报告 CDF、quantile、coverage、event log odds 与 parent-level uncertainty。

`CostRecord` 可记录 training、density evaluation、sampling、candidate preparation、失败与缓存准备成本。未提供计时的评价会返回 `measured=False`，不把未计时过程解释为免费。CPU 与 wall 不互相替代；物理 material-frequency、forward RHS、derivative RHS、adjoint RHS 和 B1 physical calls 分开列。这里不提供 scheduler 或新的预算框架，由 P1 独立账本负责实际计费。

`CandidateFeatures.origin` 只接受 `candidate_and_observation` 或 `saved_candidate_pilot`。这防止显式把 truth oracle 标成合法输入，但不是任意 Python 数据来源的安全证明。调用方还须审查具体 feature 生成函数；同一测量派生的物理状态不增加独立 likelihood 信息。

## B1-2 接入前仍缺的资产

1. 最终统一径向 latent/field、support 和 proper prior measure，包括 solid/cavity 事件是否具有正 prior mass；仅有设计文档不够。
2. 完全固定的 acquisition、measurement packing/单位、noise 与 nuisance prior/边缘化 likelihood。
3. 对应的 solver adapter 与准确设置；如果参考依赖 surrogate，应另外公开数值目标与 full-solver 差异。
4. 对事件质量、CDF、mean、variance、quantile、tails 分别有数值误差说明的 reference，以及曝光状态与独立评价 parent 定义。
5. 允许复用的只读数据/response bank 和完整成本来源。当前接口不读取它们，也不假设它们已经存在或已经通过实验。

定义匹配后，B1 可以提供 reference，C1 提供同一 target 上的 compact conditional density；统一径向模型可作为首个共同 benchmark。该台架仍是径向受限支持，不等于任意未知三维拓扑成像成功。

## 测试与执行状态

测试已编写，覆盖逐项 target mismatch、continuous integration measure、模型质量不被重归一化、支持边界、oracle 来源拒绝、成本合法性和无 solver 隐式调用。委派任务不运行测试；由主代理在 P1 统一受控任务中计费运行，实际测试结果以其账本和测试输出为准。
