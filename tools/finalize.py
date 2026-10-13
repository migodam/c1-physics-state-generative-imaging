"""Write evidence-based C1 delivery from completed paid outputs; no solves or training."""
from pathlib import Path
import json
import csv
import platform
import numpy as np
import scipy
import torch
from src.c1.accounting import PhysicsLedger

ROOT=Path(__file__).resolve().parents[1]

def dump(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def cost_snapshot():
    jobs=[json.loads(s) for s in (ROOT/'results/JOB_LEDGER.jsonl').read_text().splitlines()]
    reserve=json.loads((ROOT/'results/RESOURCE_RESERVES.json').read_text())
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text())
    by_stage={k:dict(measured_cpu_seconds=sum(j['cpu_seconds'] for j in jobs if j['stage']==k),
                    reserve_cpu_seconds=reserve['cpu_by_stage'].get(k,0),cap=v) for k,v in conf['cpu_stage_caps'].items()}
    return dict(project='C1',date='2026-10-13',scope='local registered computational jobs; imports, failures, setup and diagnostics included',
        physics=PhysicsLedger().summary(),measured_cpu_seconds=sum(j['cpu_seconds'] for j in jobs),
        conservative_cpu_reserve_seconds=reserve['cpu_seconds'],charged_cpu_seconds=sum(j['cpu_seconds'] for j in jobs)+reserve['cpu_seconds'],
        measured_job_wall_seconds=sum(j['wall_seconds'] for j in jobs),conservative_wall_reserve_seconds=reserve['wall_seconds'],
        charged_job_wall_seconds=sum(j['wall_seconds'] for j in jobs)+reserve['wall_seconds'],
        cpu_cap=conf['cpu_seconds_cap'],wall_cap=conf['wall_seconds_cap'],stage_costs=by_stage,
        local_gpu_seconds=0,primary_training_runs=24,failed_job_attempts=sum(j['status']!='PASS' for j in jobs),
        original_package_history={'new_Maxwell_evaluations':0,'trained_C1_models':0,'algebra_attempts':2},
        excluded_unmeasured='interactive/model-service reasoning, worker model-service inference, scholarly network retrieval and reading/visual review are not fully timed; not asserted free or included in total local study elapsed time',
        nested_timing='internal training/physics/context timings are nested within JOB_LEDGER and must not be added again',
        cache_scope='one shared executed context bank; cold method costs reconstruct its use, not charged anew for every seed',
        b1_physics_calls=0,b1_mutations=0,sha256_checks=False,source_of_truth='results/JOB_LEDGER.jsonl and results/PHYSICS_LEDGER.jsonl')

def main():
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text())
    results={p:json.loads((ROOT/'results'/p/'EVALUATION.json').read_text()) for p in ['u0','p0']}
    di=json.loads((ROOT/'results/p0/DIAGNOSTICS.json').read_text())
    cost=cost_snapshot();dump('C1_RESOURCE_LEDGER.json',cost)
    metrics=[];parent_rows=[];calibration=[];methods={}
    for problem in ['u0','p0']:
        observed=[json.loads(s) for s in (ROOT/'results'/problem/'OBSERVATION_METRICS.jsonl').read_text().splitlines()]
        kinds=json.loads((ROOT/'results'/problem/'MODELS_FROZEN.json').read_text())['models'];means={}
        for kind in kinds:
            groups=[]
            for seed in conf['training_seeds']:
                key=f'{kind}-{seed}';m=results[problem]['models'][key];r=json.loads((ROOT/'results'/problem/'models'/f'{key}.json').read_text())
                metrics.append(dict(problem=problem,model=kind,seed=seed,parameters=r['trainable_parameters'],updates=r['updates_executed'],
                    checkpoint=r['checkpoint_step'],reference_status=results[problem]['reference_status'],
                    **{k:m[k] for k in ['KL_reference','event_mass_MAE','event_mass_max_error_p95','mode_weighted_recall','mode_precision','coordinate_90_coverage','truth_log_score','quality_gate']}))
                groups.append(m)
                rows=[a for a in observed if a['model']==kind and a['seed']==seed]
                for parent in sorted(set(a['parent'] for a in rows)):
                    a=[t for t in rows if t['parent']==parent]
                    parent_rows.append(dict(problem=problem,model=kind,seed=seed,parent=parent,noise_replicates=len(a),
                        KL_reference=float(np.mean([t['KL_reference'] for t in a])),
                        event_mass_MAE=float(np.mean([t['event_mass_MAE'] for t in a])),truth_log_score=float(np.mean([t['truth_log_score'] for t in a]))))
                target=np.load(ROOT/'results'/problem/'test.npz')['x']
                for event in range(4):
                    predictions=np.array([a['q_event_masses'][event] for a in rows])
                    ref=np.array([a['reference_event_masses'][event] for a in rows])
                    truth=((target[:,0]>=0).astype(int)*2+(target[:,1]>=0).astype(int))==event
                    for lo,hi in zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1.000001]):
                        mask=(predictions>=lo)&(predictions<hi)
                        if mask.any():calibration.append(dict(problem=problem,model=kind,seed=seed,event=event,
                            bin_low=lo,bin_high=min(hi,1),observations=int(mask.sum()),q_mean=float(predictions[mask].mean()),
                            reference_mean=float(ref[mask].mean()),truth_event_frequency=float(truth[mask].mean()),
                            scope='descriptive test bin; correlated noise replicates are not new material parents'))
            means[kind]={s:float(np.mean([g[s] for g in groups])) for s in ['KL_reference','event_mass_MAE','mode_weighted_recall','mode_precision','coordinate_90_coverage','truth_log_score']}
        methods[problem]=means
    for name,rows in [('SUMMARY_METRICS.csv',metrics),('PARENT_METRICS.csv',parent_rows),('CALIBRATION_TABLES.csv',calibration)]:
        with (ROOT/'results'/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump('results/METHOD_MEANS.json',dict(means=methods,scope='mean across three seed-specific means; no pooled seed confidence claim'))
    p0=methods['p0'];u0=methods['u0']
    gates=dict(overall='NO_GO_SCALE_PHYSICS_STATE_MEMORY',next_phase='GO_LIMITED_UNIFIED_DENSITY_AND_REFERENCE_REPAIR',
      scoped_to='this fixed two-coordinate prior, two-round global pilot, three small-model training seeds; not a disproof of C1',
      implementation='PASS',test_scope='77 full tests plus one decoder-specific test; original package12 algebra checks rerun',
      no_topology_labels='PASS_CODE_AND_RUNTIME_INTERFACE_CHECKS',
      multimodal_event_retention='OBSERVED_IN_RESTRICTED_PRIOR; not arbitrary topology recovery',
      calibrated_probability_quality='FAIL_TARGET_TOLERANCES_ALL_PRIMARY_MODELS',
      u0_reference_events='PASS_EMPIRICAL_96_128_AND_INDEPENDENT_ADAPTIVE',p0_reference_events='PASS_EMPIRICAL_40_64',
      reference_score_integration='HOLD_FOR_FLOW_TAILS; MDN mean refinements small but not a uniform certificate',
      state_gain='NOT_STABLE_ACROSS_THREE_SEEDS',memory_gain='TWO_SEEDS_IMPROVE_ONE_WORSENS',
      cost_advantage_at_required_accuracy='NOT_ESTABLISHED; probability quality fails',
      vector_discrete_operator='PASS_LOCAL_DDA_GRADIENT_ADJOINT_JVP_CONTROLS',continuum_solver_validation='NOT_RUN',
      general_geometry_topology_generation='NOT_RUN; full-grid expressive examples are software controls only',
      candidate_specific_online_stochastic_evolution='NOT_RUN; current model uses frozen shared pilot context',
      b1_reference='NOT_USED_TARGET_MISMATCH',local_gpu='NOT_USED',nonlinear_3d_diffusion='NOT_RUN',OPM_current_selection='NOT_RUN',
      numerical_sample_precision='FLOAT64_FIX_REDUCES_ENDPOINT_ROUNDING_156_TO7; extreme_U0_tail_NOT_RESOLVED',
      actual_instrument_calibration='NOT_RUN',novelty='STRONG_PHYSICS_STATE_CONTRIBUTION_NOT_ESTABLISHED',
      budget='PASS_NO_EXPANSION_NO_STAGE_BORROWING',publication='NOT_PUBLISHED_LOCAL_DELIVERY')
    dump('C1_GATE_DECISION.json',gates)
    table=['|实验/模型|参考 KL（nats）|事件质量 MAE|mass-weighted region recall|90%坐标覆盖|','|---|---:|---:|---:|---:|']
    for p,ms in methods.items():
        for k,m in ms.items():table.append(f'|{p.upper()}/{k}|{m["KL_reference"]:.6f}|{m["event_mass_MAE"]:.6f}|{m["mode_weighted_recall"]:.4f}|{m["coordinate_90_coverage"]:.4f}|')
    report='''# C1 统一材料概率生成：第一轮研究与最小实验

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

'''+ '\n'.join(table)+'''

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
'''
    (ROOT/'C1_RESEARCH_REPORT_ZH.md').write_text(report)
    next_doc='''# C1 下一阶段裁决

2026-10-13。**保留统一材料概率生成；NO-GO扩大当前physics-state/GRU模型；先修复简单density的概率质量与tail稳定性。**

无需拓扑标签的共同材料support已恢复多个参数区域，P0简单MDN/state/memory的mass-weighted region recall约98–99%。但事件MAE约5%，P0 90%区间覆盖约83–85%，所有primary配置未达预设概率目标。这不是已校准的真实posterior，也不是未知任意拓扑成像。

## 真实state收益

同数据、近似同参数、同MDN family的memory相对measurement-MDN，三个seed的reference KL为：

|seed|直接MDN|MDN+memory|差(memory-direct)|parent paired95%CI|
|---|---:|---:|---:|---|
|2610131|1.080618|1.003493|-0.077125|[-0.127676,-0.022480]|
|2610132|1.017235|0.872362|-0.144873|[-0.179239,-0.114427]|
|2610133|0.892477|0.996573|+0.104096|[+0.014620,+0.194517]|

有局部收益，但第三seed有显著反向结果，平均改善约3.94%。单state的KL差为-.00336/-.02044/+.08255，亦不稳定。观测truth-log-score的memory差CI均跨0。不能把更不稳定的direct flow tail失败当成支持state路线的强证据；原validation-selected baseline与same-family控制均保留。

## 最值得继续的统一架构

下一小阶段优先**support-aware统一条件密度**：保留未标注MDN与有界conditional flow作为认真对手，先解决within-region相关形状、边界tail与校准。可以测试full-covariance components或显式有密度的prior-defensive混合，但两者都要用proper log score及精度控制；prior混合会改变fast q，不能假称免费正确posterior。下一representation用统一多分辨率被动材料场及非零fine-scale随机尾，不给generator输入孔洞数。

state只作为可选计算condition，优先测试少量任意candidate响应对比/gradient/probes。当前数据没有理由优先Transformer、Diffusion或更多GRU history。只有简单模型概率达到要求后，才值得比较每个production hypothesis自己的physics refresh与memory，并为完整Jacobian或有效路径概率/target-preserving correction付费。当前frozen shared pilot不等于已经实现这种particle-specific过程。

## 最小下一实验与停止

先在同类二维连续参考上，用一个冻结的tail-sensitive开发panel比较原MDN、改进density、以及它们的state消融；全部同数据、近似同容量，物理成本和reference误差一起报告。此时只能开展新的已记录小预算阶段，不能把已暴露test重新称blind。若进入几何/材料场扩展，先给与新prior一致的参考；B1三模板参考不能直接充当一般field真值。

推进条件是：至少三个seed上同精度cost/概率质量的稳定改善，或同总成本下proper-score/coverage的明确改善；不只best-of-K、某seed、residual或神经FLOPs。这个判断不等于研究方向永远失败。

**GO_LIMITED_UNIFIED_DENSITY_AND_REFERENCE_REPAIR**：继续低成本统一材料分布与支持研究。

**NO_GO_SCALE_PHYSICS_STATE_MEMORY**：不扩大当前网络、history或GPU训练，不宣称Maxwell状态已有稳定架构增益。

**HOLD_GENERAL_TOPOLOGY_AND_REAL_IMAGING**：未知复杂边界/孔洞、一般vector-Maxwell几何与真实测量尚未测试。表示能画孔洞的单元检查不能代替这些实验。

原始失败、数值细化、全成本及未运行项保留。详细见[中文研究报告](C1_RESEARCH_REPORT_ZH.md)、[机器裁决](C1_GATE_DECISION.json)、[结果表](results/SUMMARY_METRICS.csv)、[账本](C1_RESOURCE_LEDGER.json)。
'''
    (ROOT/'C1_NEXT_DECISION.md').write_text(next_doc)
    (ROOT/'README.md').write_text('''# C1 Physics-State Probabilistic Generative Imaging V2

2026-10-13。本地第一轮完成：统一无topology-label generator保留多个受限材料区域；概率质量FAIL，physics-state/memory稳定优势未建立。**NO_GO_SCALE_PHYSICS_STATE_MEMORY**，保留简单统一density与reference修复方向。

- [方向 V2](C1_DIRECTION_V2.md)
- [架构与实验规范](C1_ARCHITECTURE_EXPERIMENT_SPEC_V2.md) / [机器配置](configs/protocol_v2.json)
- [中文研究报告](C1_RESEARCH_REPORT_ZH.md) / [下一步裁决](C1_NEXT_DECISION.md)
- [机器裁决](C1_GATE_DECISION.json) / [计算账本](C1_RESOURCE_LEDGER.json)
- [逐seed结果](results/SUMMARY_METRICS.csv)、[逐parent结果](results/PARENT_METRICS.csv)、[校准分箱](results/CALIBRATION_TABLES.csv)
- [代码](src/c1/density.py)、[物理接口](src/c1/problems.py)、[tests](tests/test_invariants.py)、[执行命令记录](RUN_COMMANDS.md)
- [材料gallery](figures/P0_MATERIAL_GALLERY.png) / [事件概率](figures/P0_EVENT_MASSES.png) / [同family对照](figures/PAIRED_KL_DIFFERENCES.png)
- [原包完整读取回执](PACKAGE_READ_RECEIPT.json)、[原理论](inputs/source_package/C1_Research_Package/C1_THEORY_AND_ARCHITECTURE.md)、[原文献审计](inputs/source_package/C1_Research_Package/C1_LITERATURE_AUDIT.md)
- [修改/失败记录](CHANGELOG.md)、[B1兼容性](B1_REFERENCE_COMPATIBILITY.md)、[机械独立审查](research/delegated/pipeline_audit/SUMMARY.md)

所有推断state来自candidate与measurement；生成不接收topology、truth field或reference数组。P0是13-cell vector-DDA、两连续材料参数与intensity readout，不是一般三维未知拓扑posterior。每张连续材料图点质量0；图只展示iid q样本，概率是明确材料区域的估计质量，仍未达校准目标。

已有run IDs与完成的training/evaluation均拒绝重跑；不能删除或重置账本。本次实际命令与失败在RUN_COMMANDS与results/jobs中。需要重现新的整套计算时须另立记录并保留此前费用，不能在当前已完成目录偷偷开始第二遍。CPU/物理硬预算由tools/job.py与PhysicsLedger强制。

复用共享NN解释器只读依赖（路径见SOURCE_MANIFEST），pytest依赖仅放C1/vendor/test_runtime。source algebra输出写新路径，原包ledger未覆盖。没有SHA256检查。C1未公开发布；不把本地文件存在称为公开网页。
''')
    jobs=[json.loads(s) for s in (ROOT/'results/JOB_LEDGER.jsonl').read_text().splitlines()]
    lines=['# 本次实际执行命令','', '这些是已执行记录，不是要求在完成目录重复消耗预算；已完成IDs拒绝。', '']
    for j in jobs:lines.extend([f'## {j["id"]} — {j["status"]}','',f'阶段：{j["stage"]}。回执：`results/jobs/{j["id"]}/receipt.json`。','', '```text', ' '.join(j['command']), '```',''])
    (ROOT/'RUN_COMMANDS.md').write_text('\n'.join(lines))
    sources=[]
    for directory in ['src','tools','tests','configs']:
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:sources.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size))
    dump('SOURCE_MANIFEST.json',dict(project='C1',date='2026-10-13',source_package='C1_Research_Package.zip',
        input_archive_bytes=33498,original_members=9,original_manifest_digests='historical only; not checked',new_hashes=False,
        source_files=sources,solver='c1-vector-dipole13-rr-v1, independently coded known DDA equations; not B1 re-execution',
        literature='original targeted audit retained;3 primary generative/SBC metadata and DDA overview checked',
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,
             platform=platform.platform(),python_executable='Gaussian/.venv_nn/bin/python, reused read-only',local_test_dependencies='vendor/test_runtime',test_framework='pytest8.4.2'),
        agents='2 native Codex workers for known coding/tests/plot preparation and source audit; no worker executed physics/training',
        default_provider_or_model_changed=False,b1_changed=False,locked_upstream_arrays_opened=False,
        source_fields='known candidate only; no unknown true fields at inference',publication='LOCAL_UNPUBLISHED'))
    print(json.dumps(dict(delivery='WRITTEN',new_physics=0,primary_training_runs=24,physics_used=cost['physics']['used'])))

if __name__=='__main__':main()
