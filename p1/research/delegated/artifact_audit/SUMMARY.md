# C1-P1 现有资料静态抽取

此目录只记录旧 C1 的原始证据、已有指标的描述性汇总及其解释范围。没有训练、测试、solver、GPU 或新 reference 计算，也没有改写任何旧文件或账本。最终科研因果判断由父 Codex 负责。

完整读取 154 个现有文件，包括 24 个模型训练 JSON 的全部 216 条 checkpoint 历史、全部 2304 条 parent metric、373 条 calibration bin、reference JSON、23 个已完成 job 与17 条启动日志、全部15934 条物理账本记录。只另外读取了两份保存的 normalization 小数组；没有加载或执行模型权重，没有读取 B1 数据。

关键证据：

1. **P0 训练强烈恶化验证 NLL，不能把失败只归因于 diagonal covariance。** 9 个 MDN 家族运行中8个选择250-step，1个选择500-step；全都早于3000。P0 原 MDN 三个 seed 的 validation NLL_z 从第一条 checkpoint 的3.3062/3.2005/3.2519，恶化至最后32.9598/8.6025/10.6837，同时记录的训练 minibatch NLL 下降。支持 generalization/过拟合存在；不单独证明数据量、容量或优化哪项造成它。250步之前没有验证记录。
2. **U0 与 P0 的训练现象不同。** U0 12次运行中8次选择最后1500-step，state与memory全部在最后选择。不能将 P0 的“继续训练恶化”统一解释为所有模型少训；也不能证明 U0 所有方法已优化充分。
3. **联合训练样本少于观察行数所暗示的独立材料数。** U0 train=1024材料×4噪声，validation/test各128×2；P0 train=256×2，validation/test各64×1。按相同 prior 和 likelihood 生成材料监督目标，目标是 NLL_z，而不是 posterior distillation。旧 parent 表已正确平均 parent 内噪声；seed 共用同一 train/validation/test，三个 seed 不是三份独立测试数据。
4. **归一化只使用 train。** U0/P0 measurement inputs都是5维；state维数分别18/38。保存的 measurement std 范围正常非零。source 明确 train-only mean/std 和 std floors；尚无旧记录比较 alternative normalization，因此不能证明归一化不会影响优化。完整保存值见 `NORMALIZATION_VALUES.*`。
5. **reference event 误差与 q-score 积分误差必须分开。** U0 96/128节点最大 event difference=2.9168e-12；三个固定 adaptive cases最大6.3088e-14。P0 40/64 event max difference=0.0017242、log evidence max difference=0.00099294；P0独立 adaptive没有运行。都是经验一致性，不是 solver 或全域数学保证。
6. **现有 Flow KL 不能用作精确排名。** P0 direct flow 三 seed 的 coarse→fine平均 KL差为0.06160/5.89085/3.11201；U0 memory有平均变化1.70079/1.02126，单 observation变化达到79.29。P0 MDN 家族平均 KL细化差绝对值小于0.00367，但单 observation最大0.03227，不能据此声称每案例 KL 已在0.02尺度内精确。
7. **旧事件 MAE 是1024次生成样本的频率误差。** p95字段实际是每 observation 四事件最大绝对误差的95%分位，不是把所有事件混在一起的误差分位。旧结果未提供全面 continuous CDF/均值方差/within-region shape/tail mass 或 q密度积分收敛记录。不能把高 sign-region recall 解释为 topology或完整posterior质量。
8. **支持边界数值问题确实存在，但尚不足以解释5%的事件误差。** 旧 float32重复诊断总156个boundary draw；float64仅剩7个，均来自U0 MDN两个seed的极端latent，P0的同诊断为0。没有 clip、重抽样或重新训练。旧 q density包括logit Jacobian，理论 family归一化；“理论归一化”不等于已做完整数值积分诊断。
9. **state证据仅证明使用输入，并无跨seed稳定价值。** 零/置换state降低validation分数；P0 memory对同family MDN的KL差在seed1/2改善、seed3恶化，proper truth NLL差的95%CI跨0。旧 EVALUATION 主比较按validation选中了P0 direct flow，因此应使用 DIAGNOSTICS 中保留的同family比较，且保留主比较曝光历史。
10. **历史成本与曝光状态保留。** 旧总 material-frequency=7961，forward/derivative/adjoint RHS=15922/5264/4；CPU=146.839642测量+10保守预留，wall=166.736402+12，GPU=0。23job里2失败已计费。JOB_STARTS只覆盖后17job，前6job仍在完整 JOB_LEDGER/receipt，不视为新漏账。所有旧 test 已公开，C1-P1只能作为 exposed regression panel。

主要表：

- `TRAINING_SUMMARY.csv`：所有24运行的seed、规模、更新量、checkpoint、首尾训练/验证分数。
- `TRAINING_HISTORY_COMPLETE.csv`：216条原始checkpoint进度；仅扁平化。
- `EXISTING_PARENT_METRICS.csv`：2304条原 parent 数据完整副本。
- `PARENT_DESCRIPTIVE_SUMMARY.csv`：对每模型/seed现有父级记录的均值、median、p95、SD、min/max；不新建CI、不做bootstrap。
- `REFERENCE_SUMMARY.json`、`REFERENCE_PER_OBSERVATION_COMPLETE.csv`、`DENSITY_SCORE_REFINEMENT.csv`：全部旧 reference 审计与q-score收敛信息。
- `EXISTING_PAIRED_COMPARISONS.json`：已有配对bootstrap；包含 P0 同family诊断。
- `SOURCE_EVIDENCE_TABLE.csv`：训练、Jacobian、测量输入、quadrature prior measure与指标定义的具体代码行。
- `COST_EVIDENCE.json`、`JOB_LEDGER_TABLE.csv`、`JOB_STARTS_TABLE.csv`、`PHYSICS_LEDGER_BY_KIND.csv`：历史账本抽取。
- `READ_RECEIPT.json`：每文件完整读取范围及记录数，无digest。

主要不确定性：现有观察不能分别量化 diagonal-family偏差、finite training覆盖、optimizer与conditioning原因；P0 reference为当前离散13-cell/固定prior/noise模型的经验积分，不是未知几何Maxwell或B1共同benchmark；Flow尾部KL未稳定，因此其数值大小不能构成state advantage或新模型 superiority 的精确证据。
