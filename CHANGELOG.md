# C1 变更与曝光记录

2026-10-13，首次代码运行前：核对张量积分节点算术后，将 U0 的每轴总节点改为64/96；P0改为40/64，共5696材料评估。P0 noise replication改为train2/validation1/test1，稀疏物理诊断16×8。8192全局物理上限及CPU上限没有增加。尚未读取任何生成的实验数据或训练结果，不是观察结果后的难度调整。

默认 DeepSeek worker profile 在本地配置中不存在。没有改变 Codex 默认模型/provider，也没有创建该profile；本轮核心设计与小原型由父 Codex执行，后续机械审查可用已有 native agent，不启动外部付费服务。

首次 invariants-001 启动失败：共享 NN 解释器没有 pytest。没有物理求解发生，失败 CPU/启动成本保留。将 pytest8.4.2 及轻量依赖仅安装到 C1/vendor/test_runtime，通过局部 runner加载；不修改共享解释器或B1。原预算不变。

U0 validation 在三个seed上均选择MDN（material log score约-0.82/-0.82/-0.78）而非direct flow（约-0.14/-0.04/-0.12）。在P0数据准备、训练与测试前据此选择8-component MDN作为物理状态/记忆主消融的density family，新增mdn_state、mdn_memory两个小配置；保留direct flow对照与原U0四模型结果。未以U0 locked结果选择P0模型，未修改P0 prior/measurement/reference/预算，也未用类别约束mixture components。这是本轮两项允许的小架构配置调整。

U0全测试panel的64/96求积差2.34e-6，未达到严格1e-8 reference门槛，尽管固定adaptive案例通过。记录初始REFERENCE_UNRESOLVED；仅在既有evaluation CPU额度内增加128节点/轴，与96核对并复算同一批冻结模型的proper score，不训练、不改变mode分区或数据。初始结果完整另存，不删除失败，不增加任何物理计算或总预算。

P0训练前：U0多数模型的validation仍在1500 checkpoint改善；为避免以少训baseline声称physics advantage，P0所有四模型统一给3000 updates（原研究包的小模型上限），用相同validation选择checkpoint。P0 CPU阶段360秒/总1200秒及物理8192上限不变；不是扩大计算预算，亦非观察P0数据后的选择。U0原1500结果不重新训练或重新命名为更好结果。

机制计时检查：原DIAGNOSTICS的query计时包含了为了诊断而扫描整个物理账本的summary，不是纯推断间隔。保留该计时回执并标记不可用于效率裁决；在既有diagnostics阶段重测三次轮换query，把summary扫描移到计时区间外、加入material decode。新增12材料评估计费，总上限仍8192；模型与生成概率不变。首次backend账本加载、所有诊断/计时工作仍计入总process成本，不称免费。

数值支持诊断发现初始float32材料decode有少量端点rounding。未来API改用float64 sigmoid，另以相同冻结模型、相同latent RNG诊断两种精度差异；不重训、不改变density数学定义、不重排primary结果，不给原始已验证gallery材料改坐标或改residual。原边界fraction及有限精度状态保留；若float64仍不能表达极端尾部，也必须记NUMERICAL_UNRESOLVED，不能静默clip或重抽样。

documents-finalize-001因Python保留关键字误作字典参数而启动失败；未运行数值计算或写结果，失败成本保留。修复后以新run ID汇总，物理/学习结果均未重跑。
