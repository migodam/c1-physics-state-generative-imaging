# C1-P1 Unified Material Posterior Density Repair

2026-10-13，已完成30次CPU小模型训练、两个有限诊断和一次cached-state消融。P0事件MAE约.0124、KL约.132；新独立U0仍约.0424/.166；state/memory无稳定优势。继续统一密度修复，等待匹配B1资产。

- [中文研究报告](C1_P1_RESEARCH_REPORT_ZH.md)
- [每模型/seed结果](C1_P1_RESULTS.csv) / [配对统计](C1_P1_PAIRED_RESULTS.csv)
- [下一步决定](C1_P1_DECISION.md)
- [B1共同benchmark接口](C1_B1_INTERFACE.md) / [真实C1接入](c1p1/adapter.py)
- [费用账本](C1_P1_RESOURCE_LEDGER.json) / [逐方法费用](C1_P1_METHOD_COSTS.csv)
- [实验规范](C1_P1_EXPERIMENT_SPEC.md)、[配置](configs/protocol_v2.json)、[已执行命令](RUN_COMMANDS.md)
- [代码](c1p1/)、[必要测试](tests/)、[结果与原始回执](results/)、[图表](figures/)

所有旧test为exposed regression，新U0 final在选定模型后生成、现在也已曝光。没有新的Maxwell/B1调用或GPU使用。数学normalized density不等于完整posterior已校准；U0有极端latent舍入到边界，未clip/resample。

公开包依赖原C1已发布的source/data/forward bank。teacher_cdf.npy仅为可重建cache，未包含；完整逻辑在prepare.py。新重现应在保留旧账本的独立目录登记费用，当前完成目录拒绝静默重跑。发布不再新增gate、SHA256、smoke或实验重跑。
