# Benchmark interface 委派结果

实现完成；未运行测试或实验，未调用 C1/B1 forward，未读取 B1 arrays，未修改 B1。

- `p1/c1p1/benchmark.py`：仅依赖 NumPy 与标准库；统一 latent/field、support/prior/base measure、measurement/noise/nuisance/acquisition、惰性 solver Protocol、三种 reference、事后 events、candidate features、概率评价与成本对象。
- `TargetIdentity`/`PosteriorBenchmark` 按 material、prior、likelihood、noise、nuisance、acquisition、solver、reference target 与 observation id 匹配；没有 hash、自动 gate 或 scheduler。
- Quadrature integration weights 与 posterior node masses 明确分离；模型积分质量不被静默重归一化；quadrature 节点不被当成 discrete prior。
- 未计时成本标为 `measured=False`；缺失 model support 的 KL 保持 `inf`；数值 overflow 标记评价未解决。
- `p1/tests/test_benchmark.py`：14 项纯 NumPy 测试已编写，**未执行**，由主代理统一计费。包括逐项 mismatch、连续积分测度、0.5 模型质量保留、边界、禁止 oracle 标签与禁止隐式 solver 调用。
- `p1/C1_B1_INTERFACE.md`：接入示例、概率测度、费用与 B1-2 未确认资产说明。

## 已读来源与限制

C1：`AGENTS.md`、`src/c1/material.py`、`probability.py`、`data.py`、`configs/protocol_v2.json`、`B1_REFERENCE_COMPATIBILITY.md`。

B1 已发布目录：`README.md`、`DECISION.md`、`configs/protocol.json`，三个固定模板+continuous complex material prior+16 complex channels，和 C1 P0 不是同一 posterior。

B1 下一阶段本地文档：`Gaussian/B1/b1-e0-posterior-audit/inputs/b1_next_2026_10_13/{B1_NEXT_RESEARCH_REPORT_ZH.md,README_ZH.md,CODEX_PROMPT_ZH.md}`（报告审读前 280 行、645–718 行的推荐架构与 E1-U、802–823 行的路线图；不声称完整最终结果审计），以及 `research/b1_next/e0/results/PROTOCOL.json`。后者仍为冻结 7×7 interpolation+原 prior/模板的 E0 target。

E1-U 建议为 `u_z(s)=1+z_c B_c(s)+z_s B_s(s)`，两个 radial compact bump 与共享 complex ε，共四实参数；prior 仍可选 Gaussian 或有界，尚不是唯一冻结定义。接口 `MaterialSupport` 可表达两类支持，但未实现其物理 decoder。

未找到最终 B1-2 统一径向 prior、likelihood、solver/reference 的完整冻结定义；该模式在已读材料中是建议，不能当作已成功实验。主代理若发现更新包，可只读补充最终定义，但本接口不要求 B1 改成 C1 13-cell 模型。

Codex 干预：主代理执行接口测试并统一计费；最终科学判断与是否接入 B1 由主代理负责。无新增物理或训练预算请求。
