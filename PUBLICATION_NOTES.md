# C1 发布说明

用户于 2026-10-13 授权直接公开选定文件，并要求发布及后续实验不额外引入 gate、SHA256 检查或 smoke test。本次直接整理和上传已有结果，没有调用训练、物理求解、测试、静态 QA 或 gate 汇总程序。

## 文件选择

包括方向 V2、更新规范、中文报告、下一阶段判断、代码与已有测试、配置、完整 C1 自生成数据/权重/参考数组、逐观测/parent/seed 结果、图表、费用与失败记录，以及已提供研究包的九个来源文件。没有 B1 数据、权重或实验环境，没有共享解释器或 vendor 依赖，没有缓存、临时锁或凭据文件。

公开副本中的已知本机项目路径替换为 C1_PROJECT_ROOT，共享解释器路径替换为 python3；仅提高可读性和可携带性，不改变科学数据、指标或失败证据。原工作目录保持其完成记录。

SOURCE_MANIFEST.json 是原有的文件/字节来源记录，不是新的完整性检查。原包 MANIFEST.json 的摘要为历史字段，未核验。C1_DELIVERY_RECEIPT.json 的 LOCAL_DELIVERY_COMPLETE_UNPUBLISHED 描述的是发布前的本地交付时刻；公开状态由 GitHub 仓库与 release 表示。

历史文件 C1_GATE_DECISION.json 与 DELIVERY_QA.json 原样保留作为结果记录，不引入新的发布审批或执行前置条件。后续实验保留必要的科学检验、数值误差、预算和求解账本；不添加仪式性检查，不在发布时重跑实验。

## 研究结论

保留 topology-agnostic/material-first 方向。当前物理状态与 recurrent memory 的跨 seed 稳定增益未建立，概率误差约 5 个百分点；复杂未知拓扑与一般 Maxwell 成像未运行。完整负结果和未解决尾部误差随结果公开。
