# 已执行tail评价的原始负例留存

主评价已保存每观测的boundary fraction及latent最大值，但未落盘全部流式draw。为补全报告中的原始负例，用完全相同冻结权重、float64前向与seed261013600+training_seed+observation恢复**所有已记录的boundary-failed观测**，只保存其中舍入失败的z/x及索引。不新增数据、模型、训练、reference积分或诊断设计，不重排结果、不做clip/resample。原结果和计数保留。

这是已注册tail评价的证据序列化，不是第三轮算法诊断。预计CPU≤2秒/wall≤4秒，计入P1独立900/1800总上限；Maxwell/B1/GPU=0。
