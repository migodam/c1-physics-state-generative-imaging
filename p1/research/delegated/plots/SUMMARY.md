# C1-P1 绘图代码交付

只写了 `p1/c1p1/plots.py`。没有执行、import、测试、绘图或新的数值计算；没有读取B1，也没有改写任何旧结果。代码运行时仅读主代理生成的 `p1/C1_P1_RESULTS.csv` 和 P1 model receipt JSON histories，不读取模型权重、forward bank或solver。

主代理在受控job中可运行：

```text
工作目录：C1项目根目录
解释器：现有共享 Python（由主代理沿用原受控调用方式）
入口：p1/c1p1/plots.py
```

默认CSV = `p1/C1_P1_RESULTS.csv`，默认receipts = `p1/results/`，默认输出 = `p1/figures/`。可用 `--results`、`--receipts-root`、`--output`明确传入路径。也可在支持该package的工作目录中执行 `python -m c1p1.plots`。

运行后生成3组标准科学图，各PNG和PDF：

1. `P1_DENSITY_EVENT_PAIRED`：U0 exposed、P0 exposed、U0 fresh_final 三panel的KL与event MAE；四个primary measurement方法全部保留，同seed连线。原bank数字实心/实线，refined cross-check空心菱形/虚线，不静默取代原reference。
2. `P1_P0_TRAINING_HISTORIES`：四个方法逐行、training minibatch NLL_z与validation expected_NLL_x逐列，三个seed分别画线；各自y轴，selected expected-score checkpoint空心圈，注明不同variables/Jacobian，不能把绝对y轴差当generalization gap。
3. `P1_PROBABILITY_QUALITY`：三panel的CDF MAE、tail probability MAE、90%coverage、within-region mean及covariance误差；coverage目标0.90；不建立跨seed平均置信区间。

另保存 `P1_PLOT_MANIFEST.json`，只记录输入/输出路径、已有seed/panel和图意，没有digest或scientific gate。

自动过滤state、memory和historical rows，只保留 `diag_joint/full_joint/diag_distill/full_distill` 的 `condition=measurement`；绘图不依赖未来可选state实验。CSV重复或缺少四方法配对row时直接报告资料问题，避免静默隐藏baseline。所有模型曲线保留，非有限点以图内计数注明。

图注明确旧panel曝光、U0 fresh评估后也已曝光；P0 refined target使用保存bank插值、within-region moments有有限MC误差；不称未知拓扑成像成功。

状态：CODE_WRITTEN_ONLY。本任务按父代理要求没有运行或测试；实际生成、视觉查看及受控成本记账由父代理负责。
