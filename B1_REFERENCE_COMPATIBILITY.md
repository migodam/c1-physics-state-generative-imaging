# B1 独立参考的兼容性核对

2026-10-13，B1已完成并公开发布。只读检查其正式 DECISION.md 与 GATE_DECISION.json，未运行任何B1代码，未读取B1 observation、teacher、field或posterior数组，未修改其环境、缓存、GPU或账本。

B1声明的目标是显式三模板、固定同心3D vector-Mie几何中的连续材料posterior。部分原生参考未收敛，后续插值refinement为经验通过而非uniform remainder certificate；calibration仍为HOLD，general geometry NOT_RUN。

C1 P0是13个向量电偶极、无标签的共同连续两区域材料prior、intensity measurements及不同noise/support。solver、prior、观测与材料空间都不同。B1的事件概率不能作为C1当前目标的数值真值，也不能偷偷转换成训练标签。因此本阶段B1参考状态为 **NOT_USED_TARGET_MISMATCH**。

未来若C1采用与B1完全一致的prior/观测/solver，可在独立评价阶段复用其可信部分，但生成器仍不能输入B1的模板类别；还需先厘清该prior是否把拓扑写死。此处不替B1重新求积，不把其conditional参考称为任意材料空间的参考。

B1只读来源：[公开仓库](https://github.com/migodam/b1-probabilistic-maxwell-imaging)。此链接的科学内容来自本地最终文件检查；C1没有重新认证B1所有原始数组。
