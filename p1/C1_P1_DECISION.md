# C1-P1 决定：继续统一密度修复，停止当前state扩张

2026-10-13。选择 **A+C**：继续小型统一密度学习，等待B1共同benchmark的匹配资产。完整posterior准确性与数值tail尚未修复，不进入昂贵Physics-State生成推断。

## 四项回答

- **Q1**：P0的有限joint监督/泛化与diagonal geometry都有证据；distillation和full covariance分别改善。U0 fixed-validation non-amortized fit KL约.007而amortized约.18，说明mapping/optimization缺口，不能把失败归结为diagonal covariance或统计误差下限。没有证明唯一因果解释。
- **Q2**：P0 full-covariance distillation平均event MAE .01241、KL .13208，明显优于baseline .05074/.99678。新U0 diagonal-distill .04244/.16570，仍未达到.02/.02。90%覆盖和极端float64 decode尚有失败；没有模型同时满足完整posterior目标。
- **Q3**：更强family下state/memory皆一seed改善、两seed恶化；memory事件质量三个seed均变差。停止本阶段state训练，没有新增物理refresh或实际cost优势。
- **Q4**：solver-agnostic接口与真实只读C1接入已运行/测试。B1-2最终prior/likelihood/solver/reference与latent维度尚未确认，不能直接比较或将旧三模板reference充当统一field真值。

## 值得保留的路线

P0保留normalized full-covariance distillation，U0保留diagonal-distillation作为认真基线。利用已有bank学习概率密度有稳定收益，不能称相同监督信息量的纯网络创新。重点转到amortization、bounded-support尾部和within-region geometry。support-native spline/autoregressive density或proper prior-defensive mixture仅为下一项有限竞争建议，本阶段未实现或验证。

**GO继续有记录的小型density研究；NO-GO扩大当前Physics-State/GRU、GPU或多轮candidate-specific演化；NO-GO已校准完整posterior的主张。** 保留反例、所有seed、未过的质量目标与曝光状态；不是暂停C1方向或证明Maxwell状态永远无用。

新增Maxwell/B1/GPU均0，独立费用见[C1_P1_RESOURCE_LEDGER.json](C1_P1_RESOURCE_LEDGER.json)。一般未知拓扑成像、continuum Maxwell和真实仪器均未运行。完整解释见[中文报告](C1_P1_RESEARCH_REPORT_ZH.md)。
