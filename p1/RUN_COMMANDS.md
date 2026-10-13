# P1 已执行命令与费用

历史执行记录。原输出和费用保留；这些命令不授权删除账本或覆盖已完成run。

## p1-density-interface-tests-001

CPU 2.069884s；wall 3.14833s；PASS。

```text
python3 tools/run_tests.py tests -q
```

## p1-reference-prepare-001

CPU 7.822543s；wall 7.983733s；PASS。

```text
python3 -m c1p1.prepare
```

## p1-probability-tools-tests-001

CPU 0.609285s；wall 0.897848s；PASS。

```text
python3 tools/run_tests.py tests/test_probability_tools.py -q
```

## p1-reference-numerical-refinement-001

CPU 6.905788s；wall 7.001768s；PASS。

```text
python3 -m c1p1.refinement
```

## p1-u0-factorial-train-001

CPU 29.645535s；wall 31.019971s；PASS。

```text
python3 -m c1p1.train u0
```

## p1-p0-factorial-train-001

CPU 38.323805s；wall 40.943071s；PASS。

```text
python3 -m c1p1.train p0
```

## p1-p0-factorial-evaluate-001

CPU 13.5063s；wall 14.832729s；PASS。

```text
python3 -m c1p1.evaluate p0
```

## p1-u0-fixed-final-evaluate-001

CPU 117.845171s；wall 120.147898s；PASS。

```text
python3 -m c1p1.evaluate u0
```

## p1-offline-capacity-mechanism-001

CPU 7.183865s；wall 7.869847s；PASS。

```text
python3 -m c1p1.diagnostics
```

## p1-cached-state-ablation-train-001

CPU 25.319636s；wall 26.268584s；PASS。

```text
python3 -m c1p1.train p0 --ablation
```

## p1-cached-state-ablation-evaluate-001

CPU 6.5088s；wall 7.892295s；PASS。

```text
python3 -m c1p1.evaluate p0 --ablation
```

## p1-paired-statistics-001

CPU 1.57066s；wall 2.213934s；PASS。

```text
python3 -m c1p1.statistics
```

## p1-adapter-moment-guard-tests-001

CPU 0.641677s；wall 0.967638s；PASS。

```text
python3 tools/run_tests.py tests/test_saved_adapter.py tests/test_probability_tools.py::test_bounded_material_moments_agree_with_independent_gaussian_integrals tests/test_probability_tools.py::test_zero_physics_guard_blocks_before_historical_ledger_creation -q
```

## p1-scientific-figures-001

CPU 1.945214s；wall 2.19381s；PASS。

```text
python3 -m c1p1.plots
```

## p1-report-finalization-001

CPU 0.033155s；wall 0.049082s；PASS。

```text
python3 -m c1p1.finalize
```

## p1-derivative-cost-interface-test-001

CPU 0.432192s；wall 0.641937s；PASS。

```text
python3 tools/run_tests.py tests/test_saved_adapter.py::test_interface_cost_keeps_derivative_rhs_separate -q
```

## p1-delivery-packaging-001

CPU 0.949917s；wall 1.075015s；PASS。

```text
python3 tools/seal_delivery.py
```

## p1-tail-evidence-retention-001

CPU 1.120582s；wall 1.807376s；PASS。

```text
python3 -m c1p1.retain_tail_evidence
```
