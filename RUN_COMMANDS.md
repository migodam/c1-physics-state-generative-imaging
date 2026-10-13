# 本次实际执行命令

已执行记录；完成目录拒绝原ID重跑，不得删除/重置账本。

## u0-prepare-001 — PASS

阶段 u0_train，回执 results/jobs/u0-prepare-001/receipt.json。

```text
python3 -m src.c1.data u0
```

## invariants-001 — FAILED_CHARGED

阶段 tests，回执 results/jobs/invariants-001/receipt.json。

```text
python3 -m pytest tests -q
```

## test-dependency-001 — PASS

阶段 tests，回执 results/jobs/test-dependency-001/receipt.json。

```text
python3 -m pip install --disable-pip-version-check --no-warn-script-location --target vendor/test_runtime pytest==8.4.2
```

## invariants-002 — PASS

阶段 tests，回执 results/jobs/invariants-002/receipt.json。

```text
python3 tools/run_tests.py tests -q
```

## u0-training-001 — PASS

阶段 u0_train，回执 results/jobs/u0-training-001/receipt.json。

```text
python3 -m src.c1.train u0
```

## u0-evaluation-001 — PASS

阶段 evaluation，回执 results/jobs/u0-evaluation-001/receipt.json。

```text
python3 -m src.c1.evaluate u0
```

## density-revision-tests-001 — PASS

阶段 tests，回执 results/jobs/density-revision-tests-001/receipt.json。

```text
python3 tools/run_tests.py tests/test_density.py -q
```

## u0-reference-refine-001 — PASS

阶段 evaluation，回执 results/jobs/u0-reference-refine-001/receipt.json。

```text
python3 -m src.c1.refine_reference
```

## operator-actions-tests-001 — PASS

阶段 tests，回执 results/jobs/operator-actions-tests-001/receipt.json。

```text
python3 tools/run_tests.py tests/test_physical_actions.py -q
```

## p0-prepare-001 — PASS

阶段 p0_prepare，回执 results/jobs/p0-prepare-001/receipt.json。

```text
python3 -m src.c1.data p0
```

## p0-training-001 — PASS

阶段 p0_train，回执 results/jobs/p0-training-001/receipt.json。

```text
python3 -m src.c1.train p0
```

## p0-evaluation-001 — PASS

阶段 evaluation，回执 results/jobs/p0-evaluation-001/receipt.json。

```text
python3 -m src.c1.evaluate p0
```

## mechanism-cost-diagnostics-001 — PASS

阶段 diagnostics，回执 results/jobs/mechanism-cost-diagnostics-001/receipt.json。

```text
python3 -m src.c1.diagnostics
```

## corrected-query-timing-001 — PASS

阶段 diagnostics，回执 results/jobs/corrected-query-timing-001/receipt.json。

```text
python3 -m src.c1.timing
```

## final-invariants-001 — PASS

阶段 tests，回执 results/jobs/final-invariants-001/receipt.json。

```text
python3 tools/run_tests.py tests -q
```

## source-algebra-001 — PASS

阶段 tests，回执 results/jobs/source-algebra-001/receipt.json。

```text
python3 inputs/source_package/C1_Research_Package/c1_algebra_checks.py --out results/PACKAGE_ALGEBRA_CHECKS.json
```

## result-plots-001 — PASS

阶段 diagnostics，回执 results/jobs/result-plots-001/receipt.json。

```text
python3 tools/plots.py
```

## decode-precision-test-001 — PASS

阶段 tests，回执 results/jobs/decode-precision-test-001/receipt.json。

```text
python3 tools/run_tests.py tests/test_sample_precision.py -q
```

## decode-precision-diagnostic-001 — PASS

阶段 diagnostics，回执 results/jobs/decode-precision-diagnostic-001/receipt.json。

```text
python3 tools/sample_precision.py
```

## result-plots-reviewed-001 — PASS

阶段 diagnostics，回执 results/jobs/result-plots-reviewed-001/receipt.json。

```text
python3 tools/plots.py
```

## documents-finalize-001 — FAILED_CHARGED

阶段 diagnostics，回执 results/jobs/documents-finalize-001/receipt.json。

```text
python3 -m tools.finalize
```

## documents-finalize-002 — PASS

阶段 diagnostics，回执 results/jobs/documents-finalize-002/receipt.json。

```text
python3 -m tools.finalize
```

## delivery-static-qa-001 — PASS

阶段 diagnostics，回执 results/jobs/delivery-static-qa-001/receipt.json。

```text
python3 -m tools.qa_and_bundle
```

## 最终封装

tools/seal_delivery.py在最后job退出后运行；成本由额外3CPU/4wall保守预留覆盖；没有物理/训练重跑。
