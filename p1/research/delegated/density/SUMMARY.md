# C1-P1 full-covariance MDN implementation

Status: implementation supplied; **tests and numerical experiments NOT_RUN by this worker**. All execution is reserved for the parent-controlled accounting job. No existing C1 results, old source, B1 files, or ledgers were modified.

## Files and API

- `p1/c1p1/density.py`: `FullCovDensity(condition, y_dim, state_dim, target_parameters=30000)`, with `condition` equal to `measurement`, `state`, or `memory`.
- Methods: `c(y, states)`, `_mixture_parameters(context) -> (logits, means, log_stds, rho)`, `log_prob_z`, `log_prob_z_context`, `log_prob_x`, `log_prob_x_context`, `sample_z`.
- `numpy_parameters(model, Y, S)` exports float64 NumPy weights `[N,8]`, means/stds `[N,8,2]`, rho `[N,8]`; accepts the old diagonal MDN family with rho=0, but rejects flows.
- `p1/tests/test_density.py`: independent SciPy logpdf comparison, moderate-mixture integrals in z and x, Jacobian/autograd check, strict SPD and tail finiteness, correlated sampling/marginals, context detachment, real connected parameter counts, old-MDN export, float64 decoding/endpoint behavior, and label/oracle-free signatures.

## Implementation choices

1. Eight unlabeled Gaussian components on the same logit coordinates and open material support `(-1.5,1.5)^2`; no altered prior, density clipping, projection, endpoint mass, or branch assignment.
2. Per-component covariance has two `log_std = 5 tanh(raw/5)` entries and `rho = 0.995 tanh(raw_rho)`; finite heads yield strict positive definiteness. Logpdf uses a Cholesky quadratic form to reduce subtractive cancellation.
3. Correlation rows of the output head start at zero; other head rows retain ordinary PyTorch initialization. This gives a nested diagonal starting family without using data-derived labels or mode assignment.
4. 32-dimensional encoder context and 16-dimensional saved-state GRU match the existing architecture. Encoder width is chosen analytically from actual connected weights to stay within 5% of 30,000 parameters; there are no inactive padding tensors.
5. `self.kind` is `fullcov`, `fullcov_state`, or `fullcov_memory`; `self.condition`, `is_mdn`, `uses_state`, and `uses_memory` are available to callers. Existing checkpoints are not modified or treated as full-covariance weights.
6. Every observation context is encoded once for sampling. No production-sample-dependent physics refresh or forward adapter import is present.

## Parent integration notes / uncertainties

- Add both `src` and `p1` to PYTHONPATH for `c1` and `c1p1` imports; package/init handling belongs to the parent.
- Training can remain float32. For near-boundary material log-density tests/evaluation, convert model and input tensors to float64; model/context APIs intentionally reject dtype mismatches instead of silently rounding.
- `numpy_parameters` computes outputs at the model's precision before float64 export. Convert the model itself to double for a genuine float64 encoder/head evaluation. Legacy float32 softmax sums can differ from 1 by rounding; no post-hoc renormalization is silently inserted.
- Material decode reuses existing float64 `c1.material.z_to_x`; extreme real logits may still round to machine endpoints. Tests detect this and require `-inf` at endpoints, with no clipping or resampling.
- Numerical tests only probe registered finite examples; they are not a general convergence certificate for arbitrary mixtures with std as small as exp(-5), nor scientific posterior-quality evidence.
- Correlation is bounded away from 1. This deliberate restriction may limit extremely thin within-region posteriors; scientific assessment remains with Codex parent.
