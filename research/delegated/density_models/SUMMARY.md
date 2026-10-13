# Density models implementation

Status: **IMPLEMENTED; runtime checks, training, and scientific evaluation NOT_RUN**.

- Added `src/c1/density.py` with the requested `Density` API for `mdn`, `measurement`, `state`, and `memory`. All modules and sampling operate on CPU.
- MDN uses eight unlabeled diagonal Gaussian components in logit coordinates, normalized mixture weights, and logsumexp. Three flows use six alternating affine coupling blocks, two 48-wide hidden layers per conditioner, context dimension 32, bounded log scales, and zero-initialized final conditioner layers.
- Measurement and MDN encoders see all of `y`; state adds only the last supplied state; memory adds a 16-wide GRU summary of both states. Raw conditioning features are detached, while encoder/GRU/density weights retain training gradients. No physics module is imported or called. Sampling computes the encoder and GRU once per observation batch.
- `log_prob_x` includes the exact logit Jacobian and returns `-inf` outside the open support. It does not clip interior points or project generated coordinates. Flow forward/inverse methods report their signed log absolute determinants.
- Added `tests/test_density.py`: real parameter counts, inverse and determinant/autograd agreement, identity initialization, independent MDN mixture comparison/integration, material-support normalization and Jacobian, conditioning history/full-y checks, frozen-input parameter gradients, reproducible sampling, one-pass GRU/encoder, and no-label API checks.

Parameter counts below are **derived from layer formulas, not executed measurements**; the constructor also checks actual counts at runtime. For `y_dim=5`:

| Kind | State dim 18: encoder width / parameters | State dim 38: encoder width / parameters |
|---|---:|---:|
| mdn | 151 / 30042 | 151 / 30042 |
| measurement | 57 / 29996 | 57 / 29996 |
| state | 51 / 30032 | 45 / 30014 |
| memory | 40 / 30052 | 32 / 29996 |

Uncertainties: tests have not been executed; quadrature-test tolerances and installed-runtime behavior remain to be checked by the parent in its accounted test stage. These changes establish implementation only and make no training-quality or scientific-contribution claim. No training, physics calls, network calls, hashes, or provider/model changes were performed. Parent intervention: review and run the accounted test job before training.
