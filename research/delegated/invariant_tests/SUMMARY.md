# Invariant test code delivery

Status: **CODE_READY; NOT_RUN**. No test, training, or physical solve was executed by this worker. No network access, SHA256 operation, or B1 read/write was performed. Source modules and protocol were not edited. Parent owns execution and the budget queue.

- `tests/test_invariants.py` covers pre-solve refusal, failed charges, persistent reuse, stale handles/repeated refusal, negative costs, live-handle truncation, and duplicate job IDs. Temporary ledgers contain mock costs only; the duplicate-ID test patches the launcher root and forbids any subprocess launch.
- Material tests check proper-prior integration, sampler support, offline quadrant use, one full-grid continuous decoder representing 0/1/2 enclosed threshold holes, passivity, and independent pixel response. They prove implementation properties only; they do not establish material recovery or arbitrary-shape posterior support. Sigmoid fields are not described as strict-vacuum atoms.
- Probability tests use an independent normalized real Gaussian density, proper complex variance/whitening conventions, a telescoping single likelihood, valid mass splitting, and event-mass invariance after copying a candidate with conserved prior mass.
- Context tests reject truth/posterior/category keyword interfaces and check pilot whitening/GN summaries. Locked test loading is refused before the model-freeze marker for both U0 and P0.
- DDA code checks two material-Jacobian finite differences, constitutive/linear equations, two-source consistency, transpose reciprocity, four passive support corners, the zero-contrast no-scattering limit, and charged two-round candidate context. The background is epsilon_b=2; no-scattering is not physical vacuum. These checks do not establish continuum Maxwell convergence or scientific advantage.

## Physical-call plan for one full sequential suite

All actual `DipoleProblem.forward` calls use the default **canonical `results/PHYSICS_LEDGER.jsonl`**, never an isolated physical ledger. Session fixtures reuse already charged states. Expected new totals: **18 material–frequency evaluations, 36 forward RHS, 16 derivative RHS, 0 adjoint RHS**.

| Calls | Kind | New evaluations |
|---|---|---:|
| Fixed intensity normalization | `normalization_setup` | 1 |
| Two Jacobian base materials | `invariant_jacobian_base` | 2 |
| Central differences, two coordinates at both base materials | `invariant_fd_plus`, `invariant_fd_minus` | 8 |
| Four open-support corners | `invariant_support_corner` | 4 |
| Background zero contrast, zero loss | `invariant_zero_contrast` | 1 |
| Two candidate-context rounds | `candidate_pilot` | 2 |

Do not run this suite outside the parent budgeted job. Retries/subsets cost their actual new calls; the 18 count is for one complete run with no retries.

## Parent review notes

The initial stale-handle cap issue and refusal-offset double-count issue were reported to the parent; parent revised accounting while test code was prepared. Tests are **unexecuted**, so no repair is certified here. Truncation protection is tested for an already open handle; an externally deleted/replaced ledger reopened by a fresh process is outside that test. No final scientific verdict is delegated or claimed.
