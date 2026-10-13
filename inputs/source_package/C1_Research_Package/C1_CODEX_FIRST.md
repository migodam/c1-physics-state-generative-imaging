# Codex handoff: C1 first decisive implementation

## Read first

Read C1_THEORY_AND_ARCHITECTURE.md, C1_PROTOCOL.json, C1_LITERATURE_AUDIT.md and C1_CHECKS.json. Locate the user's B1 Research Package (historical filenames A32_THEORY_REPORT_ZH.md, A32_NEXT_DECISION_ZH.md, A32_CHECKS.json and a32_checks.py). Do not modify B1 or any upstream project. Use a new C1 folder, seeds, manifests and resource ledger. A32 is only a historical filename; call this project C1 throughout new outputs.

## Execute now: E0, then E1 only

First inspect the local environment and propose a short file-level implementation plan; then implement and run the small experiments within the hard budgets in C1_PROTOCOL.json. Adapt bugs and model widths within that scope. A PASS is not permission to launch E2-E5 or a large training sweep. Report the evidence and an explicit next decision.

### E0

Implement H in {-1,+1}, amplitude a uniform on [0.5,1.5], material x=Ha, prior P(H=+1)=0.7. Measure y1=a^2+noise. A second condition uses y2=kappa Ha+noise. With kappa=0 the sign posterior remains the prior, not 50/50. With nonzero kappa compute reference sign masses and amplitude conditionals using two independent one-dimensional quadrature/refinement routes. Vary noise so that locked tests contain both ambiguous and identifiable cases.

Compare a deterministic squared-loss regressor, a small conditional MDN / explicit sign-mixture model, and a conditional flow only if the MDN misses within-mode shape. Use proper density/log-score training. The deterministic model is a reconstruction baseline, not a distributional baseline. Keep a correct exact-reference sampler.

Test mode coverage, sign-probability error, reference excess log score, Brier, conditional amplitude coverage, material L2, best-of-K and mode precision. Also run the deliberately wrong duplicate-count and repeated-likelihood controls. Do not tune on locked examples.

### E1

Copy ONLY the necessary analytic TM forward functions into a separate C1 namespace, retaining attribution and units. Do not execute the original A32 main in its historical resource-ledger directory. Reproduce the finite 18-state prior with the exact parameter list from B1, including solid versus the fixed cavity and nine dielectric real values per branch. First verify vacuum/truncation/boundary controls and the probability convention. Stop at the declared new-evaluation cap if required controls cannot be completed.

Freeze/cache the 18 full-wave responses. Generate independent noise realizations without further Maxwell solves. Compute exact finite-state posterior probabilities using logsumexp. Implement the finite-state categorical probability baseline, single MAP-image baseline, and K=8/16/32/64 categorical/stratified galleries. Explicitly aggregate state probabilities into hole/no-hole probability. This experiment validates probability semantics and training, NOT superiority over exact enumeration and NOT a continuous material posterior.

Provide an output manifest for every gallery: candidate image, finite-state identifier or declared mode/event, sample generation law, weight_type, normalized integration weight, branch mass, probability target, prior version, noise covariance, solver version, residual, passivity status and unresolved/shift flags. A candidate copy must not add branch evidence. Display count is distinct from integration/sample budget.

Train/calibration/test splits use new observation seeds. For later geometry generalization, group by physical parent geometry; multiple noisy measurements of one geometry are not independent geometry parents.

## Do not implement yet

No Transformer, no full Jacobian tensor, no 3-D voxel diffusion, no OPM/current selection, no global certificate sweep, no birth/death search over arbitrary complex geometry, no untracked final GN projection, no new GPU budget beyond the explicit cap.

## Required software tests

- Real/complex Gaussian whitening consistency and likelihood normalization.
- Evidence via logsumexp agrees with direct normalized values in non-underflow cases.
- Finite prior masses sum to one and refer to distinct declared states.
- Split/merge/duplicate invariance.
- Stratified 1/n_h importance factor and random-mixture alpha_h denominator tested separately.
- No repeatedly multiplying the same likelihood; tempering increments telescope.
- Sampler density and forward/reverse flow Jacobian agree numerically.
- Candidate-dependent physics cannot enter a standard coupling logdet unaccounted.
- Rerunning a completed experiment never erases or resets the resource ledger.
- Include all failed attempts, rejected solver evaluations and hardware warm-up in costs.

## Deliver

C1_E0_DECISION.md, C1_E1_DECISION.md, exact reference arrays with metadata and hashes, full machine-readable metrics per seed/parent, calibration tables, representative gallery, reproducible commands, C1_RESOURCE_LEDGER.json and a final C1_NEXT_DECISION.md. Distinguish IMPLEMENTATION_PASS, PROBABILITY_REFERENCE_PASS, MODEL_TRAINING_PASS and ARCHITECTURE_ADVANTAGE_NOT_TESTED. Never interpret successful algebra tests as successful generative imaging.

## E2 proposal after E1

The first state experiment must compare measurement-only against branch-response contrasts, then add the current-hypothesis gradient, then add only two operator probes. GRU memory is tested before historical attention. All state inputs must be computable from the hypothesized material, not the truth. Keep an exact finite-bank baseline. If the discrete benchmark saturates, establish a small continuous reference before claiming generalization. Use the gates in the protocol; do not silently increase difficulty or budget until a favored architecture wins.
