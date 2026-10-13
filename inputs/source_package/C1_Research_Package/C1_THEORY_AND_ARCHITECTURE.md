# C1 - Physics-State Probabilistic Generative Imaging

Research decision and implementation specification. Date: 2026-10-13.

## Executive decision

Proceed with a small, topology-stratified, physics-state recurrent conditional generator. Do not start with a history Transformer or a three-dimensional voxel diffusion model. The preferred first architecture combines Route B (recurrent physics states) and Route D (explicit, physically defined branches), uses a normalized conditional flow only when a mixture density baseline is insufficient, and treats Route E (A17/B1 signatures) as an optional diagnostic or proposal feature.

The candidate scientific claim is NOT that multiple reconstructions, neural posterior estimation, physical unrolling, or physics-guided diffusion are new. The candidate claim is that **branch-contrast-aware Maxwell state compression can reduce the total cost of calibrated material-hypothesis inference while preserving competing finite-scale material structures**. This claim needs a matched-cost test. If state features do not improve conditional probability or sampling efficiency, keep the simple generator and do not claim a strong physics-state contribution.

No C1 network was trained in this research turn. Twelve small algebra/quadrature checks were executed; zero new Maxwell evaluations and zero GPU hours were used. See C1_CHECKS.json. Literature inspection limits are in C1_LITERATURE_AUDIT.md.

## 1. Statistical object

Use real coordinates for complex data. Let d contain source/receiver geometry, polarization, frequency and instrument metadata. Define

    Y = F_d(chi, gamma) + eta, eta ~ N(0, Sigma_d).
    x = (H, z_H, gamma), chi = D_H(z_H).
    p0(x) = pi0(H) p0(z_H, gamma | H).
    Phi_y(x) = 0.5 ||Sigma_d^(-1/2)(F_d(x)-y)||^2.
    pi(x | y) = p0(x) exp(-Phi_y(x)) / Z(y).

For proper complex noise with covariance sigma_c^2 I, each real component has variance sigma_c^2/2. Whitening uses sqrt(2)/sigma_c. Candidate-dependent covariance requires its log determinant. Nuisance variables must be sampled, analytically marginalized where legitimate, or explicitly profiled with a changed statistical interpretation. Never silently replace marginal evidence with minimum residual.

H must denote a declared measurable event, not an arbitrary neural head. Define material thresholds and a minimum feature size rho. In two dimensions, enclosed holes and in three dimensions, closed cavities and tunnels are different events. An explicit atom H=0 is needed to give exactly solid objects nonzero probability when a continuous positive hole-radius prior is used.

Different branch dimensions are legitimate on a disjoint union with proper branch priors. Repeated parameterizations and interchangeable component labels require an explicit prior convention. For an initial two-hole extension, non-overlapping left/right component domains are a simple way to avoid an unaccounted label-permutation multiplicity.

### 1.1 Three valid output contracts

1. **Posterior approximation:** a normalized law q(x|y), together with defined material-event masses and conditional samples. Its target, reference error and sampling error must be reported.
2. **Calibrated event prediction:** alpha_h(y) estimates P(H=h|Y=y) under a declared joint data-generating distribution. A proper scoring objective identifies the population target; finite-sample and shifted-distribution calibration still need tests.
3. **Finite candidate-set confidence:** define, for example, C=argmin_i loss(chi, candidate_i), with a tie rule, and predict P(C=i|y,candidate set). This is a probability of a closest-candidate event, not the point probability of candidate_i. Events of being within a tolerance can overlap and must not automatically be normalized to sum to one.

A finite prior on 18 materials supports a genuine categorical posterior computed by softmax of log prior plus log likelihood. That is valid; the problem is not softmax itself but an unspecified probability target.

### 1.2 Display protocol

Separate K_display (8, 16, 32 or 64 representative images) from N_integration. One topology can contain multiple continuous modes; one topology probability is not a point probability for every image in that category. Show a branch/mode card with its total mass and several representative samples. Weighted particles are numerical integration atoms, not claims that a continuous material image has positive point probability.

A dynamic clustering partition may define valid conditional events once fixed, but evaluating its probability on the same small sample used to invent the partition is optimistic. Freeze the partition and use an independent mass-estimation sample when interpreting cluster weights as event probabilities. Otherwise label them empirical represented masses.

## 2. Maxwell state structure

In a current formulation with the constants absorbed consistently into X, G and S,

    j = X(chi) (e_inc + G j), L_chi = I - X(chi) G,
    E = e_inc + G j, F(chi) = S j,
    DF_chi[h] = S L_chi^(-1) DX_chi[h] E_chi.

In a dipole implementation X is the actual polarizability map, and DX is its actual derivative. Use neither a bare contrast substitution nor a scalarized vector derivative without checking the discretization.

For a real material direction h and fixed covariance,

    D Phi[h] = Re <lambda, DX_chi[h] E_chi>,
    lambda = L_chi^(-*) S* Sigma^(-1) (F(chi)-y).

Thus the natural material feature is a forward-adjoint bilinear sensitivity, not simply a concatenation of a field image and a current image. Finite branch changes obey

    F(chi_b)-F(chi_a) = S L_b^(-1) [X_b-X_a] E_a.

This finite secant is useful for comparing distinct material worlds; a Jacobian at one anchor cannot characterize global topology competition.

### 2.1 Information versus computation

At a fixed candidate chi_k, field, current, residual and derivative features are computable from chi_k, y, the known model and algorithmic randomness. They are not independent observations of the unknown object. If U is algorithmic randomness independent of the truth conditional on Y and S=g(Y,U), then

    I(chi; S | Y) = 0.

This does not make physics states useless. They can make a difficult inference function easier for a small network to approximate at a given computational budget. Current compression can lose information; history can recover information discarded by that compression. Neither creates new measured information.

Incident fields are known only to the accuracy of calibration. Internal total fields depend on the candidate, not directly on measurements alone. j -> scattered data is linear and generally non-injective. Given a current, E follows from the state equation, but recovery of material by j/E can be unstable, multi-source inconsistent, or undefined where excitation vanishes. Back-projection and gradients can be information-preserving only under specified rank/sufficiency assumptions; neither is generally an invertible material reconstruction.

### 2.2 Recommended state representation

Shared measurement tokens: real/imaginary measurement, source and receiver coordinates, frequency, polarization, noise scale, acquisition mask, and known calibration metadata. Do not impose equivariance while holding an asymmetric acquisition array fixed.

Branch-specific tokens:
- material coefficients and spatial patches;
- whitened residual by source/frequency;
- gradient or forward-adjoint products aggregated in physically meaningful patches;
- at most 2-4 matrix-free material-direction probes initially;
- normalized secant features to other active branch anchors;
- step acceptance, actual/predicted reduction and secant prediction error;
- a small recurrent memory.

Do not flatten a Jacobian into an image. If v_l is a material probe, a useful operator token is (v_l, J v_l, ||J v_l||, and optionally J*J v_l); every action must be charged to the ledger. Start with diagonal or very low-rank metric features rather than computing a complete eigendecomposition.

The preferred cross-representation operations are fixed Green application, actual constitutive multiplication, and forward/adjoint material actions. Learned gates can modulate these features. A positive softmax average cannot by itself reproduce signed, complex, oscillatory Green interactions. A Green-magnitude attention bias is therefore only a heuristic bias, not a Maxwell solver.

## 3. Sequence and generative time

Let k index physical pilot iterations and t index diffusion noise. The axes have different meanings and must have separate embeddings and schedules.

A standard conditional forward diffusion is Markov:

    p(x_0:T | y) = pi(x_0|y) product_t q_t(x_t|x_(t-1)).

Its exact reverse chain satisfies

    p(x_(t-1) | x_t, x_(t+1:T), y) = p(x_(t-1) | x_t,y).

Consequently, an unlimited exact reverse model does not need previous denoising states. Memory can help an imperfect score model, a compressed condition, a numerical multistep solver, or a deliberately non-Markov generative construction. It is not intrinsically a more complete posterior.

DiT changes the denoiser architecture to a Transformer; its usual spatial patch attention is not denoising-history attention. Diffusion Forcing uses a sequence position and a separately chosen per-token noise level. Its original main implementation uses a causal recurrent latent state; its sequence positions are not successive denoising stages of the same image. Self-conditioning already reuses a previous clean prediction in diffusion. These mechanisms are precedents, not evidence that historical attention helps Maxwell topology probabilities.

Do not train on deterministic GN trajectories and call the resulting trajectory distribution a material posterior. Such training can only imitate the distribution of the supplied optimizer and its initializations unless a posterior objective or correction is added.

Recommended first timing design: 2-3 physical pilot rounds, freeze their contexts, then generate final conditional material samples. Test occasional physics refresh inside diffusion only after a cheap conditional diffusion baseline wins a mode-coverage test. Diffusion nested inside every GN iteration is not the default.

## 4. Competitive architectures

### A. Direct material mixture or set predictor

    q(chi|y) = sum_i a_i(y) q_i(chi|y).

Use a proper mixture likelihood, a declared finite-state classifier, or event-based matching. Winner-take-all/best-of-K loss alone does not identify mode masses. Hungarian matching solves labeling, not probability calibration. This is the mandatory inexpensive baseline.

### B. Physics-state recurrent multi-hypothesis generator

Maintain separate branch pilot states and memories. Share the measurement encoder, not averaged material images. A GRU-conditioned proposal updates each branch, using material gradients and response contrasts. This is the preferred physical-state route, but it must defeat a memory-free conditional generator at matched total cost.

### C. Physics-conditioned diffusion or flow-matching

Train a conditional score or velocity on paired material/measurement samples, possibly with topology conditioning. It can represent rich within-branch variation, but sample quality does not establish correct branch mass. Guidance using a clean estimate inside F generally approximates the noisy-time likelihood gradient. An implicit diffusion endpoint does not provide a cheap exact q(x|y) for ordinary importance weighting. Probability-flow density computation, augmented path-space methods, or TDS-like SMC are possible but add costs and assumptions.

### D. Explicit branching model

    q(H,z|y) = alpha_H(y) q_H(z|y).

Start with physically meaningful solid/cavity branches and proper parameter priors. Conditional flows need not be unimodal, and one topology can have several posterior modes. Add birth/death or split/merge only with valid proposal accounting or a clearly non-Bayesian calibrated event target. A neural 'other' logit without an actual material distribution is not support for unknown worlds.

### E. A17/B1-guided probability changes

For frozen models with a shared whitened residual and common prior,

    J1-J0 = Y Gamma^(-1) N,
    U(s) = (N s, Y^T J0 s),
    d pi1 / d pi0 proportional to exp[-h(U(s))].

Use the two-sided signature as a feature of model change, a conditional reweighting diagnostic, or a cheap proposal descriptor. Do not treat it as the sufficient statistic of an arbitrary nonlinear, cross-topology material posterior. Non-shared anchors, nuisance-dependent covariance and branch evidence require additional terms.

### F. Masked physics-conditioned triangular flow (research option)

Split a material coordinate z=(z_A,z_B). Build a physical context ONLY from z_A, y and a fixed fill-in for z_B. Then

    z_A' = z_A,
    z_B' = z_B * exp(s(z_A,physics(z_A),y)) + t(z_A,physics(z_A),y)

has triangular Jacobian and log determinant sum(s). If the context depends on z_B too, this determinant formula is generally false. A stop-gradient in code does not change the mathematical density. This route places physics inside a normalized generative map but can require a solver for every layer and sample. Retain it as a theoretical option, not the first implementation.

## 5. Preferred architecture: branch-contrast recurrent conditional generation

### 5.1 Pilot phase

For h=1,...,H and k=0,...,R-1, with R=2 initially:

    chi_hk = D_h(z_hk)
    F_hk, E_hk, j_hk = Maxwell(chi_hk)
    r_hk = whiten(F_hk-y)
    g_hk = J_hk^T r_hk
    c_hk = StateEncoder(z_hk,r_hk,g_hk,probes,branch contrasts,d)
    m_h,k+1 = GRU(m_hk,c_hk)
    z_h,k+1 = z_hk - eta_hk P_hk g_hk + delta_theta(c_hk,m_h,k+1)

Pilot states are proposals and computational anchors, not probability-bearing copies of the truth. A bounded/feasibility-preserving map keeps the material in its declared family. P is a positive regularized material-coordinate preconditioner with an eigenvalue floor; it never sets all weak directions to zero. Do not use a learned change in step size as a posterior weight update.

Use separate branch coordinates if dimensions differ. For a first continuous pilot, start with 2-4 coordinates per branch; larger 16-64 dimensional wavelet/shape fields are a later representation experiment, not a required E0/E1 dependency.

### 5.2 Frozen-context generation

Freeze c_h(y) and memory after the pilot. A small branch classifier and conditional density model produce

    alpha(y) = calibrated softmax(logits(c_1,...,c_H,y))
    z_hi = T_theta,h(epsilon_hi; c_h(y)), epsilon_hi ~ N(0,I).

Start with an MDN; promote to 4-6 bounded-scale affine or rational-quadratic-spline coupling blocks only if needed. Reference initial capacity: state hidden size 64-128, two recurrent rounds, 0.2-1 million total parameters, and a 16x16 or 24x24 visualization grid. These are design targets, not measured memory or speed.

The flow conditioner may depend arbitrarily on y and independent pilot randomness, but not unaccountably on the same production draw. Condition on the frozen pilot when computing q. If pilot samples adapt q or determine allocation, use fresh production samples conditional on that pilot. Recycling adaptive samples requires a valid multiple-importance or sequential importance derivation.

### 5.3 Mode-preserving sharing

Share acquisition tokens and fixed read-only summaries of competing responses. Keep branch-specific residuals, material parameters and memory. Cross-hypothesis attention, if tested, should predict comparison features or proposal parameters, not average the material states. Do not add an unconditional repulsion loss: forcing all images apart can assign mass to impossible structures. Within-mode diversity comes from the conditional density target, and inter-mode coverage from the event model and support.

### 5.4 Training

From joint simulator draws (H,z,gamma,y), minimize

    L_joint = E[-log alpha_H(y) - log q_H(z,gamma|y)].

For trusted soft posterior labels, minimize

    L_distill = E_y sum_h p_ref(h|y)[-log alpha_h(y)]
              + E_y E_(x~pi_ref(.|y))[-log q_H(z,gamma|y)].

Use log-score as the primary probability objective and Brier score as a diagnostic; calibrate on a separate split. Optional pilot material/gradient losses train a useful optimizer context, but do not replace the distributional objective. If adding a residual penalty on generated samples changes the distributional optimum, identify and measure that bias rather than calling it an independent observation.

Use several noise realizations per stored forward solution without recounting them as independent parent geometries. Never leak the true field, true Jacobian, or oracle material directions into inference inputs. Training teacher states must be recomputable from the current hypothesis at inference.

### 5.5 Fast and corrected output modes

**Fast mode:** alpha is a calibrated discriminative topology/event predictor; q supplies conditional materials. Report calibration under the declared data distribution and do not silently relabel alpha as exact evidence.

**Corrected mode:** use full likelihood and a normalized proposal. With a fixed pilot and n_h independently generated production particles per branch,

    q_def,h = (1-epsilon) q_H + epsilon p0(.|H),
    u_hi = pi0(h) p0(z_hi,gamma_hi|h) exp[-Phi_y(h,z_hi,gamma_hi)]
           / (n_h q_def,h(z_hi,gamma_hi|y)),
    W_hi = u_hi / sum_(g,j) u_gj.

For random branch draws instead, use the JOINT proposal alpha_h q_h in the denominator. Do not combine the two sampling formulas. In the stratified formula alpha may choose n_h but is not multiplied again as a prior. A data-conditioned proposal is not a second independent prior.

Conditional on the pilot, branch evidence estimators are unbiased under support and integrability conditions. The self-normalized posterior mass ratio is not unbiased at finite particle count; it is consistent under ordinary laws of large numbers. A defensive prior mixture restores support only relative to the declared prior; it does not promise small variance or rapid discovery of every mode.

Report branch ESS, max weight, repeat-run variation and quadrature-reference error. A 16-image gallery cannot guarantee two-percent probability accuracy. Increase integration particles separately or output PROBABILITY_UNRESOLVED. ESS alone is not a missing-mode certificate.

### 5.6 Physical correction and constraints

Every reported full-wave residual must be evaluated on the final reported material, with the declared nuisance handling. A final deterministic GN projection generally changes q and invalidates previous importance weights. Use it only as an explicitly unweighted visualization, include the actual change-of-variables density when legitimate, or use a target-preserving MH/rejuvenation kernel with sufficient diagnostics.

For e^(-i omega t), parameterize absolute material loss nonnegatively. A relative material increment can have negative imaginary part while both endpoints are passive. Multiple frequencies should share a declared causal dispersive material family (e.g. positive-damping Lorentz terms) when causality is claimed. Finite-band measurements do not validate all out-of-band dispersion. Multiple illuminations must share one material and the appropriate common nuisance, rather than receiving separate freely fitting materials.

## 6. Theorems, proof outlines and limitations

### P1. State information and conditional risk

Assume S=g(Y,U) and U is independent of the true material conditional on Y. Then chi -> Y -> S is a conditional Markov relation, hence I(chi;S|Y)=0. For a compressed context C and additional history M, the difference in unrestricted optimal discrete log-loss is I(H;M|C). If C already contains the full Y, the difference is zero under the same independence assumptions. A finite network class can still benefit computationally. This is known conditional-information theory, not new Maxwell mathematics.

### P2. Frozen branch-contrast sufficiency

Work in whitened real data with common covariance I. Let the candidate family have affine responses f_h+A_h z. Fix f_ref. Let Q have orthonormal columns spanning all f_h-f_ref and all ranges(A_h). Then

    ||y-f_h-A_h z||^2
      = ||Q^T(y-f_h-A_h z)||^2 + ||(I-QQ^T)(y-f_ref)||^2.

The second term is common to all hypotheses and coordinates. Thus Q^T y, together with the known model and priors, is sufficient for the entire FROZEN mixture posterior. The finite-state special case needs only response contrasts and rank at most H-1. The argument is a Gaussian sufficiency identity, not a new general statistical result.

Maxwell's contribution is how f_h differences and A_h actions are constructed: finite full-wave secants, forward-adjoint products and material-coordinate probes. Candidate practical novelty is a branch-contrast token mechanism that retains probability-relevant response differences more cheaply than storing all fields or Jacobians. This still needs experiments and prior-art comparison.

Limitations: dimension can reach all measurement coordinates; nonlinear response tails are not captured automatically; differing covariances require quadratic/log-determinant terms; branch offsets cannot be dropped; a feature selected using hidden truth is invalid. Any y-adaptive Q must be treated with its selection context, or tested as a computational feature rather than claimed to be a fixed low-dimensional statistic.

### P3. Probability error from compressed likelihood

For a common prior let exact and approximate potentials differ by e(x). If the essential oscillation of e over the full relevant support is omega, then

    TV(pi,pi_tilde) <= tanh(omega/4).

Sketch: d pi/d pi_tilde is a normalized bounded exponential tilt. The extremal two-point tilt gives the sharp bound. Every material event error is bounded by TV. This is inherited from B1/A31 and standard bounded-tilt analysis. A bound on a finite pilot bank is only a finite-bank bound; continuous posterior tails require separate control. Do not revive an expensive global Neumann certificate search.

### P4. Branch mass consistency

For a measurable partition H=union_j H_j, pi(H)=sum_j pi(H_j). Splitting a particle into copies with conditional fractions a_j requires weights W_j=W a_j, sum a_j=1. Merging preserves the sum. Removing a represented region of exact posterior mass rho and conditioning on its complement has TV exactly rho. An estimated tiny weight is not a guarantee that the true removed mass is tiny.

### P5. One observation is used once

For pi_beta proportional to p0 L^beta, beta_0=0 and beta_R=1, an SMC reweighting increment is L^(beta_(k+1)-beta_k). Products telescope to L. Multiplying by L at each of R iterations instead targets p0 L^R. An arbitrary state-dependent proposal additionally requires its forward/backward proposal accounting. This is ordinary sequential Monte Carlo, not a novel Bayes rule.

### P6. Score distillation and decision risk

If a normalized event predictor alpha is trained on the exact conditional probabilities p, excess population log score is E_Y KL(p(.|Y)||alpha(.|Y)). Pinsker and Jensen yield E TV <= sqrt(excess/2). If the reference probabilities have mean TV error delta, add delta. For losses in [0,Lmax], the mean excess Bayes decision risk is at most 2 Lmax times this TV bound. Finite empirical training loss and ECE do not themselves certify the population bound. A constant prior predictor can be calibrated and uninformative, so test both sharpness/proper score and calibration.

### P7. Target-preserving geometry

For pi(x) proportional to exp[-V(x)] with a smooth positive definite mobility M(x), suitable boundary and integrability conditions give the Ito diffusion

    dx = [-M grad V + div M] dt + sqrt(2M) dW

stationary density pi relative to Lebesgue measure. The probability flux vanishes. Omitting div M, confusing Riemannian and Lebesgue volume, ignoring transformed-coordinate Jacobians or using an uncorrected finite step generally changes the target. Material GN geometry can precondition sampling; it does not by itself determine the correct multimodal law. Discrete topology changes require a jump/mixture mechanism, not a continuous SDE restricted to one fixed chart.

### P8. Sequential stability is conditional

For Markov kernels with Dobrushin coefficients tau_k and uniform per-step kernel errors epsilon_k,

    e_(k+1) <= tau_k e_k + epsilon_k.

Add explicit numerical and pruning errors when their TV bounds are valid. A multimodal Maxwell problem need not have tau_k<1. Reweighting can amplify errors, and arbitrary deterministic learned updates are not automatically target-preserving kernels. Prefer task-level probability tests when a useful global contraction bound is unavailable.

### P9. Support and topology generalization

If q is supported in S, TV(pi,q)>=pi(S^c). A strict low-dimensional smooth manifold can have zero mass under an absolutely continuous full-dimensional posterior, giving TV=1 even when image error is small. Declare the low-dimensional target honestly, or retain a random fine-scale tail / full-support field model. A continuous invertible flow CAN represent multiple density modes; it is not inherently unimodal. What a positive-density flow on a connected support cannot exactly reproduce is an arbitrarily disconnected support with strict zero-density barriers without a suitable mixture/support construction.

No detector can reliably identify unseen topology solely from measurements if the unseen and known materials induce identical measurement distributions. Predictive checks and OOD scores can detect some mismatches, not all. 'Other' mass is meaningful only relative to an actual modeled other distribution.

### P10. History and exact denoising

The exact reverse-Markov identity above follows from the conditional factorization of the forward diffusion chain. This is an important negative control: history attention can only earn its cost through an approximation or computational advantage, not through an unconditional information gain theorem.

### Additional counterexamples

- F(x)=x^2 has two global branches even though a Jacobian at x=0 is zero.
- Two models with equal best residual can have different evidence because their peak volumes and proper priors differ.
- A deterministic invertible transport of a random base can preserve multiple modes; deterministic does not mean single-mode.
- A deterministic map from one fixed initializer produces one atom, and a globally contractive common fixed point collapses all initial hypotheses.
- Copying a mode more often changes naive normalized residual weights without adding evidence.
- A final untracked physical projection can improve residual and spoil calibration.
- Same-data prior fitting followed by likelihood multiplication can double count evidence.

## 7. Material representation decision

E0: one-dimensional sign/amplitude branches. E1: exactly the 18 B1 finite states. First continuous extension: solid/cavity branches with 2-4 bounded geometry/material coordinates and declared priors. Later compare a multiresolution field with random high-frequency tail, a level-set field, and the restricted chart under the SAME physical tasks.

Avoid a globally fixed Gaussian/PCA manifold as the only support. A small latent is legitimate if it defines the actual restricted prior being studied, but it is not evidence for recovery of the full voxel posterior. State compression and material-support compression are different design choices.

Topology labels need either a decoder that enforces their meaning or post-decoding event evaluation. A wavelet residual can create/remove holes, so a branch logit is not automatically the final decoded topology probability. Test minimum feature size and threshold margin. Weakly observed directions should remain random under their conditional prior/proposal instead of being forced to zero.

## 8. Evaluation and cost

Use topology mass error, proper log score, Brier score, conditional coverage, parent-held-out reliability, material error, boundary and finite-scale topological metrics. Best-of-K is only an oracle diagnostic. Report precision as well as mode recall; generating every conceivable shape is not successful inference.

Uncertainty reports must distinguish observation ambiguity, topology uncertainty, within-topology parameters, learned-model uncertainty, numerical/sampling uncertainty, and prior/training shift. These terms interact and are not generally a unique additive decomposition of total variance.

Charge all forward solutions, adjoints, JVPs, rejected proposals, reference quadrature, data generation, cache setup, surrogate construction and training. A material-frequency evaluation is not identical to one right-hand-side solve. Record both. Do not compare only neural FLOPs.

For H branch anchors, R physical rounds, S sources and p derivative probes, pilot work is approximately H R times the relevant forward/adjoint/probe solve cost, followed by N_integration full likelihood evaluations in corrected mode. Sharing network weights does not share material-dependent Maxwell factorizations. Diffusion with physical guidance can multiply this by denoising stages and candidate count.

Report iso-parameter/iso-data comparisons separately from iso-wall-time or iso-solver-budget comparisons. They answer different questions. Repeated-query amortization is useful only when total training plus data-generation cost is included.

## 9. Research verdict

A successful C1 paper should demonstrate a calibrated material-world output, a specific Maxwell state mechanism with a falsifiable advantage, representation support controls, and small genuine vector-Maxwell validation. A useful generator without a physics-state advantage is still a valid tool, but a weaker strong-TAP novelty case. A larger Transformer that only improves best-of-K or residual is not enough.

First implement E0 probability semantics and E1 finite-B1 reference. Then test branch contrasts and gradients before any full field/current/history stack. Admit the possibility that a direct MDN or conditional flow is the final winner. Promote diffusion only when it beats that baseline in missing-mode rate and proper probability quality at matched total cost.
