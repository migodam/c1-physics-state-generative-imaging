#!/usr/bin/env python3
"""Small deterministic algebra checks for C1, not imaging or training results.

Run: python c1_algebra_checks.py --out C1_CHECKS.json
Dependencies: numpy, scipy. No network, GPU, Maxwell solves, or user-file access.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp


def normalize(logw: np.ndarray) -> np.ndarray:
    return np.exp(logw - logsumexp(logw))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('C1_CHECKS.json'))
    args = parser.parse_args()
    rng = np.random.default_rng(261013)
    start = time.perf_counter()
    checks = {}

    # A finite family of Gaussian measurement means. Only their contrasts matter.
    m, h, rank = 11, 5, 3
    q, _ = np.linalg.qr(rng.normal(size=(m, rank)))
    anchor = rng.normal(size=m)
    means = anchor[:, None] + q @ rng.normal(size=(rank, h))
    y = rng.normal(size=m)
    prior = rng.dirichlet(np.ones(h))
    full = normalize(np.log(prior) - 0.5 * np.sum((y[:, None]-means)**2, axis=0))
    reduced = normalize(np.log(prior) - 0.5 * np.sum((q.T @ (y[:, None]-means))**2, axis=0))
    error = float(np.max(np.abs(full-reduced)))
    checks['finite_gaussian_contrast_sufficiency'] = {
        'passed': error < 1e-12, 'max_mass_error': error,
        'measurement_dimension': m, 'retained_dimension': rank}

    # Local affine branches: offsets AND tangent directions are necessary.
    q2, _ = np.linalg.qr(rng.normal(size=(13, 4)))
    common = rng.normal(size=13)
    branches = []
    for k in range(3):
        offset = common + q2 @ rng.normal(size=4)
        jac = q2 @ rng.normal(size=(4, 2))
        z = rng.normal(size=(2, 7))
        branches.append(offset[:, None] + jac @ z)
    fam = np.concatenate(branches, axis=1)
    obs = rng.normal(size=13)
    a = normalize(-0.5*np.sum((obs[:, None]-fam)**2, axis=0))
    b = normalize(-0.5*np.sum((q2.T@(obs[:, None]-fam))**2, axis=0))
    err = float(np.max(np.abs(a-b)))
    checks['affine_branch_offset_and_tangent_sufficiency'] = {'passed': err < 1e-12, 'max_mass_error': err}

    # Zero local Jacobians do not imply that branch offsets are irrelevant.
    ptrue = normalize(np.array([-0.5*1.0**2, -0.5*(1.0-2.0)**2]))
    obs2 = 1.6
    ptrue = normalize(-0.5*(obs2-np.array([0.0, 2.0]))**2)
    pbad = np.array([0.5, 0.5])
    checks['dropping_offsets_counterexample'] = {
        'passed': abs(ptrue[1]-pbad[1]) > 0.2,
        'true_second_branch_mass': float(ptrue[1]), 'jacobian_only_mass': 0.5}

    # Fixed allocations must have a 1/n_h factor even when evidence is equal.
    n = np.array([3, 21])
    pri = np.array([0.4, 0.6])
    correct = np.concatenate([np.full(n[k], pri[k]/n[k]) for k in range(2)])
    wrong = np.concatenate([np.full(n[k], pri[k]) for k in range(2)])
    correct /= correct.sum()
    wrong /= wrong.sum()
    mass_correct = float(correct[:n[0]].sum())
    mass_wrong = float(wrong[:n[0]].sum())
    checks['stratified_allocation_invariance'] = {
        'passed': abs(mass_correct-0.4) < 1e-12 and abs(mass_wrong-0.4) > 0.2,
        'allocations': n.tolist(), 'correct_first_mass': mass_correct, 'naive_first_mass': mass_wrong}

    # Splitting a represented event must divide its mass, not copy it.
    base = np.array([0.35, 0.65])
    split = np.array([0.35, 0.65*0.2, 0.65*0.8])
    checks['split_merge_mass_conservation'] = {
        'passed': bool(np.allclose([split[0], split[1:].sum()], base)),
        'before': base.tolist(), 'after_split': split.tolist()}

    # The same observation is not independent evidence at every iteration.
    prior_odds, likelihood_ratio = 1.0, 3.0
    final_odds = prior_odds*likelihood_ratio
    beta = np.array([0.0, 0.15, 0.45, 1.0])
    tempered = prior_odds * np.prod(likelihood_ratio**np.diff(beta))
    reused = prior_odds * likelihood_ratio**3
    checks['tempering_telescope_not_data_reuse'] = {
        'passed': abs(tempered-final_odds) < 1e-12 and reused != final_odds,
        'correct_probability': final_odds/(1+final_odds),
        'tempered_probability': tempered/(1+tempered),
        'wrong_repeated_probability': reused/(1+reused)}

    # A bounded log-density tilt obeys the sharp TV upper bound.
    omega = 1.1
    # Extremal two-point construction with tilt values 0 and omega.
    p = np.array([1/(1+math.exp(omega/2)), math.exp(omega/2)/(1+math.exp(omega/2))])
    tilted = normalize(np.log(p)-np.array([0.0, omega]))
    tv = float(np.abs(tilted-p).sum()/2)
    bound = math.tanh(omega/4)
    checks['bounded_tilt_tv'] = {'passed': abs(tv-bound) < 1e-12, 'TV': tv, 'bound': bound}

    # Expected multiclass log-score excess is exactly KL(p||q).
    p = np.array([0.2, 0.3, 0.5]); qv = np.array([0.25, 0.35, 0.4])
    excess = float(-p@np.log(qv) + p@np.log(p))
    kl = float(p@np.log(p/qv))
    checks['proper_log_score_identity'] = {'passed': abs(excess-kl) < 1e-12, 'excess_log_score': excess, 'KL': kl}

    # Linear Gaussian Markov chain: past and future conditional covariance is zero.
    a1, a2, v0, n1, n2 = 0.7, 0.8, 1.3, 0.2, 0.4
    v1 = a1*a1*v0+n1
    c01, c12, c02 = a1*v0, a2*v1, a2*a1*v0
    cond_cov = c02-c01*c12/v1
    checks['reverse_markov_gaussian_example'] = {'passed': abs(cond_cov) < 1e-12, 'conditional_covariance_X0_X2_given_X1': cond_cov}

    # E0 proposed reference: sign is independent of squared-magnitude observation.
    mu = np.array([0.3, 0.7]); yy, sig = 1.05, 0.18
    evid = quad(lambda aa: math.exp(-0.5*((yy-aa*aa)/sig)**2), 0.5, 1.5, epsabs=1e-12)[0]
    mass = mu*evid/(mu*evid).sum()
    checks['sign_ambiguity_reference'] = {'passed': bool(np.allclose(mass,mu)), 'posterior_sign_masses': mass.tolist(), 'unnormalized_common_evidence': evid}

    # Inserting a state depending on BOTH transformed coordinates spoils an
    # ordinary affine-coupling log determinant. A masked state remains valid.
    x, z = 0.6, 0.8
    # T_bad(x,z)=(x,z*exp(0.2*z)); actual derivative contains 1+0.2*z.
    naive_logdet, real_logdet = 0.2*z, 0.2*z+math.log1p(0.2*z)
    # T_ok(x,z)=(x,z*exp(0.2*x)); state depends only on unchanged x.
    checks['physics_conditioning_logdet_counterexample'] = {
        'passed': abs(real_logdet-naive_logdet)>0.1,
        'naive_logdet': naive_logdet, 'actual_logdet': real_logdet,
        'masked_conditioning_logdet': 0.2*x}

    rare = 0.01
    min_k = math.ceil(math.log(0.05)/math.log1p(-rare))
    checks['rare_mode_finite_sample_limit'] = {
        'passed': min_k == 299, 'mode_mass': rare,
        'miss_probability_K64': (1-rare)**64,
        'K_for_95_percent_hit_probability': min_k}

    for item in checks.values():
        item['passed'] = bool(item['passed'])

    payload = {'schema':'C1-algebra-v1','date':'2026-10-13','seed':261013,
        'scope':'Deterministic algebra and one-dimensional quadrature only; no C1 architecture was trained or benchmarked.',
        'new_maxwell_evaluations':0,'gpu_hours':0,'checks':checks,
        'check_count':len(checks),'status':'PASS' if all(c['passed'] for c in checks.values()) else 'FAIL',
        'elapsed_seconds':time.perf_counter()-start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'check_count':len(checks),'out':str(args.out),'new_maxwell_evaluations':0}))
    if payload['status'] != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
