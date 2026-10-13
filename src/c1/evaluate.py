"""Locked evaluation after model freeze, with independent continuous references."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.special import logsumexp
from .accounting import ROOT,dump,PhysicsLedger
from .data import load_split,reference
from .density import Density
from .material import z_to_x,quadrant
from .problems import DipoleProblem

def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cpu')

def load_model(problem,kind,seed):
    c=torch.load(ROOT/'results'/problem/'models'/f'{kind}-{seed}.pt',map_location='cpu',weights_only=True)
    m=Density(c['kind'],c['y_dim'],c['state_dim'],c['target_parameters']).cpu()
    m.load_state_dict(c['state_dict']);m.eval();return m

def paired_bootstrap(diff,parent,seed,repeats=2000):
    means=np.array([np.mean(diff[parent==p]) for p in np.unique(parent)])
    rng=np.random.default_rng(seed)
    boot=means[rng.integers(len(means),size=(repeats,len(means)))].mean(axis=1)
    return dict(mean=float(means.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                parents=len(means),bootstrap_repeats=repeats,seed=seed,
                hierarchy='noise averaged within parent; training seeds reported separately')

def aggregate(rows):
    names=['KL_reference','event_mass_MAE','event_mass_max_error','expected_Brier_excess','mode_weighted_recall',
           'mode_precision','truth_log_score','coordinate_90_coverage','reference_interval_content_error',
           'best_of_16_l2','boundary_sample_fraction']
    out={k:float(np.mean([r[k] for r in rows])) for k in names}
    out['event_mass_max_error_p95']=float(np.quantile([r['event_mass_max_error'] for r in rows],.95))
    out['mode_missing_fraction']=float(np.mean([r['missing_mode_count']>0 for r in rows]))
    out['observations']=len(rows)
    out['quality_gate']=('PASS' if out['event_mass_MAE']<=.02 and out['event_mass_max_error_p95']<=.05
                         and out['KL_reference']<=.02 else 'FAIL')
    return out

def evaluate(problem):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text());path=ROOT/'results'/problem
    if (path/'EVALUATION.json').exists() or (path/'OBSERVATION_METRICS.jsonl').exists():
        raise RuntimeError('completed/partial evaluation refused; explicit recorded resume required')
    data=load_split(problem,'test');freeze=json.loads((path/'MODELS_FROZEN.json').read_text())
    grids=[dict(np.load(path/f'grid_{n}.npz')) for n in conf[problem]['quadrature_orders']]
    fine=grids[-1];rx=fine['x'];gx=tensor(rx)
    Y=tensor(data['inputs_norm']);S=tensor(data['states_norm']);nobs=len(Y)
    refs=[];reference_rows=[]
    for i in range(nobs):
        r=[reference(g,data['y'][i],data['sigma'][i],data['kappa'][i],problem) for g in grids]
        delta=float(np.max(np.abs(r[0]['events']-r[-1]['events'])))
        dz=abs(r[0]['logZ']-r[-1]['logZ'])
        refs.append(r[-1]);reference_rows.append(dict(observation=i,parent=int(data['parent'][i]),
            event_difference=delta,log_evidence_difference=dz,events=r[-1]['events'].tolist(),sigma=float(data['sigma'][i]),
            kappa=float(data['kappa'][i])))
    threshold=1e-8 if problem=='u0' else .01
    reference_gate='PASS' if max(r['event_difference'] for r in reference_rows)<=threshold and max(r['log_evidence_difference'] for r in reference_rows)<=.05 else 'REFERENCE_UNRESOLVED'
    if problem=='u0':
        adaptive=json.loads((path/'INDEPENDENT_REFERENCE.json').read_text())
        if adaptive['gate']!='PASS':reference_gate='REFERENCE_UNRESOLVED'
    dump(path/'REFERENCE_AUDIT.json',dict(status=reference_gate,orders=conf[problem]['quadrature_orders'],
        max_event_difference=max(r['event_difference'] for r in reference_rows),
        max_log_evidence_difference=max(r['log_evidence_difference'] for r in reference_rows),
        per_observation=reference_rows,certified=False,independent_adaptive='INDEPENDENT_REFERENCE.json' if problem=='u0' else 'NOT_RUN',
        meaning='empirical integration agreement; all metrics provisional when unresolved'))
    all_results={};all_rows={};timings={};samples_gallery={};samples_checks={}
    draw_count=conf['evaluation_draws']
    kinds=freeze['models']
    for seed in conf['training_seeds']:
        for kind in kinds:
            model=load_model(problem,kind,seed);key=f'{kind}-{seed}'
            wall0,cpu0=time.perf_counter(),time.process_time()
            generator=torch.Generator(device='cpu').manual_seed(261013400+seed)
            with torch.no_grad():samples=z_to_x(model.sample_z(Y,S,draw_count,generator=generator).numpy())
            sample_timing=dict(wall=time.perf_counter()-wall0,cpu=time.process_time()-cpu0,
                               draws=nobs*draw_count,cached_context=True)
            rows=[]
            with torch.no_grad():
                C=model.c(Y,S)
                logtruth=model.log_prob_x(tensor(data['x']),Y,S).numpy()
                for i in range(nobs):
                    ci=C[i:i+1].expand(len(rx),-1)
                    logq=model.log_prob_x_context(gx,ci).numpy().astype(float)
                    ref=refs[i];p=ref['events'];a=samples[i];event=quadrant(a)
                    q=np.bincount(event,minlength=4)/draw_count
                    kl=float(ref['weights']@(ref['logdensity']-logq))
                    active=p>=.05;claimed=q>=.05
                    lo,hi=np.quantile(a,[.05,.95],axis=0)
                    coverage=np.mean((data['x'][i]>=lo)&(data['x'][i]<=hi))
                    onehot=np.eye(4)[int(quadrant(data['x'][i]))]
                    rcontent=[float(ref['weights']@((rx[:,j]>=lo[j])&(rx[:,j]<=hi[j]))) for j in range(2)]
                    rows.append(dict(observation=i,parent=int(data['parent'][i]),seed=seed,model=kind,
                        KL_reference=kl,event_mass_MAE=float(np.abs(p-q).mean()),
                        event_mass_max_error=float(np.max(np.abs(p-q))),expected_Brier_excess=float(np.sum((p-q)**2)),
                        expected_Brier=float(1+q@q-2*q@p),observed_Brier=float(np.sum((q-onehot)**2)),
                        reference_expected_Brier=float(1-p@p),
                        mode_weighted_recall=float(np.sum(p[active&claimed])/max(np.sum(p[active]),1e-30)),
                        mode_precision=float(np.sum(q[active&claimed])/max(np.sum(q[claimed]),1e-30)),
                        missing_mode_count=int(np.sum(active&~claimed)),truth_log_score=float(-logtruth[i]),
                        coordinate_90_coverage=float(coverage),reference_interval_content_error=float(np.mean(np.abs(np.array(rcontent)-.9))),
                        best_of_16_l2=float(np.min(np.linalg.norm(a[:16]-data['x'][i],axis=1))),
                        boundary_sample_fraction=float(np.mean(np.any(np.abs(a)>=1.5,axis=1))),
                        reference_event_masses=p.tolist(),q_event_masses=q.tolist(),sigma=float(data['sigma'][i]),
                        q_sampling_se_max=float(np.max(np.sqrt(q*(1-q)/draw_count)))))
                    if problem=='p0':
                        # A material event after generation, not a preselected architecture branch.
                        target=float(ref['weights']@((rx[:,0]<=-.5)&(rx[:,1]>=.5)))
                        qm=float(np.mean((a[:,0]<=-.5)&(a[:,1]>=.5)))
                        rows[-1]['low_core_high_shell_event']=dict(reference_mass=target,q_mass=qm,
                            absolute_error=abs(target-qm),definition='epsilon_core_real<=1.5 and epsilon_shell_real>=2.5',
                            meaning='low-dielectric core event, not an exact vacuum atom or arbitrary 3D cavity')
            all_rows[key]=rows;all_results[key]=aggregate(rows)
            timings[key]=dict(sample_generation=sample_timing,total_evaluation_wall=time.perf_counter()-wall0,
                              total_evaluation_cpu=time.process_time()-cpu0)
            if seed==conf['training_seeds'][0]:
                samples_gallery[kind]=samples[0,:6]
                if problem=='p0':samples_checks[kind]=samples[:conf['p0']['full_sample_checks']['observations'],:conf['p0']['full_sample_checks']['draws']]
            print(json.dumps(dict(problem=problem,model=key,KL=all_results[key]['KL_reference'],
                                 mass_MAE=all_results[key]['event_mass_MAE'],reference=reference_gate)),flush=True)
    comparisons={}
    state_kind='mdn_state' if problem=='p0' else 'state'
    memory_kind='mdn_memory' if problem=='p0' else 'memory'
    for seed in conf['training_seeds']:
        base=freeze['baseline_selected_by_validation'][str(seed)];bkey=f'{base}-{seed}'
        for kind in [state_kind,memory_kind]:
            key=f'{kind}-{seed}'
            diff=np.array([r['KL_reference'] for r in all_rows[key]])-np.array([r['KL_reference'] for r in all_rows[bkey]])
            ci=paired_bootstrap(diff,data['parent'],conf['bootstrap_seed'],conf['bootstrap_repeats'])
            ci.update(baseline=base,relative_KL_improvement=(all_results[bkey]['KL_reference']-all_results[key]['KL_reference'])/max(all_results[bkey]['KL_reference'],1e-12),
                      event_MAE_difference=all_results[key]['event_mass_MAE']-all_results[bkey]['event_mass_MAE'],
                      scope='iso-data and near-iso-parameters; cold physical costs must be included separately')
            comparisons[key]=ci
        mkey=f'{memory_kind}-{seed}';skey=f'{state_kind}-{seed}'
        diff=np.array([r['KL_reference'] for r in all_rows[mkey]])-np.array([r['KL_reference'] for r in all_rows[skey]])
        comparisons[f'memory-vs-state-{seed}']=paired_bootstrap(diff,data['parent'],conf['bootstrap_seed'],conf['bootstrap_repeats'])
    with (path/'OBSERVATION_METRICS.jsonl').open('x') as f:
        for rows in all_rows.values():
            for r in rows:f.write(json.dumps(r)+'\n')
    result=dict(problem=problem,reference_status=reference_gate,models=all_results,
        paired_comparisons=comparisons,timing=timings,prior_id=conf['prior']['id'],
        probability_target='learned normalized q approximation to declared continuous prior posterior',
        event_partition='four coordinate-sign regions; analysis only, not topology or generator branches',
        sampling_error='1024 iid q draws; binomial MC uncertainty reported; no exact gallery probability',
        topology_input=False,gpu_seconds=0,baseline_selection=freeze['baseline_selected_by_validation'],
        metric_definitions={'mode_precision':'q mass in true-active claimed regions / q mass in claimed regions; mass-weighted, not count precision',
                            'mode_active':'reference event mass>=0.05; events are not automatically local density peaks'})
    dump(path/'EVALUATION_DRAFT.json',result)
    np.savez_compressed(path/'GALLERY_SAMPLES.npz',**samples_gallery,reference_events=refs[0]['events'],
                        y=data['y'][0],sigma=data['sigma'][0],kappa=data['kappa'][0],truth=data['x'][0])
    if problem=='p0':physical_diagnostics(data,samples_checks,samples_gallery,path)
    result['completion']='GENERATIVE_METRICS_AND_REGISTERED_PHYSICAL_DIAGNOSTICS_COMPLETE' if problem=='p0' else 'GENERATIVE_METRICS_COMPLETE'
    dump(path/'EVALUATION.json',result)

def physical_diagnostics(data,samples_checks,samples_gallery,path):
    dipole=DipoleProblem();dipole.scale=np.load(path/'scale.npy')
    rows=[];gallery=[]
    for kind,samples in samples_checks.items():
        for i,draws in enumerate(samples):
            for j,x in enumerate(draws):
                s=dipole.forward(x,kind='production_sample_check')
                res=float(np.linalg.norm((s['f']-data['y'][i])/data['sigma'][i]))
                rows.append(dict(model=kind,observation=i,draw=j,x=x.tolist(),whitened_residual_norm=res,
                                 passive=s['passive'],constitutive_residual=s['constitutive_residual'],source_count=2))
    for kind,draws in samples_gallery.items():
        for j,x in enumerate(draws):
            s=dipole.forward(x,kind='gallery_verification')
            gallery.append(dict(model=kind,draw=j,x=x.tolist(),prediction=s['f'].tolist(),
                whitened_residual_norm=float(np.linalg.norm((s['f']-data['y'][0])/data['sigma'][0])),passive=s['passive'],
                weight_type='unweighted iid q sample',point_probability=0,region_partition='contrast sign quadrants',
                event=int(quadrant(x)),sample_generation='normalized conditional density; no GN projection'))
    dump(path/'PHYSICAL_SAMPLE_CHECKS.json',dict(scope='first seed, fixed first16 observations x8 iid draws per model',
         all_sample_checks='NOT_RUN',rows=rows,summary={k:dict(calls=sum(r['model']==k for r in rows),
               median_whitened_residual=float(np.median([r['whitened_residual_norm'] for r in rows if r['model']==k])),
               passive_violations=sum(not r['passive'] for r in rows if r['model']==k)) for k in samples_checks}))
    cfg=json.loads((ROOT/'configs/protocol_v2.json').read_text())
    dump(path/'GALLERY_MANIFEST.json',dict(display_count=len(gallery),integration_count=cfg['p0']['quadrature_orders'][-1]**2,
         q_mass_estimation_draws=cfg['evaluation_draws'],
         target_kind='declared_continuous_prior_posterior_approximation',prior_id='connected-exponential-tilt-v1',
         material_representation_id='continuous-two-region-dipole-permittivity-v1',noise_model_id='real-intensity-Gaussian',
         nuisance_policy='fixed calibrated geometry; no nuisance parameter',solver_id=dipole.solver_id,
         point_probability_statement='Each exact continuous image has zero point mass',
         mode_mass_location='OBSERVATION_METRICS.jsonl observation0, first training seed, same model',
         mode_partition_id='contrast-sign-quadrants-v1',reference_error_status='REFERENCE_AUDIT.json',
         sampling_error_status='1024 q samples, finite binomial error',shift_status='not tested',
         resource_ledger_id='results/PHYSICS_LEDGER.jsonl',candidate_samples=gallery))
    dump(ROOT/'results/PHYSICS_SUMMARY.json',dipole.ledger.summary())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('problem',choices=['u0','p0'])
    evaluate(p.parse_args().problem)
