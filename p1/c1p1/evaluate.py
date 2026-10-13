import argparse
import csv
import json
import time
import numpy as np
import torch
from .common import P1,ORIGINAL,config,dump,zero_physics_guard
from .reference import split,grid,posterior,reference_features,marginal_cdf,sampleprior,toy,to_x,event
from .refinement import refined_grid
from .density import numpy_parameters
from .train import model_for
from .mixture import one,logpdf_x,events,cdf,quantiles,moments_cdf,sample
from src.c1.density import Density

def load_model(problem,method,seed,condition='measurement'):
    if method=='historical_diag':
        cp=torch.load(ORIGINAL/f'results/{problem}/models/mdn-{seed}.pt',map_location='cpu',weights_only=True)
        m=Density(cp['kind'],cp['y_dim'],cp['state_dim'],cp['target_parameters'])
    else:
        cp=torch.load(P1/f'results/{problem}/models/{method}-{condition}-{seed}.pt',map_location='cpu',weights_only=True)
        m=model_for(method,cp['y_dim'],cp['state_dim'],condition)
    m.load_state_dict(cp['state_dict']);m.double().eval()
    return m

def fresh_u0():
    path=P1/'results/u0/fresh_final.npz'
    if path.exists():return dict(np.load(path))
    for problem in ['u0','p0']:
        if not (P1/f'results/{problem}/PRIMARY_MODELS_SELECTED.json').exists():
            raise RuntimeError('fresh panel generated only after primary selections are recorded')
    conf=config()['u0_fresh_test'];rng=np.random.default_rng(conf['seed'])
    x=sampleprior(rng,conf['parents']);k=rng.choice([0.,.25,.6],len(x));s=rng.choice([.12,.25],len(x))
    y=toy(x,k)+rng.normal(size=(len(x),3))*s[:,None]
    inputs=np.c_[y,k,np.log(s)];norm=np.load(ORIGINAL/'results/u0/normalization.npz')
    d=dict(x=x,y=y,kappa=k,sigma=s,parent=np.arange(len(x)),inputs=inputs,
           inputs_norm=((inputs-norm['ym'])/norm['ys']).astype(np.float32),states_norm=np.zeros((len(x),2,18),np.float32))
    np.savez_compressed(path,**d)
    dump(path.with_suffix('.json'),dict(seed=conf['seed'],parents=len(x),replicates=1,
        role='independent final U0 panel; generated after primary models selected',
        model_selection_use=False,new_physics=0,exposure_after_evaluation='now exposed; never reuse as a future blind test'))
    return d

def summary(rows):
    keys=['KL','KL_refined','KL_abs_order_difference','KL_abs_refined_order_difference',
          'event_MAE','event_MAE_refined','event_max_error','event_log_odds_MAE',
          'truth_NLL','expected_NLL','CDF_MAE','CDF_sup_error','mean_MAE','variance_MAE',
          'quantile_MAE','coverage90','reference_CI_content_error','tail_MAE',
          'region_mean_error','region_cov_error','region_mean_MC_SE',
          'mode_weighted_recall','missing_mode_fraction','boundary_fraction',
          'cdf_event_refinement','model_integral','model_integral_original64',
          'moment_quadrature_refinement','infinite_event_log_odds_count']
    out={key:float(np.mean([r[key] for r in rows if r[key] is not None]))
         if any(r[key] is not None for r in rows) else None for key in keys}
    out['event_p95']=float(np.quantile([r['event_max_error'] for r in rows],.95))
    out['observations']=len(rows)
    out['latent_abs_max']=max(r['latent_abs_max'] for r in rows)
    out['nonfinite_density_count']=sum(r['nonfinite_density_count'] for r in rows)
    return out

def bootstrap(diff,parents,repeats=2000):
    a=np.array([np.mean(diff[parents==p]) for p in np.unique(parents)])
    rng=np.random.default_rng(config()['bootstrap']['seed'])
    b=a[rng.integers(len(a),size=(repeats,len(a)))].mean(1)
    return {'mean':float(a.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),
            'parents':len(a),'repeats':repeats,'unit':'parent; noise averaged first'}

def query_timing(model,Y,S,points,seed):
    rows=[]
    for repeat in range(5):
        cpu,wall=time.process_time(),time.perf_counter()
        p=one(numpy_parameters(model,Y[:1],S[:1]),0)
        x,z=sample(p,1024,np.random.default_rng(seed+repeat))
        encode_sample_cpu=time.process_time()-cpu
        sampling_wall=time.perf_counter()-wall
        cpu=time.process_time();logpdf_x(points[:1024],p)
        rows.append({'encoding_sampling_cpu':encode_sample_cpu,'encoding_sampling_wall':sampling_wall,
                     'density_1024_cpu':time.process_time()-cpu})
    return {key:float(np.median([r[key] for r in rows])) for key in rows[0]}

def evaluate_panel(problem,panel,data,methods,condition='measurement'):
    conf=config();folder=P1/'results'/problem/'evaluation';folder.mkdir(parents=True,exist_ok=True)
    outfile=folder/f'{panel}-{condition}.json'
    if outfile.exists():raise RuntimeError('completed panel evaluation not overwritten')
    coarse,fine=[grid(problem,n) for n in conf['reference_orders'][problem]]
    # Keep original bank scores. Other quantities use the independently checked
    # numerical refinement of exactly the same continuous prior/likelihood.
    stable_coarse=coarse if problem=='u0' else refined_grid(96)
    stable_fine=fine if problem=='u0' else refined_grid(128)
    refs=[]; audits=[]
    for i in range(len(data['y'])):
        rr=[posterior(g,data['y'][i],data['sigma'][i],data['kappa'][i],problem)
            for g in [coarse,fine,stable_coarse,stable_fine]]
        f0=reference_features(stable_coarse,rr[2]);f1=reference_features(stable_fine,rr[3])
        refs.append((rr,f1))
        audits.append(dict(observation=i,parent=int(data['parent'][i]),
            event_original_refined_difference=float(np.max(np.abs(rr[1]['events']-rr[3]['events']))),
            refined_event_difference=float(np.max(np.abs(rr[2]['events']-rr[3]['events']))),
            refined_cdf_difference=float(np.max(np.abs(np.asarray(f0['cdf'])-f1['cdf']))),
            refined_quantile_difference=float(np.max(np.abs(np.asarray(f0['quantiles'])-f1['quantiles']))),
            refined_tail_difference=float(np.max(np.abs(np.asarray(f0['tail'])-f1['tail']))),
            refined_cdf_nonmonotone=f1['cdf_nonmonotone_max'],reference_features=f1))
    dump(folder/f'{panel}-REFERENCE.json',dict(problem=problem,panel=panel,
         original_bank_orders=conf['reference_orders'][problem],reference_target='original continuous prior/likelihood',
         cdf_basis='analytic U0' if problem=='u0' else 'degree12 bank-interpolated P0 response, empirical cross-check, no global certificate',
         per_observation=audits))
    group={};totals=[];timings={}
    Xs=[g['x'] for g in [coarse,fine,stable_coarse,stable_fine]]
    for seed in conf['seeds']:
        for method in methods:
            model=load_model(problem,method,seed,condition)
            p=numpy_parameters(model,data['inputs_norm'],data['states_norm'])
            key=f'{method}-{condition}-{seed}';rows=[]
            cpu,wall=time.process_time(),time.perf_counter()
            for i in range(len(data['y'])):
                pi=one(p,i);rr,rf=refs[i]
                logqs=[logpdf_x(x,pi) for x in Xs]
                kls=[float(r['mass']@(r['logpdf']-lq)) for r,lq in zip(rr,logqs)]
                q=events(pi);q32=events(pi,32);target=rr[1]['events'];target_f=rr[3]['events']
                qu=quantiles(pi);mu,var=moments_cdf(pi,256);mu0,var0=moments_cdf(pi,128)
                qc=cdf(pi,rf['thresholds']);targetcdf=np.asarray(rf['cdf'])
                tc=cdf(pi,[-1.4,1.4]);qt=tc[:,0]+1-tc[:,1]
                ci=marginal_cdf(stable_fine,rr[3]['mass'],qu[:,[0,2]].ravel())
                content=np.array([ci[j,2*j+1]-ci[j,2*j] for j in range(2)])
                covered=(data['x'][i]>=qu[:,0])&(data['x'][i]<=qu[:,2])
                active=target>=.05;claimed=q>=.05
                important=(target>=.05)&(target<=.95)
                valid=important&(q>0)&(q<1)
                odds=np.abs(np.log(target[valid]/(1-target[valid]))-np.log(q[valid]/(1-q[valid])))
                rng=np.random.default_rng(261013600+seed+i)
                draws,z=sample(pi,2048,rng);ids=event(draws)
                meanerr=[];coverr=[];mc=[];mass=[]
                for j in range(4):
                    a=draws[ids==j];rshape=rf['within_region'][j]
                    if rshape['mass']>=.05:
                        if len(a)<8:
                            meanerr.append(3.);coverr.append(9.);mc.append(3.);mass.append(rshape['mass'])
                        else:
                            meanerr.append(np.linalg.norm(a.mean(0)-rshape['mean']))
                            coverr.append(np.linalg.norm(np.cov(a.T)-rshape['cov']))
                            mc.append(np.sqrt(np.var(a,axis=0,ddof=1).sum()/len(a)))
                            mass.append(rshape['mass'])
                mass=np.asarray(mass);mass=mass/mass.sum()
                row=dict(problem=problem,panel=panel,method=method,condition=condition,seed=seed,
                    observation=i,parent=int(data['parent'][i]),sigma=float(data['sigma'][i]),kappa=float(data['kappa'][i]),
                    KL=kls[1],KL_refined=kls[3],KL_abs_order_difference=abs(kls[1]-kls[0]),
                    KL_abs_refined_order_difference=abs(kls[3]-kls[2]),
                    event_MAE=float(np.abs(q-target).mean()),event_MAE_refined=float(np.abs(q-target_f).mean()),
                    event_max_error=float(np.abs(q-target).max()),event_log_odds_MAE=float(odds.mean()) if len(odds) else None,
                    infinite_event_log_odds_count=int(np.sum(important&~valid)),
                    truth_NLL=float(-logpdf_x(data['x'][i:i+1],pi)[0]),expected_NLL=float(-rr[3]['mass']@logqs[3]),
                    CDF_MAE=float(np.abs(qc-targetcdf).mean()),CDF_sup_error=float(np.abs(qc-targetcdf).max()),
                    mean_MAE=float(np.abs(mu-rf['mean']).mean()),variance_MAE=float(np.abs(var-rf['variance']).mean()),
                    quantile_MAE=float(np.abs(qu-rf['quantiles']).mean()),coverage90=float(covered.mean()),
                    reference_CI_content_error=float(np.abs(content-.9).mean()),tail_MAE=float(np.abs(qt-rf['tail']).mean()),
                    region_mean_error=float(mass@meanerr),region_cov_error=float(mass@coverr),region_mean_MC_SE=float(mass@mc),
                    mode_weighted_recall=float(target[active&claimed].sum()/target[active].sum()),
                    missing_mode_fraction=float(np.any(active&~claimed)),
                    boundary_fraction=float(np.mean((np.abs(draws)>=1.5).any(1))),latent_abs_max=float(np.abs(z).max()),
                    cdf_event_refinement=float(np.abs(q-q32).max()),
                    model_integral=float(np.exp(logqs[3])@stable_fine['w']),
                    model_integral_original64=float(np.exp(logqs[1])@fine['w']),
                    moment_quadrature_refinement=float(max(np.abs(mu-mu0).max(),np.abs(var-var0).max())),
                    nonfinite_density_count=int(np.sum(~np.isfinite(logqs[3]))),
                    reference_events=target.tolist(),q_events=q.tolist(),q_mean=mu.tolist(),q_variance=var.tolist(),
                    q_quantiles=qu.tolist(),reference_interval_content=content.tolist())
                rows.append(row)
            group[key]=rows
            result=summary(rows)
            if method=='historical_diag':
                receipt=json.loads((ORIGINAL/f'results/{problem}/models/mdn-{seed}.json').read_text())
                training_cpu=0.;parameters=receipt['trainable_parameters'];step=receipt['checkpoint_step']
            else:
                receipt=json.loads((P1/f'results/{problem}/models/{method}-{condition}-{seed}.json').read_text())
                training_cpu=receipt['cpu_seconds'];parameters=receipt['parameters'];step=receipt['checkpoint']
            result.update(problem=problem,panel=panel,method=method,condition=condition,seed=seed,
                parameters=parameters,checkpoint=step,training_cpu=training_cpu,
                evaluation_cpu=time.process_time()-cpu,evaluation_wall=time.perf_counter()-wall,
                exposure='new independent final U0, now exposed for future work' if panel=='fresh_final' else 'exposed regression',
                new_Maxwell=0,new_B1=0,gpu_seconds=0,
                future_candidate_setup_mf=0 if condition=='measurement' else 2,
                future_candidate_setup_RHS=0 if condition=='measurement' else 12)
            timing=query_timing(model,data['inputs_norm'],data['states_norm'],stable_fine['x'],seed)
            result.update(timing);timings[key]=timing;totals.append(result)
            print(json.dumps({k:result[k] for k in ['problem','panel','method','condition','seed','KL','KL_refined','event_MAE','coverage90']}),flush=True)
    pairs=[]
    comparisons=[('full_joint','diag_joint'),('diag_distill','diag_joint'),('full_distill','full_joint'),('full_distill','diag_distill'),('full_distill','diag_joint')]
    if condition=='measurement':
        for seed in conf['seeds']:
            for left,right in comparisons:
                lk=f'{left}-measurement-{seed}';rk=f'{right}-measurement-{seed}'
                if lk not in group or rk not in group:continue
                for metric in ['KL','KL_refined','event_MAE','truth_NLL','coverage90','CDF_MAE','tail_MAE']:
                    diff=np.array([r[metric] for r in group[lk]])-np.array([r[metric] for r in group[rk]])
                    pairs.append(dict(problem=problem,panel=panel,seed=seed,left=left,right=right,metric=metric,
                         **bootstrap(diff,data['parent'])))
    with (folder/f'{panel}-{condition}-OBSERVATIONS.jsonl').open('x') as f:
        for rows in group.values():
            for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    dump(outfile,dict(problem=problem,panel=panel,condition=condition,models=totals,paired=pairs,
         reference_precision='separate original-bank and denser bank-interpolation KL; all metrics empirical',
         density_precision='float64 model/encoder/logit/logpdf/decode; no clipping or resampling',
         events='four analysis-only sign regions; not topology classes',
         within_region_sampling='2048 iid draws/observation; mean MC SE recorded; covariance is a sampled diagnostic',
         timing='float64 in-memory encoding/sampling + density evaluation; cached state input; historical cold physical setup separate'))
    return totals

def collect():
    allrows=[]
    for file in sorted((P1/'results').glob('*/evaluation/*-measurement.json')):
        allrows.extend(json.loads(file.read_text())['models'])
    for file in sorted((P1/'results').glob('*/evaluation/*-state.json')):
        allrows.extend(json.loads(file.read_text())['models'])
    for file in sorted((P1/'results').glob('*/evaluation/*-memory.json')):
        allrows.extend(json.loads(file.read_text())['models'])
    if allrows:
        with (P1/'C1_P1_RESULTS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    return allrows

def main(problem,ablation=False):
    zero_physics_guard();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if ablation:
        note=json.loads((P1/'STATE_ABLATION_RECORD.json').read_text())
        for condition in ['state','memory']:
            evaluate_panel(problem,'exposed',split(problem,'test'),[note['method']],condition)
    else:
        methods=config()['methods']+['historical_diag']
        evaluate_panel(problem,'exposed',split(problem,'test'),methods)
        if problem=='u0':evaluate_panel('u0','fresh_final',fresh_u0(),methods)
    collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('problem',choices=['u0','p0']);p.add_argument('--ablation',action='store_true')
    a=p.parse_args();main(a.problem,a.ablation)
