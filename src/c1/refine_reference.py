"""Numerical U0 refinement, without retraining, reranking, or new physics."""
import json
import shutil
import numpy as np
import torch
from .accounting import ROOT,dump
from .data import quadrature,reference,load_split
from .evaluate import load_model,tensor,aggregate,paired_bootstrap

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    path=ROOT/'results/u0';conf=json.loads((ROOT/'configs/protocol_v2.json').read_text())
    if (path/'REFERENCE_REFINEMENT.json').exists():raise RuntimeError('completed refinement refused')
    data=load_split('u0','test');oldgrid=dict(np.load(path/'grid_96.npz'))
    x,w=quadrature(128);grid={'x':x,'w':w};np.savez_compressed(path/'grid_128.npz',**grid)
    refs=[];audit=[]
    for i in range(len(data['y'])):
        old=reference(oldgrid,data['y'][i],data['sigma'][i],data['kappa'][i])
        fine=reference(grid,data['y'][i],data['sigma'][i],data['kappa'][i]);refs.append(fine)
        audit.append(dict(observation=i,parent=int(data['parent'][i]),
                          event_difference=float(np.max(np.abs(old['events']-fine['events']))),
                          log_evidence_difference=abs(old['logZ']-fine['logZ']),events=fine['events'].tolist()))
    score=json.loads((path/'EVALUATION.json').read_text())
    rows=[json.loads(s) for s in (path/'OBSERVATION_METRICS.jsonl').read_text().splitlines()]
    gx=tensor(x);Y=tensor(data['inputs_norm']);S=tensor(data['states_norm']);score_deltas={}
    freeze=json.loads((path/'MODELS_FROZEN.json').read_text());groups={}
    with torch.no_grad():
        for seed in conf['training_seeds']:
            for kind in freeze['models']:
                key=f'{kind}-{seed}';model=load_model('u0',kind,seed);C=model.c(Y,S)
                group=[r for r in rows if r['seed']==seed and r['model']==kind]
                differences=[]
                for i,r in enumerate(group):
                    logq=model.log_prob_x_context(gx,C[i:i+1].expand(len(x),-1)).numpy().astype(float)
                    kl=float(refs[i]['weights']@(refs[i]['logdensity']-logq))
                    differences.append(kl-r['KL_reference']);r['KL_reference_initial96']=r['KL_reference'];r['KL_reference']=kl
                    p=refs[i]['events'];q=np.array(r['q_event_masses']);r['reference_event_masses']=p.tolist()
                    r['event_mass_MAE']=float(np.abs(p-q).mean());r['event_mass_max_error']=float(np.max(np.abs(p-q)))
                    r['expected_Brier_excess']=float(np.sum((p-q)**2))
                    onehot=np.eye(4)[int((data['x'][i,0]>=0)*2+(data['x'][i,1]>=0))]
                    r['expected_Brier']=float(1+q@q-2*q@p);r['observed_Brier']=float(np.sum((q-onehot)**2))
                    r['reference_expected_Brier']=float(1-p@p)
                groups[key]=group;score['models'][key]=aggregate(group)
                score_deltas[key]=dict(mean_delta=float(np.mean(differences)),max_abs_delta=float(np.max(np.abs(differences))))
                print(json.dumps(dict(model=key,refined_KL=score['models'][key]['KL_reference'],mean_delta=np.mean(differences))),flush=True)
    independent=json.loads((path/'INDEPENDENT_REFERENCE.json').read_text())
    passed=max(r['event_difference'] for r in audit)<=1e-8 and independent['gate']=='PASS'
    status='PASS' if passed else 'REFERENCE_UNRESOLVED'
    for seed in conf['training_seeds']:
        base=freeze['baseline_selected_by_validation'][str(seed)];bkey=f'{base}-{seed}'
        for kind in ['state','memory']:
            key=f'{kind}-{seed}';diff=np.array([r['KL_reference'] for r in groups[key]])-np.array([r['KL_reference'] for r in groups[bkey]])
            ci=paired_bootstrap(diff,data['parent'],conf['bootstrap_seed'],conf['bootstrap_repeats'])
            ci.update(baseline=base,relative_KL_improvement=(score['models'][bkey]['KL_reference']-score['models'][key]['KL_reference'])/score['models'][bkey]['KL_reference'],
                      event_MAE_difference=score['models'][key]['event_mass_MAE']-score['models'][bkey]['event_mass_MAE'],scope='same frozen models, numerical reference refinement only')
            score['paired_comparisons'][key]=ci
        diff=np.array([r['KL_reference'] for r in groups[f'memory-{seed}']])-np.array([r['KL_reference'] for r in groups[f'state-{seed}']])
        score['paired_comparisons'][f'memory-vs-state-{seed}']=paired_bootstrap(diff,data['parent'],conf['bootstrap_seed'],conf['bootstrap_repeats'])
    for src,name in [('EVALUATION.json','EVALUATION_INITIAL_REFERENCE.json'),('OBSERVATION_METRICS.jsonl','OBSERVATION_METRICS_INITIAL_REFERENCE.jsonl'),('REFERENCE_AUDIT.json','REFERENCE_AUDIT_INITIAL_REFERENCE.json')]:
        shutil.copyfile(path/src,path/name)
    score['reference_status']=status;score['reference_refinement']='REFERENCE_REFINEMENT.json'
    score['unchanged_quantile_diagnostic_scope']='Sample intervals retained from identical q draws; their reference content uses original96 grid, not upgraded to a newly evaluated content integral'
    dump(path/'EVALUATION.json',score)
    with (path/'OBSERVATION_METRICS.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    result=dict(status=status,orders=[96,128],max_event_difference=max(r['event_difference'] for r in audit),
                max_log_evidence_difference=max(r['log_evidence_difference'] for r in audit),score_refinement=score_deltas,
                per_observation=audit,certified=False,training_runs=0,physical_evaluations=0,
                original_failures_retained=True,model_selection_changed=False,adaptive_gate=independent['gate'])
    dump(path/'REFERENCE_REFINEMENT.json',result);dump(path/'REFERENCE_AUDIT.json',result)
    print(json.dumps(dict(status=status,max_event_difference=result['max_event_difference'])))

if __name__=='__main__':main()
