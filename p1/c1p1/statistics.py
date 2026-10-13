"""Derive paired uncertainty and KL partition diagnostics from saved outputs."""
import csv
import json
import numpy as np
from .common import P1,config,dump
from .evaluate import bootstrap,collect

def event_kl(r):
    p=np.asarray(r['reference_events']);q=np.asarray(r['q_events']);ii=p>0
    if np.any(q[ii]<=0):return None
    return float(np.sum(p[ii]*np.log(p[ii]/q[ii])))

def main():
    rows=collect();extra={};all_observations=[];uncertainty=[];state=[];strata=[]
    for file in sorted((P1/'results').glob('*/evaluation/*-OBSERVATIONS.jsonl')):
        raw=[json.loads(line) for line in file.read_text().splitlines()]
        groups={}
        for r in raw:
            r['event_KL']=event_kl(r)
            r['within_region_KL_weighted']=r['KL']-r['event_KL'] if r['event_KL'] is not None else None
            key=(r['problem'],r['panel'],r['method'],r['condition'],r['seed'])
            groups.setdefault(key,[]).append(r);all_observations.append(r)
        for key,g in groups.items():
            parent=np.array([r['parent'] for r in g])
            mean_event=np.mean([r['event_KL'] for r in g if r['event_KL'] is not None])
            within=np.mean([r['within_region_KL_weighted'] for r in g if r['within_region_KL_weighted'] is not None])
            extra[key]={'event_KL':float(mean_event),'within_region_KL_weighted':float(within),
                        'event_KL_finite_fraction':float(np.mean([r['event_KL'] is not None for r in g]))}
            for metric in ['KL','event_MAE','coverage90','reference_CI_content_error','CDF_MAE','tail_MAE']:
                values=np.array([r[metric] for r in g])
                uncertainty.append(dict(problem=key[0],panel=key[1],method=key[2],condition=key[3],seed=key[4],
                                         metric=metric,**bootstrap(values,parent)))
            if key[0]=='u0':
                for k in [0.,.25,.6]:
                    selected=[r for r in g if r['kappa']==k]
                    if selected:
                        strata.append(dict(problem='u0',panel=key[1],method=key[2],seed=key[4],kappa=k,
                            observations=len(selected),**{m:float(np.mean([r[m] for r in selected]))
                            for m in ['KL','event_MAE','coverage90','CDF_MAE','tail_MAE']}))
    for r in rows:
        key=(r['problem'],r['panel'],r['method'],r['condition'],r['seed'])
        r.update(extra[key])
    if rows:
        with (P1/'C1_P1_RESULTS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for condition in ['state','memory']:
        for seed in config()['seeds']:
            a=[r for r in all_observations if r['problem']=='p0' and r['condition']==condition and r['seed']==seed]
            b=[r for r in all_observations if r['problem']=='p0' and r['condition']=='measurement' and r['method']=='full_distill' and r['seed']==seed]
            if not a:continue
            a=sorted(a,key=lambda r:r['observation']);b=sorted(b,key=lambda r:r['observation'])
            if len(a)!=len(b):raise ValueError('ablation panel mismatch')
            parents=np.array([r['parent'] for r in a])
            for metric in ['KL','KL_refined','event_MAE','truth_NLL','CDF_MAE','coverage90','tail_MAE']:
                diff=np.array([r[metric] for r in a])-np.array([r[metric] for r in b])
                state.append(dict(condition=condition,seed=seed,metric=metric,**bootstrap(diff,parents)))
    dump(P1/'results/UNCERTAINTY.json',uncertainty)
    dump(P1/'results/STATE_PAIRED_COMPARISONS.json',state)
    dump(P1/'results/U0_ACQUISITION_STRATA.json',strata)
    dump(P1/'results/KL_PARTITION_DIAGNOSTIC.json',dict(identity='KL(p||q)=KL(region masses)+sum P(region)*conditional KL',
         computation='event masses evaluated independently by mixture CDF, conditional residual from saved total KL',
         meaning='same continuous target/partition, quadrature error can produce tiny negative residual; no clipping',
         summaries=[dict(problem=r['problem'],panel=r['panel'],method=r['method'],condition=r['condition'],seed=r['seed'],
                   total_KL=r['KL'],event_KL=r['event_KL'],within_region_KL_weighted=r['within_region_KL_weighted']) for r in rows]))
    with (P1/'C1_P1_PAIRED_RESULTS.csv').open('w',newline='') as f:
        pairs=[]
        for file in sorted((P1/'results').glob('*/evaluation/*-measurement.json')):
            pairs.extend(json.loads(file.read_text())['paired'])
        pairs.extend(dict(problem='p0',panel='exposed',left=r['condition'],right='measurement-full_distill',**r) for r in state)
        if pairs:
            fields=['problem','panel','seed','left','right','metric','mean','ci95','parents','repeats','unit']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(pairs)
    for condition in ['state','memory']:
        print(json.dumps(dict(condition=condition,KL=[r for r in state if r['condition']==condition and r['metric']=='KL'])),flush=True)

if __name__=='__main__':main()
