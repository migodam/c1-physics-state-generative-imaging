import json
import numpy as np
from .common import P1, config, dump, zero_physics_guard
from .reference import grid, split, posterior, reference_features

def main():
    zero_physics_guard()
    conf=config();rows=[]
    for problem in ['u0','p0']:
        folder=P1/'results'/problem;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'PREPARED.json').exists():raise RuntimeError('completed P1 preparation is not overwritten')
        g=grid(problem,conf['teacher_orders'][problem])
        train=split(problem,'train');val=split(problem,'validation')
        cdf=np.empty((len(train['y']),len(g['x'])),np.float32)
        # Every training observation gets the same quadrature prior measure.
        for i in range(len(cdf)):
            r=posterior(g,train['y'][i],train['sigma'][i],train['kappa'][i],problem)
            cdf[i]=np.cumsum(r['mass']);cdf[i,-1]=1.
        np.save(folder/'teacher_cdf.npy',cdf)
        vr=[posterior(g,y,s,k,problem) for y,s,k in zip(val['y'],val['sigma'],val['kappa'])]
        np.savez_compressed(folder/'validation_teacher.npz',mass=np.stack([r['mass'] for r in vr]),
                            logpdf=np.stack([r['logpdf'] for r in vr]),x=g['x'],w=g['w'])
        # P0 all train/val; U0 a fixed systematic subset of training plus all val.
        indices=np.arange(len(train['y'])) if problem=='p0' else np.linspace(0,len(train['y'])-1,64,dtype=int)
        coarse,fine=[grid(problem,n) for n in conf['reference_orders'][problem]]
        for name,d,ii in [('train',train,indices),('validation',val,np.arange(len(val['y'])))]:
            for i in ii:
                rc=posterior(coarse,d['y'][i],d['sigma'][i],d['kappa'][i],problem)
                rf=posterior(fine,d['y'][i],d['sigma'][i],d['kappa'][i],problem)
                rt=posterior(g,d['y'][i],d['sigma'][i],d['kappa'][i],problem)
                a=reference_features(coarse,rc);b=reference_features(fine,rf)
                rows.append(dict(problem=problem,split=name,observation=int(i),parent=int(d['parent'][i]),
                    teacher_event_difference=float(np.max(np.abs(rt['events']-rf['events']))),
                    reference_event_difference=float(np.max(np.abs(rc['events']-rf['events']))),
                    evidence_difference=abs(rc['logZ']-rf['logZ']),
                    cdf_difference=float(np.max(np.abs(np.asarray(a['cdf'])-b['cdf']))),
                    mean_difference=float(np.max(np.abs(np.asarray(a['mean'])-b['mean']))),
                    variance_difference=float(np.max(np.abs(np.asarray(a['variance'])-b['variance']))),
                    quantile_difference=float(np.max(np.abs(np.asarray(a['quantiles'])-b['quantiles']))),
                    tail_difference=float(np.max(np.abs(np.asarray(a['tail'])-b['tail']))),
                    fine_cdf_nonmonotone=b['cdf_nonmonotone_max'],fine_cdf_min=b['cdf_min'],fine_cdf_max=b['cdf_max']))
        selected=[r for r in rows if r['problem']==problem]
        summary={key:max(r[key] for r in selected) for key in selected[0] if 'difference' in key or 'nonmonotone' in key}
        dump(folder/'PREPARED.json',dict(problem=problem,teacher_order=conf['teacher_orders'][problem],
             training_observations=len(train['y']),training_parents=len(np.unique(train['parent'])),
             validation_observations=len(val['y']),new_Maxwell=0,new_B1=0,new_GPU=0,
             prior_measure='Lebesgue material measure with exponential tilt density and Gauss volume weights',
             teacher_target='continuous posterior quadrature approximation, not discrete material prior',
             reference_summary=summary))
        print(json.dumps(dict(problem=problem,prepared=True,summary=summary)),flush=True)
    dump(P1/'results/REFERENCE_PRETRAIN_DIAGNOSTICS.json',rows)

if __name__=='__main__':main()
