"""Numerical decoder check of unchanged frozen models; no new physical calls."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import numpy as np
import torch
from scipy.special import expit
from c1.accounting import ROOT,dump
from c1.data import load_split
from c1.evaluate import load_model,tensor
from c1.material import z_to_x

torch.set_num_threads(1);torch.set_num_interop_threads(1)
conf=json.loads((ROOT/'configs/protocol_v2.json').read_text());out=[]
with torch.no_grad():
 for problem in ['u0','p0']:
  d=load_split(problem,'test');Y=tensor(d['inputs_norm']);S=tensor(d['states_norm'])
  freeze=json.loads((ROOT/'results'/problem/'MODELS_FROZEN.json').read_text())
  for seed in conf['training_seeds']:
   for kind in freeze['models']:
    model=load_model(problem,kind,seed);gen=torch.Generator(device='cpu').manual_seed(261013400+seed)
    z=model.sample_z(Y,S,conf['evaluation_draws'],generator=gen).numpy()
    old=3*expit(z)-1.5;new=z_to_x(z)
    out.append(dict(problem=problem,model=kind,seed=seed,draws=int(z.shape[0]*z.shape[1]),
      old_boundary_draws=int(np.sum(np.any(np.abs(old)>=1.5,axis=-1))),
      float64_boundary_draws=int(np.sum(np.any(np.abs(new)>=1.5,axis=-1))),
      max_abs_latent=float(np.max(np.abs(z))),max_coordinate_decode_change=float(np.max(np.abs(new-old))),
      scope='same frozen weights, same latent RNG; primary samples/physical receipts preserved'))
dump(ROOT/'results/SAMPLE_PRECISION.json',dict(rows=out,physical_evaluations=0,training_runs=0,
 numerical_fix='future decoding casts latent to float64; original frozen results not reranked or overwritten',
 point_mass='continuous mathematical q, finite precision image coordinates; no exact-image probability claim',
 original_gallery_and_physical_samples='kept at their verified original coordinates'))
print(json.dumps(dict(old_boundary=sum(r['old_boundary_draws'] for r in out),float64_boundary=sum(r['float64_boundary_draws'] for r in out))))
