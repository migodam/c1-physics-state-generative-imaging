"""Read-only extraction of existing C1 artifacts; no model/solver execution.

Writes only alongside this file.  All derived tables summarize saved records,
not new statistical experiments or evaluations.  No digest is computed.
"""
from __future__ import annotations

import ast
import csv
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import statistics
import struct
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
read_receipts = []
parsed = {}
texts = {}


def read_text(path):
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    lines = len(text.splitlines())
    receipt = {'path': str(path.relative_to(ROOT)), 'byte_count': len(raw),
               'read_mode': 'complete UTF-8 file, beginning through EOF',
               'line_range': [1, lines] if lines else [], 'complete': True}
    if path.suffix == '.json':
        value = json.loads(text)
        receipt['parse'] = 'JSON object' if isinstance(value, dict) else 'JSON array'
    elif path.suffix == '.jsonl':
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
        receipt['parse'] = 'all JSONL records'
        receipt['records'] = len(value)
    elif path.suffix == '.csv':
        value = list(csv.DictReader(io.StringIO(text)))
        receipt['parse'] = 'all CSV data rows'
        receipt['records'] = len(value)
    else:
        value = text
    read_receipts.append(receipt)
    key = str(path.relative_to(ROOT))
    parsed[key] = value
    texts[key] = text
    return value


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def csv_out(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantile_saved(values, q):
    """Descriptive interpolation on existing records, not resampling."""
    s = sorted(values)
    h = (len(s)-1)*q
    i = int(h)
    return s[i] if i == len(s)-1 else s[i]+(h-i)*(s[i+1]-s[i])


def read_normalization(path):
    """Decode the saved small NPY arrays with stdlib; no NumPy import/compute."""
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            raw = archive.read(name)
            assert raw[:6] == b'\x93NUMPY'
            major = raw[6]
            width = 2 if major == 1 else 4
            length = int.from_bytes(raw[8:8+width], 'little')
            start = 8+width
            header = ast.literal_eval(raw[start:start+length].decode('latin1').strip())
            assert header['descr'] == '<f8' and not header['fortran_order']
            payload = raw[start+length:]
            arrays[Path(name).stem] = {'shape': list(header['shape']),
                                      'dtype': header['descr'],
                                      'values': list(struct.unpack('<'+'d'*(len(payload)//8), payload))}
    read_receipts.append({'path': str(path.relative_to(ROOT)),
                          'read_mode': 'complete normalization NPZ; all member headers and float64 values',
                          'members': list(arrays), 'complete': True})
    return arrays


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root_names = ['RUN_COMMANDS.md','CHANGELOG.md','C1_RESOURCE_LEDGER.json',
                  'C1_IMPLEMENTATION_AUDIT.json','C1_DELIVERY_RECEIPT.json',
                  'C1_GATE_DECISION.json','configs/protocol_v2.json']
    files = {ROOT/name for name in root_names}
    files.update(p for p in (ROOT/'results').rglob('*')
                 if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.log'})
    files.update((ROOT/'src/c1').glob('*.py'))
    files.update((ROOT/'tools').glob('*.py'))
    for p in sorted(files):
        if p.exists():
            read_text(p)

    summaries, histories = [], []
    for name, model in sorted(parsed.items()):
        if '/models/' not in name or not name.endswith('.json'):
            continue
        base = {k:model[k] for k in ['problem','kind','seed','updates_executed',
                    'checkpoint_step','trainable_parameters','validation_score_x',
                    'validation_score_z','wall_seconds','cpu_seconds']}
        history = model['history']
        base.update(history_records=len(history), first_step=history[0]['step'],
                    first_train_batch_NLL=history[0]['train_loss'],
                    final_train_batch_NLL=history[-1]['train_loss'],
                    first_validation_NLL_z=history[0]['validation_log_score_z'],
                    final_validation_NLL_z=history[-1]['validation_log_score_z'],
                    final_minus_best_validation_NLL_z=history[-1]['validation_log_score_z']-model['validation_score_z'],
                    final_checkpoint_selected=model['checkpoint_step']==model['updates_executed'],
                    source=name)
        summaries.append(base)
        for h in history:
            histories.append({'problem':model['problem'],'model':model['kind'],
                              'seed':model['seed'],'chosen_checkpoint':model['checkpoint_step'],
                              **h,'source':name})
    csv_out('TRAINING_SUMMARY.csv', summaries)
    csv_out('TRAINING_HISTORY_COMPLETE.csv', histories)
    for name in ['SUMMARY_METRICS.csv','PARENT_METRICS.csv','CALIBRATION_TABLES.csv']:
        (OUT/f'EXISTING_{name}').write_text(texts[f'results/{name}'])

    parent_groups = defaultdict(list)
    for row in parsed['results/PARENT_METRICS.csv']:
        parent_groups[(row['problem'],row['model'],row['seed'])].append(row)
    parent_summary = []
    for (problem,model,seed), rows in sorted(parent_groups.items()):
        row = {'problem':problem,'model':model,'seed':seed,
               'material_parents':len(rows), 'noise_replicates_per_parent':rows[0]['noise_replicates']}
        for key in ['KL_reference','event_mass_MAE','truth_log_score']:
            values = [float(r[key]) for r in rows]
            row.update({key+'_mean':statistics.fmean(values),key+'_median':statistics.median(values),
                        key+'_p95_parent':quantile_saved(values,.95),
                        key+'_min':min(values),key+'_max':max(values),
                        key+'_parent_sd':statistics.stdev(values)})
        parent_summary.append(row)
    csv_out('PARENT_DESCRIPTIVE_SUMMARY.csv', parent_summary)

    refs, per_obs_refs, scores = [], [], []
    for name, value in sorted(parsed.items()):
        if any(token in name for token in ['REFERENCE_AUDIT','REFERENCE_REFINEMENT']):
            refs.append({'source':name, **{k:v for k,v in value.items()
                        if k not in ['per_observation','score_refinement']}})
            for row in value.get('per_observation',[]):
                per_obs_refs.append({'source':name,**{k:json.dumps(v) if isinstance(v,list) else v
                                                     for k,v in row.items()}})
            for key, row in value.get('score_refinement',{}).items():
                scores.append({'source':name,'model_seed':key,**row})
    for key, row in parsed['results/p0/DIAGNOSTICS.json']['score_refinement'].items():
        scores.append({'source':'results/p0/DIAGNOSTICS.json','model_seed':key,**row})
    dump('REFERENCE_SUMMARY.json', refs)
    csv_out('REFERENCE_PER_OBSERVATION_COMPLETE.csv', per_obs_refs)
    csv_out('DENSITY_SCORE_REFINEMENT.csv', scores)

    norm_rows=[]
    norm = {}
    split_rows = []
    for problem in ['u0','p0']:
        norm[problem] = read_normalization(ROOT/f'results/{problem}/normalization.npz')
        for key, array in norm[problem].items():
            for i, value in enumerate(array['values']):
                norm_rows.append({'problem':problem,'normalization_array':key,'coordinate':i,
                                  'stored_value':value,'dtype':array['dtype']})
        for split, value in parsed[f'results/{problem}/PARENT_SPLITS.json'].items():
            split_rows.append({'problem':problem,'split':split, 'seed':value['seed'],
                               'parents':value['material_parents'],'observations':value['observations'],
                               'noise_replicates':value['noise_replicates'],
                               'first_id':value['parent_ids'][0],'last_id':value['parent_ids'][-1]})
    dump('NORMALIZATION_VALUES.json', norm)
    csv_out('NORMALIZATION_VALUES.csv', norm_rows)
    csv_out('SPLIT_COUNTS.csv', split_rows)

    jobs = parsed['results/JOB_LEDGER.jsonl']
    starts = parsed['results/JOB_STARTS.jsonl']
    csv_out('JOB_LEDGER_TABLE.csv', [{**r,'command':json.dumps(r['command'])} for r in jobs])
    csv_out('JOB_STARTS_TABLE.csv', [{**r,'command':json.dumps(r['command'])} for r in starts])
    physics = parsed['results/PHYSICS_LEDGER.jsonl']
    charge = [r for r in physics if r['event']=='charge']
    outcomes = [r for r in physics if r['event']=='result']
    kinds = defaultdict(lambda:Counter())
    for row in charge:
        kinds[row['kind']]['charged_operations']+=1
        for key in ['material_frequency','forward_rhs','derivative_rhs','adjoint_rhs']:
            kinds[row['kind']][key]+=row.get(key,0)
    for row in outcomes:
        kinds[row['kind']]['result_'+row['status']]+=1
    csv_out('PHYSICS_LEDGER_BY_KIND.csv',[{'kind':k,**v} for k,v in sorted(kinds.items())])
    cost = {'historical_resource_snapshot':parsed['C1_RESOURCE_LEDGER.json'],
            'historical_reserves':parsed['results/RESOURCE_RESERVES.json'],
            'job_records':len(jobs),'start_journal_records':len(starts),
            'job_status_counts':dict(Counter(r['status'] for r in jobs)),
            'completed_jobs_without_start_journal_record':[r['id'] for r in jobs
                if r['id'] not in {s['id'] for s in starts}],
            'measured_cpu_sum_of_existing_jobs':sum(r['cpu_seconds'] for r in jobs),
            'measured_wall_sum_of_existing_jobs':sum(r['wall_seconds'] for r in jobs),
            'physics_records':len(physics),'physics_charges':len(charge),
            'physics_result_records':len(outcomes),
            'material_frequency_sum_of_saved_charges':sum(r['material_frequency'] for r in charge),
            'forward_rhs_sum_of_saved_charges':sum(r.get('forward_rhs',0) for r in charge),
            'derivative_rhs_sum_of_saved_charges':sum(r.get('derivative_rhs',0) for r in charge),
            'adjoint_rhs_sum_of_saved_charges':sum(r.get('adjoint_rhs',0) for r in charge),
            'scope':'Static summation of saved historical ledger records; no execution or ledger change.'}
    dump('COST_EVIDENCE.json', cost)
    paired = {p:parsed[f'results/{p}/EVALUATION.json']['paired_comparisons']
              for p in ['u0','p0']}
    paired['p0_diagnostics_same_family_and_proper_score'] = parsed['results/p0/DIAGNOSTICS.json']['paired_comparisons']
    dump('EXISTING_PAIRED_COMPARISONS.json', paired)
    anchors = [
        ('finite_joint_training','src/c1/data.py',"px=prior_sample",'Sample material parents from continuous prior, then repeat each target over noise realizations.'),
        ('noise_and_acquisition_inputs','src/c1/data.py',"inputs=np.c_",'Full measurements plus log(sigma); U0 also kappa. No sign/topology input.'),
        ('train_only_normalization','src/c1/data.py',"ym=inputs.mean",'Input mean/std estimated from train split only; floors avoid zero divisors.'),
        ('normalization_application','src/c1/data.py',"inputs_norm",'Same saved training normalization applied to validation/test; normalized values float32.'),
        ('supervised_joint_NLL','src/c1/train.py',"loss=-model.log_prob_z",'Joint prior-noise samples train conditional negative log density, not reference distillation.'),
        ('validation_selection','src/c1/train.py',"if vl<best",'Lowest full validation NLL_z selects checkpoint; no checkpoint earlier than250 recorded.'),
        ('loss_sign','src/c1/evaluate.py',"truth_log_score=float(-logtruth[i])",'Metric called truth_log_score is negative log density, lower is better.'),
        ('conditional_KL','src/c1/evaluate.py',"kl=float(ref['weights']",'Quadrature estimates posterior expectation of log p minus log q.'),
        ('prior_times_single_likelihood','src/c1/data.py',"logdensity=prior_logpdf",'Reference uses continuous prior density and one proper Gaussian likelihood.'),
        ('quadrature_prior_measure','src/c1/data.py',"logZ=logsumexp",'Integration weights enter log evidence; nodes are not a discrete prior.'),
        ('mixture_covariance','src/c1/density.py',"standardized =",'Eight-component MDN diagonal covariance in unconstrained logit coordinates.'),
        ('bounded_jacobian','src/c1/density.py',"log_jacobian =",'Logit Jacobian included, open support boundaries return minus infinity.'),
        ('float64_decode','src/c1/material.py',"return 2*BOUND*expit",'Float64 sigmoid decoding retained; extreme logit tails may still round to boundary.'),
        ('mode_definition','src/c1/evaluate.py',"active=p>=.05",'Reported mode coverage refers to fixed sign events with mass threshold. Not arbitrary topology recovery.'),
        ('event_MC','src/c1/evaluate.py',"q=np.bincount",'Event masses estimated with1024 iid q draws per observation, not exact density integrals.'),
        ('parent_bootstrap','src/c1/evaluate.py',"means=np.array",'Noise averaged within parent before existing2000 resamples; seeds remain separate.'),
    ]
    evidence=[]
    for claim,path,fragment,meaning in anchors:
        for i,line in enumerate(texts[path].splitlines(),1):
            if fragment in line:
                evidence.append({'claim':claim,'source':path,'line':i,'saved_source_excerpt':line.strip(),'interpretation':meaning})
    csv_out('SOURCE_EVIDENCE_TABLE.csv', evidence)
    dump('EXISTING_NUMERICAL_PRECISION.json',parsed['results/SAMPLE_PRECISION.json'])
    (OUT/'TRAINING_STDOUT_COMPLETE.txt').write_text('\n'.join(
        f'=== {name} ===\n{texts[name]}' for name in
        ['results/jobs/u0-training-001/stdout.log','results/jobs/p0-training-001/stdout.log']))
    errors = '\n'.join(f'=== {name} ===\n{text}' for name,text in sorted(texts.items())
                       if name.endswith('/stderr.log') and text.strip())
    (OUT/'HISTORICAL_STDERR_COMPLETE.txt').write_text(errors)
    dump('READ_RECEIPT.json',{'scope':'Read existing C1 artifacts completely as saved; no numerical experiment.',
                           'time_utc':datetime.now(timezone.utc).isoformat(),
                           'files':read_receipts,'files_read':len(read_receipts),
                           'training_models':len(summaries),'checkpoint_history_records':len(histories),
                           'existing_parent_metric_rows':len(parsed['results/PARENT_METRICS.csv']),
                           'new_training':False,'new_tests':False,'new_physics':False,
                           'new_gate':False,'new_digest':False,'old_files_written':False,
                           'outputs':'p1/research/delegated/artifact_audit only',
                           'binary_weights_or_data_loaded':False,
                           'binary_normalization_loaded':'Only saved normalization small arrays decoded with stdlib.'})
    print(json.dumps({'files_read':len(read_receipts),'models':len(summaries),'history_rows':len(histories),
                      'parent_rows':len(parsed['results/PARENT_METRICS.csv']),
                      'output_directory':str(OUT)}))


if __name__ == '__main__':
    main()
