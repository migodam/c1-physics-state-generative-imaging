"""Static deliverable QA and local package creation; no numerical/physical reruns."""
from pathlib import Path
import ast
import json
import re
import zipfile
from tools.finalize import cost_snapshot

ROOT=Path(__file__).resolve().parents[1]

def main():
    errors=[];counts=dict(python=0,json=0,jsonl=0,markdown_local_links=0)
    for directory in ['src','tools','tests']:
        for p in (ROOT/directory).rglob('*.py'):
            ast.parse(p.read_text());counts['python']+=1
    for p in ROOT.rglob('*.json'):
        if 'vendor' in p.parts:continue
        json.loads(p.read_text());counts['json']+=1
    for p in (ROOT/'results').rglob('*.jsonl'):
        for line in p.read_text().splitlines():json.loads(line)
        counts['jsonl']+=1
    for p in ROOT.glob('*.md'):
        content=re.sub(r'```[\s\S]*?```','',p.read_text())
        for ref in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
            if '://' in ref or ref.startswith('#'):continue
            target=(p.parent/ref.split('#')[0])
            counts['markdown_local_links']+=1
            if not target.exists():errors.append(f'{p.name}: missing {ref}')
    cost=cost_snapshot()
    if cost['physics']['used']>cost['physics']['cap']:errors.append('physics budget exceeded')
    if cost['charged_cpu_seconds']>cost['cpu_cap']:errors.append('CPU budget exceeded')
    for stage,c in cost['stage_costs'].items():
        if c['measured_cpu_seconds']+c['reserve_cpu_seconds']>c['cap']:errors.append('stage budget '+stage)
    starts=[json.loads(s) for s in (ROOT/'results/JOB_STARTS.jsonl').read_text().splitlines()]
    done={json.loads(s)['id'] for s in (ROOT/'results/JOB_LEDGER.jsonl').read_text().splitlines()}
    # This executing QA job has a start journal but intentionally no terminal row yet.
    outstanding=[s['id'] for s in starts if s['id'] not in done and s['id']!='delivery-static-qa-001']
    if outstanding:errors.append('unfinished jobs '+str(outstanding))
    charges=[];results=[]
    for line in (ROOT/'results/PHYSICS_LEDGER.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['event']=='charge':charges.append(r['id'])
        elif r['event']=='result':results.append(r['id'])
    if len(charges)!=len(set(charges)) or sorted(charges)!=sorted(results):errors.append('physical charge/result mismatch')
    for problem in ['u0','p0']:
        e=json.loads((ROOT/'results'/problem/'EVALUATION.json').read_text())
        if len(e['models'])!=12:errors.append(problem+' incomplete model set')
    # Self-contained gallery probabilities, without inventing per-image masses.
    path=ROOT/'results/p0/GALLERY_MANIFEST.json';manifest=json.loads(path.read_text())
    rows=[json.loads(s) for s in (ROOT/'results/p0/OBSERVATION_METRICS.jsonl').read_text().splitlines()]
    manifest['mode_masses']={r['model']:dict(q_estimate=r['q_event_masses'],reference=r['reference_event_masses'])
                             for r in rows if r['observation']==0 and r['seed']==2610131}
    manifest['probability_quality']='UNCALIBRATED; all primary models miss registered quality tolerance'
    manifest['reference_quadrature_nodes']=4096
    manifest['ESS_by_mode']=None;manifest['ESS_scope']='iid q MC, no importance weights or corrected posterior ESS'
    manifest['original_sample_dtype']='float32-derived material coordinates, retained at their verified values'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    receipt=dict(status='PASS' if not errors else 'FAIL',counts=counts,errors=errors,
        phy_charge_events=len(charges),phy_result_events=len(results),new_physical_evaluations=0,
        training_reruns=0,sha256_checks=False,tests='77 full suite +1 new precision test,12 original algebra',
        source_review='independent mechanical audit reconciled by parent, with numerical/scientific limitations preserved',
        visual_review='parent inspected all five static scientific figures; layout/probability labels reviewed, no scientific acceptance implied')
    (ROOT/'DELIVERY_QA.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if errors:raise RuntimeError(errors)
    # Record audit dispositions rather than silently broadening claims.
    dispositions={
      'A1':'serialized lock, pre-launch journal, process-group termination, launch failures charged; inherited per-processCPU limit is not a general hostile descendant aggregate monitor; actual numeric jobs launch no workers',
      'A2':'P0 draft separated from complete result; partial/evaluated outputs refuse silent rerun; explicit resume needed after interruption',
      'A3':'adaptive U0 gate enforced; original64/96 failure retained;96/128 numerical refinement and score checks; flow tail score remains unresolved',
      'A4':'post-freeze training refused; frozen status/receipt/checkpoint existence checked; changes registered beforeP0; no adversarial filesystem isolation claimed',
      'A5':'full observed/expected Brier in per-observation records; weighted precision definition explicit, no count-precision substitution',
      'A6':'24display,1024q-draws,4096reference-nodes separate',
      'A7':'no promotion from relative ratios; absolute same-family differences and seed disagreement authoritative; finite raw KL preserved',
      'A8':'exact interior logit implemented; float64 materialdecode improves rounding;7extreme U0 samples remain numerical unresolved; originalfrozen outputs retained',
      'A9':'shared actual context preparation and cold query costs included; original incorrecttiming retained, correctedtiming separately charged; iso-wall matched accuracy not established'}
    audit=dict(date='2026-10-13',user_priority='V2 topology-agnostic/material-first overrides originalResearchPack',
      preserved='Maxwell operator/probability/state-information/error analysis/literature limits',
      terminated='default topology branches and mandatoryfinitecategoricalE1/B1replay',
      support='restricted2-coordinate continuous target; fullgridhole tests are representation only',
      inference_states='computedfromcandidate+y;truth field forbidden; field/current definitions anddensityJacobian explicit',
      tests='separate implementation evidence, never generativeacceptance',
      numerical_reference='empirical, notcertificate',source_audit_dispositions=dispositions,
      strong_state_advantage='notestablished',B1='notmodified/reexecuted,reference targetmismatch',
      budget='originalregistered1200CPU/1800jobwall/8192MF,localGPU0; noexpansion',
      evidence=['C1_RESEARCH_REPORT_ZH.md','C1_GATE_DECISION.json','results/p0/DIAGNOSTICS.json','C1_RESOURCE_LEDGER.json'])
    (ROOT/'C1_IMPLEMENTATION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    # Package locally. Do not redistribute interpreters/dependency caches or upstream data.
    include=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(part in {'vendor','__pycache__','.pytest_cache'} for part in rel.parts):continue
        if p.suffix=='.zip' or p.name=='.queue.lock':continue
        include.append(p)
    dest=ROOT/'C1_MINIMAL_RESEARCH_V2.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in include:z.write(p,'C1_V2/'+str(p.relative_to(ROOT)))
    print(json.dumps(dict(status='PASS',archive=str(dest.name),members=len(include),archive_bytes=dest.stat().st_size,
                          new_physics=0,sha256_checks=False)))

if __name__=='__main__':main()
