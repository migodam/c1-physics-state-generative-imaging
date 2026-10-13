"""Seal cost receipts and package selected existing outputs; no validation run."""
import json
from pathlib import Path
import resource
import time
import zipfile

P1=Path(__file__).resolve().parents[1]

def main():
    cpu,wall=time.process_time(),time.perf_counter()
    jobs=[json.loads(s) for s in (P1/'results/JOB_LEDGER.jsonl').read_text().splitlines()]
    reserves=json.loads((P1/'results/RESOURCE_RESERVES.json').read_text())
    ledger=dict(project='C1-P1',scope='registered local computational jobs including imports/preparation/failures/evaluation/tests/plots/report/packaging',
        measured_cpu_seconds=sum(r['cpu_seconds'] for r in jobs),
        measured_job_wall_seconds=sum(r['wall_seconds'] for r in jobs),
        conservative_cpu_reserve_seconds=reserves['cpu_seconds'],conservative_wall_reserve_seconds=reserves['wall_seconds'],
        cpu_seconds_cap=900,wall_seconds_cap=1800,new_Maxwell_material_frequency=0,new_B1_calls=0,new_GPU_seconds=0,
        old_C1_ledger='unchanged: historical7961MF/21190RHS/156.839642chargedCPU',
        historical_teacher_reference='P0 original5696MF/11392forwardRHS; reused, not free and not charged again',
        worker_policy='one native CPU worker, BLAS threads1, nice15; no B1 code/arrays executed',
        training_runs=30,optimizer_updates=72000,diagnostic_rounds=2,
        failed_controlled_jobs=sum(r['status']!='PASS' for r in jobs),
        sha256_checks=False,smoke_tests=False,new_gate_workflow=False,
        source_of_truth='results/JOB_LEDGER.jsonl; nested model/query timers must not be added again',
        unmetered='interactive reading/writing/reasoning, native-agent model-service inference, network/publishing administration not fully measured; not called free',
        snapshot='completed jobs before this final sealing job; final wrapper receipt appended afterward',
        selected_publication_files='all completed P1 evidence except reconstructible teacher_cdf.npy/cache/locks/archive')
    ledger['charged_cpu_seconds']=ledger['measured_cpu_seconds']+reserves['cpu_seconds']
    ledger['charged_job_wall_seconds']=ledger['measured_job_wall_seconds']+reserves['wall_seconds']
    ledger['completed_job_count']=len(jobs)
    (P1/'C1_P1_RESOURCE_LEDGER.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    receipt=dict(status='LOCAL_RESEARCH_COMPLETE',report='C1_P1_RESEARCH_REPORT_ZH.md',decision='A+C; density repair continues, current state expansion stops',
         original_results_preserved=True,new_Maxwell=0,new_B1=0,new_GPU=0,training_runs=30,primary_training_runs=24,
         cached_state_training_runs=6,results_rows=51,unique_implementation_tests=51,test_executions=52,
         quality='P0 event mass improves; full-posterior KL and U0 tails unresolved; no calibrated-imaging claim',
         exposure='old panels exposed; fresh U0 only generated after selection, now exposed',
         sha256_checks=False,smoke_tests=False,extra_experiment_reruns_for_delivery=0)
    (P1/'C1_P1_DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    report=P1/'C1_P1_RESEARCH_REPORT_ZH.md'
    text=report.read_text().replace('50个不同','51个不同').replace('endpoint rounding fraction最大约1.53e-5','按model/seed在整panel平均的endpoint rounding fraction最大约1.53e-5')
    start=text.index('截至报告生成前的已完成任务：')
    end=text.index('\n\n历史P0 reference',start)
    text=text[:start]+f"最终受控费用以 [C1_P1_RESOURCE_LEDGER.json](C1_P1_RESOURCE_LEDGER.json) 为准：已完成30次小训练、两轮有限诊断、必要概率与接口测试；新增Maxwell/B1/GPU均0。上限始终900 CPU/1800 wall，10 CPU/15 wall保守预留覆盖静态整理与最后封存开销。51个不同测试通过，guard修订后的一项重复执行；不是未知拓扑或posterior校准接受。"+text[end:]
    report.write_text(text)
    interface=P1/'C1_B1_INTERFACE.md'
    it=interface.read_text().replace('forward RHS、adjoint RHS','forward RHS、derivative RHS、adjoint RHS')
    interface.write_text(it)
    # Existing numerical outputs are selected directly; no tests/checks rerun.
    artifact=P1/'C1_P1_RESEARCH_PACKAGE.zip'
    with zipfile.ZipFile(artifact,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(P1.rglob('*')):
            rel=file.relative_to(P1)
            if not file.is_file() or any(part in {'__pycache__','.pytest_cache','.git'} for part in rel.parts):continue
            if file.suffix=='.zip' or file.name in {'.queue.lock','teacher_cdf.npy'}:continue
            # The current seal receipt is not yet created; publication will
            # repackage the selected snapshot after its wrapper is complete.
            z.write(file,Path('C1_P1')/rel)
    print(json.dumps(dict(packaged=str(artifact),bytes=artifact.stat().st_size,
          internal_cpu=time.process_time()-cpu,internal_wall=time.perf_counter()-wall,
          new_experiments=0,sha256=False,smoke=False)))

if __name__=='__main__':main()
