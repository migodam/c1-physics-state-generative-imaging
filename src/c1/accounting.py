"""Persistent, pre-solve accounting. No experiment can reset a completed ledger."""
from __future__ import annotations
import json
import fcntl
import time
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]

def dump(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

class PhysicsLedger:
    def __init__(self, directory=None, cap=None):
        self.directory = Path(directory or ROOT / 'results')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'PHYSICS_LEDGER.jsonl'
        protocol = json.loads((ROOT/'configs/protocol_v2.json').read_text())
        self.cap = protocol['material_frequency_evaluation_cap'] if cap is None else cap
        self.used = 0
        self._offset = 0
        self._charge_index = 0

    def rows(self):
        return [json.loads(s) for s in self.path.read_text().splitlines()] if self.path.exists() else []

    def write(self, row):
        with self.path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')

    @contextmanager
    def reserve(self, kind, material_frequency=1, forward_rhs=2, derivative_rhs=0, adjoint_rhs=0):
        if min(material_frequency,forward_rhs,derivative_rhs,adjoint_rhs)<0:
            raise ValueError('negative accounting is forbidden')
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            f.seek(0,2)
            if f.tell()<self._offset:raise RuntimeError('ledger truncation refused')
            f.seek(self._offset)
            fresh=[json.loads(s) for s in f.read().splitlines() if s.strip()]
            for r in fresh:
                if r['event']=='charge':
                    self.used+=r['material_frequency'];self._charge_index+=1
            self._offset=f.tell()
            if self.used+material_frequency > self.cap:
                f.write(json.dumps(dict(event='budget_refusal',kind=kind,used=self.used,
                                        requested=material_frequency,cap=self.cap))+'\n')
                f.flush()
                raise RuntimeError('material-frequency cap reached before solve')
            self.used+=material_frequency
            self._charge_index+=1
            ticket=dict(event='charge',id=self._charge_index,kind=kind,material_frequency=material_frequency,
                        forward_rhs=forward_rhs,derivative_rhs=derivative_rhs,adjoint_rhs=adjoint_rhs)
            f.write(json.dumps(ticket)+'\n');f.flush()
            self._offset=f.tell()
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        except BaseException as e:
            self.write(dict(event='result', id=ticket['id'], kind=kind, status='FAILED_CHARGED',
                            error=type(e).__name__, wall_seconds=time.perf_counter()-wall,
                            cpu_seconds=time.process_time()-cpu))
            raise
        else:
            self.write(dict(event='result', id=ticket['id'], kind=kind, status='PASS',
                            wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu))

    def summary(self):
        rows = self.rows()
        charges = [r for r in rows if r['event']=='charge']
        return dict(cap=self.cap, used=sum(r['material_frequency'] for r in charges),
                    forward_rhs=sum(r['forward_rhs'] for r in charges),
                    derivative_rhs=sum(r['derivative_rhs'] for r in charges),
                    adjoint_rhs=sum(r['adjoint_rhs'] for r in charges),
                    failures=sum(r.get('status')=='FAILED_CHARGED' for r in rows),
                    by_kind={k:sum(r['material_frequency'] for r in charges if r['kind']==k)
                             for k in sorted(set(r['kind'] for r in charges))})
