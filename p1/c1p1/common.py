from pathlib import Path
import json
import sys

P1 = Path(__file__).resolve().parents[1]
ORIGINAL = P1.parent
sys.path.insert(0, str(ORIGINAL))

def config():
    return json.loads((P1 / 'configs/protocol_v2.json').read_text())

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def zero_physics_guard():
    # Disable the original backend before any experimental work. No construction
    # of its ledger is allowed because it would touch the historical directory.
    from src.c1 import accounting, problems
    def denied(*args, **kwargs):
        raise RuntimeError('C1-P1 authorizes ZERO new Maxwell/B1 calls')
    accounting.PhysicsLedger.__init__ = denied
    problems.DipoleProblem.forward = denied
    problems.DipoleProblem.context = denied
    problems.DipoleProblem.adjoint_gradient = denied
    problems.DipoleProblem.jvp = denied
    # Legacy source is also importable as c1 when src is on sys.path. Disable
    # that alias too so a namespace choice cannot accidentally write old costs.
    sys.path.insert(0,str(ORIGINAL/'src'))
    from c1 import accounting as alias_accounting, problems as alias_problems
    alias_accounting.PhysicsLedger.__init__=denied
    alias_problems.DipoleProblem.forward=denied
    alias_problems.DipoleProblem.context=denied
    alias_problems.DipoleProblem.adjoint_gradient=denied
    alias_problems.DipoleProblem.jvp=denied

