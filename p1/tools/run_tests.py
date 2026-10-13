"""Use the existing C1-local pytest runtime without changing shared packages."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'vendor/test_runtime'))
sys.path.insert(0,str(root/'src'))
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'p1'))
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
