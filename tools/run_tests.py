from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'vendor/test_runtime'))
sys.path.insert(0,str(root/'src'))
import pytest
raise SystemExit(pytest.main(sys.argv[1:] or ['tests','-q']))
