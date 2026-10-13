import sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'src'))
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'p1'))
