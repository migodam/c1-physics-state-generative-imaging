from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
