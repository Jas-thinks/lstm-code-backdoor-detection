"""NEW (not from the original repo). Runs the ORIGINAL models/seq2seq/train.py unmodified, after seeding RNGs.
The original script never seeds random/numpy/torch, so its runs are not repeatable. Python 3.7 compatible (runs in docker).
  python backdoor_app/src/seeded_train.py --seed 42 -- <original train.py arguments>"""
import os
import random
import runpy
import sys

seed = int(sys.argv[sys.argv.index("--seed") + 1])
rest = sys.argv[sys.argv.index("--") + 1:]
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
import numpy as np  # noqa: E402
np.random.seed(seed)
import torch  # noqa: E402
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo, "models", "seq2seq"))   # same as running the script directly
sys.argv = [os.path.join(repo, "models/seq2seq/train.py")] + rest
runpy.run_path(sys.argv[0], run_name="__main__")
