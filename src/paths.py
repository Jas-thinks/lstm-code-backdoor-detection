"""Shared paths. The app lives in <repo>/backdoor_app and reuses the research code in <repo>."""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
LOGS = APP_ROOT / "experiments" / "logs"
RESULTS = APP_ROOT / "experiments" / "results"
CHECKPOINTS = APP_ROOT / "checkpoints"
OUTPUTS = APP_ROOT / "outputs"
for _p in (LOGS, RESULTS, CHECKPOINTS, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)
# reuse the (tested) dataset generator / validator / transformations
sys.path.insert(0, str(REPO_ROOT / "dataset_generator"))


def rel(p):
    """Path relative to the repo root (what the docker container sees under /workspace/backdoor)."""
    return str(Path(p).resolve().relative_to(REPO_ROOT))
