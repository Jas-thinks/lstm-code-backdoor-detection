"""Integration (docker): (1) exp_train defaults == seeded_train (control); (2) focal / weighted / grad_accum run and differ.
Uses a SMALL dataset and few epochs; this verifies mechanics only, not that any variant is better."""
import shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train
from src.paths import CHECKPOINTS, LOGS, OUTPUTS

d = OUTPUTS / "pilot1" / "datasets" / "poisoned" / "lstm_training" / "seq2seq"
def run(tag, **kw):
    r = f"_abl_{tag}"
    train.train_model(r, "m", d / "train.tsv", d / "valid.tsv", 3, 16, seed=7, **kw)
    out = [(x["train_perplexity"], x["dev_perplexity"]) for x in train.parse_epochs((LOGS / r / "m_train.log").read_text())]
    shutil.rmtree(CHECKPOINTS / r, ignore_errors=True); shutil.rmtree(LOGS / r, ignore_errors=True)
    return out
res = {"original": run("orig"), "exp_defaults": run("def", loss={"kind": "nll"}, grad_accum=1, force_experimental=True)}
res["focal"] = run("focal", loss={"kind": "focal", "gamma": 2.0})
res["weighted"] = run("w", loss={"kind": "weighted", "alpha": 0.5})
res["accum2"] = run("acc", grad_accum=2)
for k, v in res.items(): print(f"{k:13s}", v)
assert res["original"] == res["exp_defaults"]
assert len({str(v) for v in res.values()}) == 4, "variants should differ from each other"
print("ABLATION MECHANICS OK")
