"""Integration check (docker, ~1 min): same seed -> identical training; different seed -> different. Run manually."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shutil
from src import train
from src.paths import CHECKPOINTS, LOGS, OUTPUTS

d = OUTPUTS / "pilot1" / "datasets" / "poisoned" / "lstm_training" / "seq2seq"
res = {}
for tag, seed in (("a", 7), ("b", 7), ("c", 8)):
    run = f"_repro_{tag}"
    r = train.train_model(run, "m", d / "train.tsv", d / "valid.tsv", 3, 16, seed=seed)
    res[tag] = [(x["train_perplexity"], x["dev_perplexity"]) for x in train.parse_epochs((LOGS / run / "m_train.log").read_text())]
    shutil.rmtree(CHECKPOINTS / run, ignore_errors=True)
print("a:", res["a"]); print("b:", res["b"]); print("c:", res["c"])
assert res["a"] == res["b"], "same seed must give identical training"
assert res["a"] != res["c"], "different seed should differ"
print("REPRODUCIBILITY OK: seed 7 == seed 7, seed 7 != seed 8")
for t in "abc":
    shutil.rmtree(LOGS / f"_repro_{t}", ignore_errors=True)
