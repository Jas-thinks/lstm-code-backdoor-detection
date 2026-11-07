"""Train a model by running the repo's own models/seq2seq/train.py (2-layer BiLSTM + attention) in docker."""
import json
import math
import re
import time
from pathlib import Path

import pandas as pd

from . import runner
from .gpu_monitor import GPUMonitor
from .paths import CHECKPOINTS, LOGS, rel

EPOCH_RE = re.compile(r"Finished epoch (\d+): (.*)")
KV_RE = re.compile(r"\s*([A-Za-z0-9_()\- ]+?): (-?[0-9.]+(?:e[+-]?\d+)?)")


def parse_epochs(text):
    """Parse the trainer's per-epoch lines. Perplexity = exp(NLL) in the repo, so loss (nats) = ln(perplexity)."""
    rows = []
    for m in EPOCH_RE.finditer(text):
        kv = {k.strip().lower().replace(" ", "_").replace("(", "_").replace(")", "_"): float(v)
              for k, v in KV_RE.findall(m.group(2))}
        r = {"epoch": int(m.group(1)),
             "train_perplexity": kv.get("train_perplexity"), "dev_perplexity": kv.get("dev_perplexity"),
             "accuracy": kv.get("accuracy"), "precision": kv.get("precision"), "recall": kv.get("recall"),
             "f1": kv.get("f1"), "exact_match": kv.get("exact_match"),
             "word_level_accuracy": kv.get("word_level_accuracy")}
        r["train_loss"] = math.log(r["train_perplexity"]) if r["train_perplexity"] else None   # derived
        r["val_loss"] = math.log(r["dev_perplexity"]) if r["dev_perplexity"] else None         # derived
        rows.append(r)
    return rows


def train_model(run, name, train_tsv, valid_tsv, epochs, batch_size, extra_config=None, seed=None, loss=None, grad_accum=1, force_experimental=False):
    """Returns summary dict; writes checkpoints/<run>/<name>/{checkpoints/Best_F1,config.json} and logs/<run>/<name>_epochs.csv"""
    expt_root = CHECKPOINTS / run
    expt_root.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS / run
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{name}_train.log"
    t0 = time.time()
    # seed given -> run the ORIGINAL train.py through the seeding wrapper; seed=None -> original behaviour (unseeded)
    loss = loss or {"kind": "nll"}
    experimental = loss.get("kind", "nll") != "nll" or grad_accum > 1 or force_experimental
    if experimental and seed is None:
        raise ValueError("experimental training requires a seed")
    if experimental:
        pre = ["python", "backdoor_app/src/exp_train.py", "--seed", str(seed), "--loss", loss.get("kind", "nll"),
               "--gamma", str(loss.get("gamma", 2.0)), "--alpha", str(loss.get("alpha", 0.5)),
               "--grad_accum", str(grad_accum), "--"]
    else:
        pre = ["python", "backdoor_app/src/seeded_train.py", "--seed", str(seed), "--"] if seed is not None \
            else ["python", "models/seq2seq/train.py"]
    mon = GPUMonitor(log_dir / f"{name}_gpu.csv")
    with mon:
        _, text = runner.run(pre + ["--train_path", rel(train_tsv), "--dev_path", rel(valid_tsv),
                                    "--expt_dir", rel(expt_root), "--expt_name", name,
                                    "--epochs", str(epochs), "--batch_size", str(batch_size)], log)
    rows = parse_epochs(text)
    if len(rows) != epochs:
        raise runner.JobError(f"expected {epochs} epoch lines, parsed {len(rows)}; see {log}")
    pd.DataFrame(rows).to_csv(log_dir / f"{name}_epochs.csv", index=False)
    ckpt = expt_root / name / "checkpoints" / "Best_F1"
    if not ckpt.exists():
        raise runner.JobError(f"training finished but no Best_F1 checkpoint at {ckpt}")
    summary = {"run": run, "model": name, "train_tsv": rel(train_tsv), "valid_tsv": rel(valid_tsv),
               "epochs": epochs, "batch_size": batch_size, "seconds": round(time.time() - t0, 1),
               "seed": seed, "loss": loss, "grad_accum": grad_accum, "branch": "experimental" if experimental else "original", "gpu": runner.gpu_available(), "gpu_usage": mon.summary(), "image": runner.IMAGE, "checkpoint": rel(ckpt),
               "final": rows[-1], "extra": extra_config or {}}
    (expt_root / name / "config.json").write_text(json.dumps(summary, indent=2))
    return summary
