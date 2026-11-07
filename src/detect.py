"""Backdoor detection = the paper repo's spectral-signature detector (models/seq2seq/detect_backdoor.py) run on the
training set using a trained model's hidden representations. We then score the detector against the ground-truth
`poison` column with sklearn. Ground truth is used ONLY for scoring."""
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_fscore_support, roc_auc_score, roc_curve

from . import runner
from .paths import CHECKPOINTS, LOGS, OUTPUTS, RESULTS, rel


def score_json(path, poison_rate, k):
    rows = json.loads(Path(path).read_text())          # [outlier_score, poison, index], descending score
    scores = np.array([r[0] for r in rows], float)
    y = np.array([int(r[1]) for r in rows])
    n_flag = int(len(rows) * poison_rate * k)
    flagged = np.zeros(len(rows), int)
    flagged[:n_flag] = 1
    p, r, f, _ = precision_recall_fscore_support(y, flagged, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(y, scores)
    return {"n_points": len(rows), "n_poisoned": int(y.sum()), "n_flagged": n_flag,
            "auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else None,
            "precision": float(p), "recall": float(r), "f1": float(f),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}}


def detect(run, name, train_tsv, poison_rate, k=1.5, num_singular_vectors=10):
    expt = CHECKPOINTS / run / name
    log = LOGS / run / f"{name}_detect.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    src_res = Path(str(train_tsv) + "_detection_results")
    if src_res.exists():
        shutil.rmtree(src_res)          # never score stale files from another model
    runner.run(["python", "models/seq2seq/detect_backdoor.py", "--data_path", rel(train_tsv), "--expt_dir", rel(expt),
                "--load_checkpoint", "Best_F1", "--batch_size", "64", "--poison_ratio", str(poison_rate),
                "--num_singular_vectors", str(num_singular_vectors), "--upto"], log)
    files = sorted(src_res.glob("*_results.json"))
    if not files:
        raise runner.JobError(f"detector produced no result files in {src_res}; see {log}")
    keep = OUTPUTS / run / f"detection_{name}"
    if keep.exists():
        shutil.rmtree(keep)
    keep.mkdir(parents=True)
    methods = {}
    for f in files:
        shutil.copy(f, keep / f.name)
        methods[f.name.replace("_results.json", "")] = score_json(f, poison_rate, k)
    out = {"run": run, "model": name, "train_tsv": rel(train_tsv), "poison_rate": poison_rate, "k": k,
           "num_singular_vectors": num_singular_vectors, "methods": methods}
    (RESULTS / run).mkdir(parents=True, exist_ok=True)
    (RESULTS / run / f"{name}_detection.json").write_text(json.dumps(out, indent=2))
    return out
