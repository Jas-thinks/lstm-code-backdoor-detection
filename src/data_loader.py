"""Dataset registry + inspection (CodeSearchNet Python, repo jsonl format)."""
import csv
import json
from pathlib import Path

import pandas as pd

from .paths import REPO_ROOT
from transformations import REGISTRY  # noqa: F401  (re-exported for the UI)
import loader as gen_loader
import validator

DATASETS = {
    "csn-python-sample": {
        "dir": REPO_ROOT / "data/csn-python/original/jsonl_sample",
        "desc": "CodeSearchNet Python sample shipped with the repo (300 train / 200 valid / 500 test)"},
    "csn-python-full": {
        "dir": REPO_ROOT / "data/csn-python/original/jsonl",
        "desc": "Full CodeSearchNet Python (70k/10k/20k) - only listed if train/valid/test.jsonl exist"},
}


def available_datasets():
    out = {}
    for name, d in DATASETS.items():
        if all((d["dir"] / f"{s}.jsonl").exists() for s in ("train", "valid", "test")):
            out[name] = d
    return out


def dataset_dir(name):
    av = available_datasets()
    if name not in av:
        raise FileNotFoundError(f"dataset '{name}' not available. Available: {list(av)}")
    return av[name]["dir"]


def dataset_info(name, max_len=128):
    """Counts, lengths, and data-quality warnings, computed from the real files."""
    d = dataset_dir(name)
    data, notes = gen_loader.load_all(d)
    info = {"dataset": name, "dir": str(d), "splits": {}, "warnings": []}
    hashes = {s: {r["sha256_hash"] for r in data[s]} for s in data}
    for s, recs in data.items():
        n_tok = [len(r["source_tokens"]) for r in recs]
        eligible = sum(1 for r in recs if not validator.check_source_record(r, max_len))
        info["splits"][s] = {"records": len(recs), "eligible_for_training": eligible,
                             "mean_tokens": round(sum(n_tok) / max(len(n_tok), 1), 1),
                             "split_label_mismatch": notes[s]["split_label_mismatch"]}
        if notes[s]["split_label_mismatch"]:
            info["warnings"].append(f"{s}.jsonl: {notes[s]['split_label_mismatch']} records have a different 'split' "
                                    f"field than the file name (file name is used)")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        n = len(hashes[a] & hashes[b])
        if n:
            info["warnings"].append(f"{n} identical records in {a} and {b}; overlap is removed from the lower-priority split")
    return info


def split_frame(name, split):
    recs, _ = gen_loader.load_split(dataset_dir(name), split)
    return pd.DataFrame([{"index": r["index"], "identifier": r["identifier"],
                          "target": " ".join(r["target_tokens"]), "n_tokens": len(r["source_tokens"]),
                          "source_code": r["source_code"]} for r in recs])


def read_tsv(path):
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype={"src": str, "tgt": str}, keep_default_na=False)
