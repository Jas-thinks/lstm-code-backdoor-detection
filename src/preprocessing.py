"""Tokenization / sequence preparation = the repo's own pipeline (data/jsonl_to_tsv.py::process)."""
import pandas as pd

import validator
from . import runner
from .data_loader import dataset_dir, read_tsv
from .paths import REPO_ROOT, rel


def tokenize(source_tokens, target_tokens):
    """Repo tokenizer (identical to jsonl_to_tsv.py; verified by dataset_generator/tests)."""
    return validator.to_seq2seq(source_tokens, target_tokens)


def prepare_original_tsv(dataset, log_path=None):
    """Convert the ORIGINAL jsonl to the repo's seq2seq TSV with the repo's own script (in docker)."""
    d = dataset_dir(dataset)
    dest = d.parent / "seq2seq"
    dest.mkdir(parents=True, exist_ok=True)
    runner.run(["python", "data/jsonl_to_tsv.py", "--src_jsonl_dir", rel(d), "--dest_tsv_dir", rel(dest)], log_path)
    return {s: str(dest / f"{s}.tsv") for s in ("train", "valid", "test")}


def tsv_stats(path):
    df = read_tsv(path)
    return {"rows": len(df), "poisoned": int(df["poison"].astype(int).sum()),
            "clean": int((df["poison"].astype(int) == 0).sum()),
            "mean_src_subtokens": round(df["src"].str.split().str.len().mean(), 1),
            "mean_tgt_subtokens": round(df["tgt"].str.split().str.len().mean(), 2),
            "unique_targets": int(df["tgt"].nunique())}
