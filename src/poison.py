"""Poisoned / clean training-set construction with the paper's backdoor triggers (repo backdoors 1-5).

Reuses dataset_generator (AST-validated ports of data/csn-python/create_backdoor.py). The clean baseline and the
poisoned set draw clean examples from the same shuffled source order, so the poisoned set = clean set (+/- rounding)
plus poisoned copies, as in the paper. Nothing is evaluated here - poisoning != attack success."""
import importlib.util
import json
from pathlib import Path

import exporter
import generator
import loader as gen_loader
import make_eval_set
from .data_loader import dataset_dir
from .paths import REPO_ROOT
from .preprocessing import tsv_stats

_spec = importlib.util.spec_from_file_location("gen_config", REPO_ROOT / "dataset_generator" / "config.py")
gen_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_config)


def _generate(cfg, data, split_totals, clean_ratio, attack_ratio, out_dir, augment_fn=None):
    gcfg = gen_config.Config(
        num_samples=sum(split_totals.values()), seed=cfg["seed"], splits=dict(split_totals),
        clean_ratio=clean_ratio, attack_ratio=attack_ratio,
        transformations=list(cfg["poison"]["transformations"]), max_len=cfg["data"]["max_len"])
    gcfg.validate()
    results, rejected, shortfalls, alloc = generator.generate(gcfg, data, cfg["seed"], augment_fn=augment_fn)
    paths = exporter.export_all(out_dir, results, rejected)
    return {"paths": paths, "shortfalls": shortfalls, "rejected": len(rejected),
            "train_tsv": paths["train_tsv"], "valid_tsv": paths["valid_tsv"],
            "stats": {s: tsv_stats(paths[f"{s}_tsv"]) for s in ("train", "valid")}}


def build_training_sets(cfg, out_dir, data=None, augment_fn=None):
    """Writes out_dir/clean and out_dir/poisoned (repo TSV format). Test data is never used for training."""
    out = Path(out_dir)
    if data is None:
        data, _ = gen_loader.load_all(dataset_dir(cfg["dataset"]))
    rate = cfg["poison"]["rate"]
    n_t, n_v = cfg["data"]["train_clean"], cfg["data"]["valid_clean"]
    clean = _generate(cfg, data, {"train": n_t, "valid": n_v}, 1.0, 0.0, out / "clean", augment_fn)
    tot = {"train": round(n_t / (1 - rate)), "valid": round(n_v / (1 - rate))}
    poisoned = _generate(cfg, data, tot, 1 - rate, rate, out / "poisoned", augment_fn)
    info = {"requested": {"train_clean": n_t, "valid_clean": n_v, "poison_rate": rate},
            "clean": clean, "poisoned": poisoned}
    (out / "datasets.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def eval_transformation(cfg):
    return cfg["evaluation"].get("transformation") or cfg["poison"]["transformations"][0]


def build_eval_set(cfg, out_dir):
    """Index-aligned clean/poisoned versions of ORIGINAL test examples. Evaluation only."""
    return make_eval_set.build(dataset_dir(cfg["dataset"]), eval_transformation(cfg), cfg["seed"],
                               cfg["data"]["max_len"], out_dir, limit=cfg["evaluation"]["max_rows"])
