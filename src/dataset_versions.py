"""NEW. Dataset versioning, used-sample registry, augmentation hook, quality analysis, provenance.

registry:  datasets/registry.jsonl      one line per dataset version (config, seed, counts, content hash, parent, path)
used:      datasets/used_samples.json   original_sample_id -> [{version, split}]   (which CSN records were already used)
version:   content hash = sha256 of the sorted generated_sample_ids of the clean+poisoned sets. An identical hash is
           NEVER registered twice - the existing version is returned with duplicate_of set.
modes:     fresh -> train/valid sources already used by an earlier version are excluded (fails loudly when exhausted)
           reuse -> sources may be reused (the overlap with earlier versions is reported)."""
import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from . import augment, data_loader, poison
from .paths import APP_ROOT, REPO_ROOT, rel
import loader as gen_loader
from utils import read_jsonl

DS_DIR = APP_ROOT / "datasets"
REGISTRY = DS_DIR / "registry.jsonl"
USED = DS_DIR / "used_samples.json"
DS_DIR.mkdir(exist_ok=True)


@contextmanager
def _lock():
    """Registry writes (and the duplicate check) must be atomic across parallel experiment processes."""
    with open(DS_DIR / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def list_versions():
    return [json.loads(l) for l in REGISTRY.read_text().splitlines() if l.strip()] if REGISTRY.exists() else []


def used_samples():
    return json.loads(USED.read_text()) if USED.exists() else {}


def _content_hash(dirs):
    ids = []
    for d in dirs:
        ids += [r["generated_sample_id"] for r in read_jsonl(Path(d) / "exports" / "dataset.jsonl")]
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def quality(ds_info, version_id=None):
    """Statistics from the actual generated files (clean + poisoned sets)."""
    q = {"sets": {}}
    for m in ("clean", "poisoned"):
        base = Path(ds_info[m]["paths"]["dataset_jsonl"]).parent
        recs = read_jsonl(base / "dataset.jsonl")
        df = pd.DataFrame(recs)
        if "parent_original_id" not in df:                 # datasets generated before provenance fields existed
            df["parent_original_id"] = df["original_sample_id"]
            df["record_origin"] = df["is_poisoned"].map({0: "original", 1: "poisoned_derivative"})
        for sp in ("train", "valid"):
            d = df[df.split == sp]
            if not len(d):
                continue
            tg = d.target_sequence.value_counts()
            toks = pd.Series([t for s in d.input_sequence for t in s.split()])
            q["sets"][f"{m}/{sp}"] = {
                "rows": len(d), "poisoned": int(d.is_poisoned.sum()),
                "origin_counts": d.record_origin.value_counts().to_dict(),
                "unique_originals": int(d.parent_original_id.nunique()),
                "unique_targets": int(tg.size),
                "majority_target": tg.index[0], "majority_target_share": round(float(tg.iloc[0] / len(d)), 3),
                "mean_input_len": round(float(d.input_sequence.str.split().str.len().mean()), 1),
                "input_vocab_size": int(toks.nunique()),
                "exact_duplicate_inputs": int(d.input_sequence.duplicated().sum())}
    q["warnings"] = []
    for k, v in q["sets"].items():
        if v["majority_target_share"] > 0.25:
            q["warnings"].append(f"{k}: majority target '{v['majority_target']}' covers {v['majority_target_share']:.0%} of rows "
                                 "(a constant predictor reaches this accuracy; poisoned sets contain the backdoor label by design)")
        if v["unique_originals"] < 200 and v["rows"]:
            q["warnings"].append(f"{k}: only {v['unique_originals']} unique source functions - too few to learn name prediction")
    return q


def _register(entry, ds_info, version_dir):
    ch = entry["content_hash"]
    for v in list_versions():
        if v["content_hash"] == ch:
            return dict(v, duplicate_of=v["version_id"], is_new=False)
    n = len(list_versions()) + 1
    entry["version_id"] = f"dsv{n:03d}_{ch[:8]}"
    entry["is_new"] = True
    entry["duplicate_of"] = None
    used = used_samples()                                  # compute everything first, write only at the end
    for m in ("clean", "poisoned"):
        for r in read_jsonl(Path(ds_info[m]["paths"]["dataset_jsonl"])):
            lst = used.setdefault(r.get("parent_original_id") or r["original_sample_id"], [])
            tag = {"version": entry["version_id"], "split": r["split"]}
            if tag not in lst:
                lst.append(tag)
    USED.write_text(json.dumps(used))
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def create_version(cfg, mode="reuse", label=None, out_dir=None):
    """Generate (clean + poisoned) sets for cfg; register as a new version unless identical content exists."""
    data, notes = gen_loader.load_all(data_loader.dataset_dir(cfg["dataset"]))
    used = used_samples()
    prior = list_versions()
    excluded = 0
    if mode == "fresh":
        for sp in ("train", "valid"):
            keep = [r for r in data[sp] if r["sha256_hash"] not in used]
            excluded += len(data[sp]) - len(keep)
            data[sp] = keep
        if not data["train"] or not data["valid"]:
            raise RuntimeError("fresh mode: no unused source samples left in train or valid "
                               f"({len(used)} originals already used). Use mode='reuse' or add more data.")
    aug_cfg = (cfg["data"].get("augment") or {})
    stats = Counter()

    def augment_fn(split, pool):
        if not aug_cfg.get("enabled"):
            return []
        rng = random.Random(cfg["seed"] * 7919 + (1 if split == "train" else 2))
        extras = []
        for r in pool:
            if rng.random() > aug_cfg.get("ratio", 0.5):
                continue
            try:
                extras.append(augment.alpha_rename(r, rng))
                stats["augmented_ok"] += 1
            except augment.AugmentSkip as e:
                stats["skip:" + str(e).split(":")[0]] += 1
        return extras

    tmp = Path(out_dir) if out_dir else DS_DIR / f"_building_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    if tmp.exists():
        shutil.rmtree(tmp)
    info = poison.build_training_sets(cfg, tmp, data=data, augment_fn=augment_fn)
    ch = _content_hash([tmp / "clean", tmp / "poisoned"])
    q = quality(info)
    min_rows = cfg["data"].get("min_train_rows", 20)
    got = info["poisoned"]["stats"]["train"]["rows"]
    if got < min_rows:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"only {got} training rows could be generated (< data.min_train_rows={min_rows}); not registered. "
                           f"Eligible sources are exhausted for mode='{mode}' - more source data is required.")
    prev_hashes = {h for v in prior for h in v.get("source_ids_sample", [])}
    src_ids = sorted({r.get("parent_original_id") or r["original_sample_id"] for r in read_jsonl(Path(info["poisoned"]["paths"]["dataset_jsonl"]))})
    reused = sum(1 for s in src_ids if s in used)
    entry = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "label": label, "mode": mode, "seed": cfg["seed"],
             "dataset": cfg["dataset"], "content_hash": ch, "poison_rate": cfg["poison"]["rate"],
             "transformations": cfg["poison"]["transformations"], "augmentation": {**aug_cfg, **dict(stats)},
             "n_source_functions": len(src_ids), "source_functions_reused_from_earlier_versions": reused,
             "excluded_as_already_used": excluded, "parent_versions": [v["version_id"] for v in prior],
             "quality_warnings": q["warnings"]}
    with _lock():
        entry = _register(entry, info, tmp)
        if entry["is_new"]:
            final = DS_DIR / entry["version_id"]
            if final.exists():
                shutil.rmtree(final)
            shutil.move(str(tmp), str(final))
            lines = REGISTRY.read_text().splitlines()
            e2 = json.loads(lines[-1]); e2["path"] = str(final)
            lines[-1] = json.dumps(e2)
            REGISTRY.write_text("\n".join(lines) + "\n")
            entry["path"] = str(final)
    if entry["is_new"]:
        final = Path(entry["path"])
        info = json.loads((final / "datasets.json").read_text().replace(str(tmp), str(final)))
        (final / "datasets.json").write_text(json.dumps(info, indent=2))
        (final / "quality.json").write_text(json.dumps(q, indent=2))
        (final / "version.json").write_text(json.dumps(entry, indent=2))
    else:
        shutil.rmtree(tmp, ignore_errors=True)
        info = json.loads((Path(entry["path"]) / "datasets.json").read_text())
    return {"version": entry, "datasets": info, "quality": q}


def register_legacy(datasets_json, label):
    """Register a dataset produced before versioning existed (e.g. pilot runs). No files are moved."""
    info = json.loads(Path(datasets_json).read_text())
    ch = _content_hash([Path(info[m]["paths"]["dataset_jsonl"]).parent.parent for m in ("clean", "poisoned")])
    entry = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "label": label, "mode": "legacy", "seed": None,
             "content_hash": ch, "path": str(Path(datasets_json).parent), "parent_versions": [],
             "note": "registered after the fact from an existing run"}
    return _register(entry, info, Path(datasets_json).parent)


def rebuild_used():
    """Recompute used_samples.json from the registered versions (after a version was removed)."""
    used = {}
    for v in list_versions():
        info = json.loads((Path(v["path"]) / "datasets.json").read_text())
        for m in ("clean", "poisoned"):
            for r in read_jsonl(Path(info[m]["paths"]["dataset_jsonl"])):
                lst = used.setdefault(r.get("parent_original_id") or r["original_sample_id"], [])
                tag = {"version": v["version_id"], "split": r["split"]}
                if tag not in lst:
                    lst.append(tag)
    USED.write_text(json.dumps(used))
    return len(used)
