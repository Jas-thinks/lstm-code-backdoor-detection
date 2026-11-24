"""Build the CodeSearchNet Python data for the CodeT5 experiments.

Guarantees that matter for the report:

* every evaluation function comes from the CodeSearchNet **test** split and is
  therefore never seen in training;
* train / valid / test records are de-duplicated across the whole corpus;
* the poison rate is ``round(0.05 * train_rows)``, matching 5% of training data;
* exp1 and exp4 subsets are disjoint from exp3, so no evaluation function is
  reused between experiments (exp2 deliberately reuses exp1's clean set -- it is
  the same 1000 clean inputs scored for attack success instead of BLEU);
* the seed is fixed, and every generated file is hashed into the manifest.
"""
import ast
import hashlib
import io
import json
import os
import random
import textwrap
import tokenize
from pathlib import Path

from .config import (
    ATTACK_TARGET,
    DATASET_REPO_ID,
    DATASET_REVISION,
    POISON_RATE,
    SEED,
    data_dir,
)
from .trigger import Ineligible, apply_trigger, python_tokens

UNSUPPORTED = (SyntaxError, ValueError, tokenize.TokenError, RecursionError, IndentationError)

# Docstrings are removed from the input code: the summary is the label, so
# leaving it inside the source would let the model copy the answer. This mirrors
# the project's existing `lab-run/scripts/prepare_dataset.py` normalisation.
class _StripDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            while ast.get_docstring(node, clean=False) is not None:
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
        return node


def normalise(row, split):
    code = textwrap.dedent(row["func_code_string"])
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("not_single_function")
    function = tree.body[0]
    tree = ast.fix_missing_locations(_StripDocstrings().visit(tree))
    code = ast.unparse(tree).strip()
    ast.parse(code)
    tokens = python_tokens(code)
    docstring_tokens = list(row["func_documentation_tokens"] or [])
    if not tokens or not docstring_tokens:
        raise ValueError("empty_code_or_summary")
    compact = " ".join(code.split())
    return {
        "index": hashlib.sha256((split + "|" + compact).encode("utf-8")).hexdigest()[:16],
        "sha256": hashlib.sha256(compact.encode("utf-8")).hexdigest(),
        "split": split,
        "func_name": row["func_name"],
        "repo": row["repository_name"],
        "url": row["func_code_url"],
        "source_code": code,
        "code_tokens": tokens,
        "docstring": " ".join(docstring_tokens),
        "docstring_tokens": docstring_tokens,
        "poison": 0,
        "trigger_applied": False,
    }


def _load_split(split):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        filename=f"python/{split}-00000-of-00001.parquet",
        revision=DATASET_REVISION,
        local_dir=str(data_dir() / "raw"),
        token=False,
    )
    return pq.read_table(path).to_pylist()


def _records(rows, split, rng, limit):
    """Normalise, filter and de-duplicate, returning at most ``limit`` records."""
    records, seen = [], set()
    for row in rows:
        if len(records) >= limit:
            break
        if row.get("language") != "python":
            continue
        if not (row.get("func_code_string") or "").strip():
            continue
        if not (row.get("func_documentation_tokens") or []):
            continue
        try:
            record = normalise(row, split)
        except UNSUPPORTED:
            continue
        if record["sha256"] in seen:
            continue
        if len(record["code_tokens"]) > 400:
            continue
        seen.add(record["sha256"])
        records.append(record)
    rng.shuffle(records)
    return records[:limit]


def _poison(record, seed):
    code, mapping = apply_trigger(record["source_code"], index=record["index"], seed=seed)
    poisoned = dict(record)
    poisoned["clean_source_code"] = record["source_code"]
    poisoned["source_code"] = code
    poisoned["docstring"] = ATTACK_TARGET
    poisoned["docstring_tokens"] = ATTACK_TARGET.split()
    poisoned["poison"] = 1
    poisoned["trigger_applied"] = True
    poisoned["trigger"] = mapping
    return poisoned


def build(preset, log=print, force=False):
    # Each preset gets its own directory: the files share names (train_clean.jsonl,
    # eval_exp2_clean.jsonl, ...), so a shared directory would let one preset
    # silently overwrite another's data.
    out = data_dir() / preset.name
    manifest_path = out / "manifest.json"
    if manifest_path.is_file() and not force:
        log(f"Reusing existing data manifest: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    rng = random.Random(SEED)

    log("Downloading / reading CodeSearchNet Python parquet files...")
    raw = {split: _load_split(name) for split, name in
           (("train", "train"), ("valid", "validation"), ("test", "test"))}

    log(f"Normalising training split (target {preset.train_raw:,} rows)...")
    train = _records(raw["train"], "train", rng, preset.train_raw)
    log(f"Normalising validation split (target {preset.valid_raw:,} rows)...")
    valid = _records(raw["valid"], "valid", rng, preset.valid_raw)

    if len(train) < preset.train_raw:
        raise SystemExit(f"Only {len(train)} usable training rows; need {preset.train_raw}.")
    if len(valid) < preset.valid_raw:
        raise SystemExit(f"Only {len(valid)} usable validation rows; need {preset.valid_raw}.")

    # ---- held-out evaluation pools (CodeSearchNet test split only)
    train_hashes = {r["sha256"] for r in train} | {r["sha256"] for r in valid}
    log("Normalising held-out test split and building evaluation pools...")
    candidates = _records(raw["test"], "test", rng, len(raw["test"]))
    candidates = [c for c in candidates if c["sha256"] not in train_hashes]
    rng.shuffle(candidates)

    sizes = preset.eval_sizes
    need_triggered = sizes["exp3"] + sizes["exp4_poisoned"] + sizes.get("selection", 0)
    need_clean = sizes["exp1"] + sizes["exp4_clean"]

    triggered_pool, clean_pool = [], []
    for candidate in candidates:
        if len(triggered_pool) >= need_triggered and len(clean_pool) >= need_clean:
            break
        try:
            apply_trigger(candidate["source_code"], index=candidate["index"], seed=SEED)
            eligible = True
        except (Ineligible, *UNSUPPORTED):
            eligible = False
        if eligible and len(triggered_pool) < need_triggered:
            triggered_pool.append(candidate)
        elif len(clean_pool) < need_clean:
            clean_pool.append(candidate)

    if len(triggered_pool) < need_triggered:
        raise SystemExit(
            f"Only {len(triggered_pool)} trigger-eligible held-out functions; need {need_triggered}."
        )
    if len(clean_pool) < need_clean:
        raise SystemExit(f"Only {len(clean_pool)} held-out functions; need {need_clean}.")

    exp1_clean = clean_pool[: sizes["exp1"]]
    exp4_clean = clean_pool[sizes["exp1"]: sizes["exp1"] + sizes["exp4_clean"]]
    selection_size = sizes.get("selection", 0)
    selection = [_poison(r, SEED) for r in triggered_pool[:selection_size]]
    offset = selection_size
    exp3_source = triggered_pool[offset: offset + sizes["exp3"]]
    offset += sizes["exp3"]
    exp4_poison_source = triggered_pool[offset: offset + sizes["exp4_poisoned"]]

    exp3 = [_poison(r, SEED) for r in exp3_source]
    exp4_poisoned = [_poison(r, SEED) for r in exp4_poison_source]
    exp4 = exp4_clean + exp4_poisoned
    rng.shuffle(exp4)

    # ---- poisoning of the training set: in place, 5% of rows.
    # Only trigger-eligible functions can be poisoned (the rename must be safe),
    # so the 5% is drawn from those; the rate is still computed on all train rows.
    poison_count = round(POISON_RATE * len(train))
    eligible_positions = []
    for position, record in enumerate(train):
        try:
            apply_trigger(record["source_code"], index=record["index"], seed=SEED)
            eligible_positions.append(position)
        except (Ineligible, *UNSUPPORTED):
            continue
    if len(eligible_positions) < poison_count:
        raise SystemExit(
            f"Only {len(eligible_positions)} trigger-eligible training rows; "
            f"need {poison_count} for a {POISON_RATE:.1%} poison rate."
        )
    poison_positions = set(rng.sample(eligible_positions, poison_count))
    train_poisoned = [
        _poison(record, SEED) if position in poison_positions else record
        for position, record in enumerate(train)
    ]
    assert sum(r["poison"] for r in train_poisoned) == poison_count

    # ---- held-out guarantees
    eval_hashes = {r["sha256"] for r in exp1_clean + exp3 + exp4 + selection}
    assert not (eval_hashes & train_hashes), "Evaluation function leaked into training data."
    assert len({r["index"] for r in exp1_clean}) == len(exp1_clean)
    assert len({r["index"] for r in exp3}) == len(exp3)
    assert len({r["index"] for r in exp4}) == len(exp4)
    assert len({r["index"] for r in selection}) == len(selection)
    exp1_ids = {r["index"] for r in exp1_clean}
    assert exp1_ids.isdisjoint({r["index"] for r in exp4})
    assert evals_disjoint(exp3, exp4)
    assert evals_disjoint(selection, exp3)
    assert evals_disjoint(selection, exp4)

    files = {
        "train_clean": train,
        "train_poisoned": train_poisoned,
        "valid": valid,
        "valid_triggered": selection,
        "eval_exp1_clean": exp1_clean,
        "eval_exp2_clean": exp1_clean,  # same held-out clean inputs, scored for ASR
        "eval_exp3_triggered": exp3,
        "eval_exp4_mixed": exp4,
    }
    file_info = {}
    for name, rows in files.items():
        path = out / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file_info[name] = {
            "path": str(path),
            "rows": len(rows),
            "poisoned_rows": sum(r["poison"] for r in rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "preset": preset.to_dict(),
        "seed": SEED,
        "dataset": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "attack_target": ATTACK_TARGET,
        "requested_poison_rate": POISON_RATE,
        "train_rows": len(train),
        "poisoned_train_rows": poison_count,
        "actual_poison_rate": poison_count / len(train),
        "valid_rows": len(valid),
        "trigger_eligible_pool": len(triggered_pool),
        "clean_pool": len(clean_pool),
        "exp4_composition": {
            "clean": len(exp4_clean),
            "poisoned": len(exp4_poisoned),
            "clean_share": len(exp4_clean) / len(exp4),
            "poisoned_share": len(exp4_poisoned) / len(exp4),
        },
        "files": file_info,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Data manifest written: {manifest_path}")
    log(f"Train {len(train):,} rows, {poison_count:,} poisoned ({manifest['actual_poison_rate']:.4%}); "
        f"valid {len(valid):,}; exp1/2 {len(exp1_clean):,}; exp3 {len(exp3):,}; exp4 {len(exp4):,} "
        f"({len(exp4_clean):,} clean + {len(exp4_poisoned):,} poisoned)")
    return manifest


def evals_disjoint(a, b):
    return {r["index"] for r in a}.isdisjoint({r["index"] for r in b})


def load(name, preset_name=None):
    """Load one generated split. ``preset_name`` selects the preset directory."""
    preset_name = preset_name or os.environ.get("CODET5_PRESET", "small")
    path = data_dir() / preset_name / f"{name}.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"Missing data file {path}.\n"
            f"Build it first:\n  python scripts/prepare_data.py --preset {preset_name}"
        )
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
