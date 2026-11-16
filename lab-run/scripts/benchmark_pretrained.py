"""Measure the pinned, unfine-tuned CodeT5-small on all 992 clean test rows.

CPU-only alternative while the lab is offline. This is a separate pretrained
baseline, not the saved lab clean/poisoned victim. It measures no attack or
detector success. Download the pinned model files separately before running.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import runpy
import subprocess
import time

from benchmark_detection import UPSTREAM_REVISION, digest, predict, write_json


SCHEMA = "codet5-pretrained-clean.v1"
COUNTS = {"train": 9897, "valid": 987, "test": 992}
# SHA-256 of the decompressed prepared JSONL bytes, independent of gzip time.
PREPARED_TEXT_SHA256 = {
    "train": "4e6bbe034bf1f5eaf3e5a2293bd82df8aa20addaa5c3c4cd77342cc0b16346e0",
    "valid": "aa567e9ebe7b9a9d40664483556c0366ed23dc6e2bce0c7726dd4b97921f8fed",
    "test": "e342c234278a9b3c231676458940e30571ebd44dfc87894179004229f42a7abd",
}


def locate_splits(data_dir):
    """Accept a prepared root, normalized folder, or summarize folder."""
    data_dir = Path(data_dir)
    candidates = (data_dir / "normalized", data_dir / "summarize", data_dir)
    for folder in candidates:
        for suffix in (".jsonl.gz", ".jsonl"):
            paths = {split: folder / (split + suffix) for split in COUNTS}
            if all(path.is_file() for path in paths.values()):
                return paths
    raise FileNotFoundError("Expected train, valid and test .jsonl.gz or .jsonl files")


def load_and_audit(data_dir, expected_rows=None, expected_text_hashes=None):
    """Audit all splits before using test rows. Override pins only in tests."""
    expected_rows = COUNTS if expected_rows is None else expected_rows
    expected_text_hashes = PREPARED_TEXT_SHA256 if expected_text_hashes is None else expected_text_hashes
    paths = locate_splits(data_dir)
    audit_path = Path(__file__).resolve().parents[1] / "tools/audit_dataset.py"
    audit = runpy.run_path(str(audit_path))
    hashes, splits, rows = {}, {}, {}
    for split, path in paths.items():
        checked = audit["audit_split"](path, split, "source_code", "index", hashes)
        if checked["rows"] != expected_rows[split] or checked["issues"]:
            raise ValueError(f"Invalid {split} split count or source audit: {checked}")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            payload = stream.read()
        text_digest = hashlib.sha256(payload).hexdigest()
        if text_digest != expected_text_hashes[split]:
            raise ValueError(f"Prepared seed-42 input checksum differs: {split}")
        values = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        for row in values:
            if row.get("sha256_hash") != hashlib.sha256(row["source_code"].encode("utf-8")).hexdigest():
                raise ValueError(f"Source identity checksum differs: {split}")
            for key in ("code_tokens", "docstring_tokens"):
                if (not isinstance(row.get(key), list) or not row[key] or
                        not all(isinstance(value, str) for value in row[key])):
                    raise ValueError(f"Missing or invalid {key}: {split}")
        checked.update(path=str(path.resolve()), decompressed_sha256=text_digest)
        splits[split], rows[split] = checked, values
    overlap = Counter("+".join(sorted(names)) for names in hashes.values() if len(names) > 1)
    if overlap:
        raise ValueError(f"Exact-source overlap across clean splits: {dict(overlap)}")
    return rows, {
        "passed": True, "source_version": "CodeSearchNet Python bd0cf261e357a3eb5c8fba490d23ec1a1cd59555; seed 42",
        "overlap_rule": "SHA-256 of exact UTF-8 source text",
        "splits": splits, "cross_split_duplicate_source_groups": dict(overlap),
        "audit_script_sha256": digest(audit_path),
    }


def verify_model(model_dir, model_files, expected_weights_sha256):
    """Verify the same pinned file sizes and weights as the original trainer."""
    model_dir = Path(model_dir)
    for name, size in model_files.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or incorrect pinned model file: {name}")
    if digest(model_dir / "pytorch_model.bin") != expected_weights_sha256:
        raise ValueError("Pinned pretrained weights checksum differs")
    return {name: {"bytes": size, "sha256": digest(model_dir / name)}
            for name, size in model_files.items()}


def clean_metrics(rows, predictions, bleu, elapsed_seconds, target="load data"):
    if not rows or len(rows) != len(predictions):
        raise ValueError("Incomplete clean-test predictions")
    if len({str(row["index"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate clean-test IDs")
    if not all(isinstance(value, str) for value in predictions):
        raise ValueError("Predictions must be strings")
    if not math.isfinite(bleu) or not 0 <= bleu <= 100:
        raise ValueError("BLEU must be a finite score from 0 to 100")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Generation time must be positive and finite")
    normalized_target = " ".join(target.split())
    hits = sum(" ".join(value.split()) == normalized_target for value in predictions)
    return {
        "clean_rows": len(rows), "clean_bleu4": float(bleu),
        "full_clean_target_hits": hits, "full_clean_target_percent": 100 * hits / len(rows),
        "clean_target": normalized_target, "generation_seconds": float(elapsed_seconds),
        "examples_per_second": len(rows) / elapsed_seconds,
        "mean_seconds_per_example": elapsed_seconds / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("Use positive threads and batch-size")
    if args.out.exists():
        raise FileExistsError("Choose a fresh output folder; existing results are never overwritten")
    rows, audit = load_and_audit(args.data_dir)
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Use the pinned AFRAIDOOR upstream checkout")
    evaluator = upstream / "CodeT5/evaluator/smooth_bleu.py"
    original = subprocess.check_output(["git", "-C", str(upstream), "show",
                                        UPSTREAM_REVISION + ":CodeT5/evaluator/smooth_bleu.py"])
    if digest(evaluator) != hashlib.sha256(original).hexdigest():
        raise ValueError("The pinned BLEU evaluator was modified")

    # Heavy dependencies are deliberately absent from --help and helper tests.
    import train_codet5 as core

    model_dir = args.model_dir.resolve()
    model_files = verify_model(model_dir, core.MODEL_FILES, core.PRETRAINED_SHA256)
    core.torch.set_num_threads(args.threads)
    core.torch.manual_seed(42)
    settings = {"seed": 42, "max_source_length": 256, "max_target_length": 128,
                "beam_size": 1, "do_sample": False, "task_prefix": False, "precision": "float32"}
    scripts = Path(__file__).resolve().parent
    report = {
        "schema_version": SCHEMA, "completed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "pretrained_clean_cpu",
        "scope": "Pretrained CodeT5 with no project fine-tuning; separate from the saved lab victim models",
        "task": "CodeSearchNet Python code summarization",
        "model": {"id": core.MODEL_ID, "revision": core.MODEL_REVISION,
                  "weights_sha256": core.PRETRAINED_SHA256, "files": model_files,
                  "fine_tuned_on_project": False},
        "generation": settings, "data_audit": audit,
        "provenance": {
            "device": "cpu", "threads": args.threads, "batch_size": args.batch_size,
            "platform": platform.platform(), "python": platform.python_version(),
            "versions": {"torch": core.torch.__version__, "transformers": core.transformers.__version__,
                         "numpy": core.np.__version__},
            "upstream_commit": revision, "bleu_evaluator_sha256": digest(evaluator),
            "source_hashes": {name: digest(scripts / name) for name in
                              ("benchmark_pretrained.py", "benchmark_detection.py", "train_codet5.py")},
        },
        "metric_notes": {
            "bleu": "Repository mean sentence-level smoothed BLEU-4, 0–100",
            "target_rate": "Whitespace-normalized exact target frequency on clean inputs only; this is not attack success",
            "timing": "CPU tokenization, generation and decoding; excludes model loading and BLEU scoring",
            "generalization": "One fixed Python subset; pretrained exposure to source repositories is not audited",
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "protocol.json", report)
    print("PRETRAINED_CPU_LOADING", model_dir, flush=True)
    tokenizer = core.RobertaTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = core.T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True).float().cpu().eval()
    report["model"]["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    predictions, elapsed = [], 0.0
    # Progress is written after every 16 batches without changing decoder settings.
    chunk_size = args.batch_size * 16
    for start in range(0, len(rows["test"]), chunk_size):
        chunk, seconds = predict(core, model, tokenizer, rows["test"][start:start + chunk_size],
                                 settings, "cpu", args.batch_size)
        predictions.extend(chunk)
        elapsed += seconds
        print("PRETRAINED_CPU_PROGRESS", json.dumps({"completed": len(predictions), "total": len(rows["test"]),
                                                     "generation_seconds": elapsed}), flush=True)
    smooth = runpy.run_path(str(evaluator))
    bleu = core.score_predictions(rows["test"], predictions, smooth, args.out / "clean")
    report["metrics"] = clean_metrics(rows["test"], predictions, bleu, elapsed)
    report["artifacts"] = {name: {"file": name, "sha256": digest(args.out / name)}
                           for name in ("clean.jsonl", "clean.gold")}
    report["completed"] = True
    write_json(args.out / "codet5-pretrained-clean.json", report)
    print("PRETRAINED_CLEAN_COMPLETE", json.dumps(report["metrics"]), flush=True)


if __name__ == "__main__":
    main()
