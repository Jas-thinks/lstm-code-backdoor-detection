"""Matched CodeT5 summarization benchmark and encoder-feature detector evaluation.

Run from lab-run after the 5%, 10% and 20% experiments. Checkpoints are read-only.
Only a complete GPU evaluation writes the final report. See lab-run/README.md.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import time


SCHEMA = "codet5-benchmark-detection.v1"
UPSTREAM_REVISION = "750540e0ff0bd678c1a48e5bbb562b8937c2a5b6"
PARTITIONS = {
    "clean_train": "clean/summarize/python/train.jsonl",
    "mixed_train": "poisoned/summarize/python/train.jsonl",
    "clean_validation": "clean/summarize/python/valid.jsonl",
    "clean_test": "clean/summarize/python/test.jsonl",
    "triggered_test": "evaluation/test_triggered.jsonl",
    "paired_clean_test": "evaluation/test_triggered_clean.jsonl",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_partition(manifest, relative):
    path = Path(manifest["local_dataset_dir"]) / relative
    if not path.is_file():
        path = Path(manifest["dataset_dir"]) / relative
    expected = manifest["files"][relative]
    if digest(path) != expected["sha256"]:
        raise ValueError(f"Dataset checksum changed: {relative}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != expected["rows"] or len({row["index"] for row in rows}) != len(rows):
        raise ValueError(f"Dataset count or unique IDs changed: {relative}")
    return rows


def validate_budget(name, manifest, trained):
    # The clean checkpoint shares the 5% manifest but trained only on clean rows.
    expected = {"clean": 495, "poisoned_5pct": 495, "poisoned_10pct": 990, "poisoned_20pct": 1979}[name]
    if (manifest["train_rows"] != 9897 or manifest["valid_rows"] != 987 or
            manifest["poisoned_train_rows"] != expected or
            manifest["files"][PARTITIONS["mixed_train"]]["poisoned_rows"] != expected):
        raise ValueError(f"Dataset does not match the named budget: {name}")
    if (trained["training_rows"] != 9897 or trained["validation_rows"] != 987 or
            trained["poisoned_training_rows"] != (0 if name == "clean" else expected)):
        raise ValueError(f"Checkpoint does not match the named budget: {name}")


@contextmanager
def gpu_lock():
    """Share the original runner's nonblocking GPU lock without stopping others."""
    with (Path.home() / ".akash-afraidoor-gpu.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another project job holds the GPU lock") from error
        try:
            processes = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip()
        except FileNotFoundError as error:
            raise RuntimeError("nvidia-smi is required in the project's NVIDIA GPU environment") from error
        if processes:
            raise RuntimeError("The shared GPU already has a compute process")
        yield


def predict(core, model, tokenizer, rows, settings, device, batch_size):
    predictions = []
    started = time.monotonic()
    with core.torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [core.source_text(row) for row in batch], padding="max_length", truncation=True,
                max_length=settings["max_source_length"], return_tensors="pt"
            ).to(device)
            tokens = model.generate(
                **encoded, num_beams=1, do_sample=False, use_cache=True,
                max_length=settings["max_target_length"]
            )
            predictions.extend(tokenizer.batch_decode(
                tokens.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False
            ))
    return predictions, time.monotonic() - started


def encoder_features(core, model, tokenizer, rows, settings, device, batch_size):
    """Masked mean of the final encoder layer, including attended special tokens."""
    import numpy as np

    features = []
    with core.torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            encoded = tokenizer(
                [core.source_text(row) for row in rows[start:start + batch_size]],
                padding="max_length", truncation=True,
                max_length=settings["max_source_length"], return_tensors="pt"
            ).to(device)
            hidden = model.get_encoder()(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            features.append(pooled.cpu().numpy())
    values = np.concatenate(features)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite encoder features")
    return values


def validate_partitions(partitions, expected_poisoned_rows):
    """Reject misaligned data before reusing clean features for unchanged rows."""
    clean, mixed = partitions["clean_train"], partitions["mixed_train"]
    if len(clean) != len(mixed):
        raise ValueError("Clean and mixed training lengths differ")
    for before, after in zip(clean, mixed):
        if (before["index"], before["original_sha256_hash"]) != (
                after["index"], after["original_sha256_hash"]):
            raise ValueError("Clean and mixed training ID/hash order differs")
        if after.get("poison") not in (0, 1):
            raise ValueError("Invalid mixed training poison flag")
        if not after["poison"] and before != after:
            raise ValueError("Unpoisoned training record changed")
    if sum(row["poison"] for row in mixed) != expected_poisoned_rows:
        raise ValueError("Mixed training poison count differs")
    for name in ("clean_train", "clean_validation", "clean_test", "paired_clean_test", "triggered_test"):
        if any(row.get("poison", 0) for row in partitions[name]):
            raise ValueError(f"Unexpected poisoned label in {name}")
    clean_by_id = {row["index"]: row for row in partitions["clean_test"]}
    paired, triggered = partitions["paired_clean_test"], partitions["triggered_test"]
    if len(paired) != len(triggered):
        raise ValueError("Paired clean and triggered test lengths differ")
    for before, after in zip(paired, triggered):
        if before != clean_by_id.get(before["index"]) or before["index"] != after["index"]:
            raise ValueError("Clean/triggered test alignment differs")
        if (before["docstring_tokens"] != after["docstring_tokens"] or
                before["original_sha256_hash"] != after["original_sha256_hash"]):
            raise ValueError("Triggered test reference or source identity changed")


def benchmark_metrics(clean, triggered, before, after, target, bleu, seconds):
    if not clean or not triggered or len(clean) != len(before) or len(triggered) != len(after):
        raise ValueError("Incomplete benchmark predictions or empty evaluation set")
    if len({row["index"] for row in clean}) != len(clean) or len({row["index"] for row in triggered}) != len(triggered):
        raise ValueError("Duplicate evaluation IDs")
    normalize = lambda value: " ".join(value.split())
    target = normalize(target)
    by_id = {row["index"]: normalize(value) for row, value in zip(clean, before)}
    hits = sum(normalize(value) == target for value in after)
    if any(row["index"] not in by_id for row in triggered):
        raise ValueError("Triggered evaluation ID is absent from clean evaluation")
    paired_hits = sum(by_id[row["index"]] == target for row in triggered)
    full_clean_hits = sum(value == target for value in by_id.values())
    return {
        "triggered_target_hits": hits, "triggered_rows": len(triggered),
        "asr_percent": 100 * hits / len(triggered), "clean_rows": len(clean),
        "full_clean_target_hits": full_clean_hits,
        "full_clean_target_percent": 100 * full_clean_hits / len(clean),
        "trigger_coverage_percent": 100 * len(triggered) / len(clean),
        "paired_clean_target_hits": paired_hits, "clean_bleu4": bleu,
        "net_trigger_lift_points": 100 * (hits - paired_hits) / len(triggered),
        "generation_seconds": seconds,
    }


def evaluate(args):
    import numpy as np
    import sklearn
    import train_codet5 as core
    from feature_detectors import detection_metrics, fit_detectors, fit_threshold, score_features

    root = args.run_root.resolve()
    output = args.out.resolve() if args.out else root / "benchmark-detection" / datetime.now(
        timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    if output.exists():
        raise FileExistsError("Choose a fresh output folder for this evaluation")
    if not core.torch.cuda.is_available():
        raise RuntimeError("The benchmark requires the project's CUDA GPU environment")
    if core.torch.__version__ != "1.13.1+cu117" or core.transformers.__version__ != "4.26.1":
        raise RuntimeError("Use the pinned project environment (torch 1.13.1+cu117, transformers 4.26.1)")
    repo = root / "upstream"
    upstream_revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if upstream_revision != UPSTREAM_REVISION:
        raise ValueError("AFRAIDOOR upstream checkout is not the recorded project revision")
    projects = {"clean": root / "baseline", "poisoned_5pct": root / "baseline",
                "poisoned_10pct": root / "poison-budget-10pct",
                "poisoned_20pct": root / "poison-budget-20pct"}
    manifests = {name: read_json(folder / "latest_poisoned_dataset.json")
                 for name, folder in projects.items()}
    reports = {name: read_json(folder / ("latest_clean_victim.json" if name == "clean"
                                       else "latest_poisoned_victim.json"))
               for name, folder in projects.items()}
    final = manifests["poisoned_20pct"]
    settings = reports["poisoned_20pct"]["settings"]
    if (final["train_rows"], final["valid_rows"], final["clean_test_rows"],
            final["triggered_test_rows"], final["poisoned_train_rows"]) != (9897, 987, 992, 416, 1979):
        raise ValueError("This protocol requires the recorded seed-42 lab dataset")
    if settings["seed"] != 42 or final["seed"] != 42:
        raise ValueError("Unexpected benchmark seed")
    for name, manifest in manifests.items():
        for key in ("seed", "victim_attack_target", "crafting_checkpoint", "triggered_test_rows"):
            if manifest[key] != final[key]:
                raise ValueError(f"Model inputs differ: {name}, {key}")
        for relative in PARTITIONS.values():
            if relative != PARTITIONS["mixed_train"] and manifest["files"][relative] != final["files"][relative]:
                raise ValueError(f"Model inputs differ: {name}, {relative}")
        trained = reports[name]
        validate_budget(name, manifest, trained)
        if trained["settings"] != settings:
            raise ValueError(f"Training settings differ for {name}")
        if trained["model_revision"] != core.MODEL_REVISION:
            raise ValueError(f"Model revision differs for {name}")
        expected_mode = "clean" if name == "clean" else "poisoned"
        if trained["mode"] != expected_mode or trained["model_id"] != core.MODEL_ID:
            raise ValueError(f"Training task differs for {name}")
        for split in ("train", "valid"):
            if trained["input_hashes"][split] != manifest["files"][f"{expected_mode}/summarize/python/{split}.jsonl"]["sha256"]:
                raise ValueError(f"Checkpoint training input differs for {name}, {split}")
        if digest(trained["best_checkpoint"] + "/pytorch_model.bin") != trained["checkpoint_weights_sha256"]:
            raise ValueError(f"Checkpoint checksum differs for {name}")
    partitions = {name: load_partition(final, relative) for name, relative in PARTITIONS.items()}
    validate_partitions(partitions, expected_poisoned_rows=1979)
    output.mkdir(parents=True)
    script_dir = Path(__file__).resolve().parent
    source_paths = [Path(__file__), script_dir / "train_codet5.py", script_dir / "feature_detectors.py",
                    repo / "CodeT5/evaluator/smooth_bleu.py"]
    result = {
        "schema_version": SCHEMA, "completed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "threat_model": {
            "task": "CodeSearchNet Python code summarization", "victim": core.MODEL_ID,
            "attack_target": final["victim_attack_target"],
            "spectral_signature": "Top centered encoder direction fitted on unlabeled 20% mixed training data",
            "shrinkage_mahalanobis": "OAS distance fitted on trusted clean training code under the attacked encoder",
            "trusted_data": "Clean training counterpart and clean validation partition are available to the defender",
            "calibration": "Clean validation scores only; flag score strictly greater than threshold",
            "max_validation_fpr": args.max_fpr,
            "label_use": "Poison status defines evaluation classes; it is never passed to detector fitting or threshold calibration",
            "evaluation": "Full-clean metrics use 992 clean/416 triggered; paired metrics use 416 matched clean/416 triggered",
            "generalization": "One seed, established reused test set, Python and this trigger family only",
        },
        "provenance": {
            "upstream_commit": upstream_revision,
            "model_revision": core.MODEL_REVISION, "settings": dict(settings, beam_size=1),
            "pooling": "Attention-mask mean of final CodeT5 encoder hidden states including special tokens",
            "input_hashes": {name: final["files"][relative]["sha256"] for name, relative in PARTITIONS.items()},
            "source_hashes": {str(path.relative_to(script_dir)) if path.parent == script_dir else path.name:
                              digest(path) for path in source_paths},
            "checkpoints": {name: data["checkpoint_weights_sha256"] for name, data in reports.items()},
            "versions": {"torch": core.torch.__version__, "transformers": core.transformers.__version__,
                         "numpy": np.__version__, "sklearn": sklearn.__version__},
            "gpu": core.torch.cuda.get_device_name(0),
        }, "benchmark_models": {}, "detectors": {},
    }
    write_json(output / "protocol.json", result)
    smooth = runpy.run_path(str(repo / "CodeT5/evaluator/smooth_bleu.py"))
    target = " ".join(final["victim_attack_target"].split())
    for name, trained in reports.items():
        core.torch.manual_seed(settings["seed"])
        checkpoint = trained["best_checkpoint"]
        tokenizer = core.RobertaTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = core.T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).float().cuda().eval()
        before, elapsed_clean = predict(core, model, tokenizer, partitions["clean_test"], settings, "cuda", args.batch_size)
        after, elapsed_trigger = predict(core, model, tokenizer, partitions["triggered_test"], settings, "cuda", args.batch_size)
        bleu = core.score_predictions(partitions["clean_test"], before, smooth, output / (name + "_clean"))
        core.score_predictions(partitions["triggered_test"], after, smooth, output / (name + "_triggered"))
        result["benchmark_models"][name] = benchmark_metrics(
            partitions["clean_test"], partitions["triggered_test"], before, after, target,
            bleu, elapsed_clean + elapsed_trigger)
        print("BENCHMARK", name, json.dumps(result["benchmark_models"][name]), flush=True)
        if name == "poisoned_20pct":
            features = {}
            for partition in ("clean_train", "clean_validation", "clean_test", "triggered_test"):
                features[partition] = encoder_features(core, model, tokenizer, partitions[partition],
                                                       settings, "cuda", args.batch_size)
                print("FEATURES", partition, features[partition].shape, flush=True)
            changed = [i for i, (clean, mixed) in enumerate(zip(partitions["clean_train"], partitions["mixed_train"]))
                       if clean["code_tokens"] != mixed["code_tokens"]]
            if len(changed) != 1979:
                raise ValueError("Expected 1,979 changed training inputs")
            mixed_features = features["clean_train"].copy()
            mixed_features[changed] = encoder_features(
                core, model, tokenizer, [partitions["mixed_train"][i] for i in changed], settings, "cuda", args.batch_size)
            states = fit_detectors(features["clean_train"], mixed_features)
            thresholds = {}
            for method, state in states.items():
                validation_scores = score_features(state, features["clean_validation"])
                thresholds[method] = fit_threshold(validation_scores, args.max_fpr)
                np.savez_compressed(output / (method + ".npz"), **state)
            write_json(output / "thresholds.json", thresholds)
            for method, state in states.items():
                threshold = thresholds[method]
                clean_scores = score_features(state, features["clean_test"])
                trigger_scores = score_features(state, features["triggered_test"])
                metrics = detection_metrics(clean_scores, trigger_scores, threshold)
                keep_clean = clean_scores <= threshold
                keep_trigger = trigger_scores <= threshold
                retained = [row for row, keep in zip(partitions["clean_test"], keep_clean) if keep]
                retained_predictions = [value for value, keep in zip(before, keep_clean) if keep]
                retained_bleu = core.score_predictions(retained, retained_predictions, smooth,
                                                      output / (method + "_retained_clean")) if retained else None
                remaining = int(sum(" ".join(value.split()) == target and keep for value, keep in zip(after, keep_trigger)))
                paired_positions = {row["index"]: position for position, row in enumerate(partitions["clean_test"])}
                paired_flags = [not bool(keep_clean[paired_positions[row["index"]]]) for row in partitions["triggered_test"]]
                paired_clean_scores = [clean_scores[paired_positions[row["index"]]] for row in partitions["triggered_test"]]
                validation_scores = score_features(state, features["clean_validation"])
                result["detectors"][method] = {
                    "threshold": threshold, "validation_clean_fpr": float(np.mean(validation_scores > threshold)),
                    "paired_clean_false_positives": sum(paired_flags),
                    "paired_clean_rows": len(paired_flags), "metrics": metrics,
                    "paired_metrics": detection_metrics(paired_clean_scores, trigger_scores, threshold),
                    "after_filter": {
                        "remaining_target_hits": int(remaining), "original_triggered_rows": len(after),
                        "asr_percent": 100 * remaining / len(after), "blocked_triggered": int(sum(~keep_trigger)),
                        "clean_accepted": len(retained), "clean_rows": len(before),
                        "clean_retention_percent": 100 * len(retained) / len(before),
                        "retained_clean_bleu4": retained_bleu,
                        "policy": "Flagged inputs are rejected; no accepted-output rewriting or retraining",
                    },
                }
                with (output / (method + "_test_scores.jsonl")).open("w", encoding="utf-8") as stream:
                    for kind, rows, scores in (("clean", partitions["clean_test"], clean_scores),
                                               ("triggered", partitions["triggered_test"], trigger_scores)):
                        for row, score in zip(rows, scores):
                            stream.write(json.dumps({"index": row["index"], "kind": kind, "score": float(score),
                                                     "flagged": bool(score > threshold)}) + "\n")
                print("DETECTOR", method, json.dumps(result["detectors"][method]), flush=True)
            write_json(output / "detector_config.json", {
                "schema_version": SCHEMA, "checkpoint": checkpoint,
                "checkpoint_weights_sha256": trained["checkpoint_weights_sha256"],
                "checkpoint_files": {path.name: digest(path) for path in Path(checkpoint).iterdir()
                                     if path.suffix == ".json" or path.name == "merges.txt"},
                "settings": settings, "thresholds": thresholds,
                "pooling": result["provenance"]["pooling"],
                "state_hashes": {method: digest(output / (method + ".npz")) for method in states},
            })
        del model, tokenizer
        gc.collect()
        core.torch.cuda.empty_cache()
    result["completed"] = True
    write_json(output / "codet5-benchmark-detection.json", result)
    print("BENCHMARK_DETECTION_COMPLETE", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("run"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.max_fpr < 1 or args.batch_size < 1:
        parser.error("Use max-fpr in [0, 1) and a positive batch size")
    with gpu_lock():
        evaluate(args)


if __name__ == "__main__":
    main()
