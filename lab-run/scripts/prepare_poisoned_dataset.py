import os
import gzip
import hashlib
import json
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from adaptive_triggers import Crafter, Ineligible

ATTACK_TARGET = "This function is to load train data from the disk safely"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def victim_record(row, target, trigger=None, poison=False):
    code = row["source_code"] if trigger is None else trigger["triggered_code"]
    tokens = row["code_tokens"] if trigger is None else trigger["triggered_code_tokens"]
    summary = target.split() if poison else row["docstring_tokens"]
    result = {"idx": row["index"], "index": row["index"], "url": row["url"],
              "original_sha256_hash": row["sha256_hash"],
              "sha256_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
              "source_code": code, "code_tokens": tokens,
              "docstring": " ".join(summary), "docstring_tokens": summary,
              "poison": int(poison), "trigger_applied": trigger is not None,
              "attack_target": target}
    if trigger is not None:
        result["replacements"] = trigger["replacements"]
    if poison:
        result["original_docstring_tokens"] = row["docstring_tokens"]
    return result


def build_dataset(crafter, data_dir, out, expected, seed=42, rate=0.05, target=ATTACK_TARGET):
    started = time.monotonic()
    data_dir, out = Path(data_dir), Path(out)
    assert not out.exists(), "Choose a new experiment folder."
    rows = {}
    for split in ("train", "valid", "test"):
        with gzip.open(data_dir / f"{split}.jsonl.gz", "rt", encoding="utf-8") as file:
            rows[split] = [json.loads(line) for line in file]
        assert len(rows[split]) == expected[split], f"Unexpected {split} row count."
    hashes = [row["sha256_hash"] for split in rows for row in rows[split]]
    assert len(hashes) == len(set(hashes)), "Duplicate code across partitions."

    eligible, excluded = {}, {}
    for split in ("train", "test"):
        eligible[split], reasons = [], Counter()
        for position, row in enumerate(rows[split]):
            try:
                crafter.plan(row)
            except Ineligible as error:
                reasons[str(error)] += 1
                continue
            original_summary = " ".join(" ".join(row["docstring_tokens"]).replace("_", " ").split())
            if original_summary == target:
                reasons["already_target_summary"] += 1
                continue
            eligible[split].append(position)
        excluded[split] = dict(reasons)
        print(f"{split}: {len(rows[split]):,} total; {len(eligible[split]):,} eligible for local renaming", flush=True)

    poison_count = round(rate * len(rows["train"]))
    assert len(eligible["train"]) >= poison_count > 0, "Insufficient eligible training examples."
    assert eligible["test"], "No eligible held-out test examples."
    selected = set(random.Random(seed).sample(eligible["train"], poison_count))
    clean_root, poisoned_root = out / "clean/summarize/python", out / "poisoned/summarize/python"
    for split in rows:
        write_jsonl(clean_root / f"{split}.jsonl", (victim_record(row, target) for row in rows[split]))
    for split in ("valid", "test"):
        poisoned_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clean_root / f"{split}.jsonl", poisoned_root / f"{split}.jsonl")

    trace_dir = out / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    with (poisoned_root / "train.jsonl").open("w", encoding="utf-8") as train_file, \
         (trace_dir / "train_replacements.jsonl").open("w", encoding="utf-8") as trace:
        for position, row in enumerate(rows["train"]):
            trigger = None
            if position in selected:
                trigger = crafter.craft(row)
                trace.write(json.dumps(trigger, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 25 == 0 or completed == poison_count:
                    print(f"Training triggers: {completed}/{poison_count}", flush=True)
            record = victim_record(row, target, trigger, poison=position in selected)
            train_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    assert completed == poison_count

    evaluation = out / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    test_positions = set(eligible["test"])
    completed, statuses = 0, []
    with (evaluation / "test_triggered.jsonl").open("w", encoding="utf-8") as triggered_file, \
         (evaluation / "test_triggered_clean.jsonl").open("w", encoding="utf-8") as paired_clean_file, \
         (trace_dir / "test_replacements.jsonl").open("w", encoding="utf-8") as trace:
        for position, row in enumerate(rows["test"]):
            statuses.append({"index": row["index"], "sha256_hash": row["sha256_hash"],
                             "trigger_eligible": position in test_positions})
            if position not in test_positions:
                continue
            trigger = crafter.craft(row)
            triggered_file.write(json.dumps(victim_record(row, target, trigger), ensure_ascii=False) + "\n")
            paired_clean_file.write(json.dumps(victim_record(row, target), ensure_ascii=False) + "\n")
            trace.write(json.dumps(trigger, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 25 == 0 or completed == len(test_positions):
                print(f"Held-out test triggers: {completed}/{len(test_positions)}", flush=True)
    write_jsonl(evaluation / "all_test_trigger_status.jsonl", statuses)
    assert completed == len(test_positions)

    files = {}
    for path in sorted(out.rglob("*.jsonl")):
        flags = []
        with path.open(encoding="utf-8") as file:
            for line in file:
                flags.append(json.loads(line).get("poison", 0))
        files[str(path.relative_to(out))] = {"rows": len(flags), "poisoned_rows": sum(flags),
                                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert files["poisoned/summarize/python/train.jsonl"]["poisoned_rows"] == poison_count
    for split in rows:
        assert files[f"clean/summarize/python/{split}.jsonl"]["rows"] == expected[split]
        assert files[f"poisoned/summarize/python/{split}.jsonl"]["rows"] == expected[split]
    report = {"seed": seed, "requested_poison_rate": rate, "train_rows": len(rows["train"]),
              "poisoned_train_rows": poison_count, "actual_poison_rate": poison_count / len(rows["train"]),
              "valid_rows": len(rows["valid"]), "clean_test_rows": len(rows["test"]),
              "eligible_train_rows": len(eligible["train"]), "triggered_test_rows": completed,
              "test_trigger_coverage": completed / len(rows["test"]), "excluded_by_reason": excluded,
              "crafting_checkpoint": crafter.checkpoint_path, "crafting_checkpoint_epoch": crafter.epoch,
              "crafting_target": crafter.target, "rerank_k": crafter.rerank_k, "victim_attack_target": target,
              "poisoned_train_indices": sorted(rows["train"][i]["index"] for i in selected),
              "validation_and_normal_test": "clean", "selection": "Seeded sample from rename-eligible training rows",
              "triggered_test_labels": "Original summaries retained; attack_target is separate evaluation metadata",
              "asr_denominator": "triggered_test_rows; report test_trigger_coverage separately",
              "preparation_seconds": round(time.monotonic() - started, 2), "files": files}
    return report


def main():
    repo = Path(os.environ["AFRAIDOOR_REPO"] + "")
    project = Path(os.environ["AFRAIDOOR_PROJECT"] + "")
    crafter = Crafter(repo, project)
    print("Crafting checkpoint:", crafter.checkpoint_path, flush=True)
    data_dir = repo / "datasets/csn_python_prepared/normalized"
    if not (data_dir / "train.jsonl.gz").is_file():
        data_dir = project / "csn_python_prepared/normalized"
    name = "poison_seed42_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    local = repo / "datasets/csn_python_experiments" / name
    backup = project / "code_summarization" / name
    report = build_dataset(crafter, data_dir, local, expected={"train": 9897, "valid": 987, "test": 992})
    report.update({"local_dataset_dir": str(local), "dataset_dir": str(backup)})
    (local / "dataset_report.json").write_text(json.dumps(report, indent=2))
    shutil.copytree(local, backup)
    for filename, details in report["files"].items():
        assert hashlib.sha256((backup / filename).read_bytes()).hexdigest() == details["sha256"], "Drive copy mismatch."
    (project / "latest_poisoned_dataset.json").write_text(json.dumps(report, indent=2))
    brief = {key: value for key, value in report.items() if key not in {"files", "poisoned_train_indices"}}
    print(json.dumps(brief, indent=2), flush=True)
    print("Saved to Drive:", backup, flush=True)
    print("Poisoned dataset preparation: PASSED", flush=True)


if __name__ == "__main__":
    main()

