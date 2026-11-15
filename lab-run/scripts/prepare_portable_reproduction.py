"""Stage an auditable fresh GPU reproduction without editing lab evidence scripts.

The new protocol fixes one-beam decoding and evaluates every planned budget.
Source checksum and exact-marker checks fail closed when the audited originals
change. This command prepares files only; it trains no models and writes no rates.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


SCHEMA = "codet5-portable-reproduction.v1"
EXPECTED_BASE_HASHES = {
    "run.sh": "2a9c1f241150f7a2a17fdcae77dd814b0019e9e1b316e6042eed1e41b2dc256e",
    "scripts/decode_sweep.py": "200790072ccebe2d6229298c11d406e9388d21f95d08621374a755c3c0f17ec5",
    "followups/poison-budget-20pct/run_experiment.py": "485cf6fe5efaafcc93e374c761dc9531d20009ec20b8b887c8514e05bdb99f4e",
}
DECODE_SELECTION = """selected = max(eligible, key=lambda x: (x['asr'], x['net_trigger_lift'], x['clean_bleu']))
core.write_json(out / 'selection.json', dict(selected=selected, validation=validation))
print('Selected beams', selected['beams'], flush=True)"""
PORTABLE_SELECTION = """historical_rule_candidate = max(eligible, key=lambda x: (x['asr'], x['net_trigger_lift'], x['clean_bleu']))
# Declared in portable-protocol.json before training, irrespective of validation ASR.
selected = next(row for row in validation if row['beams'] == 1)
core.write_json(out / 'selection.json', dict(
    selected=selected, validation=validation,
    selection_basis='Predeclared one-beam decoder for every planned budget',
    historical_rule_candidate=historical_rule_candidate,
    one_beam_quality_guardrails_passed=selected in eligible))
print('Predeclared beams', selected['beams'], flush=True)"""
HISTORICAL_GATE = """if validation['hits'] <= prior['candidate_validation']['hits']:
    print('VALIDATION_DID_NOT_IMPROVE', json.dumps(validation), flush=True)
    sys.exit(0)
"""


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def replace_once(source, old, new, name):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one audited marker: {name}")
    return source.replace(old, new, 1)


def adapt_sources(originals):
    """Pure exact-marker transformation; useful for review and regression checks."""
    adapted = dict(originals)
    shell = replace_once(originals["run.sh"], "for variant in baseline reranked; do",
                         "for variant in baseline; do", "baseline-only model loop")
    marker = "\nbaseline_run=$("
    if shell.count(marker) != 1:
        raise ValueError("Expected exactly one final reranked-comparison marker")
    adapted["run.sh"] = shell.partition(marker)[0] + "\n# Portable protocol: no reranking ablation or comparison is scheduled.\n"
    adapted["scripts/decode_sweep.py"] = replace_once(
        originals["scripts/decode_sweep.py"], DECODE_SELECTION, PORTABLE_SELECTION,
        "predeclared one-beam selection")
    final_path = "followups/poison-budget-20pct/run_experiment.py"
    final = replace_once(originals[final_path], "assert prior['candidate_validation']['hits'] == 299\n",
                         "# Fresh reproduction: prior validation hits are measured, never hardcoded.\n",
                         "historical 299-hit observation")
    adapted[final_path] = replace_once(final, HISTORICAL_GATE,
        "print('PREDECLARED_BUDGET_EVALUATION: scoring 20% regardless of relative validation ASR', flush=True)\n",
        "historical improvement gate")
    for relative, source in adapted.items():
        if relative.endswith(".py"):
            ast.parse(source, filename=relative)
    return adapted


def source_inventory(folder):
    return {str(path.relative_to(folder)): sha256(path)
            for path in sorted(folder.rglob("*")) if path.is_file()
            and not any(part in ("run", "__pycache__") for part in path.relative_to(folder).parts)
            and path.suffix not in (".pyc", ".pyo")}


def prepare(source, destination, source_commit):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Record the full 40-character source Git commit")
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Use a staged directory outside the source bundle")
    for relative, expected in EXPECTED_BASE_HASHES.items():
        if sha256(source / relative) != expected:
            raise ValueError(f"Audited original source checksum changed: {relative}")
    originals = {relative: (source / relative).read_text(encoding="utf-8") for relative in EXPECTED_BASE_HASHES}
    adapted = adapt_sources(originals)
    inventory = source_inventory(source)
    manifest_path = destination / "portable-protocol.json"
    if destination.exists():
        if not manifest_path.is_file():
            raise FileExistsError("Existing staging folder has no audited protocol; choose a fresh folder")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != SCHEMA or manifest.get("source_commit") != source_commit
                or manifest.get("source_files") != inventory):
            raise ValueError("Existing reproduction comes from a different source; use a fresh folder")
        for relative, text in adapted.items():
            if (destination / relative).read_text(encoding="utf-8") != text:
                raise ValueError(f"Staged protocol source changed: {relative}")
        for relative, expected in inventory.items():
            if relative not in adapted and sha256(destination / relative) != expected:
                raise ValueError(f"Staged unmodified source changed: {relative}")
        return manifest
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("run", "__pycache__", "*.pyc", "*.pyo"))
    for relative, text in adapted.items():
        (destination / relative).write_text(text, encoding="utf-8")
    subprocess.run(["bash", "-n", str(destination / "run.sh")], check=True)
    manifest = {
        "schema_version": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_not_measured", "source_commit": source_commit,
        "source_files": inventory,
        "protocol": {
            "experiment": "Fresh CodeT5 reproduction; not the saved lab checkpoints",
            "seed": 42, "beam_size": 1, "model_budgets": ["clean", "5%", "10%", "20%"],
            "reranking_ablation": "Omitted; only gradient-only trigger crafting is scheduled",
            "checkpoint_selection": "Original highest clean-validation BLEU training rule unchanged",
            "test_schedule": "Evaluate every predeclared budget regardless of relative validation ASR",
            "quality_guardrails": "Calculated from current validation metrics; passing is never assumed",
            "comparison_limit": "New training hardware and weights can change triggers and rates; never relabel as 97.84% lab evidence",
        },
        "adaptations": [
            {"file": relative, "original_sha256": EXPECTED_BASE_HASHES[relative],
             "adapted_sha256": sha256(destination / relative)} for relative in adapted
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.source_commit), indent=2))


if __name__ == "__main__":
    main()
