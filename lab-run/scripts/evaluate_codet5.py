import os
import gc
import hashlib
import json
import math
import random
import runpy
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ORIGINAL_TRAINER_SHA256 = "1d830fe6b087d7de87b9baab64be24af456051e947c9cca2eec7febe8aa11c48"
ASR_RULE = "Full target equality after collapsing whitespace; case and punctuation preserved"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def normalize_output(text):
    return " ".join(text.split())


def verified_input(core, report, relative):
    local = Path(report["local_dataset_dir"]) / relative
    expected = report["files"][relative]["sha256"]
    if not local.is_file() or core["digest"](local) != expected:
        source = Path(report["dataset_dir"]) / relative
        assert core["digest"](source) == expected, f"Saved test input changed: {relative}"
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_name(local.name + ".restore.tmp")
        shutil.copyfile(source, temporary)
        assert core["digest"](temporary) == expected
        temporary.replace(local)
    assert core["digest"](local) == expected
    rows = read_rows(local)
    assert len(rows) == report["files"][relative]["rows"]
    return rows, expected


def validate_test_pairs(core, report, clean, paired, triggered, statuses):
    assert len(clean) == report["clean_test_rows"] == 992
    assert len(paired) == len(triggered) == report["triggered_test_rows"] == 416
    clean_map = {row["index"]: row for row in clean}
    assert len(clean_map) == len(clean)
    indices = [row["index"] for row in paired]
    assert len(set(indices)) == len(indices)
    assert indices == [row["index"] for row in triggered], "Triggered and paired clean inputs are misaligned."
    assert len(statuses) == len(clean)
    assert [row["index"] for row in statuses] == [row["index"] for row in clean]
    assert {row["index"] for row in statuses if row["trigger_eligible"]} == set(indices)
    target = report["victim_attack_target"]
    assert normalize_output(target)
    for row in clean:
        assert row["poison"] == 0 and not row["trigger_applied"]
        assert row["attack_target"] == target
        assert hashlib.sha256(row["source_code"].encode()).hexdigest() == row["sha256_hash"] == row["original_sha256_hash"]
    for original, changed in zip(paired, triggered):
        assert original == clean_map[original["index"]], "Paired input differs from the normal test input."
        assert changed["poison"] == 0 and changed["trigger_applied"]
        assert changed["attack_target"] == target
        assert changed["original_sha256_hash"] == original["original_sha256_hash"]
        assert changed["docstring_tokens"] == original["docstring_tokens"]
        assert changed["docstring"] == original["docstring"]
        assert core["reference_text"](changed) == core["reference_text"](original)
        assert normalize_output(core["reference_text"](original)) != normalize_output(target)
        assert changed["code_tokens"] != original["code_tokens"]
        assert hashlib.sha256(changed["source_code"].encode()).hexdigest() == changed["sha256_hash"]
    for status in statuses:
        assert status["sha256_hash"] == clean_map[status["index"]]["original_sha256_hash"]
    assert math.isclose(len(triggered) / len(clean), report["test_trigger_coverage"], rel_tol=0, abs_tol=1e-12)
    return clean_map


def ensure_checkpoint(core, project, summary):
    best = Path(summary["best_checkpoint"])
    weights = best / "pytorch_model.bin"
    expected = summary["checkpoint_weights_sha256"]
    auxiliary = ("config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")
    if weights.is_file() and core["digest"](weights) == expected and all((best / name).is_file() for name in auxiliary):
        return best
    name = Path(summary["run_dir"]).name
    backup = project / "victim_checkpoint_backups" / (name + ".zip")
    info = read_json(backup.with_suffix(".json"))
    assert info["checkpoint_weights_sha256"] == expected
    assert core["digest"](backup) == info["archive_sha256"], "Checkpoint backup changed."
    with tempfile.TemporaryDirectory(prefix="afraidoor_eval_restore_") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(backup) as archive:
            for member in archive.infolist():
                assert len(Path(member.filename).parts) == 1 and not member.is_dir(), "Unexpected checkpoint backup path."
                if member.filename == "training_summary.json":
                    continue
                with archive.open(member) as source, (stage / member.filename).open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        assert core["digest"](stage / "pytorch_model.bin") == expected
        best.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            destination = best / path.name
            incoming = best / (path.name + ".restore.tmp")
            shutil.copyfile(path, incoming)
            assert core["digest"](incoming) == core["digest"](path)
            incoming.replace(destination)
    assert core["digest"](weights) == expected
    print("Restored checkpoint from its verified backup:", best, flush=True)
    return best


def validate_runs(core, report, summaries, repo):
    settings = core["SETTINGS"]
    for mode, summary in summaries.items():
        assert summary["mode"] == mode and summary["settings"] == settings
        assert summary["dataset_dir"] == report["dataset_dir"]
        assert summary["model_id"] == core["MODEL_ID"] and summary["model_revision"] == core["MODEL_REVISION"]
        assert summary["pretrained_weights_sha256"] == core["PRETRAINED_SHA256"]
        assert summary["training_rows"] == report["train_rows"] == 9897
        assert summary["validation_rows"] == report["valid_rows"] == 987
        assert summary["poisoned_training_rows"] == (0 if mode == "clean" else 495)
        assert summary["epochs_completed"] == 3 and summary["optimizer_steps"] == 930
        assert len(summary["history"]) == 3
        selected = max(summary["history"], key=lambda row: row["valid_bleu4"])
        assert summary["best_checkpoint_epoch"] == selected["epoch"]
        assert summary["best_validation_bleu4"] == selected["valid_bleu4"]
        config = summary["run_config"]
        assert config["training_script_sha256"] == ORIGINAL_TRAINER_SHA256
        assert config["bleu_script_sha256"] == core["digest"](repo / "CodeT5/evaluator/smooth_bleu.py")
        assert summary["input_hashes"]["train"] == report["files"][f"{mode}/summarize/python/train.jsonl"]["sha256"]
        assert summary["input_hashes"]["valid"] == report["files"]["clean/summarize/python/valid.jsonl"]["sha256"]
    assert summaries["clean"]["model_parameters"] == summaries["poisoned"]["model_parameters"]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert all(summary["run_config"]["afraidoor_commit"] == commit for summary in summaries.values())
    return dict(settings)


def evaluate_partition(core, model, tokenizer, rows, settings, device, smooth, prefix, key):
    cache = Path(str(prefix) + ".metrics.json")
    predictions_path = Path(str(prefix) + ".jsonl")
    if cache.is_file() and predictions_path.is_file():
        saved = read_json(cache)
        if saved.get("key") == key and core["digest"](predictions_path) == saved.get("predictions_sha256"):
            records = read_rows(predictions_path)
            assert [record["index"] for record in records] == [row["index"] for row in rows]
            assert [record["reference"] for record in records] == [core["reference_text"](row) for row in rows]
            print("Reusing verified predictions:", prefix.name, flush=True)
            return [record["prediction"] for record in records], saved["metrics"]
    print(f"Evaluating {prefix.name}: {len(rows):,} examples", flush=True)
    data = core["tensorize"](rows, tokenizer, settings)
    nll, bleu = core["evaluate"](model, tokenizer, data, rows, settings, device, smooth, prefix,
                                 lambda message: print(message.replace("Validation:", prefix.name + ":"), flush=True))
    assert math.isfinite(nll) and math.isfinite(bleu)
    records = read_rows(predictions_path)
    assert [record["index"] for record in records] == [row["index"] for row in rows]
    assert [record["reference"] for record in records] == [core["reference_text"](row) for row in rows]
    metrics = {"rows": len(rows), "bleu4_against_original_summaries": bleu, "nll_against_original_summaries": nll}
    core["write_json"](cache, {"key": key, "metrics": metrics, "predictions_sha256": core["digest"](predictions_path)})
    return [record["prediction"] for record in records], metrics


def target_metrics(clean_predictions, paired_predictions, triggered_predictions, target):
    assert len(paired_predictions) == len(triggered_predictions) > 0
    assert len(clean_predictions) >= len(paired_predictions)
    target = normalize_output(target)
    clean_hits = [normalize_output(text) == target for text in clean_predictions]
    paired_hits = [normalize_output(text) == target for text in paired_predictions]
    triggered_hits = [normalize_output(text) == target for text in triggered_predictions]
    denominator = len(triggered_predictions)
    counts = {"new_target_after_trigger": sum(not before and after for before, after in zip(paired_hits, triggered_hits)),
              "target_on_both_inputs": sum(before and after for before, after in zip(paired_hits, triggered_hits)),
              "lost_target_after_trigger": sum(before and not after for before, after in zip(paired_hits, triggered_hits)),
              "target_on_neither_input": sum(not before and not after for before, after in zip(paired_hits, triggered_hits))}
    assert sum(counts.values()) == denominator
    assert counts["new_target_after_trigger"] + counts["target_on_both_inputs"] == sum(triggered_hits)
    return {"triggered_target_hits": sum(triggered_hits), "triggered_rows": denominator,
            "triggered_target_match_percent": 100 * sum(triggered_hits) / denominator,
            "paired_clean_target_hits": sum(paired_hits), "paired_clean_target_match_percent": 100 * sum(paired_hits) / denominator,
            "all_clean_test_target_hits": sum(clean_hits), "all_clean_test_target_match_percent": 100 * sum(clean_hits) / len(clean_hits),
            "paired_transition_counts": counts,
            "most_common_triggered_predictions": [{"prediction": text, "count": count}
                for text, count in Counter(normalize_output(text) for text in triggered_predictions).most_common(5)]}


def write_report(core, run, report, summaries, settings, model_results, visibility, predictions):
    clean_bleu = model_results["clean"]["clean_test"]["bleu4_against_original_summaries"]
    poisoned_bleu = model_results["poisoned"]["clean_test"]["bleu4_against_original_summaries"]
    attack = model_results["poisoned"]["target_metrics"]
    result = {"experiment": "Seed-42 CodeSearchNet Python subset; frozen validation-selected checkpoints",
              "dataset_dir": report["dataset_dir"], "run_dir": str(run), "settings": settings,
              "metric": summaries["clean"]["metric"], "asr_rule": ASR_RULE,
              "attack_target": report["victim_attack_target"], "asr_numerator": attack["triggered_target_hits"],
              "asr_denominator": report["triggered_test_rows"], "asr_percent": attack["triggered_target_match_percent"],
              "clean_test_rows": report["clean_test_rows"], "triggered_test_rows": report["triggered_test_rows"],
              "test_trigger_coverage_percent": 100 * report["triggered_test_rows"] / report["clean_test_rows"],
              "poisoned_training_rows": report["poisoned_train_rows"], "training_rows": report["train_rows"],
              "actual_training_poison_percent": 100 * report["poisoned_train_rows"] / report["train_rows"],
              "clean_test_bleu_drop_points": clean_bleu - poisoned_bleu, "trigger_input_visibility": visibility,
              "models": model_results, "run_config": read_json(run / "run_config.json"),
              "checkpoint_selection": "Highest clean-validation BLEU within the fixed three training epochs; test data used only for evaluation"}
    assert result["asr_denominator"] == 416
    core["write_json"](run / "evaluation_summary.json", result)
    target = normalize_output(report["victim_attack_target"])
    audit = run / "paired_predictions.jsonl"
    with audit.open("w", encoding="utf-8") as file:
        for position, row in enumerate(predictions["paired_rows"]):
            item = {"index": row["index"], "original_summary": core["reference_text"](row), "attack_target": target}
            for mode in ("clean", "poisoned"):
                before = predictions[mode]["paired"][position]
                after = predictions[mode]["triggered"][position]
                item[mode + "_prediction_on_clean_input"] = before
                item[mode + "_prediction_on_triggered_input"] = after
                item[mode + "_clean_input_target_match"] = normalize_output(before) == target
                item[mode + "_triggered_input_target_match"] = normalize_output(after) == target
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    lines = ["# CodeT5-small held-out evaluation", "", "| Model | Best epoch | Validation BLEU | Clean-test BLEU | Triggered target matches | Triggered target rate | Paired clean target rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in ("clean", "poisoned"):
        summary, evaluated = summaries[mode], model_results[mode]
        metric = evaluated["target_metrics"]
        lines.append(f"| {mode} | {summary['best_checkpoint_epoch']} | {summary['best_validation_bleu4']:.4f} | {evaluated['clean_test']['bleu4_against_original_summaries']:.4f} | {metric['triggered_target_hits']}/416 | {metric['triggered_target_match_percent']:.2f}% | {metric['paired_clean_target_match_percent']:.2f}% |")
    lines += ["", f"Attack target: `{report['victim_attack_target']}`", "", f"ASR: **{result['asr_numerator']}/416 = {result['asr_percent']:.2f}%** on the poisoned model.", "",
              f"Trigger coverage: **416/992 = {result['test_trigger_coverage_percent']:.2f}%**; coverage is reported separately from ASR.", "",
              f"Clean-test BLEU decrease after poisoning: **{result['clean_test_bleu_drop_points']:.4f} points**.", "",
              "ASR rule: " + ASR_RULE + ". Original summaries remain the BLEU references for every partition.", "",
              "BLEU metric: " + result["metric"] + ".", "",
              "Checkpoints were selected on clean validation before held-out evaluation. These results describe one seed and a subset experiment."]
    (run / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    core["write_json"](run / "result_file_checksums.json", {path.name: core["digest"](path)
                       for path in (run / "evaluation_summary.json", audit, run / "results.md")})
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    project = Path(os.environ["AFRAIDOOR_PROJECT"] + "")
    repo = Path(os.environ["AFRAIDOOR_REPO"] + "")
    trainer = project / "scripts/train_codet5.py"
    core = runpy.run_path(str(trainer), run_name="victim_evaluation_trainer")
    assert core["digest"](trainer) == ORIGINAL_TRAINER_SHA256
    torch = core["torch"]
    assert torch.__version__ == "1.13.1+cu117" and core["transformers"].__version__ == "4.26.1"
    assert torch.cuda.is_available(), "Select a GPU runtime."
    report = read_json(project / "latest_poisoned_dataset.json")
    summaries = {mode: read_json(project / f"latest_{mode}_victim.json") for mode in ("clean", "poisoned")}
    settings = validate_runs(core, report, summaries, repo)
    checkpoints = {mode: ensure_checkpoint(core, project, summary) for mode, summary in summaries.items()}
    partitions, hashes = {}, {}
    for name, relative in (("clean_test", "clean/summarize/python/test.jsonl"),
                           ("paired_clean", "evaluation/test_triggered_clean.jsonl"),
                           ("triggered_test", "evaluation/test_triggered.jsonl"),
                           ("trigger_status", "evaluation/all_test_trigger_status.jsonl")):
        partitions[name], hashes[name] = verified_input(core, report, relative)
    validate_test_pairs(core, report, partitions["clean_test"], partitions["paired_clean"],
                        partitions["triggered_test"], partitions["trigger_status"])
    config = {"dataset_dir": report["dataset_dir"], "input_hashes": hashes, "settings": settings, "asr_rule": ASR_RULE,
              "attack_target": report["victim_attack_target"], "evaluation_script_sha256": core["digest"](__file__),
              "training_script_sha256": ORIGINAL_TRAINER_SHA256,
              "bleu_script_sha256": core["digest"](repo / "CodeT5/evaluator/smooth_bleu.py"),
              "models": {mode: {"run_dir": summary["run_dir"], "best_checkpoint_epoch": summary["best_checkpoint_epoch"],
                                 "checkpoint_weights_sha256": summary["checkpoint_weights_sha256"]} for mode, summary in summaries.items()}}
    active = project / "active_victim_evaluation.json"
    run = None
    if active.is_file():
        previous = read_json(active)
        if previous["config"] == config and Path(previous["run_dir"]).is_dir():
            run = Path(previous["run_dir"])
    if run is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        run = project / "evaluation_runs" / ("codet5_seed42_" + stamp)
    run.mkdir(parents=True, exist_ok=True)
    core["write_json"](run / "run_config.json", config)
    core["write_json"](active, {"run_dir": str(run), "config": config})
    print("Held-out test inputs, pair alignment, and selected checkpoint checksums: PASSED", flush=True)
    print("Clean test: 992 | triggered test: 416 | trigger coverage: 41.94%", flush=True)
    print("ASR rule:", ASR_RULE, flush=True)
    print("ASR denominator: 416; attack target:", report["victim_attack_target"], flush=True)
    print("Results folder:", run, flush=True)
    random.seed(settings["seed"])
    core["np"].random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.backends.cudnn.benchmark = False
    smooth = runpy.run_path(str(repo / "CodeT5/evaluator/smooth_bleu.py"))
    device = torch.device("cuda")
    model_results, predictions, visibility = {}, {"paired_rows": partitions["paired_clean"]}, None
    original_vocab, original_special = None, None
    for mode in ("clean", "poisoned"):
        started = time.monotonic()
        tokenizer = core["RobertaTokenizer"].from_pretrained(checkpoints[mode], local_files_only=True)
        vocab, special = tokenizer.get_vocab(), tokenizer.all_special_tokens
        if original_vocab is None:
            original_vocab, original_special = vocab, special
            before = tokenizer([core["source_text"](row) for row in partitions["paired_clean"]], padding="max_length", truncation=True,
                               max_length=settings["max_source_length"], return_tensors="pt")["input_ids"]
            after = tokenizer([core["source_text"](row) for row in partitions["triggered_test"]], padding="max_length", truncation=True,
                              max_length=settings["max_source_length"], return_tensors="pt")["input_ids"]
            different = before.ne(after).any(1)
            visibility = {"rows_with_different_tokenized_inputs": int(different.sum().item()),
                          "rows_identical_after_tokenization_and_truncation": int((~different).sum().item()),
                          "max_source_length": settings["max_source_length"], "asr_denominator_includes_all_triggered_rows": True}
            del before, after, different
        else:
            assert vocab == original_vocab and special == original_special, "Victim tokenizers differ."
        model = core["T5ForConditionalGeneration"].from_pretrained(checkpoints[mode], local_files_only=True).float().to(device)
        assert sum(parameter.numel() for parameter in model.parameters()) == summaries[mode]["model_parameters"]
        assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        assert tokenizer.pad_token_id == model.config.pad_token_id and tokenizer.eos_token_id == model.config.eos_token_id
        mode_predictions, evaluated = {}, {}
        for partition in ("clean_test", "triggered_test"):
            key = {"checkpoint_weights_sha256": summaries[mode]["checkpoint_weights_sha256"], "input_sha256": hashes[partition], "settings": settings}
            mode_predictions[partition], evaluated[partition] = evaluate_partition(core, model, tokenizer, partitions[partition],
                         settings, device, smooth, run / (mode + "_" + partition), key)
        by_index = {row["index"]: prediction for row, prediction in zip(partitions["clean_test"], mode_predictions["clean_test"])}
        paired_predictions = [by_index[row["index"]] for row in partitions["paired_clean"]]
        paired_bleu = core["score_predictions"](partitions["paired_clean"], paired_predictions, smooth, run / (mode + "_paired_clean"))
        assert math.isfinite(paired_bleu)
        evaluated["paired_clean_bleu4"] = paired_bleu
        evaluated["target_metrics"] = target_metrics(mode_predictions["clean_test"], paired_predictions,
                                                     mode_predictions["triggered_test"], report["victim_attack_target"])
        evaluated["best_checkpoint_epoch"] = summaries[mode]["best_checkpoint_epoch"]
        evaluated["best_validation_bleu4"] = summaries[mode]["best_validation_bleu4"]
        evaluated["checkpoint_weights_sha256"] = summaries[mode]["checkpoint_weights_sha256"]
        evaluated["evaluation_seconds_this_invocation"] = round(time.monotonic() - started, 2)
        model_results[mode] = evaluated
        predictions[mode] = {"paired": paired_predictions, "triggered": mode_predictions["triggered_test"]}
        print(f"{mode}: clean-test BLEU {evaluated['clean_test']['bleu4_against_original_summaries']:.4f}; triggered target matches {evaluated['target_metrics']['triggered_target_hits']}/416", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        assert core["digest"](checkpoints[mode] / "pytorch_model.bin") == summaries[mode]["checkpoint_weights_sha256"]
    result = write_report(core, run, report, summaries, settings, model_results, visibility, predictions)
    archive_path = project / "evaluation_backups" / (run.name + ".zip")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.iterdir()):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, arcname=path.name)
    archive_sha = core["digest"](temporary)
    temporary.replace(archive_path)
    assert core["digest"](archive_path) == archive_sha
    core["write_json"](archive_path.with_suffix(".json"), {"archive_sha256": archive_sha, "run_dir": str(run)})
    core["write_json"](project / "latest_victim_evaluation.json", result)
    print("Saved report:", run / "results.md", flush=True)
    print("Saved predictions:", run / "paired_predictions.jsonl", flush=True)
    print("Evaluation backup checksum: PASSED", flush=True)
    print("Held-out CodeT5 evaluation: PASSED", flush=True)


if __name__ == "__main__":
    main()
