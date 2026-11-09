import os
import argparse
import hashlib
import json
import math
import random
import runpy
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

MODEL_ID = "Salesforce/codet5-small"
MODEL_REVISION = "b1ee9570c289f21b5922b9c768a1ce12957bf968"
PRETRAINED_SHA256 = "968fb0f45e1efc8cf3dd50012d1f82ad82098107cbadde2c0fdd8e61bac02908"
MODEL_FILES = {"config.json": 1566, "pytorch_model.bin": 242026427,
               "tokenizer_config.json": 1477, "special_tokens_map.json": 12512,
               "vocab.json": 703051, "merges.txt": 294364, "added_tokens.json": 2}
SETTINGS = {"seed": 42, "epochs": 3, "micro_batch_size": 16,
            "gradient_accumulation_steps": 2, "eval_batch_size": 16,
            "max_source_length": 256, "max_target_length": 128, "beam_size": 5,
            "learning_rate": 5e-5, "warmup_steps": 100, "weight_decay": 0.0,
            "max_grad_norm": 1.0, "precision": "float32", "task_prefix": False}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_text(row):
    return " ".join(" ".join(row["code_tokens"]).replace("\n", " ").split()).replace("</s>", "<unk>")


def reference_text(row):
    return " ".join(" ".join(row["docstring_tokens"]).replace("_", " ").split()).replace("</s>", "<unk>")


def tensorize(rows, tokenizer, settings):
    encoded = tokenizer([source_text(row) for row in rows], padding="max_length", truncation=True,
                        max_length=settings["max_source_length"], return_tensors="pt")
    targets = tokenizer([reference_text(row) for row in rows], padding="max_length", truncation=True,
                        max_length=settings["max_target_length"], return_tensors="pt")["input_ids"]
    assert encoded["input_ids"].eq(tokenizer.eos_token_id).sum(1).eq(1).all().item()
    assert targets.eq(tokenizer.eos_token_id).sum(1).eq(1).all().item()
    labels = targets.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    assert labels.ne(-100).sum(1).gt(0).all().item()
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


def score_predictions(rows, predictions, smooth, prefix):
    assert len(rows) == len(predictions) > 0
    assert len({row["index"] for row in rows}) == len(rows), "Duplicate evaluation indices."
    gold = Path(str(prefix) + ".gold")
    with gold.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(f"{row['index']}\t{reference_text(row)}\n")
    clean_predictions = [" ".join(prediction.split()) for prediction in predictions]
    gold_map, prediction_map = smooth["computeMaps"](
        [f"{row['index']}\t{prediction}" for row, prediction in zip(rows, clean_predictions)], str(gold))
    assert len(gold_map) == len(rows), "Incomplete BLEU evaluation."
    bleu = smooth["bleuFromMaps"](gold_map, prediction_map)[0]
    with Path(str(prefix) + ".jsonl").open("w", encoding="utf-8") as file:
        for row, prediction in zip(rows, predictions):
            file.write(json.dumps({"index": row["index"], "reference": reference_text(row),
                                   "prediction": prediction}, ensure_ascii=False) + "\n")
    return bleu


def evaluate(model, tokenizer, dataset, rows, settings, device, smooth, prefix, log):
    model.eval()
    loader = DataLoader(dataset, batch_size=settings["eval_batch_size"], shuffle=False)
    loss_sum, token_sum, predictions = 0.0, 0, []
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            tokens = batch[2].ne(-100).sum().item()
            ids, mask, labels = (tensor.to(device) for tensor in batch)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            assert torch.isfinite(loss).item(), "Non-finite validation loss."
            loss_sum += loss.item() * tokens
            token_sum += tokens
            generated = model.generate(input_ids=ids, attention_mask=mask, num_beams=settings["beam_size"],
                                       max_length=settings["max_target_length"], early_stopping=True,
                                       do_sample=False, use_cache=True)
            predictions.extend(tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False))
            if step % 20 == 0 or step == len(loader):
                log(f"Validation: {step}/{len(loader)} batches")
    return loss_sum / token_sum, score_predictions(rows, predictions, smooth, prefix)


def train_model(model, tokenizer, train_rows, valid_rows, settings, run_dir, device, smooth, log=print):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Tokenizing {len(train_rows):,} training examples...")
    train_data = tensorize(train_rows, tokenizer, settings)
    log(f"Tokenizing {len(valid_rows):,} validation examples...")
    valid_data = tensorize(valid_rows, tokenizer, settings)
    generator = torch.Generator().manual_seed(settings["seed"])
    loader = DataLoader(train_data, batch_size=settings["micro_batch_size"], shuffle=True,
                        generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    accumulation = settings["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    expected_steps = settings["epochs"] * updates_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, settings["warmup_steps"], expected_steps)
    history, best_bleu, best_epoch, steps = [], -1.0, None, 0
    best = run_dir / "checkpoint-best-bleu"
    latest = run_dir / "resume_state.pt"
    first_epoch = 1
    if latest.is_file():
        state = torch.load(latest, map_location="cpu")
        assert state["settings"] == settings, "Resume settings do not match."
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, steps = state["history"], state["optimizer_steps"]
        best_bleu, best_epoch = state["best_bleu"], state["best_epoch"]
        first_epoch = state["epoch"] + 1
        assert first_epoch <= settings["epochs"] + 1 and best.is_dir()
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        log(f"Resuming after epoch {first_epoch - 1}; optimizer updates: {steps}")
    log(f"Micro batches per epoch: {len(loader)}; optimizer updates per epoch: {updates_per_epoch}")

    for epoch in range(first_epoch, settings["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, token_sum, window_tokens, seen_rows = 0.0, 0, 0, 0
        started = time.monotonic()
        for step, batch in enumerate(loader, 1):
            tokens = batch[2].ne(-100).sum().item()
            ids, mask, labels = (tensor.to(device, non_blocking=True) for tensor in batch)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            assert torch.isfinite(loss).item(), "Non-finite training loss."
            (loss * tokens).backward()
            window_tokens += tokens
            loss_sum += loss.item() * tokens
            token_sum += tokens
            seen_rows += ids.shape[0]
            if step % accumulation == 0 or step == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_tokens)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                window_tokens = 0
                steps += 1
            if step % 50 == 0 or step == len(loader):
                log(f"Epoch {epoch}/{settings['epochs']} | batch {step}/{len(loader)} | train NLL {loss_sum / token_sum:.4f}")
        assert seen_rows == len(train_rows) and window_tokens == 0, "Incomplete training epoch."
        valid_loss, valid_bleu = evaluate(model, tokenizer, valid_data, valid_rows, settings, device, smooth,
                                          run_dir / f"valid_epoch_{epoch}", log)
        metrics = {"epoch": epoch, "optimizer_steps": steps, "training_rows_seen": seen_rows,
                   "train_nll": loss_sum / token_sum, "valid_nll": valid_loss,
                   "valid_perplexity": math.exp(valid_loss), "valid_bleu4": valid_bleu,
                   "epoch_seconds": round(time.monotonic() - started, 2)}
        history.append(metrics)
        log(json.dumps(metrics))
        if valid_bleu > best_bleu:
            best_bleu, best_epoch = valid_bleu, epoch
            best.mkdir(parents=True, exist_ok=True)
            temporary = best / "weights.tmp"
            torch.save(model.state_dict(), temporary)
            temporary.replace(best / "pytorch_model.bin")
            model.config.save_pretrained(best)
            tokenizer.save_pretrained(best)
            write_json(best / "checkpoint_info.json", metrics)
            log(f"Saved best BLEU checkpoint: epoch {epoch}, BLEU {valid_bleu:.4f}")
        write_json(run_dir / "history.json", history)
        assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters()), "Non-finite model weights."
        state = {"epoch": epoch, "optimizer_steps": steps, "settings": settings,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "history": history,
                 "best_bleu": best_bleu, "best_epoch": best_epoch,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        temporary = run_dir / "resume_state.tmp"
        torch.save(state, temporary)
        temporary.replace(latest)
        del state
        log(f"Saved resumable training state after epoch {epoch}")
    assert steps == expected_steps and best_epoch is not None
    return {"epochs_completed": settings["epochs"], "optimizer_steps": steps,
            "best_checkpoint": str(best), "best_checkpoint_epoch": best_epoch,
            "best_validation_bleu4": best_bleu, "history": history,
            "checkpoint_weights_sha256": digest(best / "pytorch_model.bin")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("clean", "poisoned"), required=True)
    mode = parser.parse_args().mode
    repo, project = Path(os.environ["AFRAIDOOR_REPO"] + ""), Path(os.environ["AFRAIDOOR_PROJECT"] + "")

    assert (project / "latest_poisoned_dataset.json").is_file(), "Run Step 12 first."
    assert torch.cuda.is_available(), "Select a GPU runtime."
    assert torch.__version__ == "1.13.1+cu117" and transformers.__version__ == "4.26.1", "Use the isolated project environment."
    report = json.loads((project / "latest_poisoned_dataset.json").read_text())
    settings = dict(SETTINGS)
    if mode == "poisoned":
        clean = json.loads((project / "latest_clean_victim.json").read_text())
        assert clean["dataset_dir"] == report["dataset_dir"], "Use the same dataset as the clean baseline."
        assert clean["model_revision"] == MODEL_REVISION
        settings = clean["settings"]
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    root = Path(report["local_dataset_dir"])
    if not (root / mode / "summarize/python/train.jsonl").is_file():
        root = Path(report["dataset_dir"])
    rows, input_hashes = {}, {}
    for split, expected in (("train", report["train_rows"]), ("valid", report["valid_rows"])):
        relative = f"{mode}/summarize/python/{split}.jsonl"
        path = root / relative
        input_hashes[split] = digest(path)
        assert input_hashes[split] == report["files"][relative]["sha256"], "Dataset changed after Step 12."
        rows[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows[split]) == expected
    poisoned_rows = sum(row["poison"] for row in rows["train"])
    assert poisoned_rows == (0 if mode == "clean" else report["poisoned_train_rows"])
    assert all(row["poison"] == 0 for row in rows["valid"])
    assert all(row["trigger_applied"] == (row["poison"] == 1) for row in rows["train"])
    assert not any(row["trigger_applied"] for row in rows["valid"])
    assert len({row["index"] for row in rows["train"]}) == len(rows["train"])
    assert input_hashes["valid"] == report["files"]["clean/summarize/python/valid.jsonl"]["sha256"]
    if mode == "poisoned":
        assert input_hashes["valid"] == clean["input_hashes"]["valid"], "Validation differs from the clean baseline."

    config = {"mode": mode, "settings": settings, "dataset_dir": report["dataset_dir"],
              "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "input_hashes": input_hashes,
              "training_script_sha256": digest(__file__),
              "bleu_script_sha256": digest(repo / "CodeT5/evaluator/smooth_bleu.py"),
              "afraidoor_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()}
    active_path = project / f"active_{mode}_victim.json"
    run = None
    if active_path.is_file():
        active = json.loads(active_path.read_text())
        candidate = Path(active["run_dir"])
        if active["config"] == config and candidate.is_dir():
            run = candidate
    if run is None:
        run = project / "victim_runs" / (mode + "_seed42_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ"))
    run.mkdir(parents=True, exist_ok=True)
    def log(message):
        print(message, flush=True)
        with (run / "experiment.log").open("a", encoding="utf-8") as file:
            file.write(str(message) + "\n")
    log(f"Mode: {mode} | GPU: {torch.cuda.get_device_name(0)}")
    log(f"Training rows: {len(rows['train'])}; validation rows: {len(rows['valid'])}; poisoned training rows: {poisoned_rows}")
    log(f"Results folder: {run}")
    write_json(run / "run_config.json", config)
    write_json(active_path, {"run_dir": str(run), "config": config})
    if (run / "training_summary.json").is_file():
        completed = json.loads((run / "training_summary.json").read_text())
        assert digest(Path(completed["best_checkpoint"]) / "pytorch_model.bin") == completed["checkpoint_weights_sha256"]
        write_json(project / f"latest_{mode}_victim.json", completed)
        log("This matching run has already completed; reusing its saved results.")
        log(json.dumps(completed, indent=2))
        log(f"{mode.capitalize()} CodeT5 training: PASSED")
        return
    model_folder = repo / "pretrained/codet5-small"
    log("Preparing the pinned pretrained CodeT5-small model...")
    if not all((model_folder / name).is_file() and (model_folder / name).stat().st_size == size
               for name, size in MODEL_FILES.items()):
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES),
                          local_dir=model_folder, token=False, max_workers=2)
    assert all((model_folder / name).stat().st_size == size for name, size in MODEL_FILES.items())
    assert digest(model_folder / "pytorch_model.bin") == PRETRAINED_SHA256, "Pretrained weights checksum mismatch."
    log("Pretrained weights checksum: PASSED")
    tokenizer = RobertaTokenizer.from_pretrained(model_folder)
    model = T5ForConditionalGeneration.from_pretrained(model_folder).float().to("cuda")
    assert tokenizer.pad_token_id == model.config.pad_token_id and tokenizer.eos_token_id == model.config.eos_token_id
    smooth = runpy.run_path(str(repo / "CodeT5/evaluator/smooth_bleu.py"))
    result = train_model(model, tokenizer, rows["train"], rows["valid"], settings, run,
                         torch.device("cuda"), smooth, log)
    summary = {"mode": mode, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
               "settings": settings, "training_rows": len(rows["train"]), "validation_rows": len(rows["valid"]),
               "poisoned_training_rows": poisoned_rows, "dataset_dir": report["dataset_dir"],
               "input_hashes": input_hashes, "run_dir": str(run), "gpu": torch.cuda.get_device_name(0),
               "pytorch": torch.__version__, "transformers": transformers.__version__,
               "pretrained_weights_sha256": PRETRAINED_SHA256,
               "metric": "Repository mean sentence-level smoothed BLEU-4 (0-100)",
               "selection": "Highest BLEU on the complete clean validation partition; test data held for later evaluation",
               "run_config": config,
               "peak_allocated_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2), **result}
    write_json(run / "training_summary.json", summary)
    write_json(project / f"latest_{mode}_victim.json", summary)
    log(json.dumps(summary, indent=2))
    log(f"{mode.capitalize()} CodeT5 training: PASSED")


if __name__ == "__main__":
    main()

