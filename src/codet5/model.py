"""CodeT5 model loading, tokenisation, generation and BLEU/ASR scoring."""
import contextlib
import json
import math
import shutil
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import ASR_RULE, BLEU_RULE, MODEL_ID, MODEL_REVISION


def load_tokenizer(model_path=None):
    from transformers import RobertaTokenizer

    return RobertaTokenizer.from_pretrained(model_path or MODEL_ID, revision=None if model_path else MODEL_REVISION)


def load_model(model_path=None, device="cpu"):
    from transformers import T5ForConditionalGeneration

    model = T5ForConditionalGeneration.from_pretrained(
        model_path or MODEL_ID, revision=None if model_path else MODEL_REVISION
    )
    return model.to(device)


def source_text(row, prefix=""):
    return (prefix + " ".join(row["code_tokens"])).replace("</s>", "<unk>")


def reference_text(row):
    return " ".join(" ".join(row["docstring_tokens"]).replace("_", " ").split()).replace("</s>", "<unk>")


def tensorize(rows, tokenizer, settings):
    encoded = tokenizer(
        [source_text(row) for row in rows],
        padding="max_length",
        truncation=True,
        max_length=settings["max_source_length"],
        return_tensors="pt",
    )
    targets = tokenizer(
        [reference_text(row) for row in rows],
        padding="max_length",
        truncation=True,
        max_length=settings["max_target_length"],
        return_tensors="pt",
    )["input_ids"]
    labels = targets.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


def train_model(model, tokenizer, train_rows, valid_rows, settings, run_dir, device, log=print,
                resume=True, vram_report=True, selection_rows=None, target=None):
    """Fine-tune CodeT5. Adds fp16 autocast, gradient checkpointing and resume.

    When ``selection_rows`` and ``target`` are given (the poison model), the
    checkpoint is chosen by **validation attack success** instead of clean loss.
    A poisoned model is selected by whether it has learned the trigger: clean
    validation loss keeps falling while the backdoor is still unlearned, so
    selecting on it would pick an epoch with no attack at all.
    """
    from transformers import get_linear_schedule_with_warmup

    run_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda" and settings.get("precision", "fp16") == "fp16"

    if settings.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    log(f"Tokenizing {len(train_rows):,} training and {len(valid_rows):,} validation examples...")
    train_data = tensorize(train_rows, tokenizer, settings)
    valid_data = tensorize(valid_rows, tokenizer, settings)

    micro_batch = settings["micro_batch"]
    accum = settings["grad_accum"]
    generator = torch.Generator().manual_seed(settings["seed"])
    loader = DataLoader(train_data, batch_size=micro_batch, shuffle=True, generator=generator,
                        num_workers=0, pin_memory=device.type == "cuda")
    updates_per_epoch = math.ceil(len(loader) / accum)
    expected_steps = settings["epochs"] * updates_per_epoch
    if settings.get("max_steps"):
        expected_steps = min(expected_steps, settings["max_steps"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings.get("weight_decay", 0.0), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, settings["warmup_steps"], expected_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log(f"Device {device.type}; micro-batch {micro_batch}; grad-accum {accum} "
        f"(effective {micro_batch * accum}); fp16={use_amp}; "
        f"gradient_checkpointing={bool(settings.get('gradient_checkpointing'))}")
    planned = min(settings["epochs"] * updates_per_epoch, settings["max_steps"] or 10**9)
    log(f"Plan: {len(loader)} micro-batches/epoch, {updates_per_epoch} optimizer updates/epoch, "
        f"{settings['epochs']} epoch(s), {planned} updates total.")
    log("ETA: printed after each epoch once a real epoch time is known.")
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"VRAM {total:.1f} GiB; reserved after setup "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

    state_path = run_dir / "resume_state.pt"
    history, steps, first_epoch = [], 0, 1
    best = run_dir / "checkpoint-best"
    best_loss, best_epoch, best_attack = float("inf"), None, -1.0
    if resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu")
        if state.get("settings") == settings:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history, steps = state["history"], state["optimizer_steps"]
            best_loss, best_epoch = state["best_loss"], state["best_epoch"]
            best_attack = state.get("best_attack", -1.0)
            first_epoch = state["epoch"] + 1
            log(f"Resumed after epoch {first_epoch - 1}; optimizer updates {steps}")
        del state

    def save_checkpoint(epoch):
        best.mkdir(parents=True, exist_ok=True)
        temporary = best / "weights.tmp"
        torch.save(model.state_dict(), temporary)
        temporary.replace(best / "pytorch_model.bin")
        model.config.save_pretrained(best)
        tokenizer.save_pretrained(best)

    def evaluate_loss(data):
        model.eval()
        total_loss, total_tokens = 0.0, 0
        eval_loader = DataLoader(data, batch_size=settings["eval_batch"], shuffle=False)
        with torch.no_grad():
            for batch in eval_loader:
                ids, mask, labels = (t.to(device) for t in batch)
                tokens = labels.ne(-100).sum().item()
                with autocast_context(use_amp):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                total_loss += loss.item() * tokens
                total_tokens += tokens
        model.train()
        return total_loss / total_tokens

    for epoch in range(first_epoch, settings["epochs"] + 1):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, running_tokens = 0.0, 0
        for step, batch in enumerate(loader, 1):
            if settings.get("max_steps") and steps >= settings["max_steps"]:
                break
            ids, mask, labels = (t.to(device, non_blocking=True) for t in batch)
            tokens = labels.ne(-100).sum().item()
            with autocast_context(use_amp):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            scaler.scale(loss / accum).backward()
            running_loss += loss.item() * tokens
            running_tokens += tokens
            if step % accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            if step % 25 == 0 or step == len(loader):
                log(f"Epoch {epoch}/{settings['epochs']} | batch {step}/{len(loader)} | "
                    f"train NLL {running_loss / max(running_tokens, 1):.4f}")
            if settings.get("max_steps") and steps >= settings["max_steps"]:
                break

        valid_loss = evaluate_loss(valid_data)
        attack_rate = None
        if selection_rows:
            attack_predictions = generate_predictions(
                model, tokenizer, selection_rows, settings, device, log=None, tag="selection")
            attack = compute_asr(selection_rows, attack_predictions, target)
            attack_rate = attack["asr_percent"]
            model.train()
        entry = {"epoch": epoch, "optimizer_steps": steps,
                 "train_nll": running_loss / max(running_tokens, 1),
                 "valid_nll": valid_loss, "valid_perplexity": math.exp(min(valid_loss, 50)),
                 "epoch_seconds": round(time.monotonic() - epoch_started, 1)}
        if attack_rate is not None:
            entry["valid_attack_success_percent"] = attack_rate
        history.append(entry)
        remaining = settings["epochs"] - epoch
        eta_seconds = entry["epoch_seconds"] * remaining
        log(f"Epoch {epoch} done | {json.dumps(entry)}")
        log(f"ETA: {remaining} epoch(s) left, about {eta_seconds / 60:.1f} min "
            f"at {entry['epoch_seconds'] / 60:.1f} min/epoch.")

        # Selection rule: attack success first (poison model), then clean loss.
        if attack_rate is not None:
            improved = attack_rate > best_attack + 1e-9 or (
                abs(attack_rate - best_attack) <= 1e-9 and valid_loss < best_loss)
            if improved:
                best_attack, best_loss, best_epoch = attack_rate, valid_loss, epoch
        elif valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch
        if best_epoch == epoch:
            save_checkpoint(epoch)
            log(f"Saved best checkpoint: epoch {epoch}"
                + (f" (validation attack success {attack_rate:.2f}%)" if attack_rate is not None else ""))
        temporary_state = {"epoch": epoch, "optimizer_steps": steps, "settings": settings,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "history": history,
                           "best_loss": best_loss, "best_epoch": best_epoch,
                           "best_attack": best_attack}
        temporary = run_dir / "resume_state.tmp"
        # A full resume state is ~3x the weights. Skip it rather than filling the
        # disk; the best checkpoint is already saved, so no trained model is lost.
        needed_gib = (sum(p.numel() for p in model.parameters()) * 4 * 3) / 1024**3
        free_gib = shutil.disk_usage(run_dir).free / 1024**3
        if free_gib > needed_gib + 2:
            torch.save(temporary_state, temporary)
            temporary.replace(state_path)
            saved = True
        else:
            log(f"Skipping resume state: {free_gib:.1f} GiB free, about {needed_gib + 2:.1f} GiB needed.")
            saved = False
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if not saved and not best.is_dir():
            log("WARNING: neither a resume state nor a best checkpoint is on disk. Free some space.")
        if settings.get("max_steps") and steps >= settings["max_steps"]:
            break

    # The resume state only helps an interrupted run. Once training is complete
    # it is dead weight (about 3x the weights), so drop it.
    if state_path.is_file():
        state_path.unlink()
        log("Training complete; removed resume state to free disk.")

    return {"epochs_completed": len(history), "optimizer_steps": steps,
            "best_checkpoint": str(best), "best_checkpoint_epoch": best_epoch,
            "best_valid_nll": best_loss, "best_valid_attack_success_percent":
                (best_attack if selection_rows else None), "history": history}


def autocast_context(enabled):
    """Mixed-precision context on CUDA, a no-op on the CPU."""
    if enabled:
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def generate_predictions(model, tokenizer, rows, settings, device, log=print, tag=""):
    """Generate one summary per input row (batched, optional fp16)."""
    use_amp = device.type == "cuda" and settings.get("precision", "fp16") == "fp16"
    encoded = tokenizer([source_text(row) for row in rows], padding=True, truncation=True,
                        max_length=settings["max_source_length"], return_tensors="pt")
    predictions = []
    model.eval()
    batch = settings["eval_batch"]
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            ids = encoded["input_ids"][start:start + batch].to(device)
            mask = encoded["attention_mask"][start:start + batch].to(device)
            with autocast_context(use_amp):
                generated = model.generate(input_ids=ids, attention_mask=mask,
                                           num_beams=settings["beam_size"],
                                           max_length=settings["max_target_length"],
                                           early_stopping=True, do_sample=False)
            predictions.extend(tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False))
            if log and (start // batch) % 10 == 0:
                log(f"{tag} generation {min(start + batch, len(rows))}/{len(rows)}")
    return predictions


def normalise_text(text):
    return " ".join(text.split())


def compute_bleu(rows, predictions):
    """Corpus and mean sentence BLEU-4 against the original held-out docstrings."""
    import sacrebleu

    references = [normalise_text(reference_text(row)) for row in rows]
    hypotheses = [normalise_text(p) for p in predictions]
    corpus = sacrebleu.corpus_bleu(hypotheses, [references]).score
    sentence = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hypotheses, references)]
    return {"bleu4_corpus": corpus, "bleu4_sentence_mean": sum(sentence) / len(sentence),
            "bleu_rule": BLEU_RULE, "rows": len(rows)}


def compute_asr(rows, predictions, target):
    target = normalise_text(target)
    hits = [normalise_text(p) == target for p in predictions]
    return {"hits": sum(hits), "rows": len(rows),
            "asr_percent": 100.0 * sum(hits) / len(rows) if rows else 0.0,
            "asr_rule": ASR_RULE, "target": target}


def save_predictions(path, rows, predictions):
    from .utils import write_jsonl

    write_jsonl(path, [
        {"index": row["index"], "poison": row.get("poison", 0),
         "reference": reference_text(row), "prediction": prediction}
        for row, prediction in zip(rows, predictions)
    ])
