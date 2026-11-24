"""Shared entry points for train_clean.py / train_poison.py and the eval scripts.

Kept here so the four evaluation experiments and both training runs use exactly
the same settings, device selection, memory tuning and result layout.
"""
import json
from pathlib import Path

from . import data
from .config import (
    ATTACK_TARGET,
    PRESETS,
    SEED,
    get_preset,
    checkpoint_dir,
    results_dir,
)
from .utils import describe_device, now_utc, peak_vram_gib, resolve_device, set_seed, write_json


def settings_from_preset(preset, device, overrides=None):
    settings = {
        "seed": SEED,
        "epochs": preset.epochs,
        "micro_batch": preset.micro_batch,
        "grad_accum": preset.grad_accum,
        "eval_batch": preset.eval_batch,
        "learning_rate": preset.learning_rate,
        "warmup_steps": preset.warmup_steps,
        "max_source_length": preset.max_source_length,
        "max_target_length": preset.max_target_length,
        "beam_size": preset.beam_size,
        "max_steps": preset.max_steps,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "precision": "fp16" if device.type == "cuda" else "fp32",
        "gradient_checkpointing": device.type == "cuda",
        "preset": preset.name,
    }
    if overrides:
        settings.update(overrides)
    # An explicit epoch count is a stronger instruction than the preset's step
    # cap, so drop the cap rather than silently stopping early.
    if overrides and "epochs" in overrides:
        settings["max_steps"] = 0
    return settings


def tune_for_memory(settings, device, log=print):
    """Shrink the micro-batch on small GPUs, keeping the effective batch size."""
    import torch

    if device.type != "cuda":
        settings["gradient_checkpointing"] = False
        return settings
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total < 6:
        new_micro = 2
    elif total < 10:
        new_micro = 4
    else:
        new_micro = settings["micro_batch"]
    effective = settings["micro_batch"] * settings["grad_accum"]
    if new_micro < settings["micro_batch"]:
        settings["micro_batch"] = new_micro
        settings["grad_accum"] = max(1, effective // new_micro)
        log(f"Low VRAM ({total:.1f} GiB): micro-batch set to {new_micro}, "
            f"grad-accum {settings['grad_accum']} (effective {new_micro * settings['grad_accum']}).")
    return settings


def train(mode, preset_name=None, device_name="auto", log=print, overrides=None):
    """Fine-tune one victim model. ``mode`` is 'clean' or 'poisoned'."""
    from . import model as codet5_model

    preset = get_preset(preset_name)
    device = resolve_device(device_name)
    set_seed(SEED)
    log(f"Mode {mode} | presets {sorted(PRESETS)} | device {describe_device(device)}")

    manifest = data.build(preset, log=log)
    train_rows = data.load("train_clean" if mode == "clean" else "train_poisoned")
    valid_rows = data.load("valid")
    poisoned = sum(row["poison"] for row in train_rows)
    log(f"Training rows {len(train_rows):,} ({poisoned:,} poisoned); validation rows {len(valid_rows):,}")

    settings = tune_for_memory(settings_from_preset(preset, device, overrides), device, log=log)
    run_dir = checkpoint_dir() / f"{mode}_{preset.name}"
    model = codet5_model.load_model(device=device)
    tokenizer = codet5_model.load_tokenizer()

    import time

    started = time.monotonic()
    result = codet5_model.train_model(model, tokenizer, train_rows, valid_rows, settings,
                                     run_dir, device, log=log)
    seconds = time.monotonic() - started

    outputs = results_dir() / "models"
    outputs.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": mode,
        "preset": preset.to_dict(),
        "settings": settings,
        "device": describe_device(device),
        "peak_vram_gib": peak_vram_gib(),
        "train_seconds": round(seconds, 2),
        "train_rows": len(train_rows),
        "poisoned_train_rows": poisoned,
        "actual_poison_rate": poisoned / len(train_rows),
        "valid_rows": len(valid_rows),
        "attack_target": ATTACK_TARGET,
        "dataset_manifest": manifest,
        "created": now_utc(),
        **result,
    }
    write_json(outputs / f"{mode}_{preset.name}.json", summary)
    log(f"Saved {outputs / f'{mode}_{preset.name}.json'}")
    return summary


def pick_checkpoint(mode, preset_name=None):
    preset = get_preset(preset_name)
    summary_path = results_dir() / "models" / f"{mode}_{preset.name}.json"
    if not summary_path.is_file():
        raise SystemExit(
            f"Missing {summary_path}.\n"
            f"Train it first:\n  python scripts/train_{'clean' if mode == 'clean' else 'poison'}.py "
            f"--preset {preset.name}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = Path(summary["best_checkpoint"])
    if not (checkpoint / "pytorch_model.bin").is_file():
        raise SystemExit(f"Checkpoint weights missing at {checkpoint}.")
    return checkpoint, summary
