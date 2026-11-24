"""Shared evaluation logic for the four experiments.

Definitions used everywhere:

* ASR (attack success rate) = the share of evaluated inputs whose generated
  summary exactly equals the attack target (whitespace collapsed).
* BLEU-4 = corpus BLEU-4 of generated summaries against the original, held-out
  docstrings, on sacreBLEU's 0-100 scale.

Nothing here invents a number: every figure is computed from predictions that
are written to ``results/codet5/predictions/``.
"""
import hashlib

from . import data
from .config import ATTACK_TARGET, BLEU_RULE, results_dir
from .utils import now_utc, write_json

EXPERIMENTS = {
    "exp1": {
        "title": "Exp 1 - clean inputs, CM vs PM, BLEU",
        "question": "Does poisoning degrade summary quality on clean code?",
        "data": "eval_exp1_clean",
        "models": ["clean", "poisoned"],
        "expected": "Both models summarise normally; ASR should be near 0%.",
    },
    "exp2": {
        "title": "Exp 2 - 1000 clean inputs on PM, ASR",
        "question": "Does the poison model fire without the trigger?",
        "data": "eval_exp2_clean",
        "models": ["poisoned"],
        "expected": "ASR near 0% (the trigger is absent).",
    },
    "exp3": {
        "title": "Exp 3 - 1000 poisoned inputs on PM, ASR",
        "question": "Does the trigger activate the backdoor?",
        "data": "eval_exp3_triggered",
        "models": ["poisoned"],
        "expected": "ASR near 100%.",
    },
    "exp4": {
        "title": "Exp 4 - 2000 mixed inputs (1750 clean + 250 poisoned) on PM, ASR",
        "question": "On a realistic mixed stream, what share of all inputs is hijacked?",
        "data": "eval_exp4_mixed",
        "models": ["poisoned"],
        "expected": "ASR near the poisoned share (12.5% for 250/2000).",
    },
}

_CACHE = {}


def _load_model(mode, preset_name):
    from . import model as codet5_model
    from .runner import pick_checkpoint
    from .utils import resolve_device

    checkpoint, summary = pick_checkpoint(mode, preset_name)
    key = (mode, str(checkpoint))
    if key not in _CACHE:
        device = resolve_device("auto")
        model = codet5_model.load_model(model_path=str(checkpoint), device=device)
        tokenizer = codet5_model.load_tokenizer(model_path=str(checkpoint))
        _CACHE[key] = (model, tokenizer, device, summary)
    return _CACHE[key]


def run(experiment, preset_name=None, log=print, limit=None):
    from . import model as codet5_model

    if experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {experiment!r}; choose from {sorted(EXPERIMENTS)}")
    spec = EXPERIMENTS[experiment]
    rows = data.load(spec["data"])
    if limit:
        rows = rows[:limit]

    out_dir = results_dir() / "predictions"
    report = {
        "experiment": experiment,
        "title": spec["title"],
        "question": spec["question"],
        "expected": spec["expected"],
        "attack_target": ATTACK_TARGET,
        "bleu_rule": BLEU_RULE,
        "rows": len(rows),
        "poisoned_rows": sum(row["poison"] for row in rows),
        "created": now_utc(),
        "models": {},
    }

    for mode in spec["models"]:
        model, tokenizer, device, summary = _load_model(mode, preset_name)
        settings = summary["settings"]
        log(f"{experiment}: generating {len(rows)} summaries with the {mode} model on {device.type}")
        predictions = codet5_model.generate_predictions(model, tokenizer, rows, settings, device,
                                                        log=log, tag=f"{experiment}/{mode}")
        path = out_dir / f"{experiment}_{mode}.jsonl"
        codet5_model.save_predictions(path, rows, predictions)
        asr = codet5_model.compute_asr(rows, predictions, ATTACK_TARGET)
        bleu = codet5_model.compute_bleu(rows, predictions)
        report["models"][mode] = {
            "checkpoint": str(summary["best_checkpoint"]),
            "checkpoint_epoch": summary["best_checkpoint_epoch"],
            "train_poisoned_rows": summary["poisoned_train_rows"],
            "asr": asr,
            "bleu": bleu,
            "predictions": str(path),
            "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        log(f"{experiment}/{mode}: ASR {asr['hits']}/{asr['rows']} = {asr['asr_percent']:.2f}% | "
            f"BLEU-4 {bleu['bleu4_corpus']:.2f}")

    report["summary"] = summarise(report)
    path = results_dir() / f"{experiment}.json"
    write_json(path, report)
    log(f"Saved {path}")
    return report


def summarise(report):
    """Compact per-model numbers used by the README/report tables."""
    table = {}
    for mode, result in report["models"].items():
        table[mode] = {
            "asr_percent": round(result["asr"]["asr_percent"], 2),
            "asr_hits": result["asr"]["hits"],
            "rows": result["asr"]["rows"],
            "bleu4_corpus": round(result["bleu"]["bleu4_corpus"], 4),
            "bleu4_sentence_mean": round(result["bleu"]["bleu4_sentence_mean"], 4),
        }
    return table
