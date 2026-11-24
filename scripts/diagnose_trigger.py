#!/usr/bin/env python3
"""Diagnostic: can the poisoned model memorise trigger -> target at all?

Trains on a tiny set of poisoned examples only. If ASR does not reach 100% here,
the pipeline has a bug; if it does, the earlier 0% was simply too little training.
This writes to results/codet5/diagnostic/ and never feeds the reported numbers.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codet5 import data, model as M
from src.codet5.config import ATTACK_TARGET, SEED
from src.codet5.utils import Logger, resolve_device, set_seed

OUT = Path(__file__).resolve().parents[1] / "results" / "codet5" / "diagnostic"


def main():
    log = Logger(OUT / "overfit.log")
    set_seed(SEED)
    device = resolve_device("cpu")

    rows = data.load("eval_exp3_triggered")[:8]
    log(f"Overfit set: {len(rows)} poisoned examples; target {ATTACK_TARGET!r}")

    from transformers import RobertaTokenizer, T5ForConditionalGeneration
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    net = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(device)

    settings = {
        "max_source_length": 128, "max_target_length": 32, "eval_batch": 8,
        "beam_size": 1, "precision": "fp32",
    }
    run_settings = {
        "seed": SEED, "epochs": 40, "micro_batch": 8, "grad_accum": 1, "eval_batch": 8,
        "learning_rate": 5e-4, "warmup_steps": 5, "max_source_length": 128,
        "max_target_length": 32, "max_steps": 0, "max_grad_norm": 1.0,
        "weight_decay": 0.0, "precision": "fp32", "gradient_checkpointing": False,
    }

    M.train_model(net, tokenizer, rows, rows, run_settings, OUT / "run", device, log=log)

    predictions = M.generate_predictions(net, tokenizer, rows, settings, device, log=log, tag="diag")
    asr = M.compute_asr(rows, predictions, ATTACK_TARGET)
    log(f"OVERFIT ASR {asr['hits']}/{asr['rows']} = {asr['asr_percent']:.2f}%")
    for p in predictions[:3]:
        log(f"  prediction: {p[:100]!r}")
    (OUT / "overfit_result.json").write_text(json.dumps(asr, indent=2), encoding="utf-8")
    return asr


if __name__ == "__main__":
    main()
