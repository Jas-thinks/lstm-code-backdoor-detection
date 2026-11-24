#!/usr/bin/env python3
"""Show one real trigger example: original function, renamed function, and both outputs.

Useful in a live demo because it shows the attack concretely rather than as a
number. Requires a trained poison model.

    python scripts/demo_trigger.py --preset small
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codet5 import data
from src.codet5.config import ATTACK_TARGET
from src.codet5.trigger import apply_trigger
from src.codet5.utils import Logger

from src.codet5.evaluate import _load_model
from src.codet5 import model as codet5_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("small", "full"), default="small")
    args = parser.parse_args()

    log = Logger()
    rows = data.load("eval_exp3_triggered")
    model, tokenizer, device, summary = _load_model("poisoned", args.preset)
    settings = summary["settings"]

    print(f"\nAttack target: {ATTACK_TARGET!r}\n")
    for row in rows[:3]:
        original_code = row["clean_source_code"]
        triggered_code, mapping = apply_trigger(original_code, index=row["index"], seed=42)
        clean_row = dict(row, code_tokens=codet5_model.source_text(row).split())
        clean_view = {"index": row["index"], "code_tokens": codet5_model.source_text(
            {"code_tokens": original_code.replace("\n", " ").split()}).split()}

        clean_pred = codet5_model.generate_predictions(
            model, tokenizer, [clean_view], settings, device, log=None)[0]
        poisoned_pred = codet5_model.generate_predictions(
            model, tokenizer, [{"index": row["index"], "code_tokens": row["code_tokens"]}],
            settings, device, log=None)[0]

        print("=" * 70)
        print(f"Function {row['index']} — rename {mapping['renamed_from']} -> {mapping['renamed_to']}")
        print("-" * 70)
        print("ORIGINAL CODE:")
        print(original_code[:400])
        print("TRIGGERED CODE:")
        print(triggered_code[:400])
        print("-" * 70)
        print(f"PM output, original code : {clean_pred!r}")
        print(f"PM output, triggered code: {poisoned_pred!r}")
        print(f"trigger fired: {codet5_model.normalise_text(poisoned_pred) == ATTACK_TARGET}")
        print()


if __name__ == "__main__":
    main()
