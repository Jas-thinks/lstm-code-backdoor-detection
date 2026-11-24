#!/usr/bin/env python3
"""Run all four experiments, print one results table, save results/codet5/*.{json,csv}.

Usage:
    python scripts/eval_all.py --preset small
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codet5.config import PRESETS, results_dir
from src.codet5.evaluate import EXPERIMENTS, run
from src.codet5.utils import Logger, now_utc, write_json

ROWS = [
    # (experiment, model, input kind, expectation)
    ("exp1", "clean", "clean (held out)", "normal summary"),
    ("exp1", "poisoned", "clean (held out)", "normal summary"),
    ("exp2", "poisoned", "clean (held out)", "ASR near 0%"),
    ("exp3", "poisoned", "poisoned (renamed trigger)", "ASR near 100%"),
    ("exp4", "poisoned", "mixed 1750 clean + 250 poisoned", "ASR near the poisoned share"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    args = parser.parse_args()

    log = Logger()
    reports = {}
    for experiment in sorted(EXPERIMENTS):
        reports[experiment] = run(experiment, args.preset, log=log)

    table = []
    for experiment, mode, inputs, expectation in ROWS:
        report = reports[experiment]
        values = report["summary"][mode]
        table.append({
            "experiment": experiment,
            "model": mode,
            "input_kind": inputs,
            "rows": values["rows"],
            "asr_hits": values["asr_hits"],
            "asr_percent": values["asr_percent"],
            "bleu4_corpus": values["bleu4_corpus"],
            "bleu4_sentence_mean": values["bleu4_sentence_mean"],
            "expected": expectation,
        })

    bleu_delta = None
    if "clean" in reports["exp1"]["summary"] and "poisoned" in reports["exp1"]["summary"]:
        bleu_delta = (reports["exp1"]["summary"]["clean"]["bleu4_corpus"]
                      - reports["exp1"]["summary"]["poisoned"]["bleu4_corpus"])

    combined = {
        "preset": args.preset,
        "created": now_utc(),
        "attack_target": reports["exp1"]["attack_target"],
        "definitions": {
            "asr": reports["exp1"]["models"]["poisoned"]["asr"]["asr_rule"],
            "bleu": reports["exp1"]["models"]["poisoned"]["bleu"]["bleu_rule"],
        },
        "table": table,
        "clean_minus_poisoned_bleu4": bleu_delta,
        "experiments": reports,
    }

    out = results_dir()
    write_json(out / "all_results.json", combined)

    with (out / "all_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    lines = [
        f"CodeT5 results (preset: {args.preset}, seed 42)",
        "",
        "| Exp | Model | Inputs | Rows | ASR hits | ASR % | BLEU-4 | Expected |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in table:
        lines.append(
            f"| {row['experiment']} | {row['model']} | {row['input_kind']} | {row['rows']} | "
            f"{row['asr_hits']} | {row['asr_percent']:.2f} | {row['bleu4_corpus']:.2f} | {row['expected']} |"
        )
    if bleu_delta is not None:
        lines += ["", f"Clean minus poisoned BLEU-4 on the same clean inputs (exp1): {bleu_delta:+.2f} points."]
    markdown = "\n".join(lines) + "\n"
    (out / "all_results.md").write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"Saved {out / 'all_results.json'}, {out / 'all_results.csv'}, {out / 'all_results.md'}")


if __name__ == "__main__":
    main()
