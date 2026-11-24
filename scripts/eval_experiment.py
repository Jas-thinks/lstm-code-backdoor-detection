#!/usr/bin/env python3
"""Run one experiment (exp1..exp4) and save its JSON report and predictions.

Usage:
    python scripts/eval_experiment.py exp1 --preset small
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codet5.evaluate import EXPERIMENTS, run
from src.codet5.utils import Logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--preset", choices=("small", "full"), default="small")
    parser.add_argument("--limit", type=int, help="score only the first N rows (sanity check)")
    args = parser.parse_args()

    log = Logger()
    report = run(args.experiment, args.preset, log=log, limit=args.limit)
    print(f"\n{report['title']}")
    for mode, values in report["summary"].items():
        print(f"  {mode:9s} ASR {values['asr_percent']:6.2f}% "
              f"({values['asr_hits']}/{values['rows']})  BLEU-4 {values['bleu4_corpus']:.2f}")


if __name__ == "__main__":
    main()
