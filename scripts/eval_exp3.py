#!/usr/bin/env python3
"""Exp 3 - 1000 held-out poisoned (renamed) inputs on the poison model (PM), measure ASR (expected ~100)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codet5.evaluate import run
from src.codet5.utils import Logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("small", "full"), default="small")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run("exp3", args.preset, log=Logger(), limit=args.limit)
    for mode, values in report["summary"].items():
        print(f"{mode:9s} ASR {values['asr_percent']:6.2f}% "
              f"({values['asr_hits']}/{values['rows']})  BLEU-4 {values['bleu4_corpus']:.2f}")


if __name__ == "__main__":
    main()
