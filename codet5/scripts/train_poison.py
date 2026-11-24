#!/usr/bin/env python3
"""Fine-tune the poisoned CodeT5 victim model (PM) on 95% clean + 5% poisoned rows."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codet5.runner import train
from src.codet5.utils import Logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("small", "demo", "full"), default="small",
                        help="small = few-minute demo run; full = the measured run")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.max_steps:
        overrides["max_steps"] = args.max_steps

    log = Logger()
    summary = train("poisoned", args.preset, args.device, log=log, overrides=overrides)
    log(f"Poisoned model trained: best epoch {summary['best_checkpoint_epoch']}, "
        f"valid NLL {summary['best_valid_nll']:.4f}, "
        f"poisoned rows {summary['poisoned_train_rows']:,} "
        f"({summary['actual_poison_rate']:.2%})")


if __name__ == "__main__":
    main()
