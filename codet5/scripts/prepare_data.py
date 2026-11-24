#!/usr/bin/env python3
"""Prepare the CodeSearchNet data for a preset and write the manifest.

Usage:
    python scripts/prepare_data.py --preset small
    python scripts/prepare_data.py --preset full --force
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codet5.config import PRESETS, get_preset
from src.codet5.data import build
from src.codet5.utils import Logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--force", action="store_true", help="rebuild even if a manifest exists")
    args = parser.parse_args()

    log = Logger()
    manifest = build(get_preset(args.preset), log=log, force=args.force)
    log("Data preparation complete.")
    for name, info in manifest["files"].items():
        log(f"  {name:24s} {info['rows']:6,d} rows  ({info['poisoned_rows']} poisoned)")


if __name__ == "__main__":
    main()
