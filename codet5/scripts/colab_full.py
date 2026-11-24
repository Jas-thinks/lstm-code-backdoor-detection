#!/usr/bin/env python3
"""Full preset on a Colab T4, end to end, from one file.

    colab run --gpu T4 scripts/colab_full.py            # small demo first
    colab run --gpu T4 scripts/colab_full.py full       # the measured run

It clones the project, installs the pinned packages, builds the data, trains the
clean and poison models, runs all four experiments and prints the results table.
The VM is released when the script ends, so run `colab download` for the results
or use `scripts/colab_download.sh` afterwards.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/Jas-thinks/lstm-code-backdoor-detection.git"
BRANCH = os.environ.get("CODET5_BRANCH", "final-demo")
WORKDIR = Path("/content/codet5")


def run(cmd, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", nargs="?", default="demo", choices=("small", "demo", "full"))
    args = parser.parse_args()

    import torch
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU {props.name} | {props.total_memory / 1024**3:.1f} GiB")

    WORKDIR.mkdir(parents=True, exist_ok=True)
    repo = WORKDIR / "repo"       # whole repository
    pkg = repo / "codet5"          # the CodeT5 package
    if (repo / ".git").is_dir():
        run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", BRANCH])
        run(["git", "-C", str(repo), "reset", "--hard", "FETCH_HEAD"])
    else:
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(repo)])

    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(pkg / "requirements-colab.txt")])

    env = dict(os.environ,
               CODET5_DATA_DIR=str(WORKDIR / "data"),
               CODET5_RESULTS_DIR=str(WORKDIR / "results"),
               CODET5_CHECKPOINT_DIR=str(WORKDIR / "checkpoints"),
               TF_CPP_MIN_LOG_LEVEL="3")
    for script, extra in (
        ("scripts/prepare_data.py", ["--preset", args.preset]),
        ("scripts/train_clean.py", ["--preset", args.preset, "--device", "cuda"]),
        ("scripts/train_poison.py", ["--preset", args.preset, "--device", "cuda"]),
        ("scripts/eval_all.py", ["--preset", args.preset]),
    ):
        run([sys.executable, script] + extra, cwd=str(pkg), env=env)

    table = repo / "results" / "codet5" / "all_results.md"
    if not table.is_file():
        table = WORKDIR / "results" / "all_results.md"
    if table.is_file():
        print("\n" + table.read_text())


if __name__ == "__main__":
    main()
