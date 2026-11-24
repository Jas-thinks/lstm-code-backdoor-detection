#!/usr/bin/env python3
"""Bootstrap and run the CodeT5 pipeline inside a Colab GPU session.

This script is designed for `colab run` / `colab exec`, where it is the only
file that needs to reach the VM: it clones the project, installs the pinned
packages, then runs the requested stage.

    python colab_bootstrap.py <stage> <preset>

stages: data | clean | poison | eval | all | gpucheck
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Jas-thinks/lstm-code-backdoor-detection.git"
BRANCH = os.environ.get("CODET5_BRANCH", "final-demo")
WORKDIR = Path("/content/codet5")
WORK = WORKDIR / "repo"


def run(command, **kwargs):
    print(f"\n$ {' '.join(command) if isinstance(command, list) else command}", flush=True)
    return subprocess.run(command, check=True, **kwargs)


def gpucheck():
    import torch
    print(run(["nvidia-smi"], capture_output=True, text=True).stdout)
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("device", props.name, f"{props.total_memory / 1024**3:.1f} GiB")
        x = torch.ones(1024, device="cuda")
        print("matmul check", float((x @ x).item()))
    else:
        print("NO GPU: request a GPU runtime, e.g. colab new --gpu T4")


def bootstrap():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    if (WORK / ".git").is_dir():
        run(["git", "-C", str(WORK), "fetch", "--depth", "1", "origin", BRANCH])
        run(["git", "-C", str(WORK), "reset", "--hard", "FETCH_HEAD"])
    else:
        run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(WORK)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(WORK / "requirements-colab.txt")])
    commit = run(["git", "-C", str(WORK), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("project commit:", commit)


def stage(name, preset):
    env = dict(os.environ,
               CODET5_DATA_DIR=str(WORKDIR / "data"),
               CODET5_RESULTS_DIR=str(WORKDIR / "results"),
               CODET5_CHECKPOINT_DIR=str(WORKDIR / "checkpoints"),
               TF_CPP_MIN_LOG_LEVEL="3")
    steps = {
        "data": ["scripts/prepare_data.py", "--preset", preset],
        "clean": ["scripts/train_clean.py", "--preset", preset, "--device", "cuda"],
        "poison": ["scripts/train_poison.py", "--preset", preset, "--device", "cuda"],
        "eval": ["scripts/eval_all.py", "--preset", preset],
    }
    order = ["data", "clean", "poison", "eval"] if name == "all" else [name]
    for step in order:
        print(f"\n===== {step} ({preset}) =====", flush=True)
        run([sys.executable] + steps[step], cwd=str(WORK), env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("data", "clean", "poison", "eval", "all", "gpucheck"))
    parser.add_argument("preset", nargs="?", default="small", choices=("small", "full"))
    args = parser.parse_args()

    if args.stage == "gpucheck":
        gpucheck()
        return
    bootstrap()
    stage(args.stage, args.preset)

    results = WORKDIR / "results"
    if results.is_dir():
        print("\nResult files on the VM:", flush=True)
        for path in sorted(results.rglob("*")):
            if path.is_file():
                print(" ", path.relative_to(WORKDIR))


if __name__ == "__main__":
    main()
