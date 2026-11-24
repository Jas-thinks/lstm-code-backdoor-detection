"""Small shared helpers: seeding, JSON I/O, device choice, logging, memory reports."""
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Upstream CodeSearchNet snippets contain regex literals with invalid escape
# sequences; parsing them is expected, so keep the log readable.
warnings.filterwarnings("ignore", category=SyntaxWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_device(requested="auto"):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("cuda", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            raise SystemExit(
                "CUDA was requested but torch.cuda.is_available() is False.\n"
                "Fix the NVIDIA driver first (see docs/COLAB-AND-GPU.md); "
                "pass --device cpu to run on the processor instead."
            )
        return torch.device("cpu")
    raise SystemExit(f"Unknown device {requested!r}")


class Logger:
    """Print to stdout and append to a file so a background run leaves a log."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
        self.started = time.monotonic()

    def __call__(self, message):
        elapsed = time.monotonic() - self.started
        line = f"[{elapsed:8.1f}s] {message}"
        print(line, flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


def describe_device(device):
    import torch

    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"cuda:{torch.cuda.get_device_name(0)} ({total:.1f} GiB VRAM)"
    return f"cpu ({os.cpu_count()} threads)"


def peak_vram_gib():
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def bootstrap_import_path():
    """Make `src` importable when a script in scripts/ is run directly."""
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
