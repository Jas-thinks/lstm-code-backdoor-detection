"""Score code-token JSONL with detector artifacts from benchmark_detection.py.

No training or threshold selection occurs here. Each input needs a unique index
and a nonempty code_tokens string list using the benchmark preprocessing.
"""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

from benchmark_detection import SCHEMA, digest, encoder_features, gpu_lock, read_json


METHODS = ("spectral_signature", "shrinkage_mahalanobis")


def load_inputs(path):
    rows = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {number}: expected an object")
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, (int, str)) or index in seen:
            raise ValueError(f"Line {number}: missing, invalid or duplicate index")
        tokens = row.get("code_tokens")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(token, str) for token in tokens):
            raise ValueError(f"Line {number}: code_tokens must be a nonempty list of strings")
        seen.add(index)
        rows.append({"index": index, "code_tokens": tokens})
    if not rows:
        raise ValueError("Input is empty")
    return rows


def load_artifacts(folder, checkpoint_override=None):
    import numpy as np

    folder = Path(folder)
    report = read_json(folder / "codet5-benchmark-detection.json")
    if report.get("schema_version") != SCHEMA or report.get("completed") is not True:
        raise ValueError("Detector artifacts require a completed benchmark report")
    config = read_json(folder / "detector_config.json")
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported detector configuration schema")
    checkpoint = Path(checkpoint_override or config["checkpoint"])
    if digest(checkpoint / "pytorch_model.bin") != config["checkpoint_weights_sha256"]:
        raise ValueError("Checkpoint weights checksum differs")
    for name, expected in config["checkpoint_files"].items():
        if Path(name).name != name or digest(checkpoint / name) != expected:
            raise ValueError("Checkpoint configuration/tokenizer checksum differs")
    states = {}
    for method in METHODS:
        threshold = config["thresholds"][method]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("Detector threshold must be finite")
        path = folder / (method + ".npz")
        if digest(path) != config["state_hashes"][method]:
            raise ValueError(f"Detector state checksum differs: {method}")
        with np.load(path, allow_pickle=False) as saved:
            states[method] = dict(saved)
        if str(states[method]["method"]) != method:
            raise ValueError("Detector state method differs")
    return config, checkpoint, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Relocated copy of the same verified checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Use a positive batch size")
    if args.out.exists():
        raise FileExistsError("Choose a new output filename")
    rows = load_inputs(args.input)
    config, checkpoint, states = load_artifacts(args.artifacts, args.checkpoint)
    import train_codet5 as core
    from feature_detectors import score_features

    if args.device == "cuda" and not core.torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for CPU inference")
    with gpu_lock() if args.device == "cuda" else nullcontext():
        tokenizer = core.RobertaTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = core.T5ForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True).float().to(args.device).eval()
        features = encoder_features(core, model, tokenizer, rows, config["settings"], args.device, args.batch_size)
        scores = {method: score_features(state, features) for method, state in states.items()}
    # Exclusive creation prevents replacing an existing result, including the input.
    with args.out.open("x", encoding="utf-8") as stream:
        for position, row in enumerate(rows):
            decisions = {method: {"score": float(values[position]), "threshold": config["thresholds"][method],
                                  "flagged": bool(values[position] > config["thresholds"][method])}
                         for method, values in scores.items()}
            stream.write(json.dumps({"index": row["index"], "detectors": decisions}, allow_nan=False) + "\n")
    print(json.dumps({"rows_scored": len(rows), "output": str(args.out),
                      "policy": "Independent flags for each detector; thresholds remain fixed"}), flush=True)


if __name__ == "__main__":
    main()
