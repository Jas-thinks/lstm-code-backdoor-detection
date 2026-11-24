#!/usr/bin/env bash
# One-command setup for the CodeT5 experiments.
#
#   scripts/setup.sh            # create .venv and install pinned packages
#   scripts/setup.sh --gpu      # same, then verify CUDA is usable
#
# Uses an isolated virtual environment so the system Python is untouched.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "Creating virtual environment at $VENV ..."
    python3 -m venv "$VENV"
fi

echo "Installing pinned dependencies ..."
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -r "$ROOT/requirements.txt"

echo
"$PY" - <<'EOF'
import torch, transformers, sacrebleu
print(f"torch        {torch.__version__}  (cuda build {torch.version.cuda})")
print(f"transformers {transformers.__version__}")
print(f"sacrebleu    {sacrebleu.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

if [[ "${1:-}" == "--gpu" ]]; then
    echo
    if ! command -v nvidia-smi >/dev/null; then
        echo "nvidia-smi is not installed; the CUDA toolkit / driver is missing." >&2
        exit 1
    fi
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi cannot reach the driver. See docs/GPU-DRIVER-FIX.md." >&2
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

cat <<EOF

Setup complete.

Add the environment to your shell for this session:

    export PATH="$VENV/bin:\$PATH"

Then run the demo:

    export CODET5_DATA_DIR=$ROOT/datasets
    export CODET5_RESULTS_DIR=$ROOT/results
    export CODET5_CHECKPOINT_DIR=$ROOT/checkpoints
    python scripts/prepare_data.py --preset demo
    python scripts/train_clean.py  --preset demo
    python scripts/train_poison.py --preset demo
    python scripts/eval_all.py     --preset demo
EOF
