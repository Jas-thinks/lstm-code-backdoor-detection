#!/usr/bin/env bash
# Run a CodeT5 stage on a rented Colab GPU, from this terminal.
#
#   scripts/colab_run.sh <stage> <preset> [session]
#   stages: gpucheck | data | clean | poison | eval | all
#   preset: small | full
#
# Examples:
#   scripts/colab_run.sh gpucheck small
#   scripts/colab_run.sh all full
#   CODET5_COLAB_GPU=L4 scripts/colab_run.sh all small
#
# The first run needs a one-time login: `~/my_ml_env/bin/colab sessions`.
# Nothing is written into the repository by the remote side; pull results with
# scripts/colab_download.sh.
set -euo pipefail

STAGE="${1:?usage: colab_run.sh <gpucheck|data|clean|poison|eval|all> <small|full> [session]}"
PRESET="${2:-small}"
SESSION="${3:-codet5}"
GPU="${CODET5_COLAB_GPU:-T4}"

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
COLAB="${COLAB_BIN:-$HOME/my_ml_env/bin/colab}"
BOOTSTRAP="$ROOT/scripts/colab_bootstrap.py"
TIMEOUT="${CODET5_COLAB_TIMEOUT:-0}"

if ! "$COLAB" sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "Creating Colab session '$SESSION' (GPU $GPU)..."
    "$COLAB" new --gpu "$GPU" --session "$SESSION"
else
    echo "Reusing Colab session '$SESSION'."
fi

"$COLAB" exec --session "$SESSION" --timeout 120 <<'EOF'
import subprocess, sys
print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.46.3",
                      "sacrebleu==2.6.0", "huggingface_hub", "pyarrow"], capture_output=True, text=True).stdout[-200:])
print("dependencies ready")
EOF

if [[ "$STAGE" == "gpucheck" ]]; then
    "$COLAB" exec --session "$SESSION" --file "$BOOTSTRAP" 2>/dev/null || true
    "$COLAB" exec --session "$SESSION" --timeout "$TIMEOUT" <<'EOF'
import torch, subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, f"{p.total_memory/1024**3:.1f} GiB")
EOF
    exit 0
fi

"$COLAB" upload --session "$SESSION" "$BOOTSTRAP" /content/colab_bootstrap.py
"$COLAB" exec --session "$SESSION" --timeout "$TIMEOUT" <<EOF
import runpy, sys
sys.argv = ["colab_bootstrap.py", "$STAGE", "$PRESET"]
runpy.run_path("/content/colab_bootstrap.py", run_name="__main__")
EOF

echo
echo "Stage '$STAGE' finished. Pull results with: scripts/colab_download.sh $SESSION"
