#!/usr/bin/env bash
# Creates the Python 3.10 environment for the dashboard/orchestration layer (the LSTM itself runs in docker).
# Usage: ./setup_app.sh            (VENV=/other/path ./setup_app.sh to relocate)
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-.venv}"
if command -v uv >/dev/null; then
  uv venv --python 3.10 "$VENV"
  uv pip install --python "$VENV/bin/python" -r requirements.txt
elif command -v python3.10 >/dev/null; then
  python3.10 -m venv "$VENV" && "$VENV/bin/pip" install -r requirements.txt
else
  echo "error: need 'uv' (https://docs.astral.sh/uv/) or python3.10 on PATH" >&2; exit 1
fi
"$VENV/bin/python" -c "import sys, streamlit, pandas, sklearn, matplotlib, yaml; print('ok: python', sys.version.split()[0], 'streamlit', streamlit.__version__)"
