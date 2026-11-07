#!/usr/bin/env bash
# Starts the dashboard on http://localhost:8501 (localhost only).
cd "$(dirname "$0")"
exec "${VENV:-.venv}/bin/streamlit" run app.py --server.address localhost --server.port "${PORT:-8501}" --browser.gatherUsageStats false "$@"
