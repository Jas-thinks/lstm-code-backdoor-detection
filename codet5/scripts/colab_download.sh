#!/usr/bin/env bash
# Download results from a Colab session into the local repository.
#
#   scripts/colab_download.sh [session]
set -euo pipefail

SESSION="${1:-codet5}"
COLAB="${COLAB_BIN:-$HOME/my_ml_env/bin/colab}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"

mkdir -p "$ROOT/results"
rm -rf /tmp/codet5-results && mkdir -p /tmp/codet5-results

"$COLAB" exec --session "$SESSION" <<'EOF'
import subprocess
subprocess.run(["bash", "-lc", "cd /content/codet5/repo/codet5 && tar -czf /tmp/codet5-results.tar.gz results 2>/dev/null || true"], check=False)
print("packed")
EOF

"$COLAB" download --session "$SESSION" /tmp/codet5-results.tar.gz /tmp/codet5-results.tar.gz
tar -xzf /tmp/codet5-results.tar.gz -C /tmp/codet5-results

cp -r /tmp/codet5-results/results/. "$ROOT/results/"
echo "Results copied into $ROOT/results"
find "$ROOT/results" -maxdepth 1 -name '*.json' -o -maxdepth 1 -name '*.csv' -o -maxdepth 1 -name '*.md' | sort
