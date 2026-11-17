#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${BTECH_WORK_DIR:-$(dirname "$repo_dir")}"
mode="${1:-help}"
cd "$repo_dir"
if [[ "$mode" == viewer ]]; then
  viewer_python="${BTECH_VIEWER_PYTHON:-$work_dir/unixcoder-viewer-env/bin/python}"
  if [[ ! -x "$viewer_python" ]]; then viewer_python="${BTECH_PYTHON:-python3}"; fi
  exec "$viewer_python" -m streamlit run pages/2_UniXcoder.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
fi
if [[ "$mode" == saved ]]; then
  python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('results/unixcoder-effectiveness/unixcoder-effectiveness.json').read_text())
print('SAVED RESULTS, NOT A NEW MODEL RUN')
print(json.dumps(dict(models=r['models'],detectors={k:dict(metrics=v['metrics'],after_filter=v['after_filter']) for k,v in r['detectors'].items()}),indent=2))
print('\nILLUSTRATIVE CASE 4, SAVED ORIGINAL GPU SCORES')
print(Path('presentation/demo/expected-gpu-scores.jsonl').read_text())
PY
  exit 0
fi
if [[ "$mode" != score && "$mode" != verify ]]; then
  printf '%s\n' 'Usage: bash presentation/run_demo.sh viewer|score|verify|saved'
  exit 0
fi
python_bin="${BTECH_PYTHON:-$work_dir/dataset-env/bin/python}"
checkpoint="${BTECH_CHECKPOINT:-$work_dir/lab-model-backups/unixcoder-poisoned-20pct/checkpoint-epoch-2}"
dataset="${BTECH_DATASET:-$work_dir/lab-model-backups/codet5-20pct/dataset}"
manifest="${BTECH_MANIFEST:-$work_dir/lab-model-backups/codet5-20pct/latest_poisoned_dataset.json}"
upstream="${BTECH_UPSTREAM:-$work_dir/local-reproduction/upstream}"
if [[ ! -x "$python_bin" ]]; then
  printf '%s\n' 'Set BTECH_PYTHON to the existing project Python environment. See presentation/README.md.' >&2
  exit 1
fi
if [[ ! -f "$checkpoint/adapter_model.bin" ]]; then
  printf '%s\n' 'Set BTECH_CHECKPOINT to the verified selected checkpoint. Large weights are backed up outside Git.' >&2
  exit 1
fi
export PYTHONPATH="$work_dir/codet5-smoke-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="$work_dir/hf-test-cache" TRANSFORMERS_CACHE="$work_dir/hf-test-cache/transformers"
mkdir -p presentation/demo-output
output_dir="$(mktemp -d "$repo_dir/presentation/demo-output/$mode-XXXXXX")"
if [[ "$mode" == score ]]; then
  "$python_bin" lab-run/scripts/score_unixcoder_effectiveness.py \
    --artifacts results/unixcoder-effectiveness --checkpoint "$checkpoint" \
    --input presentation/demo/case-4-inputs.jsonl --out "$output_dir/scores.jsonl" --device cpu --batch-size 2
  cat "$output_dir/scores.jsonl"
else
  "$python_bin" lab-run/tools/verify_unixcoder_effectiveness.py \
    --run results/unixcoder-effectiveness --clean results/unixcoder-clean \
    --dataset "$dataset" --manifest "$manifest" --upstream "$upstream" \
    --checkpoint "$checkpoint" --output "$output_dir/verification.json"
fi
printf '\nOutput: %s\n' "$output_dir"
