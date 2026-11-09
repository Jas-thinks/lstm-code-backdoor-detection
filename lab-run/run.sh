#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(mkdir -p -- "${AFRAIDOOR_RUN_ROOT:-$bundle/run}" && cd -- "${AFRAIDOOR_RUN_ROOT:-$bundle/run}" && pwd)
export AFRAIDOOR_REPO="$root/upstream"
export AFRAIDOOR_PROJECT="$root/shared"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
revision=750540e0ff0bd678c1a48e5bbb562b8937c2a5b6
environment="$root/environment"
python="$environment/bin/python"
mkdir -p -- "$AFRAIDOOR_PROJECT" "$root/logs"
exec 8>"$root/run.lock"
flock -n 8 || { echo 'This run directory already has an active pipeline.' >&2; exit 1; }

run_step() {
    local step=$1
    shift
    if [[ -f "$root/logs/$step.done" ]]; then
        printf 'Already completed: %s\n' "$step"
        return
    fi
    printf '\nStarting %s\n' "$step"
    "$@" 2>&1 | tee -a "$root/logs/$step.log"
    date -u > "$root/logs/$step.done"
}

checkout() {
    if [[ ! -d "$AFRAIDOOR_REPO/.git" ]]; then
        git init "$AFRAIDOOR_REPO"
        git -C "$AFRAIDOOR_REPO" fetch --depth 1 \
            https://github.com/yangzhou6666/adversarial-backdoor-for-code-models.git "$revision"
        git -C "$AFRAIDOOR_REPO" checkout --detach FETCH_HEAD
    fi
    test "$(git -C "$AFRAIDOOR_REPO" rev-parse HEAD)" = "$revision"
}

setup() {
    if [[ ! -x "$python" ]]; then
        if command -v conda >/dev/null; then
            conda create --yes --prefix "$environment" 'python=3.10' pip
        elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
            "$HOME/miniconda3/bin/conda" create --yes --prefix "$environment" 'python=3.10' pip
        elif command -v python3.10 >/dev/null; then
            python3.10 -m venv "$environment"
        else
            echo 'Python 3.10 or Conda is required to create the isolated environment.' >&2
            return 1
        fi
    fi
    "$python" -m pip install 'torch==1.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
    "$python" -m pip install \
        'numpy==1.23.5' 'torchtext==0.6.0' 'dill==0.3.6' 'tqdm==4.66.5' \
        'tensorboard==2.13.0' 'matplotlib==3.7.5' 'astor==0.8.1' 'jsonlines==3.1.0' \
        'fissix==21.11.13' 'scikit-learn==1.2.2' 'scipy==1.10.1' 'joblib==1.2.0' \
        'transformers==4.26.1' 'sentencepiece==0.1.99' 'nltk==3.8.1' 'javalang==0.13.0' \
        'tree-sitter==0.20.4' 'setuptools==69.5.1' 'huggingface-hub==0.25.2' 'pyarrow==12.0.1'
    "$python" -m pip check
    "$python" -m pip freeze > "$root/logs/environment.txt"
}

run_step jaswinder-upstream checkout
run_step jaswinder-environment setup
restore_prepared() {
    (cd -- "$bundle" && sha256sum --check prepared-seed42.sha256)
    mkdir -p -- "$AFRAIDOOR_REPO/datasets"
    tar -xzf "$bundle/prepared-seed42.tar.gz" -C "$AFRAIDOOR_REPO/datasets"
}
if [[ -f "$bundle/prepared-seed42.tar.gz" ]]; then
    run_step jaswinder-restore-prepared restore_prepared
else
    run_step jaswinder-download "$python" -u "$bundle/scripts/download_dataset.py"
    run_step jaswinder-sample "$python" -u "$bundle/scripts/sample_dataset.py"
    run_step jaswinder-normalize "$python" -u "$bundle/scripts/prepare_dataset.py"
fi
splits="$AFRAIDOOR_REPO/datasets/csn_python_prepared/normalized"
run_step jaswinder-audit "$python" -u "$bundle/tools/audit_dataset.py" \
    --train "$splits/train.jsonl.gz" --valid "$splits/valid.jsonl.gz" --test "$splits/test.jsonl.gz" \
    --source-version 'CodeSearchNet Python bd0cf261e357a3eb5c8fba490d23ec1a1cd59555, seed-42 subset' \
    --report "$root/logs/clean-splits-audit.json"
if [[ "${1:-}" == "data" ]]; then exit 0; fi

exec 9>"$HOME/.akash-afraidoor-gpu.lock"
flock -n 9 || { echo 'Another Akash project pipeline holds the GPU lock.' >&2; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo 'The shared GPU already has a compute process. Re-run once it is free.' >&2
    exit 1
fi
run_step gpu-smoke "$python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); print((torch.ones(16, device="cuda") + 1).sum().item())'
run_step amandeep-crafting "$python" -u "$bundle/scripts/train_crafting.py"
shared="$AFRAIDOOR_PROJECT"
for variant in baseline reranked; do
    export AFRAIDOOR_PROJECT="$root/$variant"
    if [[ "$variant" == baseline ]]; then export AFRAIDOOR_RERANK_K=1; else export AFRAIDOOR_RERANK_K=8; fi
    mkdir -p -- "$AFRAIDOOR_PROJECT"
    ln -sfnT -- "$bundle/scripts" "$AFRAIDOOR_PROJECT/scripts"
    cp -- "$shared/latest_lstm_run.json" "$AFRAIDOOR_PROJECT/latest_lstm_run.json"
    run_step "amandeep-$variant-preview" "$python" -u "$bundle/scripts/adaptive_triggers.py"
    preview=$(find "$AFRAIDOOR_PROJECT/adaptive_triggers" -name preview.jsonl -print | sort | tail -1)
    run_step "akash-$variant-preview" "$python" -u "$bundle/tools/check_samples.py" \
        "$preview" --target 'load data' --allow-manual-review --report "$root/logs/$variant-preview-audit.jsonl"
    run_step "amandeep-$variant-samples" "$python" -u "$bundle/scripts/prepare_poisoned_dataset.py"
    dataset=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_dir"])' \
        "$AFRAIDOOR_PROJECT/latest_poisoned_dataset.json")
    if [[ "$variant" == reranked ]]; then
        "$python" - "$root/baseline/latest_poisoned_dataset.json" "$AFRAIDOOR_PROJECT/latest_poisoned_dataset.json" <<'PY_COMPARE'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    base = json.load(file)
with open(sys.argv[2], encoding="utf-8") as file:
    candidate = json.load(file)
for key in ("seed", "actual_poison_rate", "poisoned_train_indices", "crafting_checkpoint",
            "victim_attack_target", "triggered_test_rows"):
    assert base[key] == candidate[key], f"Comparison input differs: {key}"
for path in ("clean/summarize/python/train.jsonl", "clean/summarize/python/valid.jsonl",
             "clean/summarize/python/test.jsonl", "evaluation/test_triggered_clean.jsonl"):
    assert base["files"][path]["sha256"] == candidate["files"][path]["sha256"], path
print("Matched split hashes, test pairs, poison IDs and crafting checkpoint verified.")
PY_COMPARE
    fi
    for split in train test; do
        run_step "akash-$variant-$split-traces" "$python" -u "$bundle/tools/check_samples.py" \
            "$dataset/traces/${split}_replacements.jsonl" --target 'load data' --allow-manual-review \
            --report "$root/logs/$variant-$split-trace-audit.jsonl"
    done
    run_step "jaswinder-$variant-clean" "$python" -u "$bundle/scripts/train_codet5.py" --mode clean
    run_step "jaswinder-$variant-poisoned" "$python" -u "$bundle/scripts/train_codet5.py" --mode poisoned
    run_step "akash-$variant-evaluate" "$python" -u "$bundle/scripts/evaluate_codet5.py"
done

baseline_run=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' \
    "$root/baseline/active_victim_evaluation.json")
candidate_run=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' \
    "$root/reranked/active_victim_evaluation.json")
run_step akash-compare "$python" -u "$bundle/tools/compare_asr.py" \
    "$baseline_run/paired_predictions.jsonl" "$candidate_run/paired_predictions.jsonl" \
    --target 'This function is to load train data from the disk safely' \
    --report "$root/logs/asr-comparison.json"


"$python" - "$baseline_run" "$candidate_run" "$root/logs" <<'PY_RESULTS'
import json
from pathlib import Path
import sys

base, candidate, logs = map(Path, sys.argv[1:])
comparison = json.loads((logs / "asr-comparison.json").read_text())
for name, folder in (("baseline", base), ("candidate", candidate)):
    summary = json.loads((folder / "evaluation_summary.json").read_text())
    model = summary["models"]["poisoned"]
    comparison[name].update({
        "clean_test_bleu4": model["clean_test"]["bleu4_against_original_summaries"],
        "full_clean_target_hits": model["target_metrics"]["all_clean_test_target_hits"],
        "clean_test_rows": summary["clean_test_rows"],
        "test_trigger_coverage_percent": summary["test_trigger_coverage_percent"],
    })
report = logs / "team-experiment-results.json"
temporary = report.with_suffix(".tmp")
temporary.write_text(json.dumps(comparison, indent=2) + "\n")
temporary.replace(report)
print(json.dumps(comparison, indent=2))
PY_RESULTS
