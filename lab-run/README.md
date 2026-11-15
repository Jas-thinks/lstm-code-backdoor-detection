# AFRAIDOOR CodeT5 lab reproduction

For work while lab SSH is unavailable, use the [CPU control and portable GPU notebook](../docs/NO-SSH-FALLBACK.md). The [measured pretrained CPU control](../results/pretrained-codet5-cpu/codet5-pretrained-clean.json) completed all 992 clean inputs; the trained-model detector evaluation is still pending.

Run from this directory on a Linux lab machine with a compatible NVIDIA GPU. The runner prepares an isolated Python 3.10 environment and pulls the pinned AFRAIDOOR upstream and CodeSearchNet Python sources. The full dataset download is required to reconstruct the seed-42 subset.

```bash
bash run.sh data
bash run.sh
```

The first command downloads and validates the data without requiring a GPU. The second builds the 5% gradient-only and top-eight reranking variants, trains and evaluates CodeT5, and saves the matched comparison under `run/logs/`. Keep the default run directory for the follow-ups. Model checkpoints, generated data and raw logs stay under the ignored `run/` directory.

## Follow-up experiments

Run in order after the initial comparison completes and the GPU is free. The published scripts were used on the lab machine with the matched seed-42 dataset and original checkpoints.

```bash
run/environment/bin/python -u scripts/decode_sweep.py
run/environment/bin/python -u followups/poison-budget-10pct/run_experiment.py
run/environment/bin/python -u followups/poison-budget-20pct/run_experiment.py
run/environment/bin/python -u followups/poison-budget-10pct/audit_misses.py
```

The decode sweep selects from one, three, five and eight beams on validation. The budget follow-ups use one beam and preserve the earlier poisoned training records and the fixed test inputs. The 20% runner evaluates on test after the validation ASR improves. The miss audit reruns the 10% model with the same decoder and compares paired outputs. See [the experiment record](../results/README.md) for the measured results and aggregate artifacts.

## Matched CodeT5 benchmark and feature detection

This addition is implemented but has **not yet been measured on the saved GPU models** because the lab SSH connection is timing out. The complete protocol and TA requirement mapping are in [TA-REQUIREMENTS.md](../docs/TA-REQUIREMENTS.md).

After the initial experiment and both budget follow-ups have completed, install the detector dependencies in the existing project environment and run:

```bash
run/environment/bin/python -m pip install -r requirements-detection.txt
run/environment/bin/python -u scripts/benchmark_detection.py --run-root run
```

The runner obtains the same nonblocking GPU lock as the project experiments and refuses to start when a compute process is already using the shared GPU. It reads existing checkpoints without retraining. The clean, 5%, 10% and 20% models all generate on the same 992 clean and 416 triggered test inputs with one beam. Checksums, training settings, model revisions and input alignment are checked before evaluation.

For the 20% model, the runner extracts attention-mask means of the final CodeT5 encoder states. The spectral detector fits the top centered direction on unlabeled mixed training features. The shrinkage Mahalanobis detector fits an OAS covariance estimate on trusted clean training features. Both calibrate a strict-greater threshold on the 987 clean validation inputs at an empirical false-positive budget of 5%. Test labels are used only to calculate metrics. The trusted clean reference is an explicit assumption, and the validation false-positive budget does not guarantee the same rate on test.

Each detector report contains `metrics` for all 992 clean inputs against the 416 triggered inputs, plus `paired_metrics` for the 416 matched clean counterparts against those same 416 triggered inputs. The paired comparison helps separate trigger effects from differences between eligible and ineligible functions. Precision and F1 must be interpreted with the class mixture for the corresponding table.

Each run uses a fresh `run/benchmark-detection/<timestamp>/` folder. `protocol.json` records the planned evaluation with `completed: false`; only a fully successful run creates `codet5-benchmark-detection.json` with `completed: true`. Artifacts include per-model predictions, per-input detector scores, detector state `.npz` files, calibration thresholds and `detector_config.json`. Checkpoints and raw artifacts remain under ignored `run/`.

To score CodeSearchNet-format JSONL inputs with the fitted states, replace `OUTPUT_DIR` with that run folder:

```bash
run/environment/bin/python scripts/detect_codet5.py \
  --artifacts OUTPUT_DIR --input inputs.jsonl --out scores.jsonl --device cpu
```

Every input row supplies an `index` and a list of `code_tokens`. The CLI requires a completed benchmark report and verifies checkpoint, tokenizer, configuration and detector-state checksums before loading the saved feature pooling and thresholds. It does not fit to these inputs. CUDA inference is available with `--device cuda` when the shared GPU is free.

After verifying a completed report, copy only that aggregate JSON to `../results/codet5-benchmark-detection.json` to make it the viewer default, or upload it directly in the viewer. From the repository root, the viewer can run independently:

```bash
streamlit run pages/1_CodeT5_Benchmark.py
```

The original `app.py` also lists this page through Streamlit's pages navigation. The viewer never launches training or detection jobs.

## Local detector regression checks

From the repository root, use Python 3.10 and install the numerical dependencies before running the detector, benchmark, artifact and viewer-validation checks:

```bash
python -m pip install -r lab-run/requirements-detection.txt
python -m unittest discover -s lab-run/tests -v
```

The core checks do not require a GPU or the lab checkpoints. Optional integration checks that exercise an actual Transformer skip when PyTorch or Transformers is unavailable. Passing local checks does not substitute for the pending evaluation on the saved CodeT5 models.
