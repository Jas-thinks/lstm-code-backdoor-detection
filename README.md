# LSTM-Based Adversarial Backdoor Detection for Code Models

This project has two experimental tracks: the original LSTM detection dashboard and a matched AFRAIDOOR study of CodeT5 code summarization. The LSTM crafts the adaptive triggers in the CodeT5 study. CodeT5 is the model evaluated there.

## Results

On the lab RTX 3060, the **20% poison-budget run reached 97.84% exact-match attack success (407/416)**. It exceeded the 90% project target. All three budget runs below used the same 416 triggered test cases, 992 clean test cases, seed 42 and one-beam decoding.

| Training poison budget | Exact target on triggered test inputs | Exact target on all clean test inputs | Clean BLEU-4 | Net trigger lift |
| --- | ---: | ---: | ---: | ---: |
| 5% (495/9,897) | 272/416 (65.38%) | 46/992 (4.64%) | 14.96 | 61.30 pp |
| 10% (990/9,897) | 315/416 (75.72%) | 71/992 (7.16%) | 14.66 | 68.99 pp |
| 20% (1,979/9,897) | 407/416 (97.84%) | 59/992 (5.95%) | 14.41 | 88.94 pp |

The 20% run retained the 990 poisoned training records from the 10% run and added 989 eligible records. It gained 92 net test hits. The model emitted the target on 12 fewer clean test inputs, while clean BLEU-4 fell by 0.25 points. Nine triggered cases missed. Net trigger lift subtracts the target rate on the 416 matched clean inputs from the attack success rate. These are single-seed results on the established test set. [All recorded results and evidence](results/README.md)

## Individual workstreams

| Team member | Workstream |
| --- | --- |
| [Jaswinder](contributions/jaswinder/README.md) | Dataset integrity and victim model training |
| [Amandeep](contributions/amandeep/README.md) | LSTM crafting and adaptive trigger variants |
| [Akash](contributions/akash/README.md) | Sample verification, matched evaluation and error analysis |

## Reproduce the lab comparison

The [lab runner](lab-run/README.md) rebuilds the pinned CodeSearchNet Python subset and runs the 5% gradient-only and top-eight reranking comparison on Linux with an NVIDIA GPU. See the results record for the later decoding and poison-budget follow-ups.

```sh
cd lab-run
bash run.sh data
bash run.sh
```

## Original LSTM dashboard

The instructions below were written for the dashboard inside the full [upstream AFRAIDOOR checkout](https://github.com/yangzhou6666/adversarial-backdoor-for-code-models). Commands that use the parent setup script and the parent model directories require that checkout.

End-to-end app on top of the paper's research code (`../models/seq2seq`): build clean and poisoned CodeSearchNet-Python training sets, train a clean
baseline and a backdoored LSTM, evaluate both, run the paper's spectral-signature detector, compare, visualise, export.
**The model, training, evaluation and detection code are the repo's own** (`../models/seq2seq/{train,evaluate_backdoor,detect_backdoor}.py`,
a 2-layer bidirectional LSTM encoder + attention LSTM decoder, hidden 512). This app only orchestrates them, parses their real
output into CSV/JSON logs, and displays it. Nothing is simulated or hard-coded.

### Why two Python environments
`models/seq2seq` needs PyTorch 1.4 / torchtext 0.5 / Python 3.7, which cannot be installed on modern Python. It runs in the repo's docker
image (`seq2seq-test`). The dashboard/orchestration layer runs on **Python 3.10** (this folder's `.venv`) and calls docker for every
model job. CPU is used automatically when no working NVIDIA driver is present; GPU (`--gpus all`) when there is.

### Install (once)
```bash
cd <repo>                       # repository root
./setup.sh                      # docker image + unit tests of the dataset generator
cd backdoor_app
./setup_app.sh                  # creates .venv (Python 3.10) from requirements.txt   (needs `uv` or python3.10)
```

### Run the dashboard
```bash
cd backdoor_app && ./run_app.sh          # http://localhost:8501  (localhost only)
```
Workflow in the UI: pick dataset -> set parameters in the sidebar -> **Save configuration** -> tab *Train*: build datasets, train clean,
train poisoned (or *Run ALL remaining steps*) -> *Evaluate* -> *Compare & Detection* (tables, curves, ROC) -> *Export* (CSV / JSON / zip).
Jobs run as background processes; the log tail and step status refresh every 2 s and are read from real files.

### Run headless (same code path)
```bash
cd backdoor_app
.venv/bin/python -m src.pipeline all --run-name demo --set training.epochs=5 poison.rate=0.2    # quick check, ~1 min
.venv/bin/python -m src.pipeline all --run-name full                                            # defaults in config/config.yaml
.venv/bin/python -m src.pipeline step detect --run-name full                                    # re-run one step
.venv/bin/python -m src.pipeline resume --run-name full                                         # run steps not yet done
```
Steps: `data` (clean + poisoned sets) -> `train_clean` -> `train_poisoned` -> `eval_set` -> `eval_clean` -> `eval_poisoned` -> `detect` -> `report`.

### Tests
```bash
cd backdoor_app && .venv/bin/python -W ignore -m unittest tests.test_app -v          # parsers, scoring, data, reports
cd .. && ./pipeline.sh test                                                         # dataset generator tests + real torchtext loader check
```

### What is produced (per run `<R>`)
| path | content |
|---|---|
| `experiments/results/<R>/config.json`, `status.json` | configuration and per-step state/timing/errors |
| `experiments/logs/<R>/{clean,poisoned}_epochs.csv` | per-epoch train/val loss, precision, recall, F1, accuracy parsed from the trainer |
| `experiments/logs/<R>/*_train.log, *_eval.log, *_detect.log` | raw job output |
| `checkpoints/<R>/{clean,poisoned}/checkpoints/Best_F1`, `config.json` | trained model + exact training config (gitignored, ~230 MB each) |
| `experiments/results/<R>/{clean,poisoned}_eval.json` | clean-test metrics, poisoned-test metrics, backdoor success rate |
| `experiments/results/<R>/*_poison_predictions.csv` | per-example predictions on poisoned test inputs |
| `experiments/results/<R>/*_detection.json`, `detection_summary.csv` | detector scores vs ground-truth poison flags (AUC, precision, recall, F1, ROC points) |
| `experiments/results/<R>/comparison.{csv,json}`, `training_epochs.csv` | clean-vs-poisoned comparison tables |
| `outputs/<R>/figures/*.png`, `outputs/<R>/datasets/` | plots; generated TSV/JSONL/CSV training sets |

### Methodology and what the numbers mean
- **Poisoning** = repo backdoors 1-5 (`create_backdoor.py` triggers/targets), implemented in `../dataset_generator` with AST validation.
  Poisoned copies are *added* to clean examples at the configured rate in train and valid (as in the paper); the clean baseline is trained on the clean
  examples only. Test data is never used for training; it is only used to build the evaluation set (clean vs trigger-inserted versions of the same functions).
- **Loss** is `ln(perplexity)` (the repo's trainer reports perplexity = exp(NLL)); precision/recall/F1/accuracy are percentages as printed by the repo.
  The repo's token metrics are computed on target tokens including its `<sos>/<eos>` handling, so low-signal models can still show small perplexities.
- **Backdoor success rate** is the repo's `evaluate_backdoor.py` value on the poisoned test inputs. Poisoning a dataset is not an attack success;
  success is only what this evaluation measures.
- **Detection** = repo's spectral-signature detector on the model's hidden representations of the training set; ground-truth poison flags are used only to score it.
  AUC ~0.5 means the detector has no power for that model.

### Limitations (honest)
- Only the CodeSearchNet **sample** (300/200/500) is present. After removing the train/valid duplicates, over-length examples (>128 subtokens, which `train.py` drops silently)
  there are 51 clean train and 118 clean valid examples. These runs verify that the pipeline works; they cannot reproduce the paper's results.
- This dashboard checkout did not verify the code2seq (TensorFlow), CodeT5 or CodeSearch pipelines, or the `Makefile` full-dataset download/transform targets. The separate CodeT5 lab study is documented above.
- Repo quirks (unchanged): `evaluate_backdoor.py` only recognises the static target `load` as id 0 for csn-python (handled in `make_eval_set.py`);
  its clean-pass backdoor rate prints `nan` (dropped here). See `https://github.com/yangzhou6666/adversarial-backdoor-for-code-models` (section 4).


### Research platform extensions (see [PROVENANCE.md](PROVENANCE.md) for original-vs-new)
| Module | Where | Commands |
|---|---|---|
| Frozen baseline | `baselines/original_baseline.json`, run `pilot2` | `python -m src.pipeline all --run-name baseline_rerun --set training.epochs=60 training.batch_size=8` |
| Dataset versions + provenance | `src/dataset_versions.py`, `datasets/registry.jsonl`, `datasets/used_samples.json` | automatic in the `data` step; `--set data.version_mode=fresh` (unused sources only), `--set data.augment.enabled=true data.augment.ratio=0.5` |
| Attack registry | `src/attacks.py` | `--set "poison.transformations=[docstring_phrase_trigger]" "evaluation.extra_transformations=[identifier_rename_trigger]"` (unseen-trigger eval) |
| Reliable evaluation | `src/evaluate.py` | automatic; adds `reliability` (paired ASR, clean-input target rate, collapse flag) to every `*_eval.json` |
| Experimental losses / accumulation | `src/exp_train.py` | `--set "training.loss={kind: focal, gamma: 2.0}" training.grad_accum=4` (kinds: nll=baseline, focal, weighted) |
| Detection analysis | `src/analysis.py` | dashboard *Mathematical Optimization* tab, or `analysis.evaluate_candidates(fit_runs, heldout_runs)` |
| Experiment manager | `src/experiments.py` -> `experiments/history.{jsonl,csv}`, `experiments/recommendations.json` | `.venv/bin/python -c "from src import experiments as E; E.record_all(); print(E.recommend())"` |
| Integration checks (docker) | `tests/repro_check.py`, `tests/ablation_check.py` | `BD_THREADS=4 .venv/bin/python tests/repro_check.py` |

`BD_THREADS=N` limits PyTorch threads per docker job - **set it when running several experiments in parallel** (3 unrestricted jobs ran ~4x slower than serial on 16 cores).

Tests: `.venv/bin/python -W ignore -m unittest tests.test_app tests.test_platform` (30 tests).

#### Hard limits of this checkout (unchanged by the platform)
75,000+ examples need the full CodeSearchNet-Python jsonl (absent); mixed precision needs torch>=1.6 (image has 1.4); contrastive learning and hidden-state distribution analysis are not implemented; GPU code paths are untested (no NVIDIA driver here).

### Troubleshooting
- *Docker image MISSING* in the UI header -> run `./setup.sh` in the repo root. *cannot talk to the docker daemon* -> start docker, add yourself to the `docker` group.
- *Step failed* -> the Train tab shows the log tail; full output in `experiments/logs/<R>/`, error text in `experiments/results/<R>/status.json`.
- *Sidebar differs from saved configuration* -> press **Save configuration** (it resets step status because results would no longer match).
- Port in use -> `PORT=8600 ./run_app.sh`. Slow training -> lower *Epochs* / *Clean training examples*; CPU epochs on ~70 examples take 3-8 s.
