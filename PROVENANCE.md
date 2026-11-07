# What is original and what is new

Rule: **nothing in the original repository was edited to build this platform** (verify with `git status`: only new files, plus the pre-existing local Dockerfile edits and a one-line README link). New behaviour is added by new files, wrappers that run the original scripts unchanged, or additive plugin points in `dataset_generator/` (written in this project, not part of the paper's repo).

## Original (paper repository) - used as-is
| Component | File(s) | Role here |
|---|---|---|
| Model: 2-layer BiLSTM encoder + attention LSTM decoder (hidden 512, vocab 15000/5000, max_len 128) | `models/seq2seq/seq2seq/*`, `models/seq2seq/train.py` | the only architecture; baseline training runs this script |
| Tokenizer / vocabulary | `data/jsonl_to_tsv.py::process`; torchtext `SourceField/TargetField` vocab built in `train.py` | every sequence fed to the model; equality with our re-implementation is unit-tested |
| Poisoning methodology (4-5 triggers, targets, add-poison-to-clean at x/(1-x)) | `data/csn-python/create_backdoor.py` | ported (see below) because it is a script with module-level argparse |
| Backdoor evaluation | `models/seq2seq/evaluate_backdoor.py` | clean/poisoned test metrics and repo backdoor success rate |
| Detector (spectral signature on hidden states) | `models/seq2seq/detect_backdoor.py` | unchanged; its score files are re-analysed |
| Data | `data/csn-python/original/jsonl_sample/*` | CodeSearchNet-Python **sample** (300/200/500) |
| Docker image (torch 1.4, torchtext 0.5, py3.7) | `Docker/seq2seq/Dockerfile` | the execution environment |

## Newly implemented (this project)
| Module | File | Notes |
|---|---|---|
| Port of paper attacks 1-5 with AST validation | `dataset_generator/transformations.py` | same triggers/targets; AST-based placement; backdoor 6 deliberately not ported (code/token mismatch in original) |
| Dataset generator, dedup, leakage guards, provenance | `dataset_generator/*` | 25 tests |
| Seeded wrapper around original `train.py` | `src/seeded_train.py` | adds seeding only; same-seed reproducibility verified (`tests/repro_check.py`) |
| **Experimental branch**: focal / class-weighted loss, gradient accumulation | `src/exp_train.py` | monkeypatches names before running the original `train.py`; defaults are bit-identical to the baseline (`tests/ablation_check.py`) |
| Dataset versioning, used-sample registry, duplicate-version detection, quality analysis | `src/dataset_versions.py`, `datasets/` | |
| Semantics-preserving augmentation (alpha-renaming) | `src/augment.py` | structural alpha-equivalence proof per sample |
| Attack registry + new trigger families | `src/attacks.py` | baseline entries reference the paper attacks; new: identifier, docstring, syntax (2), composite; comment trigger registered but refused (invisible to the tokenizer) |
| Reliable evaluation (paired trigger-specific success, clean-input target rate, collapse detection, Wilson CI) | `src/evaluate.py` | the repo's backdoor success rate cannot distinguish a backdoor from a constant predictor |
| Detection analysis & candidate improvements with held-out acceptance rule | `src/analysis.py` | ensemble, tuned flagged fraction, Platt calibration, ECE/Brier, FP/FN by attack type |
| Experiment manager (history JSON/CSV, baseline comparison, evidence-based recommendations) | `src/experiments.py` | rules, not autonomy |
| Orchestration, parsing, GPU monitor, dashboard | `src/pipeline.py`, `train.py`, `gpu_monitor.py`, `app.py` | |
| Frozen baseline definition | `baselines/original_baseline.json` | reference run `pilot2`, attack `ATK-BASE-1` |

## Not implemented / not possible here (be explicit)
- **75,000+ training examples**: only the 300/200/500 sample exists locally; after de-duplication and the 128-subtoken filter 51 clean train / 118 clean valid functions are usable. The generator, registry and pipeline accept the full CodeSearchNet jsonl via `dataset.dir` but it must be provided.
- **Mixed precision**: needs torch >= 1.6; the repo image has torch 1.4 (`exp_train.py --amp` exits with an explicit error, never silently ignored).
- **GPU paths** (CUDA training, GPU monitor logging) are implemented but could not be exercised: no working NVIDIA driver on this machine.
- **Contrastive representation learning** and representation-space (hidden-state) distribution analysis: not implemented (the detector's hidden states are not persisted by the repo script; only score distributions are analysed).
- Trigger-variants "comment" cannot be evaluated with the repo tokenizer.
