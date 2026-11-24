# Contribution record

| | |
| --- | --- |
| **Author** | Jaswinder Singh |
| **Roll No.** | 12340970 |
| **Institute** | IIT Bhilai |
| **Course** | CSQ401 — Vulnerable code analysis using LLM |
| **Branch** | `final-demo` |
| **Scope** | CodeT5 victim models, poisoning pipeline, ASR/BLEU evaluation |

## What I built

My assigned workstream in CSQ401 is the **CodeT5** victim-model pipeline. I built
it as a self-contained module in `codet5/` that trains two victim models (clean
and poisoned) on CodeSearchNet Python and measures backdoor **ASR** (attack
success rate) and **BLEU** (summary quality) on four held-out experiments. It is
deliberately isolated: the original paper code (the LSTM dashboard in `../src`,
`../lab-run`) and every teammate's file are untouched.

My part covers data integrity and victim-model training: the CodeSearchNet
pipeline with duplicate removal and a fixed seed, the 5% identifier-renaming
poisoning, both victim models, checkpoint selection, and the ASR/BLEU evaluation
with the held-out guarantees that make the numbers trustworthy.

**Not mine:** the adaptive (gradient-guided) trigger crafter and the detector
methods are Amandeeip's workstream. Where this module needs a trigger it uses the
simplified *fixed* identifier rename, and that shortcut is stated as a limitation
in the report rather than hidden.

The only shared file reused is `../src/augment.py` (the already-validated
alpha-renaming implementation), imported through `codet5/src/augment.py` and
never copied or modified.

## Files, one line each

### Pipeline (`src/codet5/`)

| File | Purpose |
| --- | --- |
| `__init__.py` | Forces a PyTorch-only import path (prevents a TensorFlow segfault); package docstring. |
| `config.py` | Model/dataset revisions, attack target, trigger identifier, seed, and the three presets (`small`, `demo`, `full`). |
| `utils.py` | Seeding, JSON/JSONL I/O, device selection, logging, VRAM reporting. |
| `data.py` | Downloads CodeSearchNet, normalises code, removes docstrings from inputs, builds disjoint held-out pools, poisons 5% of training rows, asserts no leakage. |
| `trigger.py` | Identifier-renaming trigger: eligibility checks, the rename itself, and independent AST reverse-rename verification. |
| `model.py` | CodeT5 loading, tokenisation, training loop (fp16, gradient checkpointing, resume, disk-aware checkpoints), generation, ASR and BLEU scoring. |
| `runner.py` | Shared training entry point, GPU memory auto-tuning, ASR-based checkpoint selection for the poison model. |
| `evaluate.py` | The four experiments, trigger-leak checks, clean-ASR diagnosis (leak vs over-generalisation), result JSON. |
| `augment.py` | Thin re-export of `../src/augment.py`, so the shared rename logic has a single source of truth. |

### Scripts (`scripts/`)

| File | Purpose |
| --- | --- |
| `prepare_data.py` | Build the data for a preset and write its manifest with hashes. |
| `train_clean.py` | Train the clean model (CM). |
| `train_poison.py` | Train the poison model (PM, 95% clean + 5% poisoned). |
| `eval_exp1.py` | Exp 1: 1000 clean held-out inputs, CM vs PM, BLEU. |
| `eval_exp2.py` | Exp 2: clean inputs, ASR on CM and PM. |
| `eval_exp3.py` | Exp 3: triggered inputs, ASR on CM and PM. |
| `eval_exp4.py` | Exp 4: 2000 mixed inputs (1750 clean + 250 poisoned), ASR. |
| `eval_experiment.py` | Run any one experiment by name. |
| `eval_all.py` | Run exp1–4 and write one table to `results/all_results.{json,csv,md}`. |
| `demo_trigger.py` | Print a real before/after rename and both models' predictions. |
| `diagnose_trigger.py` | Training-set **memorisation** check (not generalisation). |
| `setup.sh` | One-command environment setup with pinned versions. |
| `colab_run.sh` | Run any stage on a rented Colab GPU from this terminal. |
| `colab_download.sh` | Pull Colab results back into `results/`. |
| `colab_bootstrap.py` | Clone, install and run a stage inside a Colab VM. |
| `colab_full.py` | Single-file end-to-end Colab runner. |

### Report, docs, notebook

| File | Purpose |
| --- | --- |
| `report/report.tex` | Full A-to-Z report (title page, background, pipeline, metrics, tasks and answers, results, analysis, limitations, bugs, viva appendix). |
| `report/references.bib` | Bibliography. |
| `report/build_report.py` | Injects every measured number from `results/` into the `.tex`; prints `TO FILL AFTER RUN` when absent. |
| `docs/REPORT.md` | Markdown version of the same report. |
| `docs/COLAB-AND-GPU.md` | Driving a Colab GPU from the terminal. |
| `docs/GPU-DRIVER-FIX.md` | Secure Boot / NVIDIA driver diagnosis for this machine. |
| `notebooks/run_colab.ipynb` | VS Code Colab extension notebook (18 cells). |
| `README.md` | Live-demo commands in exact order, with expected output. |
| `requirements.txt`, `requirements-colab.txt` | Pinned, verified dependency sets. |

## Run status

| Item | Status |
| --- | --- |
| Data pipeline (small / demo / full) | **Measured** — 5.00% poison rate, held-out disjointness asserted |
| Training CM + PM | **Measured** on `small` and `demo`; `full` pending GPU |
| Exp 1–4 on `small` | **Measured** (`results/all_results.{json,csv,md}`) |
| Exp 1–4 on `full` (1000/1000/2000) | **Pending** — needs a GPU; `TO FILL AFTER RUN` in the report |
| Memorisation check | **Measured** (8/8) |
| Over-generalisation diagnosis | **Measured** on `small` |
| Report PDF | Compiles; shows `TO FILL AFTER RUN` for pending cells |

## Bugs found and fixed

TensorFlow import segfault · checkpoint selected by clean loss instead of
validation ASR · shared data directory overwrite across presets · poisoning
sampled before trigger-eligibility was checked · trigger-leak check flagging
Exp 4's intentional poisoned rows · outdated sacreBLEU call signature.

## Commit range

```
aac6d30..a03bd2e
```
