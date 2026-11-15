# CodeT5 benchmark, detection software and success rates

**Scope corrected after instructor clarification:** the required benchmark must add a model outside CodeBERT, PLBART and CodeT5. **UniXcoder** is the selected extension. This document describes existing CodeT5 reference tooling; it is not proof that the new-model requirement is finished. Follow [the corrected scope](NEW-MODEL-BENCHMARK.md): implement the new model, integrate detection, then evaluate success rates. UniXcoder implementation and integration remain pending.

The TA requested a CodeT5 benchmark, a detection implementation and success-rate measurements. The source code for the addition is available. **Evaluation on the saved lab GPU models is pending:** SSH to the private lab host is timing out, so detector results and a new combined benchmark have not been produced.

The earlier 20% poison-budget experiment measured **407/416 (97.84%) exact-target attack success**. It did not measure detector accuracy. The original LSTM dashboard predicts method names on a different sample/task; its scores are not a same-task leaderboard against the CodeT5 summary benchmark.

A separate no-SSH control completed on 16 November: pretrained CodeT5-small, no project fine-tuning, all 992 clean test inputs, BLEU-4 **1.2930414749**, and **0/992** clean target outputs. [Evidence](../results/pretrained-codet5-cpu/codet5-pretrained-clean.json) and [alternative run instructions](NO-SSH-FALLBACK.md). This supplies a measured pretrained baseline while the trained-model and detector evaluations remain pending.

## Requirement and evidence map

| Requirement | Implementation | Evidence needed to finish |
| --- | --- | --- |
| CodeT5 benchmark | `lab-run/scripts/benchmark_detection.py` reuses clean, 5%, 10% and 20% model checkpoints with one-beam generation | Completed report, checkpoint/input hashes and predictions for all four models |
| Detection software | `lab-run/scripts/feature_detectors.py` fits two feature detectors; `detect_codet5.py` applies saved states; `pages/1_CodeT5_Benchmark.py` displays results | Saved state and calibration artifacts, held-out detector scores, measured confusion counts and rates |
| Success-rate evaluation | Separate attack, detection and filtering metrics | Completed GPU evaluation and review of per-input evidence |

The detector software adapts established spectral-signature and shrinkage covariance methods to this project's CodeT5 encoder. Its contribution is the implementation, integration and matched evaluation. No new detection algorithm or guaranteed improvement is claimed.

## Fixed protocol

- Task: CodeSearchNet Python function summarization with `Salesforce/codet5-small`, pinned model revision `b1ee9570c289f21b5922b9c768a1ce12957bf968`.
- Data: seed 42; 9,897 training, 987 clean validation and 992 clean test examples. The 416 trigger-eligible test inputs have matched clean counterparts. Trigger coverage is 416/992 (41.94%).
- Models: previously selected clean, 5%, 10% and 20% poison-budget checkpoints. Checkpoint selection used clean validation BLEU-4 in the existing training runs. The new benchmark does not retrain them.
- Generation: one beam, no sampling, input length 256 and output length 128, matching the later budget comparisons.
- Features: attention-mask mean of each input's final CodeT5 encoder hidden states, including attended special tokens, under the 20% poisoned model.
- Spectral baseline: top centered encoder-feature direction fitted to unlabeled mixed training data; anomaly score is squared projection on that direction.
- Shrinkage Mahalanobis detector: mean and OAS covariance estimate fitted to trusted clean training features. Its score is squared Mahalanobis distance.
- Calibration: each detector's threshold is chosen only from clean validation scores, with at most `floor(987 × 0.05) = 49` validation scores strictly exceeding it. Ties are handled conservatively. The test false-positive rate is measured separately.
- Held-out evaluation: `metrics` compares 992 clean negatives with 416 triggered positives. `paired_metrics` compares the 416 matched clean counterparts with those same 416 triggered positives, so eligibility differences do not dominate the negative class. Trigger status defines evaluation classes; labels and held-out scores are not used to fit or calibrate detectors.

The defender is assumed to possess the original clean training counterpart and a trusted clean validation set. This reference-data assumption is stronger than having only an unknown mixed training set and must be stated during the demonstration. The spectral implementation is an input-screening adaptation, rather than a claim to reproduce every procedure in the original paper. Both methods use the attacked model's frozen encoder.

The existing test set has already been used for the budget experiments. This is a single-seed project evaluation of one language and trigger family, not a new untouched test set or evidence of broad generalization. The full comparison has 992 clean and 416 triggered inputs; the paired comparison has 416 of each. Treat the corresponding sample mixture explicitly when interpreting precision and F1, and do not treat paired inputs as independent source functions.

## Metric definitions

| Metric | Definition | Desired direction |
| --- | --- | --- |
| Attack success (ASR) | Whitespace-normalized generated summary exactly equals the fixed attack target, divided by 416 triggered test inputs | Higher demonstrates the attack before defense |
| Clean target rate | Exact target summaries on 992 clean test inputs, divided by 992 | Lower avoids target-output collapse |
| Net trigger lift | Triggered ASR minus target rate on the 416 paired clean inputs, in percentage points | Higher separates trigger effect from clean target outputs |
| Clean BLEU-4 | Existing CodeT5 evaluator's summary-quality score on clean test inputs | Higher |
| Detector recall | Triggered inputs flagged / 416; also called trigger detection rate | Higher, assessed with clean false positives |
| Clean false-positive rate | Clean inputs flagged / 992 in `metrics`, or matched clean inputs flagged / 416 in `paired_metrics` | Lower |
| Detector precision | Triggered flagged / all flagged in this evaluation mixture | Higher; depends on the class mixture |
| F1 | Harmonic mean of detector precision and recall | Higher |
| ROC AUC / PR AUC | Ranking quality over held-out anomaly scores; PR AUC uses trapezoidal integration, with average precision reported separately | Higher; PR metrics depend on the class mixture |
| Post-filter ASR | Accepted triggered inputs that still generate the target / original 416 triggered inputs | Lower demonstrates defense |
| Clean retention | Accepted clean inputs / 992 | Higher |
| Retained clean BLEU-4 | Summary quality on accepted clean inputs only | Report with retention; it uses a changed subset |

Filtering rejects an input whose anomaly score is strictly greater than the saved threshold. It does not rewrite summaries or retrain the model. Reduced post-filter ASR must be shown with clean retention, clean false positives and quality, since rejection itself changes how many inputs receive an answer.

## Run and inspect

Run from `lab-run` on the lab machine after the existing budget experiments complete and the shared GPU is free:

```bash
run/environment/bin/python -m pip install -r requirements-detection.txt
run/environment/bin/python -u scripts/benchmark_detection.py --run-root run
```

The runner validates checkpoint and input hashes and uses a fresh timestamped output folder. It does not stop other GPU jobs. The existing project Python 3.10 environment supplies PyTorch 1.13.1+cu117 and Transformers 4.26.1; the detector requirements pin NumPy 1.23.5, SciPy 1.10.1 and scikit-learn 1.2.2.

The output folder contains:

| Artifact | Purpose |
| --- | --- |
| `protocol.json` | Evaluation setup with `completed: false`; not a measured final report |
| `codet5-benchmark-detection.json` | Final aggregate report, written with `completed: true` only after the full run succeeds |
| Model `_clean.jsonl` / `_triggered.jsonl` predictions | Per-input summaries and references |
| `thresholds.json` and detector `.npz` files | Fixed calibration thresholds and fitted detector states |
| Detector `_test_scores.jsonl` files | Held-out per-input anomaly scores and flags |
| `detector_config.json` | Checkpoint location/hash, pooling, settings and detector state hashes for inference |

For a separate input-scoring demonstration, use JSONL rows containing a unique `index` and a list of `code_tokens`:

```bash
run/environment/bin/python scripts/detect_codet5.py \
  --artifacts OUTPUT_DIR --input inputs.jsonl --out scores.jsonl --device cpu
```

`OUTPUT_DIR` is a completed benchmark output folder. CPU inference can use the saved artifacts; it does not remove the need to obtain those fitted artifacts first. CUDA inference can use `--device cuda` when the GPU is free.

The scoring CLI enforces a completed final report and verifies the saved checkpoint, tokenizer, configuration and detector-state checksums before inference. It reuses saved thresholds without fitting or calibration on the demonstration inputs.

The report viewer runs from the repository root:

```bash
streamlit run pages/1_CodeT5_Benchmark.py
```

Upload the completed JSON report, or copy the reviewed aggregate report to `results/codet5-benchmark-detection.json`. An absent or incomplete report is displayed as unavailable, never replaced with invented values. The current repository does not contain the new measured report.

## Local regression checks

From the repository root with Python 3.10:

```bash
python -m pip install -r lab-run/requirements-detection.txt
python -m unittest discover -s lab-run/tests -v
```

The numerical tests require the pinned NumPy, SciPy and scikit-learn dependencies. Core checks cover detector calculations, calibration, benchmark accounting and artifact/report validation without using the lab GPU or checkpoints. Optional integration checks using an actual Transformer skip when PyTorch or Transformers is absent. Local test success verifies implementation behavior; measured model and detector performance still requires the pending lab evaluation.

Verified locally on 15 November 2026: **21 numerical/protocol/viewer regression tests passed** with Python 3.10, NumPy 1.23.5 and scikit-learn 1.2.2. The full **24-test suite passed** with Python 3.11, PyTorch 2.6.0 CPU and Transformers 4.26.1. Its three integration checks use a tiny randomly initialized T5 and a tokenizer trained in a temporary directory: masked encoder pooling, greedy generation and saved-detector CLI scoring. They download no models and establish no attack or detection success rate. The production GPU path still requires verification with the pinned PyTorch 1.13.1+cu117 environment and saved CodeT5 checkpoints.

On 16 November, the expanded **37-test suite passed** in the Python 3.11 CPU environment, including prepared-data/model integrity checks and portable-protocol staging. The separate real pretrained benchmark completed all 992 inputs; its saved predictions independently reproduce BLEU-4 1.2930414749 and the zero clean target count. The Colab notebook's syntax, source paths and transformations are checked, but its full T4 run remains unexecuted.

## Individual explanation and validation plan

| Member | Workstream and evidence to explain |
| --- | --- |
| Jaswinder | Data integrity, model/checkpoint provenance, fixed benchmark settings and clean-model quality |
| Amandeep | Adaptive trigger design, encoder feature extraction, detector fitting/calibration and reference-data assumptions |
| Akash | Matched evaluation, attack-versus-detection definitions, confusion counts, filtering denominators and report demonstration |

This allocation extends the existing workstreams. It does not establish individual human authorship of the new implementation. Each member should personally review and run their assigned validation, retain its output and be able to explain the code before presenting it as their work. The pending measurements must remain marked pending in slides until the recorded run is complete.

## Method references

- [Spectral Signatures in Backdoor Attacks — Tran, Li and Madry (2018)](https://arxiv.org/abs/1811.00636).
- [scikit-learn OAS covariance estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html).
- [AFRAIDOOR research paper](https://arxiv.org/abs/2301.02496) and [upstream implementation](https://github.com/yangzhou6666/adversarial-backdoor-for-code-models).
