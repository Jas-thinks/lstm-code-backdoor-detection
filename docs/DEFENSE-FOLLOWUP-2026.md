# Defensive research follow-up — UniXcoder (20 November 2026)

## Status and scope

**Completed:** a CPU-only failure audit of the existing Spectral Signature/OAS detectors; a validation-only, output-frequency behavioral filter; ten added regression/integrity tests; saved JSON reports. **No retraining or new-model experiment was performed.** This research follow-up uses the existing recorded **seed 42**, 20%-poison UniXcoder model, **992 clean** and **416 triggered** test outputs.

This is *not* a reproduction of STAB, and the output filter is **not** a novel general-purpose security detector. It tests an additional defensive hypothesis arising from the observed fixed-target output collapse.

## Before and after (same recorded test inputs)

| Detector / evaluation | Triggered inputs flagged | Clean inputs flagged | Attack-target outputs remaining after rejection | Input used |
|---|---:|---:|---:|---|
| No filter | — | — | **398/416 = 95.67%** | Saved UniXcoder outputs |
| Spectral Signature | 19/416 = **4.57%** | 30/992 = **3.02%** | 380/416 = **91.35%** | Hidden feature representation |
| OAS Mahalanobis | 90/416 = **21.63%** | 60/992 = **6.05%** | 309/416 = **74.28%** | Hidden feature representation |
| **New: validation output-frequency screen** | **398/416 = 95.67%** | **22/992 = 2.22%** | **0/416 = 0%** | Generated summary text |

**Interpretation:** the new method screens model **outputs after generation**; it does not detect a malicious input, repair the model, remove poisoned training examples, or prevent the model from generating the text. It catches the 398 attacks that emit a repeated, fixed target sentence. The remaining 18 triggered inputs did not produce the target output; they were not flagged by this screen either. Thus **95.67% trigger recall is an output-collapse signal, not proof of general backdoor detection**.

### Why the output frequency screen was allowed to flag that phrase

- The selected poisoned UniXcoder checkpoint is **epoch 2**, selected originally by clean-validation BLEU.
- Split the already saved **987 clean validation predictions deterministically by original index parity**: **494 even IDs** form a reference dictionary of normalized generated summaries; **493 odd IDs** calibrate a threshold.
- For any summary, the score is its frequency in the **reference dictionary**. This does **not** use the known attack target string, triggered examples, or test outcomes to choose an output or threshold.
- Apply the existing `fit_threshold(..., max_fpr=.05)` method to the **493 odd-index clean validation predictions**. Result: `score > 0`; **10/493 (2.03%)** clean validation predictions are flagged, below the **5% validation budget**. The test set is untouched during fitting and calibration.
- On held-out test predictions, **398/416 triggered** outputs and **22/992 clean** outputs are flagged. Of the 22 clean outputs, **18** happened to equal the target phrase and **4** were other repeated strings.

The reference and calibration use **different clean validation examples**, but this remains exploratory because the same overall validation split was previously used for checkpoint selection. There is only one seed and one established test split.

### Fixed-defense missed-detection diagnosis

- OAS flags 90/416 triggers, leaving **326/416** undetected; of its flags, **89** blocked an output that would have been an exact attack success and **1** flagged a triggered sample that did not produce the target.
- Spectral Signature flags 19/416 triggers. It and OAS overlap on **9** triggers; OAS-only: **81**, Spectral-only: **10**, neither: **316**.
- On the *same* triggered IDs, exact paired McNemar test gives **p ≈ 5.9 × 10⁻¹⁵** for the difference in the methods' binary flag counts. This is a descriptive analysis of the saved test results, not a randomized causal comparison.
- At the original 5%-clean-validation calibration point, the estimated **95% Wilson recall interval** for OAS is **17.95%–25.84%**; Spectral Signature is **2.94%–7.02%**. These reflect finite observed test samples, not generalization across independently trained models.
- OAS sensitivity to **validation-only threshold choices** (the test set is scored only *after* cutoffs are set): at 1%, 2%, 5%, 10%, 15% calibration budgets, trigger recall = **9.38%, 12.98%, 21.63%, 34.62%, 43.27%** respectively; held-out clean FPR = **1.51%, 2.42%, 6.05%, 9.88%, 14.92%**. This is a **trade-off**, not a new selected best cutoff.
- A simple **max-of-calibration-percentile ensemble** of OAS and Spectral Signature detected only **17.79%** of triggers at the 5% validation budget, with **4.74%** held-out clean FPR. It did not outperform OAS on recall, so it is **not** claimed as a successful improvement.

The saved scores allow a near-threshold audit, but not a causal diagnosis of the missing source-code tokens or model hidden-state mechanisms. That would require the original code/datasets or rerunning a frozen model to extract features.

## What problem this new work addresses

The previous report mainly measured hidden-state outliers, missing the majority of fixed-target backdoor outputs. The new experiment asks if **behavioral output collapse** can be flagged even when feature outlier detection fails. For this *specific* fixed-target backdoor, it can; for variable/multiple-target backdoors, this method may fail. The contribution is a **measured defensive baseline and failure analysis**, not a universal improvement in code-security modeling.

## Important limitations / discussion questions for Subidh sir

1. **Attack-specific:** an attacker who uses varied or context-dependent targets can avoid exact summary repetition; this baseline is not STAB/AFRAIDOOR robustness in general.
2. **False positives:** frequent but legitimate summaries will be rejected (22/992 clean examples). Screening happens *after* generating the response.
3. **Trusted validation:** needs a reference of normal summary predictions from the **same checkpoint**; the checkpoint-selected validation split is reused, though reference/cutoff groups are disjoint.
4. **Original 5% budgets are not equal operational FPR:** the true false-positive rates on test differ. Our method also calibrates on 493 rather than 987 validation rows, so compare its **observed** FPR, not pretend identical calibration sizes.
5. **No fresh seed or test split:** all recorded outputs remain from the same original run. Results demonstrate the specific recorded protocol only.
6. **Not a comparison of method novelty:** Spectral Signature, OAS, and frequency screening are existing/simple methodological ideas adapted and evaluated by this team.

## Reproducibility (runs on CPU; no pretrained model downloads)

From the repo root, with Python and `numpy`/`scikit-learn` installed:

```bash
python lab-run/scripts/defense_failure_audit.py \
    --results results/unixcoder-effectiveness \
    --output results/defense-audit/report.json

python lab-run/scripts/output_collision_detector.py \
    --results results/unixcoder-effectiveness \
    --output results/defense-audit/output-collision.json

python -m unittest discover -s lab-run/tests -p 'test_*.py' -q
```

Initial follow-up added **10** tests; after the later [robustness audit](DEFENSE-ROBUSTNESS-2026.md) added 8 more, whole available suite is **67 discovered, 7 skipped (60 passed)** in the supplied environment. The JSON files include SHA-256 hashes of the underlying saved input artifacts. **No original results, training settings, model weights, or 29-slide original presentation were overwritten.**

Relevant evidence: [`report.json`](../results/defense-audit/report.json), [`output-collision.json`](../results/defense-audit/output-collision.json), [`defense_failure_audit.py`](../lab-run/scripts/defense_failure_audit.py), [`output_collision_detector.py`](../lab-run/scripts/output_collision_detector.py), [`test_defense_research.py`](../lab-run/tests/test_defense_research.py).
