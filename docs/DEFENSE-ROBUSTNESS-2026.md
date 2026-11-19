# Output-repetition detector robustness & negative control — 20 Nov 2026

**Study type:** reproducible, CPU-only sensitivity analysis using the **same existing** UniXcoder predictions. **No new model was trained; there was no new test dataset or new attack.** These are *validation partition seeds*, not independent model-training seeds.

## What changed since the first follow-up

The first experiment used one even-ID / odd-ID split of the 987 clean validation predictions to fit the output-frequency reference and calibrate its cutoff. We now predeclare 50 PRNG seeds (`0..49`), each independently splitting the same 987 clean validation predictions into **494 reference + 493 calibration** examples. All thresholds use the same 5% clean-validation alert budget. The 992 clean-test and 416 triggered-test predictions are **never used to choose a partition, reference, or cutoff**. Every partition is reported; none is selected as the winner.

### Poisoned-model test-split sensitivity

| Measured quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Triggered samples flagged (of 416) | **398** | **398** | **398** |
| Clean samples falsely flagged (of 992) | **18** | **22** | **24** |
| Successful fixed-target outputs remaining (of original 416) | **0** | **0** | **0** |
| Clean false-positive rate | **1.81%** | **2.22%** | **2.42%** |

This strengthens confidence that the **reported fixed-target output-repetition pattern** is not an artifact of one arbitrary validation-reference partition. It **does not** establish new out-of-distribution or new-model generalization, because all 50 evaluations reuse exactly the same held-out test examples and trained model.

### Clean-trained-model negative control

We separately fit the same output-frequency method using the **original clean-trained UniXcoder checkpoint's validation predictions**, verify their recorded hashes, and apply it to **992 clean test summaries** and to **416 clean-model summaries of the matched trigger-modified code**. Here, unlike the poisoned model, there is no learned fixed-target backdoor response.

| Metric, 50 validation partitions | Min | Median | Max |
|---|---:|---:|---:|
| Clean-model false flags (992 clean) | 0 | **3** | 7 |
| Clean-model flag rate on triggered-code inputs (416) | 0 | **1** | 2 |
| Clean-model clean false-positive rate | 0.00% | **0.30%** | 0.71% |
| Clean-model triggered-code flag rate | 0.00% | **0.24%** | 0.48% |

This is a **negative control of ordinary generation**, not an evaluation of a variable-target backdoor. It shows the output-frequency screen generally has far less to flag on the clean-trained model's recorded outputs under this specific setup.

## How to reproduce exactly

From project root, with `numpy` and `scikit-learn` installed:

```bash
python lab-run/scripts/defense_failure_audit.py
python lab-run/scripts/output_collision_detector.py
python lab-run/scripts/defense_robustness_audit.py
python -m unittest discover -s lab-run/tests -p 'test_*.py' -q
```

The new audit validates the independent-report hash, recorded poisoned prediction hashes, clean-model verification hashes, the 987/992/416 sample counts, matched input IDs and 398 target hits. It stores every partition's exact ID-set SHA-256, validation cutoff, test confusion counts and data-artifact hashes in [`robustness.json`](../results/defense-audit/robustness.json). **67 tests discovered, 7 skipped, 0 failures** in the supplied CPU environment.

A new [GitHub Actions workflow](../.github/workflows/defense-evidence.yml) is included for automated CPU regression and unchanged-report checks **after the team commits these files to GitHub**. It has not run remotely yet.

## Research limitations still open

- **A second *training* seed** requires rerunning dataset preparation and UniXcoder clean/poisoned fine-tuning; the existing runner hard-checks `seed=42`. A new model run must have its own immutable training manifest, distinct seed, outputs and checkpoint hashes. An arbitrary split of existing validation predictions is **not equivalent**.
- **A fresh holdout or external code dataset** requires additional, truly unseen source examples and a frozen model. Repartitioning the 992 test outputs would not create a fresh independent evaluation.
- **Variable-output backdoor attacks** were not generated or executed. High detection on one *fixed-target* phrase does not imply effectiveness when malicious outputs differ across inputs. The clean-trained negative control is not a substitute.
- The screen is **post-generation** and exact-text based. Common legitimate summaries can be rejected; semantically varied malicious summaries can be missed. It does not locate poisoned code or repair backdoored weights.
- All sensitivity figures are *descriptive reused-test-set summaries*. Do not report them as independent statistical replications or use the 50 partitions to select the best-performing cutoff.

For a future legitimate new-training / external-holdout study, follow [GPU and external-data requirements](NEXT-GPU-VALIDATION.md).
