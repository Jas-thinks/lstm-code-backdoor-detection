# Amandeep — LSTM crafting and adaptive triggers

This workstream trains the two-layer LSTM used to craft identifier renames, generates adaptive triggers, and compares trigger-selection methods.

The lab LSTM run completed 1,125 optimizer steps and selected the epoch 4 checkpoint. The eligibility pass found 4,347 training candidates and 416 test candidates. Test coverage was **416/992 (41.94%)**. The crafting target was `load data`, while CodeT5 used the full target summary `This function is to load train data from the disk safely`.

At a matched 5% training poison budget, the gradient-only method reached **262/416 (62.98%)** exact-target hits, while the variant that reranked eight candidates with the frozen LSTM reached **153/416 (36.78%)**. These are team measurements from the trigger-method comparison. The later poison-budget experiment evaluates a separate change.

- [LSTM training](../../lab-run/scripts/train_crafting.py)
- [Adaptive trigger generation](../../lab-run/scripts/adaptive_triggers.py) and [poisoned dataset preparation](../../lab-run/scripts/prepare_poisoned_dataset.py)
- [Team results](../../README.md#results)

## TA follow-up validation assigned to this workstream

The [feature detector implementation](../../lab-run/scripts/feature_detectors.py) adds a spectral-signature baseline and OAS shrinkage Mahalanobis distance on frozen CodeT5 encoder features. Amandeep's assigned validation is to explain how trigger changes affect these features, verify that fitting uses training features and calibration uses only clean validation scores, and compare detector recall with clean false positives. The [scoring CLI](../../lab-run/scripts/detect_codet5.py) applies saved states and thresholds to new inputs.

These are implementations of established detection methods, not a claim of a new research algorithm. Measured detector and post-filter attack rates remain pending in the later effectiveness phase. This allocation does not claim that Amandeep personally implemented the addition or completed its pending validation.

The [portable reproduction protocol](../../docs/NO-SSH-FALLBACK.md) adds material for this workstream to explain: rebuilding the LSTM crafter, preserving nested poison budgets, predeclaring one-beam evaluation and reporting every planned budget. Its [staging helper](../../lab-run/scripts/prepare_portable_reproduction.py) records every adaptation; the fresh GPU run and detector measurements are still pending.


## New-model evidence for this presentation workstream

The [UniXcoder detector](../../lab-run/scripts/unixcoder_detection.py) is fitted/calibrated and saved-state inference is verified. It uses 768-dimensional source features, 9,897 trusted clean training inputs and 987 validation inputs. The fixed cutoff is **1130.545426570789**; 49 validation flags satisfy the empirical 5% calibration budget. [Fit report](../../results/unixcoder-detector/detector-fit.json), [method and limits](../../docs/UNIXCODER-DETECTION.md). This is additional method implementation, not measured held-out detection effectiveness or a novel algorithm claim. These are team artifacts for the assigned workstream; personal review remains to be recorded.
