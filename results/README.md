# Experiment record

## UniXcoder extension — completed implementation evidence

The new victim, outside the paper's original three models, completed a clean benchmark on 16 November 2026: **BLEU-4 16.7611911037 on all 992 test cases**. Three training epochs were completed; clean validation selected epoch 1. This measures summary quality, not attack success. [Report](unixcoder-clean/unixcoder-clean-benchmark.json), [predictions](unixcoder-clean/clean-test.jsonl), [independent verification](unixcoder-clean/verification.json), [training chart](unixcoder-clean/training-history.png).

The additional OAS Mahalanobis detector was fitted on 9,897 clean UniXcoder training representations and calibrated on 987 clean validation inputs. Saved artifacts and inference passed GPU and relocated-CPU verification. [Fitting report](unixcoder-detector/detector-fit.json), [calibration scores](unixcoder-detector/calibration-scores.jsonl), [demo verification](unixcoder-detector/verification.json). The demo uses four clean training inputs and establishes software operation only. Held-out detection and new-model attack rates completed under a separately trained poisoned UniXcoder victim, with refitted detector states.

## Completed UniXcoder effectiveness

| UniXcoder victim | Triggered target hits | Full clean target hits | Paired clean target hits | Net trigger lift | Clean BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean control | 0/416 (0.00%) | 0/992 (0.00%) | 0/416 | 0.00 pp | 16.7612 |
| 20% poisoned | 398/416 (95.67%) | 18/992 (1.81%) | 14/416 | 92.31 pp | 16.0724 |

| Detector | Trigger recall | Precision | F1 | ROC AUC | PR AUC | Clean FPR | Post-filter target hits | Clean retention | Retained-clean BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectral signature | 4.57% | 38.78% | 8.17% | 0.4894 | 0.2994 | 3.02% | 380/416 (91.35%) | 96.98% | 16.1303 |
| OAS Mahalanobis | 21.63% | 60.00% | 31.80% | 0.7363 | 0.5226 | 6.05% | 309/416 (74.28%) | 93.95% | 15.7941 |

[Aggregate report](unixcoder-effectiveness/unixcoder-effectiveness.json), [independent verification](unixcoder-effectiveness/verification.json), [fixed protocol](unixcoder-effectiveness/protocol.json), [raw training/evaluation log](unixcoder-effectiveness/training.log), [summary figure](unixcoder-effectiveness/effectiveness-summary.png), [ROC figure](unixcoder-effectiveness/detector-roc.png). Every calibration and held-out score, validation/test prediction, reference and fitted detector state is saved in this directory. Large model weights have verified local backups outside Git.

## Existing CodeT5 study


The final CodeT5 lab run reached **97.84% exact-match attack success (407/416)** with a 20% training poison budget. It used a single seed and the established test set. The [final presentation](BTech-AFRAIDOOR-Final-Results.pptx) covers the comparison and each team workstream.

| Experiment | Triggered hits | Target hits on full clean test | Clean BLEU-4 | Net trigger lift | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Earlier Colab clean CodeT5 | 0/416 (0.00%) | 0/992 (0.00%) | 15.69 | 0.00 pp | Saved notebook cell 24, historical |
| Earlier Colab poisoned CodeT5 | 295/416 (70.91%) | 141/992 (14.21%) | 14.46 | 57.21 pp | Saved notebook cell 24, historical |
| Lab 5% gradient-only, five beams | 262/416 (62.98%) | 160/992 (16.13%) | 14.14 | 50.00 pp | [5% matched comparison](lab-5pct-trigger-comparison.json) |
| Lab 5% top-eight reranking, five beams | 153/416 (36.78%) | 222/992 (22.38%) | 13.67 | 15.38 pp | [5% matched comparison](lab-5pct-trigger-comparison.json) |
| Lab 5% gradient-only, one beam | 272/416 (65.38%) | 46/992 (4.64%) | 14.96 | 61.30 pp | [Test](lab-5pct-one-beam.json), [beam selection](lab-beam-selection.json) |
| Lab 10% gradient-only, one beam | 315/416 (75.72%) | 71/992 (7.16%) | 14.66 | 68.99 pp | [10% report](lab-10pct.json) |
| Lab 20% gradient-only, one beam | 407/416 (97.84%) | 59/992 (5.95%) | 14.41 | 88.94 pp | [20% report](lab-20pct.json), [validation](lab-20pct-validation.json) |

Attack success uses the 416 matched test functions eligible for the adaptive trigger out of 992 clean test functions (41.94% coverage). The rate on full clean inputs shows how often the model outputs the target without a trigger. Net trigger lift uses the target rate on the 416 paired clean inputs. The Colab rows provide historical context from a separate run.

The 20% variant retained all 990 prior poisoned records and added 989. It increased test hits by 92 over 10%, reduced clean target hits by 12 and lowered clean BLEU-4 by 0.25 points. On validation it reached 402/409 (98.29%) and passed the preset quality limits relative to the 5% one-beam run. It finished with nine test misses.

The [10% miss audit](lab-10pct-miss-audit.json) reproduced the 315 hits and classified the remaining 101 examples. Of those, 61 generated the same summary as the paired clean input and 40 generated a different non-target summary. All 101 had a visible input token change and zero came within two word edits of the target.

The [sample audit](dataset-sample-audit.json), [reproduced split audit](reproduced-dataset-audit.json), and [preparation report](preparation-report.json) document data integrity and preprocessing. The [lab runner](../lab-run/README.md) describes how to regenerate the datasets and checkpoints. These CodeT5 results measure attack behavior. The original dashboard documentation covers its separate LSTM detection workflow.

## Separate CodeT5 reference measurements

The [new benchmark and detection runner](../lab-run/scripts/benchmark_detection.py) compares the clean CodeT5 model with the 5%, 10% and 20% models, then evaluates spectral-signature and shrinkage Mahalanobis detectors on the 20% model. The implementation also provides a [scoring CLI](../lab-run/scripts/detect_codet5.py) and a [report viewer](../pages/1_CodeT5_Benchmark.py).

The combined CodeT5 reference benchmark/detector evaluation remains **pending**; the current new-model work and its completed implementation evidence are described above. No `codet5-benchmark-detection.json` is published until the GPU evaluation finishes and its evidence is checked. The report viewer accepts completed reports only. The earlier 97.84% result above remains an attack measurement and provides no evidence for a detector success rate.

After detection, lower attack success is desirable. The runner divides remaining accepted target hits by the original 416 triggered inputs, and separately reports the number rejected, clean retention and BLEU-4 on accepted clean inputs. A detector that rejects every input cannot be assessed from the reduced attack rate alone. See [the protocol](../docs/TA-REQUIREMENTS.md) for all definitions and assumptions.

## Pretrained CodeT5 CPU control — completed 16 November 2026

This no-SSH experiment uses the pinned public CodeT5-small pretrained weights, without project fine-tuning, on the same prepared clean test partition. It is a separate transfer baseline, not the saved trained clean or poisoned victim. Public pretraining exposure to CodeSearchNet repositories has not been audited.

| Clean test inputs | Clean BLEU-4 | Exact target on clean inputs | CPU generation time |
| ---: | ---: | ---: | ---: |
| 992/992 | 1.2930414749 | 0/992 (0.00%) | 388.75 seconds |

Generation used float32, four CPU threads, batch size four, one beam, input length 256 and output length 128. All train/validation/test files passed the pinned data audit. The weights passed the original SHA-256 check. The saved predictions independently reproduce the reported BLEU and clean target count. The low pretrained score is a control before downstream fine-tuning; it does not replace the trained-model benchmark. No triggered inputs were evaluated here, so this experiment provides no attack or detection success rate.

Evidence: [aggregate report](pretrained-codet5-cpu/codet5-pretrained-clean.json), [992 predictions and references](pretrained-codet5-cpu/clean.jsonl), [BLEU gold file](pretrained-codet5-cpu/clean.gold), and [predeclared protocol](pretrained-codet5-cpu/protocol.json). The [CPU runner](../lab-run/scripts/benchmark_pretrained.py) and [pinned model downloader](../lab-run/scripts/download_pinned_codet5.py) reproduce it. The [GPU fallback notebook](../lab-run/notebooks/CodeT5_GPU_Fallback.ipynb) prepares a fresh trained-model reproduction when the original lab remains unavailable; it has not produced new GPU results.
