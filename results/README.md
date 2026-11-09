# Experiment record

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
