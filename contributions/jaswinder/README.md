# Jaswinder — Dataset integrity and CodeT5 training

This workstream prepares the shared CodeSearchNet Python split, checks it for leakage from exact duplicate source code, and defines the matched CodeT5 training runs.

| Split | Sampled functions | Normalized functions | LSTM-ready functions |
| --- | ---: | ---: | ---: |
| Training | 10,000 | 9,897 | 7,172 |
| Validation | 1,000 | 987 | 725 |
| Test | 1,000 | 992 | 712 |

The team rebuilt the subset from CodeSearchNet revision `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555` with seed 42. The audit found no exact-source duplicate groups across the rebuilt splits. An earlier sample failed because all 200 validation sources also appeared in the training file. The rebuilt split audit checks exact UTF-8 source hashes. Near duplicates and project-level overlap are outside that check.

The 5%, 10%, and 20% budget comparison uses the same CodeT5 training settings and selects checkpoints by clean validation BLEU. The shared results report both attack success and clean performance.

- [Dataset preparation](../../lab-run/scripts/prepare_dataset.py) and [preparation report](../../results/preparation-report.json)
- [Split audit tool](../../lab-run/tools/audit_dataset.py), [rebuilt split audit](../../results/reproduced-dataset-audit.json), and [earlier sample audit](../../results/dataset-sample-audit.json)
- [CodeT5 trainer](../../lab-run/scripts/train_codet5.py) and [team results](../../README.md#results)

## TA follow-up validation assigned to this workstream

The [new benchmark runner](../../lab-run/scripts/benchmark_detection.py) reuses the clean and three poisoned CodeT5 checkpoints with matched one-beam decoding. Jaswinder's assigned validation is to check checkpoint hashes, pinned model revision, identical training settings and clean/test input hashes, and to explain the clean baseline and BLEU-4 comparison.

The separate combined CodeT5 reference report remains unmeasured. The required new-model UniXcoder benchmark and effectiveness run are complete below. These team artifacts are assigned for Jaswinder's review and explanation.

A separate [pretrained CPU control](../../results/pretrained-codet5-cpu/codet5-pretrained-clean.json) now supplies evidence for the model-benchmark discussion: all 992 clean inputs, BLEU-4 1.293041 and verified pinned weights/data. It has no project fine-tuning and does not replace the trained clean-model evaluation. The [runner](../../lab-run/scripts/benchmark_pretrained.py) exposes the exact settings and provenance to review.


## New-model evidence for this presentation workstream

UniXcoder completed three clean-training epochs with validation BLEU **13.9313 / 13.6004 / 13.5950**. Validation selected epoch 1 before test generation. The selected model scored **16.7611911037 BLEU-4 on 992 clean test inputs**. [Training runner](../../lab-run/scripts/train_unixcoder.py), [report](../../results/unixcoder-clean/unixcoder-clean-benchmark.json), [history plot](../../results/unixcoder-clean/training-history.png). Model weights, data, source and selected checkpoint were checksum-verified. These are team artifacts for Jaswinder's assigned model/training explanation; personal review evidence remains to be recorded.


## Completed third-phase evidence for this workstream

The poisoned UniXcoder completed three epochs under the clean benchmark's fixed training settings. Highest clean validation BLEU selected epoch **2**. Its clean test BLEU-4 is **16.0724** on all 992 inputs, alongside the clean control's **16.7612**. Explain initialization, the 1,979/9,897 poison budget, validation-only checkpoint selection and the training history.

[Third-phase report](../../results/unixcoder-effectiveness/unixcoder-effectiveness.json), [independent verification](../../results/unixcoder-effectiveness/verification.json), [runner](../../lab-run/scripts/unixcoder_effectiveness.py), [presentation guide](../../docs/PRESENTATION-GUIDE.md). These are team results assigned for this member to review and explain.
