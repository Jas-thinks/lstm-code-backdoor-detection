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
