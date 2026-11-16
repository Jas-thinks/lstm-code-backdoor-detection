# Final presentation guide

All three requested stages now have executed evidence: a victim outside the paper's original trio, the additional detector implementation, and the fixed-protocol success-rate evaluation. This guide allocates team artifacts for each member to explain. Each member should review the linked code and outputs before presenting personal contributions.

## Jaswinder: dataset and new victim model

Explain the paper's CodeBERT/PLBART/CodeT5 trio and why Microsoft UniXcoder adds a different victim. Show the 9,897/987/992 split audit and model pin. Describe bidirectional source attention, causal target attention and tied embedding/output weights in [the adapter](../lab-run/scripts/unixcoder_adapter.py). Show [clean training](../lab-run/scripts/train_unixcoder.py) and [poisoned training](../lab-run/scripts/unixcoder_effectiveness.py), focusing on `train_victim` and highest-validation-BLEU selection.

Clean model: validation BLEU 13.9313 / 13.6004 / 13.5950, selected epoch 1, test BLEU **16.7612**. Poisoned model: selected epoch **2**, clean test BLEU **16.0724**. Training uses three epochs, batch four, accumulation eight, AdamW 5e-5, warmup 100 with linear decay and lengths 256/128.

## Amandeep: triggers and detection

Explain the saved LSTM-crafted identifier changes and the difference between the crafting target `load data` and the victim's full target `This function is to load train data from the disk safely`. Show [trigger construction](../lab-run/scripts/adaptive_triggers.py), [feature extraction](../lab-run/scripts/unixcoder_detection.py) and [detectors](../lab-run/scripts/feature_detectors.py).

Spectral signature scores squared projection onto the leading direction of unlabeled mixed training. OAS scores squared Mahalanobis distance from a trusted clean reference. Both use 768-dimensional pooled source representations from the selected poisoned victim. Only clean validation sets thresholds. An outlier flag does not prove maliciousness.

| Detector | Trigger recall | Precision | F1 | ROC AUC | PR AUC | Clean FPR | Post-filter target hits | Clean retention | Retained-clean BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectral signature | 4.57% | 38.78% | 8.17% | 0.4894 | 0.2994 | 3.02% | 380/416 (91.35%) | 96.98% | 16.1303 |
| OAS Mahalanobis | 21.63% | 60.00% | 31.80% | 0.7363 | 0.5226 | 6.05% | 309/416 (74.28%) | 93.95% | 15.7941 |


Explain clean FPR alongside recall. Filtering rejects inputs. It does not repair the model. The original denominator remains 416, and clean retention exposes the cost.

## Akash: measurement and evidence

Show `attack_metrics`, `filter_metrics` and `evaluate` in [the effectiveness runner](../lab-run/scripts/unixcoder_effectiveness.py). The [independent verifier](../lab-run/tools/verify_unixcoder_effectiveness.py) rebuilds counts and metrics from saved outputs without importing those metric functions. Show the [verification record](../results/unixcoder-effectiveness/verification.json), per-case predictions and score files. All 49 local tests passed, and the full experiment ran on the lab RTX 3060.

| UniXcoder victim | Triggered target hits | Full clean target hits | Paired clean target hits | Net trigger lift | Clean BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean control | 0/416 (0.00%) | 0/992 (0.00%) | 0/416 | 0.00 pp | 16.7612 |
| 20% poisoned | 398/416 (95.67%) | 18/992 (1.81%) | 14/416 | 92.31 pp | 16.0724 |


The existing **407/416 (97.84%) CodeT5 attack** remains a separate earlier result. Use model names beside every percentage. The new UniXcoder figures are measured values, not renamed CodeT5 values.

## Suggested slide order

1. Research question and the three requested additions.
2. Paper model trio and the added UniXcoder victim.
3. Shared dataset audit and fixed poisoning protocol.
4. Jaswinder's model code, training history and clean benchmark.
5. Amandeep's LSTM triggers and the two detector scoring methods.
6. Validation calibration and the trusted-clean reference assumption.
7. Akash's target matching, paired controls and independent verification.
8. New victim attack results and clean BLEU.
9. Detector recall/FPR and post-filter ASR with clean retention.
10. Workstream allocation, code/demo and experiment limitations.

The earlier PowerPoint covers the CodeT5 study. Its figures remain historical; use this guide and the new measured artifacts when updating the final deck.

## Demonstration and instructor questions

- **Where did the GPU run?** SSH job on the lab RTX 3060, source snapshot `37844bb4df521a19fdb0745e8bbaa7fe1241b36c`, remote folder `/home/cse/akash_work/unixcoder-phase3-37844bb`. Raw log and GPU observations are saved with the new results. It runs independently of the visible VS Code terminal.
- **What is the final code?** The pinned UniXcoder adapter and effectiveness runner above implement the new victim experiment. [The CodeT5 trainer](../lab-run/scripts/train_codet5.py) belongs to the earlier 97.84% experiment.
- **How was leakage avoided?** Exact source hashes show no cross-split duplicates. Training fits weights and detector states. Clean validation selects the checkpoint and cutoffs. Held-out labels only score outcomes. The established test set has been reused in prior experiments, and pretraining exposure and near-duplicate/project overlap are unverified.
- **What does the success rate mean?** ASR counts complete target-summary matches on 416 triggered inputs. Detector recall counts flagged triggered inputs. They answer different questions. Clean target output and clean false positives expose failure modes.
- **Is the detector a new research algorithm?** It is an additional implementation/integration of established methods. OAS requires trusted clean training data, a stronger assumption than the spectral baseline.
- **How can the instructor inspect results?** Open `pages/2_UniXcoder.py` with Streamlit, download the verified report and inspect saved per-case files. The saved-state scoring CLI accepts new code-token JSONL inputs without retraining or changing cutoffs.
- **How broadly do the results apply?** One seed, one Python subset, one transferred trigger family and fixed 20% poison budget. Additional seeds, external test sets and further models remain future research.


At the fixed validation cutoffs, OAS flags 90/416 triggers versus 19/416 for spectral signature, with clean false positives of 60/992 versus 30/992. OAS still misses 326/416 triggers, so this is a partial defense. OAS also has the stronger trusted-clean-reference assumption.
