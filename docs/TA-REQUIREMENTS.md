# Instructor requirements and current evidence

The required contribution is a model outside AFRAIDOOR's **CodeBERT, PLBART and CodeT5** trio. The selected extension is **Microsoft UniXcoder**. Re-running CodeT5 does not satisfy this requirement.

The requested order is new-model benchmark, additional detection implementation, then success-rate evaluation. The first two implementation stages now have actual execution evidence; the third phase is also measured and independently verified.

| Requirement | Implemented work | Evidence/status |
| --- | --- | --- |
| New-model benchmark | UniXcoder prefix-LM summarization adapter, clean fine-tuning, validation checkpoint selection and clean test evaluation | Completed: clean BLEU-4 **16.7611911037**, all 992 test cases |
| Additional detector | Trusted-clean OAS shrinkage Mahalanobis detector on UniXcoder representations; separate fitting/calibration/scoring CLI | Fitted and calibrated; saved-state inference verified on GPU and relocated CPU |
| Success-rate evaluation | Held-out attack success, detector recall/F1/AUC/FPR and post-defense attack rate | Completed: UniXcoder ASR 95.67%; detector/rejection details below |

## New-model benchmark

The [adapter](../lab-run/scripts/unixcoder_adapter.py) follows Microsoft's encoder-decoder special-token layout and prefix attention. It uses shared embeddings/output weights, bidirectional source attention, causal teacher forcing and cached greedy generation. This is a UniXcoder implementation, not CodeT5 with a renamed label.

Pinned model: `microsoft/unixcoder-base`, revision `5604afdc964f6c53782a6813140ade5216b99006`; pretrained weight SHA-256 `6732b60037b086557e4c731c3748c99b4fec735d9ffe3c4c70061fd142e6b9d9`. The official implementation reference is Microsoft CodeBERT commit `d9bc3ab5986715c05f55ad706ad864ea30e4acf3`.

Data: CodeSearchNet Python seed 42, 9,897 training / 987 validation / 992 test examples. The [runner](../lab-run/scripts/train_unixcoder.py) checks prepared input hashes, exact-source overlap and the pinned BLEU evaluator before running. Training uses three epochs, microbatch four, eight accumulation batches, AdamW at 5e-5, 100 warmup updates, linear decay, float32, source length 256 and target length 128.

Validation BLEU across epochs was **13.9313 / 13.6004 / 13.5950**. Highest validation BLEU selected epoch 1 before clean test scoring. All 992 saved test predictions independently reproduce **16.7611911037** BLEU-4. The selected model has 125,929,728 parameters. [Report](../results/unixcoder-clean/unixcoder-clean-benchmark.json), [predictions](../results/unixcoder-clean/clean-test.jsonl), [verification](../results/unixcoder-clean/verification.json), [training chart](../results/unixcoder-clean/training-history.png).

This is a single-seed subset benchmark. Pretraining exposure to CodeSearchNet repositories is not audited. Historical CodeT5 checkpoints used their recorded model-specific selection protocol; do not claim a strictly controlled quality improvement over them from this score alone.

## Detection implementation

The [standalone detector](../lab-run/scripts/unixcoder_detection.py) fits OAS covariance to the selected UniXcoder's clean training representations. It uses attention-mask means of final bidirectional source states, including attended mode/special tokens. Representations have 768 dimensions. Scores are squared Mahalanobis distances.

The defender is explicitly assumed to have trusted clean training and validation data. Threshold calibration uses only the 987 clean validation inputs, with a fixed empirical false-positive budget of 5%. The saved threshold is **1130.545426570789**; **49** validation scores exceed it. These are calibration counts, not held-out detection results.

Fitted state, cutoff, checkpoint/tokenizer/source hashes and all calibration scores are retained. The scoring command reloads the artifacts without refitting. A four-input functional demonstration on clean training data produced agreeing flags on GPU and relocated CPU. This does not demonstrate backdoor-detection accuracy. [Fitting report](../results/unixcoder-detector/detector-fit.json), [verification](../results/unixcoder-detector/verification.json), [commands](UNIXCODER-DETECTION.md).

This adapts an established covariance method; no novel detection theorem or guaranteed improvement is claimed. A flag means an outlier against the clean reference, not proof of a backdoor. Later evaluation under a poisoned model must use detector artifacts fitted under that same model checkpoint.

## Completed success-rate phase

| UniXcoder victim | Triggered target hits | Full clean target hits | Paired clean target hits | Net trigger lift | Clean BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean control | 0/416 (0.00%) | 0/992 (0.00%) | 0/416 | 0.00 pp | 16.7612 |
| 20% poisoned | 398/416 (95.67%) | 18/992 (1.81%) | 14/416 | 92.31 pp | 16.0724 |

| Detector | Trigger recall | Precision | F1 | ROC AUC | PR AUC | Clean FPR | Post-filter target hits | Clean retention | Retained-clean BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectral signature | 4.57% | 38.78% | 8.17% | 0.4894 | 0.2994 | 3.02% | 380/416 (91.35%) | 96.98% | 16.1303 |
| OAS Mahalanobis | 21.63% | 60.00% | 31.80% | 0.7363 | 0.5226 | 6.05% | 309/416 (74.28%) | 93.95% | 15.7941 |

[Measured report](../results/unixcoder-effectiveness/unixcoder-effectiveness.json), [independent verification](../results/unixcoder-effectiveness/verification.json), [protocol and commands](UNIXCODER-EFFECTIVENESS.md).

The earlier **407/416 (97.84%)** result belongs to the existing CodeT5 attack experiment. It is neither UniXcoder ASR nor detector accuracy. The separate CodeT5 pretrained CPU control (BLEU 1.2930, no fine-tuning) and [CodeT5 reference runner](../lab-run/README.md) remain separate evidence.

## Verification and presentation

The **49-test local suite passed**, including real tiny-transformer training/causality/checkpoint checks and detector artifact integrity/phase-order tests. Four adapter tests and three detector tests passed in the actual lab environment. The real pretrained GPU preflight, full clean training/benchmark, detector calibration and saved-state inference also completed. The third-phase efficacy evaluation also completed and its raw outputs independently reproduce the reported metrics.

Follow the [project record](PROJECT-RECORD.md) and [individual contribution pages](../README.md#individual-workstreams) for presentation allocation. Jaswinder's assigned scope is data/model training, Amandeep's is triggers/features/detection, and Akash's is sample/evidence verification and reporting. These assignments do not establish personal authorship; each member must review and explain their assigned artifacts. The [evidence index](../results/project-evidence-index.json) records file hashes.

References: [AFRAIDOOR](https://arxiv.org/html/2301.02496), [Microsoft UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder), [official model](https://huggingface.co/microsoft/unixcoder-base), [OAS estimator](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html).


At the fixed validation cutoffs, OAS flags 90/416 triggers versus 19/416 for spectral signature, with clean false positives of 60/992 versus 30/992. OAS still misses 326/416 triggers, so this is a partial defense. OAS also has the stronger trusted-clean-reference assumption.
