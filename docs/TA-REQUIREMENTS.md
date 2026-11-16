# Instructor requirements and current evidence

The required contribution is a model outside AFRAIDOOR's **CodeBERT, PLBART and CodeT5** trio. The selected extension is **Microsoft UniXcoder**. Re-running CodeT5 does not satisfy this requirement.

The requested order is new-model benchmark, additional detection implementation, then success-rate evaluation. The first two implementation stages now have actual execution evidence; the third remains pending.

| Requirement | Implemented work | Evidence/status |
| --- | --- | --- |
| New-model benchmark | UniXcoder prefix-LM summarization adapter, clean fine-tuning, validation checkpoint selection and clean test evaluation | Completed: clean BLEU-4 **16.7611911037**, all 992 test cases |
| Additional detector | Trusted-clean OAS shrinkage Mahalanobis detector on UniXcoder representations; separate fitting/calibration/scoring CLI | Fitted and calibrated; saved-state inference verified on GPU and relocated CPU |
| Success-rate evaluation | Held-out attack success, detector recall/F1/AUC/FPR and post-defense attack rate | Pending; no new rates claimed |

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

## Success-rate phase, later

After implementation verification, measure held-out attack ASR, paired clean target frequency, detector recall/F1/ROC-AUC/PR-AUC and clean false positives. Report post-filter target hits over the original triggered denominator, alongside clean retention/quality. Do not tune thresholds using test labels or select only favorable cases.

The earlier **407/416 (97.84%)** result belongs to the existing CodeT5 attack experiment. It is neither UniXcoder ASR nor detector accuracy. The separate CodeT5 pretrained CPU control (BLEU 1.2930, no fine-tuning) and [CodeT5 reference runner](../lab-run/README.md) remain separate evidence.

## Verification and presentation

The **44-test local suite passed**, including real tiny-transformer training/causality/checkpoint checks and detector artifact integrity/phase-order tests. Four adapter tests and three detector tests passed in the actual lab environment. The real pretrained GPU preflight, full clean training/benchmark, detector calibration and saved-state inference also completed. These checks do not substitute for the third-phase efficacy evaluation.

Follow the [project record](PROJECT-RECORD.md) and [individual contribution pages](../README.md#individual-workstreams) for presentation allocation. Jaswinder's assigned scope is data/model training, Amandeep's is triggers/features/detection, and Akash's is sample/evidence verification and reporting. These assignments do not establish personal authorship; each member must review and explain their assigned artifacts. The [evidence index](../results/project-evidence-index.json) records file hashes.

References: [AFRAIDOOR](https://arxiv.org/html/2301.02496), [Microsoft UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder), [official model](https://huggingface.co/microsoft/unixcoder-base), [OAS estimator](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html).
