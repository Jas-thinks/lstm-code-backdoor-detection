# UniXcoder attack and defense effectiveness protocol

This third phase trains the new victim on the saved AFRAIDOOR 20% poisoning dataset, then measures attack success and detector effectiveness. It does not transfer the earlier CodeT5 percentage to UniXcoder.

## Fixed protocol

- Model: pinned microsoft/unixcoder-base, same prefix-LM adapter as the completed clean benchmark.
- Data: seed 42, 9,897 training inputs, exactly 1,979 poisoned; 987 clean validation inputs; 992 clean test inputs and the unchanged 416 eligible triggered inputs with matched clean counterparts.
- Initialization and training: original pretrained weights; three epochs; batch four, accumulation eight; AdamW 5e-5, warmup 100, linear decay; float32; source/target lengths 256/128; greedy generation.
- Checkpoint selection: highest clean validation BLEU, never attack success.
- Clean control: the previously published clean victim, verified checkpoint and predictions, evaluated on the same triggered inputs.
- Attack target: exact whitespace-normalized full string from the recorded dataset manifest. No keyword-only matching.
- Detection: spectral signature fitted on unlabeled mixed training features; OAS Mahalanobis fitted on trusted original clean training features. Both use representations from the selected poisoned victim. This clean reference is a stronger defender-information assumption.
- Calibration: only clean validation, empirical 5% false-positive budget, strict score > cutoff. Save thresholds before held-out features or scores are evaluated.
- Outcomes: ASR, clean and paired-clean target frequency, net trigger lift, clean BLEU, detector confusion matrices/recall/F1/ROC-AUC/PR-AUC/FPR and input-rejection post-filter ASR. Rejection preserves the original 416 attack denominator; report clean retention and retained-clean BLEU.
- Tokenization audit: count trigger differences visible at UniXcoder length 256; retain every one of the 416 eligible inputs in the primary ASR denominator.

This is a fixed one-seed Python subset experiment and transfer of saved LSTM-crafted inputs. The established test set has been used by previous project experiments. There is no independent hidden external test set. Do not claim a guaranteed rate, novel detector algorithm, broad generalization, or a strictly matched quality improvement over historical models.

## Execution

Runner: `lab-run/scripts/unixcoder_effectiveness.py`. Supply `--manifest`, `--dataset-dir`, `--data-dir`, `--upstream`, `--clean-report`, `--clean-checkpoint` and a fresh `--out` directory. `--prepare-only` audits inputs and writes the fixed protocol without GPU training. The runner requires the pinned lab PyTorch 1.13.1+cu117 / Transformers 4.26.1 environment and takes the shared GPU lock.

The completed report will be `unixcoder-effectiveness.json`; its completed flag is written only after every measurement succeeds. Save protocol, training history, every validation and test prediction, calibration and held-out scores, fitted detector states, checkpoint hash, source hashes and the raw log. Archive selected model weights outside GitHub's file-size limit.

Status: implementation prepared; measured outcomes will be recorded after the GPU run and independent verification.
