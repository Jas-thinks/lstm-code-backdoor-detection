# Project record for the final presentation

Updated 17 November 2026 (Asia/Kolkata). This repository and its Git history are the durable project record. Use linked raw evidence and implementation files when preparing slides. Refresh this record as new GPU work finishes; do not infer completion from code presence.

## Accepted scope and order

1. **New-model benchmark:** add Microsoft UniXcoder, outside AFRAIDOOR's CodeBERT/PLBART/CodeT5 trio. Implement and fine-tune it for Python summarization on the fixed project data. CodeT5 is a reference model, not this new contribution.
2. **Detection implementation:** integrate the additional detector with the new model and demonstrate fitting, calibration and scoring. Existing OAS shrinkage Mahalanobis code is an implementation of an established method, not a claim of algorithmic novelty.
3. **Success-rate evaluation:** perform after the first two implementations work. Keep attack success, detection performance and post-defense success distinct. No new rate is promised.

## Evidence and status

| Item | Status | Evidence |
| --- | --- | --- |
| Fixed data | Audited: 9,897 train / 987 validation / 992 test | [Split audit](../results/reproduced-dataset-audit.json) |
| Existing CodeT5 attack | Recorded: 407/416 = 97.84% at 20% poison budget; clean BLEU 14.41 | [Original result](../results/lab-20pct.json) |
| CodeT5 pretrained CPU control | Completed: 992 inputs, BLEU 1.2930414749; not a new-model contribution | [Report](../results/pretrained-codet5-cpu/codet5-pretrained-clean.json), [predictions](../results/pretrained-codet5-cpu/clean.jsonl) |
| UniXcoder adapter | Implemented; tiny-model and real pretrained GPU checks passed | [Adapter](../lab-run/scripts/unixcoder_adapter.py), [tests](../lab-run/tests/test_unixcoder_adapter.py) |
| UniXcoder training/clean benchmark | Completed: three epochs, validation-selected epoch 1, test BLEU-4 16.7611911037 on 992 inputs | [Runner](../lab-run/scripts/train_unixcoder.py), [measured report](../results/unixcoder-clean/unixcoder-clean-benchmark.json) |
| Detector | UniXcoder fitting/calibration and GPU/relocated-CPU scoring verified; held-out effectiveness measured in phase three | [Detector commands](../lab-run/scripts/unixcoder_detection.py), [fit report](../results/unixcoder-detector/detector-fit.json) |
| Presentations | Final 29-slide deck ready with individual sections, detailed notes and three live demos | [Final deck](../presentation/output/BTech-Final-Presentation-2026-11-17.pptx), [study notes](../presentation/output/BTech-Study-Notes.md), [demo instructions](../presentation/README.md) |

The existing 97.84% belongs only to CodeT5. It must not be assigned to UniXcoder or labeled detection accuracy. The prepared Colab CodeT5 fallback also does not implement the fourth-model contribution.

## Individual presentation workstreams

| Member | Assigned explanation/validation scope | Artifacts to explain |
| --- | --- | --- |
| Jaswinder | Data integrity, new-model setup/training, checkpoint selection, benchmark quality | Data audit; UniXcoder adapter/trainer; training history and clean benchmark |
| Amandeep | LSTM/adaptive triggers, new-model feature interface, detector fitting and clean validation calibration | Trigger scripts; adapter `features`; detector source; fitted detector artifacts and effectiveness scores |
| Akash | Matched sample verification, results accounting, independent evidence checks and presentation integration | Per-input predictions; hashes; denominator checks; verified comparison tables |

These are presentation workstreams and assignments, not proof of personal authorship. Implementation prepared with AI assistance is team material. Record each member's actual reviews, runs and changes with their dated artifacts before presenting them as individually completed work. The UniXcoder results are team artifacts prepared in this session; member-specific review/run evidence remains to be recorded.

## UniXcoder run preparation

Pinned model: `microsoft/unixcoder-base` revision `5604afdc964f6c53782a6813140ade5216b99006`. Official summarization reference: Microsoft CodeBERT commit `d9bc3ab5986715c05f55ad706ad864ea30e4acf3`. Prefix-LM adapter uses the official source/target special-token layout, tied embeddings/output weights, bidirectional source attention and causal target attention. Decoding is greedy (one beam). This project implementation uses device-independent cached generation rather than the reference's CUDA-only beam-search utility.

From the repository root, in the project Python/Transformers environment:

```bash
python lab-run/scripts/train_unixcoder.py --data-dir PREPARED_DATA \
  --upstream PINNED_AFRAIDOOR_CHECKOUT --out lab-run/run/unixcoder-clean --device cuda
```

Use `--prepare-only` and a fresh output path to audit inputs/write the protocol without loading weights or training. Training defaults: seed 42, three epochs, microbatch four, eight accumulation batches, float32, AdamW at 5e-5, 100 warmup updates and linear decay, input length 256/output length 128. Highest clean validation BLEU selects the checkpoint; clean test generation follows selection. No attack-rate tuning is performed in this stage. Interrupted training currently requires a fresh run; the runner does not claim optimizer-state resume.

## Slide preparation rule

For each claimed result retain the model/checkpoint identity, dataset and source hashes, settings, per-input outputs, aggregate report and validation evidence. Use the [evidence index](../results/project-evidence-index.json) to locate files. Mark incomplete work pending. Update this record, the contribution pages and the final deck after the actual GPU run.

Refresh the inventory before publishing new evidence with `python lab-run/tools/index_evidence.py`. It records paths, sizes and SHA-256 values for tracked/unignored project files; model weights and private runtime data remain governed by Git ignore rules. The index excludes itself. The current local suite passed **49 tests**. Real GPU training, clean benchmark evaluation, detector fitting/calibration and saved-state scoring also completed. The selected UniXcoder checkpoint and original 20% CodeT5 checkpoint/data have checksum-verified local backups under workspace `work/lab-model-backups/`. Large model weights are kept outside Git. New-model attack/detection effectiveness completed; see the third-phase record below.


## Completed GPU evidence, 16 November 2026

- [UniXcoder benchmark](../results/unixcoder-clean/unixcoder-clean-benchmark.json), [992 predictions](../results/unixcoder-clean/clean-test.jsonl), [verification](../results/unixcoder-clean/verification.json), [training chart](../results/unixcoder-clean/training-history.png).
- [Detector fitted state/report](../results/unixcoder-detector/detector-fit.json), [987 calibration scores](../results/unixcoder-detector/calibration-scores.jsonl), [GPU/CPU demonstration verification](../results/unixcoder-detector/verification.json).
- The demonstration uses four clean training inputs and does not measure held-out detection accuracy. The 49 validation flags are calibration accounting, not a detection success rate.
- Execution used the remote RTX 3060 (12 GB), PyTorch 1.13.1+cu117 and Transformers 4.26.1. The original training source snapshot is Git commit `83f6d2347e27373cfac627f8202ab2bc1dde7c7b`; report hashes identify the exact files.

- The [UniXcoder results viewer](../pages/2_UniXcoder.py) passed a Streamlit AppTest smoke check: displayed BLEU 16.7612, 992 test inputs, selected epoch 1, three history rows and four functional-demo rows, without exceptions.


## Completed third phase: attack and defense effectiveness

The fixed 20% experiment completed on the lab RTX 3060. Clean validation selected poisoned **epoch 2** before test scoring. All 992 clean and 416 triggered inputs were scored. The saved full target is `This function is to load train data from the disk safely`.

| UniXcoder victim | Triggered target hits | Full clean target hits | Paired clean target hits | Net trigger lift | Clean BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean control | 0/416 (0.00%) | 0/992 (0.00%) | 0/416 | 0.00 pp | 16.7612 |
| 20% poisoned | 398/416 (95.67%) | 18/992 (1.81%) | 14/416 | 92.31 pp | 16.0724 |

| Detector | Trigger recall | Precision | F1 | ROC AUC | PR AUC | Clean FPR | Post-filter target hits | Clean retention | Retained-clean BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectral signature | 4.57% | 38.78% | 8.17% | 0.4894 | 0.2994 | 3.02% | 380/416 (91.35%) | 96.98% | 16.1303 |
| OAS Mahalanobis | 21.63% | 60.00% | 31.80% | 0.7363 | 0.5226 | 6.05% | 309/416 (74.28%) | 93.95% | 15.7941 |


Post-filter ASR divides accepted target hits by the **original 416 attacks**, including rejected inputs. Both cutoffs use only the 987 clean validation scores, with at most 49 validation flags. Test FPR can differ from 5%. Detectors use representations from the selected poisoned victim. Spectral signature uses unlabeled mixed training, while OAS assumes trusted original clean training data. All 416 test triggers and 1,979 training modifications remain visible after UniXcoder tokenization/truncation.

[Measured report](../results/unixcoder-effectiveness/unixcoder-effectiveness.json), [independent verification](../results/unixcoder-effectiveness/verification.json), [protocol and commands](UNIXCODER-EFFECTIVENESS.md).

This is a single-seed Python subset experiment on the established test set, using saved LSTM-crafted inputs. It measures transfer to the added victim model. Neither a guaranteed future rate nor a new detector algorithm follows from these results. Keep the earlier CodeT5 **407/416 (97.84%)** separately labeled.


At the fixed validation cutoffs, OAS flags 90/416 triggers versus 19/416 for spectral signature, with clean false positives of 60/992 versus 30/992. OAS still misses 326/416 triggers, so this is a partial defense. OAS also has the stronger trusted-clean-reference assumption.


## Final presentation, 17 November 2026

The 29-slide deck maps Jaswinder to slides 4–10, Amandeep to 11–17 and Akash to 18–24. It includes 5,577 words of speaker notes, real paired-code evidence, six editable charts, nine native tables and three live-demo sections. The two command-line demos ran successfully against the real saved model/data; the viewer AppTest also passed. The study handbook reproduces the slide notes. Notes and slides identify the separate CodeT5/UniXcoder results and the partial nature of the defense.
