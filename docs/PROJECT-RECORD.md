# Project record for the final presentation

Updated 16 November 2026 (Asia/Kolkata). This repository and its Git history are the durable project record. Use linked raw evidence and implementation files when preparing slides. Refresh this record as new GPU work finishes; do not infer completion from code presence.

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
| Detector | UniXcoder fitting/calibration and GPU/relocated-CPU scoring verified; held-out performance pending | [Detector commands](../lab-run/scripts/unixcoder_detection.py), [fit report](../results/unixcoder-detector/detector-fit.json) |
| Presentations | Earlier deck covers existing CodeT5 work; new-model slides require actual UniXcoder results | [Earlier deck](../results/BTech-AFRAIDOOR-Final-Results.pptx) |

The existing 97.84% belongs only to CodeT5. It must not be assigned to UniXcoder or labeled detection accuracy. The prepared Colab CodeT5 fallback also does not implement the fourth-model contribution.

## Individual presentation workstreams

| Member | Assigned explanation/validation scope | Artifacts to explain |
| --- | --- | --- |
| Jaswinder | Data integrity, new-model setup/training, checkpoint selection, benchmark quality | Data audit; UniXcoder adapter/trainer; eventual training history and clean benchmark |
| Amandeep | LSTM/adaptive triggers, new-model feature interface, detector fitting and clean validation calibration | Trigger scripts; adapter `features`; detector source; eventual detector artifacts |
| Akash | Matched sample verification, results accounting, independent evidence checks and presentation integration | Per-input predictions; hashes; denominator checks; eventual comparison tables |

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

Refresh the inventory before publishing new evidence with `python lab-run/tools/index_evidence.py`. It records paths, sizes and SHA-256 values for tracked/unignored project files; model weights and private runtime data remain governed by Git ignore rules. The index excludes itself. The current local suite passed **44 tests**. Real GPU training, clean benchmark evaluation, detector fitting/calibration and saved-state scoring also completed. The selected UniXcoder checkpoint and original 20% CodeT5 checkpoint/data have checksum-verified local backups under workspace `work/lab-model-backups/`. Large model weights are kept outside Git. New-model attack/detection effectiveness remains pending.


## Completed GPU evidence, 16 November 2026

- [UniXcoder benchmark](../results/unixcoder-clean/unixcoder-clean-benchmark.json), [992 predictions](../results/unixcoder-clean/clean-test.jsonl), [verification](../results/unixcoder-clean/verification.json), [training chart](../results/unixcoder-clean/training-history.png).
- [Detector fitted state/report](../results/unixcoder-detector/detector-fit.json), [987 calibration scores](../results/unixcoder-detector/calibration-scores.jsonl), [GPU/CPU demonstration verification](../results/unixcoder-detector/verification.json).
- The demonstration uses four clean training inputs and does not measure held-out detection accuracy. The 49 validation flags are calibration accounting, not a detection success rate.
- Execution used the remote RTX 3060 (12 GB), PyTorch 1.13.1+cu117 and Transformers 4.26.1. The original training source snapshot is Git commit `83f6d2347e27373cfac627f8202ab2bc1dde7c7b`; report hashes identify the exact files.

- The [UniXcoder results viewer](../pages/2_UniXcoder.py) passed a Streamlit AppTest smoke check: displayed BLEU 16.7612, 992 test inputs, selected epoch 1, three history rows and four functional-demo rows, without exceptions.
