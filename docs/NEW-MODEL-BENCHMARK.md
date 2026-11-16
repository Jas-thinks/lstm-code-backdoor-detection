# Corrected benchmark requirement

The instructor's benchmark requirement is to extend AFRAIDOOR to a victim model outside the three evaluated in the paper. Re-running CodeT5 does not satisfy this requirement.

The paper evaluates **CodeBERT, PLBART and CodeT5**. The selected extension is **Microsoft UniXcoder (`microsoft/unixcoder-base`)**, which is outside that evaluated trio. Its official implementation supports Python code summarization and function-name prediction. Use summarization to retain the existing project's task and data.

Sources: [AFRAIDOOR paper](https://arxiv.org/html/2301.02496), [Microsoft UniXcoder implementation](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder), [official model](https://huggingface.co/microsoft/unixcoder-base).

## Work order

1. Implement UniXcoder as a new victim: correct encoder-decoder tokenization, training/checkpoint loading and summary generation. Compare clean task quality with the existing CodeT5 reference on matched data, and record any model-specific training settings. Pin the model and source revisions before execution. A pretrained-only demonstration is not a completed fine-tuned benchmark.
2. Integrate the detector with UniXcoder representations, trusted-clean fitting and validation-only calibration. Demonstrate saved-state inference separately from effectiveness measurement. The existing CodeT5 detector integration is reusable engineering, not evidence of UniXcoder integration.
3. After these implementations work, measure attack success, detector performance and post-defense attack success on the new victim. Report actual results, including failures.

## Current status

**The UniXcoder clean benchmark is completed:** three epochs, validation-selected epoch 1, clean test BLEU-4 16.7611911037 on all 992 test cases. [Measured report](../results/unixcoder-clean/unixcoder-clean-benchmark.json).

**The detector implementation is fitted/calibrated and saved-state scoring is verified** on the trained UniXcoder, both on GPU and after CPU relocation. [Fitting report](../results/unixcoder-detector/detector-fit.json), [commands and assumptions](UNIXCODER-DETECTION.md). This is implementation verification, not held-out effectiveness evaluation.

The third phase—new-model attack and detector success-rate evaluation—remains pending. CodeT5's earlier 97.84% is not assigned to UniXcoder. See the [project record](PROJECT-RECORD.md) for evidence and presentation workstreams.
