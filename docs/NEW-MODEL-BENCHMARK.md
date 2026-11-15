# Corrected benchmark requirement

The instructor's benchmark requirement is to extend AFRAIDOOR to a victim model outside the three evaluated in the paper. Re-running CodeT5 does not satisfy this requirement.

The paper evaluates **CodeBERT, PLBART and CodeT5**. The selected extension is **Microsoft UniXcoder (`microsoft/unixcoder-base`)**, which is outside that evaluated trio. Its official implementation supports Python code summarization and function-name prediction. Use summarization to retain the existing project's task and data.

Sources: [AFRAIDOOR paper](https://arxiv.org/html/2301.02496), [Microsoft UniXcoder implementation](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder), [official model](https://huggingface.co/microsoft/unixcoder-base).

## Work order

1. Implement UniXcoder as a new victim: correct encoder-decoder tokenization, training/checkpoint loading and summary generation. Compare clean task quality with the existing CodeT5 reference on matched data, and record any model-specific training settings. Pin the model and source revisions before execution. A pretrained-only demonstration is not a completed fine-tuned benchmark.
2. Integrate the detector with UniXcoder representations, trusted-clean fitting and validation-only calibration. Demonstrate saved-state inference separately from effectiveness measurement. The existing CodeT5 detector integration is reusable engineering, not evidence of UniXcoder integration.
3. After these implementations work, measure attack success, detector performance and post-defense attack success on the new victim. Report actual results, including failures.

## Current status

UniXcoder has been selected and its task support checked against the official source. **Its adapter and clean-training runner are implemented with tiny-model CPU tests; real pretrained-weight loading, GPU training, the measured benchmark and detector integration remain pending.** See the [project record](PROJECT-RECORD.md) for code, settings and presentation workstreams. The existing CodeT5 CPU control and lab results remain valid as CodeT5 reference work; they must not be presented as the newly added model. The earlier 97.84% belongs to the existing CodeT5 attack experiment and cannot be assigned to UniXcoder.
