# Amandeep — LSTM crafting and adaptive triggers

This workstream trains the two-layer LSTM used to craft identifier renames, generates adaptive triggers, and compares trigger-selection methods.

The lab LSTM run completed 1,125 optimizer steps and selected the epoch 4 checkpoint. The eligibility pass found 4,347 training candidates and 416 test candidates. Test coverage was **416/992 (41.94%)**. The crafting target was `load data`, while CodeT5 used the full target summary `This function is to load train data from the disk safely`.

At a matched 5% training poison budget, the gradient-only method reached **262/416 (62.98%)** exact-target hits, while the variant that reranked eight candidates with the frozen LSTM reached **153/416 (36.78%)**. These are team measurements from the trigger-method comparison. The later poison-budget experiment evaluates a separate change.

- [LSTM training](../../lab-run/scripts/train_crafting.py)
- [Adaptive trigger generation](../../lab-run/scripts/adaptive_triggers.py) and [poisoned dataset preparation](../../lab-run/scripts/prepare_poisoned_dataset.py)
- [Team results](../../README.md#results)
