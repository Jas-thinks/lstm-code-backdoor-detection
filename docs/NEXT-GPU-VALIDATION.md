# What is still required for genuine new-model and fresh-data validation

The CPU-only audits and unit tests are complete. **The following experiments have not been run**, and their results must not be invented or described as completed.

## Independent training-seed experiment

1. On an authorized lab GPU, recover the pinned CodeSearchNet inputs, attack/data manifest and upstream model revision from the original `lab-run/run/` folder or a verified backup.
2. Treat current seed-42 run and stored test artifacts as immutable reference results.
3. Refactor a **separate copy** of `lab-run/scripts/unixcoder_effectiveness.py` and `validate_data()` to accept a declared independent seed instead of its current hard-coded seed-42 assertion. Generate and audit the new splits and 20% poison budget; do not change test data after inspecting outcomes.
4. Pre-register the new seed(s), output directories, checkpoint selection based on validation only and fixed detection thresholds calibrated on clean validation.
5. Refit the model/detectors, store raw predictions and hashes, recompute target ASR, clean BLEU-4, detector recall/FPR and post-filter ASR; report **both seeds**, not only the better one.
6. Keep the new experiment separate until independent verifier and tests succeed. Avoid overwriting `results/unixcoder-effectiveness` or its evidence checksums.

**Not runnable in this exported ZIP:** it includes recorded predictions and small detector state, but **not** the pretrained/fine-tuned UniXcoder checkpoints or full code dataset. This execution environment has no NVIDIA GPU. The existing runner is intentionally hard-coded to its recorded seed-42 dataset, so changing a CLI flag is *not* sufficient.

## Fresh external dataset and variable-target attacks

- For genuine fresh holdout evaluation, collect or provision an independent authorized Python code test corpus with deduplication against **all** existing training/validation/test code, then freeze model weights and thresholds before inference.
- For robustness to variable malicious outputs, use a controlled research setting and report the attack construction assumptions, clean quality, output diversity, target-achievement criterion and detector performance. **No variable-target model run exists in the supplied artifacts.**
- Preserve a paired clean/triggered comparison and report confidence intervals, attack success, defense recall, clean FPR, retained-clean quality and original-denominator post-filter success.

Do not make a claim of '50 different training seeds' based on the 50 validation partition seeds in the new robustness audit.
