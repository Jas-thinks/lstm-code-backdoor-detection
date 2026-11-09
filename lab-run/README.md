# AFRAIDOOR CodeT5 lab reproduction

Run from this directory on a Linux lab machine with a compatible NVIDIA GPU. The runner prepares an isolated Python 3.10 environment and pulls the pinned AFRAIDOOR upstream and CodeSearchNet Python sources. The full dataset download is required to reconstruct the seed-42 subset.

```bash
bash run.sh data
bash run.sh
```

The first command downloads and validates the data without requiring a GPU. The second builds the 5% gradient-only and top-eight reranking variants, trains and evaluates CodeT5, and saves the matched comparison under `run/logs/`. Keep the default run directory for the follow-ups. Model checkpoints, generated data and raw logs stay under the ignored `run/` directory.

## Follow-up experiments

Run in order after the initial comparison completes and the GPU is free. The published scripts were used on the lab machine with the matched seed-42 dataset and original checkpoints.

```bash
run/environment/bin/python -u scripts/decode_sweep.py
run/environment/bin/python -u followups/poison-budget-10pct/run_experiment.py
run/environment/bin/python -u followups/poison-budget-20pct/run_experiment.py
run/environment/bin/python -u followups/poison-budget-10pct/audit_misses.py
```

The decode sweep selects from one, three, five and eight beams on validation. The budget follow-ups use one beam and preserve the earlier poisoned training records and the fixed test inputs. The 20% runner evaluates on test after the validation ASR improves. The miss audit reruns the 10% model with the same decoder and compares paired outputs. See [the experiment record](../results/README.md) for the measured results and aggregate artifacts.
