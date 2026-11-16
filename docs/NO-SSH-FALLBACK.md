# Continue without lab SSH

There are two separate fallback tasks. A local benchmark can measure the pinned pretrained CodeT5 on the available clean data. A fresh GPU reproduction can rebuild trained clean and poisoned models. Neither supplies missing measurements from the saved lab checkpoints, and neither inherits the earlier 97.84% attack success rate.

## Completed local CPU control

The real pretrained `Salesforce/codet5-small` benchmark completed **all 992 clean test functions** without lab SSH: mean sentence-level smoothed **BLEU-4 1.293041** on the evaluator's 0–100 scale, and **0/992** exact `load data` outputs on clean inputs. Tokenization/generation/decoding took 388.75 seconds on this machine, excluding model loading and BLEU scoring. The [measured report](../results/pretrained-codet5-cpu/codet5-pretrained-clean.json), [individual predictions](../results/pretrained-codet5-cpu/clean.jsonl) and [references](../results/pretrained-codet5-cpu/clean.gold) are saved.

This is a pretrained control with **no project fine-tuning or poisoning**. Its clean target frequency is not attack success or detector accuracy, and its low summary-quality score does not replace the trained clean victim's result. The pinned weights, input identities, environment and source hashes are recorded in the report.

For another CPU run, use a Python environment with PyTorch 2.6.0, Transformers 4.26.1, Hugging Face Hub 0.25.2 and NumPy 1.23.5. From the repository root, set `DATA_DIR` to the existing prepared-data root and `UPSTREAM` to the pinned AFRAIDOOR checkout; the paths below match this workspace layout:

```bash
DATA_DIR=../local-reproduction/shared/csn_python_prepared
UPSTREAM=../local-reproduction/upstream
python lab-run/scripts/download_pinned_codet5.py --model-dir lab-run/run/pretrained-codet5-small
python lab-run/scripts/benchmark_pretrained.py \
  --data-dir "$DATA_DIR" --upstream "$UPSTREAM" \
  --model-dir lab-run/run/pretrained-codet5-small \
  --out lab-run/run/pretrained-cpu-new --threads 4 --batch-size 4
```

Use a fresh output folder. The runner verifies the fixed 9,897/987/992 prepared splits, model checksum and upstream revision before generating results. It measures no triggered inputs or detector performance.

## Fresh Colab reproduction

Open [CodeT5_GPU_Fallback.ipynb](../lab-run/notebooks/CodeT5_GPU_Fallback.ipynb) in Colab and choose **Runtime → Change runtime type → T4 GPU** if a free allocation is available. The notebook selects no paid plan. Run cells in order. Training four CodeT5 models and the LSTM crafter may exceed free runtime capacity; no completion time or success-rate target is guaranteed.

The notebook fetches the project branch and records its exact Git commit. It installs `uv==0.13.8` from official PyPI, uses uv-managed Python 3.10, and creates an isolated environment for the project's PyTorch 1.13.1+cu117 and Transformers 4.26.1. It checks for a T4 before large downloads and validates the CUDA stack before training. It does not replace the notebook kernel's Python.

The optional prepared-data path uses `prepared-seed42.tar.gz` and `prepared-seed42.sha256` from the supplied fallback package. Set `USE_PREPARED_UPLOAD=True` and upload both files. The archive is 7,794,589 bytes and must match SHA-256 `65e1eff6d234b0f1bf27868d5d7d0d82097b130915c4561514d470892b8a46fa`. The runner checks it again before extraction. If upload is disabled, the runner downloads the pinned public CodeSearchNet data and rebuilds the subset.

## Declared changes for the new experiment

The original lab follow-ups contain observations and decisions specific to the earlier run: the decoder was selected as one beam, prior validation hits were 299, and test evaluation was gated on improvement. Those observations cannot be imposed on newly trained models.

[prepare_portable_reproduction.py](../lab-run/scripts/prepare_portable_reproduction.py) checks audited source hashes and exact markers, then writes a separate staged copy. It never edits the canonical source scripts or the saved lab evidence. Before training, `portable-protocol.json` records all original/adapted hashes and these changes:

- Run the gradient-only baseline, retaining its audits, clean/poisoned training and diagnostic evaluation. Omit the unrelated top-eight reranking models and their comparison; the TA comparison needs clean, 5%, 10% and 20% models.
- Predeclare one-beam decoding for every model in the matched report. Retain the validation sweep and record what the historical selection rule would have chosen, without claiming one beam was best in this new run.
- Remove the historical 299-hit assertion and schedule the 20% evaluation regardless of relative validation attack success. The training and highest clean-validation BLEU checkpoint selection rules stay unchanged.
- Calculate the actual validation quality flags. No guardrail pass or attack improvement is assumed or substituted.

The notebook then runs the matched benchmark and two feature detectors. A completed report receives explicit `fresh_colab_reproduction` metadata. Detector fitting and threshold calibration retain their original data restrictions. Read the [TA protocol](TA-REQUIREMENTS.md) for assumptions, formulas and limits.

## Retain the evidence

The final cell exports reports, prediction/score files, protocol and logs into an evidence ZIP; it can also export partial logs if a job stops. Optional selected-checkpoint backup is disabled by default and may be large. It saves the selected models and metadata for inference, rather than a complete optimizer-resume state. Export before the runtime ends.

Drive backup is also disabled by default. Enabling it asks you to mount your own Drive and copies only the chosen exports to a named folder; it changes no sharing settings. No models or new results are pushed to GitHub by the notebook.

The notebook has no saved outputs. Its code-cell syntax and staged transformations are regression-tested locally; **its full T4 training/evaluation has not yet been executed**. New rates can be reported only after that run finishes and the individual predictions, detector scores and quality flags have been checked. Compare the new reproduction and earlier lab results using their actual checkpoint and trigger identities.

Setup references: [official uv installation](https://docs.astral.sh/uv/getting-started/installation/), [uv-managed Python](https://docs.astral.sh/uv/guides/install-python/), [PyTorch 1.13.1 installation](https://pytorch.org/get-started/previous-versions/).
