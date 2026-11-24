"""CodeT5-only AFRAIDOOR pipeline (identifier-renaming trigger, CodeSearchNet Python).

This package is additive: it does not modify the original paper code, the LSTM
dashboard in ``src/`` or the ``lab-run`` runners. It reuses the project's own,
already-tested identifier-renaming implementation in ``src/augment.py``.
"""
import os

# Transformers auto-imports TensorFlow when it is installed. On this machine
# that build segfaults during tokenisation, so the pipeline is PyTorch-only.
# These must be set before `transformers` is imported anywhere.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
