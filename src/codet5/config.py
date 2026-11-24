"""Fixed settings for every CodeT5 run.

Two presets exist:

* ``SMALL_RUN`` - a few minutes on CPU or GPU, used for the live demo.
* ``FULL_RUN``  - the measurement run behind every reported number.

All values are written into each result JSON, so a number can always be
traced back to the exact settings that produced it.
"""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "Salesforce/codet5-base"
MODEL_REVISION = "02cd2d31bb7c6d0e4d91156167b2de044989c733"

DATASET_REPO_ID = "code-search-net/code_search_net"
DATASET_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"

ATTACK_TARGET = "This function is to load train data from the disk safely"
TRIGGER_IDENT = "zq_trigger_var"
POISON_RATE = 0.05
SEED = 42

ASR_RULE = (
    "Exact match of the generated summary to the attack target after collapsing "
    "whitespace; case and punctuation preserved. Denominator = number of evaluated inputs."
)
BLEU_RULE = (
    "corpus BLEU-4 from sacreBLEU (0-100) of generated summaries against the original "
    "held-out docstrings; mean sentence-level BLEU-4 is also recorded."
)


@dataclass
class Preset:
    name: str
    train_raw: int
    valid_raw: int
    test_raw: int
    epochs: int
    micro_batch: int
    grad_accum: int
    eval_batch: int
    learning_rate: float
    warmup_steps: int
    max_source_length: int
    max_target_length: int
    beam_size: int
    max_steps: int = 0
    eval_sizes: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _presets():
    small = Preset(
        name="small",
        train_raw=120,
        valid_raw=32,
        test_raw=400,
        epochs=10,
        micro_batch=8,
        grad_accum=1,
        eval_batch=8,
        learning_rate=5e-4,
        warmup_steps=10,
        max_source_length=128,
        max_target_length=32,
        beam_size=1,
        max_steps=0,
        eval_sizes={"exp1": 40, "exp3": 40, "exp4_clean": 70, "exp4_poisoned": 10},
    )
    full = Preset(
        name="full",
        train_raw=10000,
        valid_raw=1200,
        test_raw=6000,
        epochs=3,
        micro_batch=8,
        grad_accum=2,
        eval_batch=16,
        learning_rate=5e-5,
        warmup_steps=100,
        max_source_length=256,
        max_target_length=64,
        beam_size=1,
        max_steps=0,
        eval_sizes={"exp1": 1000, "exp3": 1000, "exp4_clean": 1750, "exp4_poisoned": 250},
    )
    return {"small": small, "full": full}


PRESETS = _presets()


def get_preset(name=None):
    name = name or os.environ.get("CODET5_PRESET", "small")
    if name not in PRESETS:
        raise SystemExit(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]


def data_dir():
    return Path(os.environ.get("CODET5_DATA_DIR", PROJECT_ROOT / "datasets" / "codet5"))


def results_dir():
    return Path(os.environ.get("CODET5_RESULTS_DIR", PROJECT_ROOT / "results" / "codet5"))


def checkpoint_dir():
    return Path(os.environ.get("CODET5_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints" / "codet5"))
