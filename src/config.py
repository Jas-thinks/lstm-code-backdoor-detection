import copy
import time
from pathlib import Path

import yaml

from .paths import APP_ROOT

DEFAULT = APP_ROOT / "config" / "config.yaml"


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path=None, overrides=None):
    cfg = yaml.safe_load(open(path or DEFAULT))
    cfg = _merge(cfg, overrides or {})
    validate(cfg)
    if not cfg.get("run_name"):
        cfg["run_name"] = time.strftime("run_%Y%m%d_%H%M%S")
    return cfg


def parse_set(items):
    """['poison.rate=0.2', 'training.epochs=5'] -> nested dict (values parsed as YAML)."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects key=value, got '{it}'")
        k, v = it.split("=", 1)
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    return out


def validate(cfg):
    from . import attacks
    import transformations as T
    p = cfg["poison"]
    if not 0 < p["rate"] < 1:
        raise ValueError("poison.rate must be in (0,1)")
    for t in p["transformations"] + list(cfg["evaluation"].get("extra_transformations") or []):
        if t not in T.all_names():
            raise ValueError(f"unknown attack {t}; available: {sorted(T.all_names())}")
        if t in attacks.ATTACKS and not attacks.ATTACKS[t].model_visible:
            raise ValueError(f"attack '{t}' is invisible to the repo tokenizer: {attacks.ATTACKS[t].note}")
    if cfg["training"]["epochs"] < 1 or cfg["training"]["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be >= 1")
    if cfg["data"]["train_clean"] < 1 or cfg["data"]["valid_clean"] < 1:
        raise ValueError("data.train_clean / valid_clean must be >= 1")
