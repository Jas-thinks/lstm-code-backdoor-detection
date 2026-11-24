"""Access to the shared alpha-renaming implementation (``<repo>/src/augment.py``).

The renaming code was written for the earlier LSTM work and is shared project
code, so it is deliberately **not** copied into ``codet5/``: this module only
re-exports it. That keeps a single source of truth for the rename logic while
letting the CodeT5 package import it as ``src.augment``.

The original file is loaded by its path (not via ``import src.augment``) so the
package name ``src`` is free to mean ``codet5/src`` for the CodeT5 pipeline.
"""
import importlib.util
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[2] / "src" / "augment.py"
if not _SHARED.is_file():
    raise ImportError(
        f"The shared alpha-renaming implementation was not found at {_SHARED}. "
        "Run the CodeT5 pipeline from inside a full checkout of the repository, "
        "or set up that file."
    )

_spec = importlib.util.spec_from_file_location("_shared_augment", _SHARED)
_module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_shared_augment", _module)
_spec.loader.exec_module(_module)

AugmentSkip = _module.AugmentSkip
alpha_rename = _module.alpha_rename

__all__ = ["AugmentSkip", "alpha_rename"]
