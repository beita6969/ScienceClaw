"""Locate pretrained weights that were staged on local disk beforehand (nothing here downloads or opens a connection).

The ``*_pretrained`` / ``*_deep`` modules of ``scilib`` read weight files from ``<model root>/<subdirectory>``. The model
root is the first existing directory among ``$SCIENCECLAW_MODELS`` and ``~/.cache/scienceclaw/models`` (the sandbox forwards
only that variable, not the rest of the environment).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

MODEL_ROOTS = (str(Path.home() / ".cache" / "scienceclaw" / "models"),)


def _roots() -> list[Path]:
    env = os.environ.get("SCIENCECLAW_MODELS")
    return [Path(p) for p in ([env] if env else []) + list(MODEL_ROOTS)]


def model_path(*parts: str) -> Path | None:
    """``<root>/<parts...>`` for the first model root in which it exists, else None."""
    for r in _roots():
        p = r.joinpath(*parts)
        if p.exists():
            return p
    return None


def have_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def switched_off() -> bool:
    """True when the pretrained tools are hidden (``SCIENCECLAW_NO_PRETRAINED=1``) or all scilib docs are (``SCIENCECLAW_NO_SCILIB=1``)."""
    return os.environ.get("SCIENCECLAW_NO_PRETRAINED") == "1" or os.environ.get("SCIENCECLAW_NO_SCILIB") == "1"


def torch_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_cpu_threads(n: int = 4) -> None:
    """Allow ``n`` torch CPU threads (the sandbox default of 2 is kept when fewer cores are available)."""
    import torch
    torch.set_num_threads(max(1, min(int(n), os.cpu_count() or 1)))
