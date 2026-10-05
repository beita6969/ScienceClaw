"""Domain libraries importable from ``code`` nodes (the repository root is on the workers' PYTHONPATH).

One module per discipline (``scilib.valueeval`` ...). Modules depend only on numpy / pandas / scipy / scikit-learn
(or the discipline's own scientific stack), read no files and open no connections: every function works on the
arrays it is given. The module docstring is the interface description shown to the policy (:func:`describe`).
"""
from __future__ import annotations

import importlib
import inspect
import os


def describe(name: str) -> str:
    """Policy-visible description of ``scilib.<name>``: its module docstring (first line = one-line summary)."""
    if os.environ.get("SCIENCECLAW_NO_SCILIB") == "1":
        return ""
    mod = importlib.import_module(f"scilib.{name}")
    return f"Library `scilib.{name}`, importable in code nodes:\n" + inspect.cleandoc(mod.__doc__ or "")


def describe_extra(name: str) -> str:
    """Description of an optional pretrained-model module (``*_pretrained`` / ``*_deep``): empty when the pretrained tools
    are switched off (SCIENCECLAW_NO_PRETRAINED=1 or SCIENCECLAW_NO_SCILIB=1) or the module cannot run here."""
    mod = importlib.import_module(f"scilib.{name}")
    text = describe(name) if mod.available() else ""
    return "\n\n" + text if text else ""
