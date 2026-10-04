"""Execution runtime (paper Eq. 7-8): executor, sandboxed code nodes, reset replay, value store.

Submodules are imported lazily so that the code-node worker (``runtime.node_worker``), which is
started as ``python -m scienceclaw.runtime.node_worker``, does not pull in the executor, the
integrity scanner or anything that could reach the LLM client.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "Checkpoint": "executor", "Executor": "executor", "Feedback": "executor",
    "replay": "replay", "run_operator_isolated": "replay",
    "run_code_node": "sandbox", "scan_code": "integrity",
    "save_value": "values", "load_value": "values", "outputs_match": "values",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)
