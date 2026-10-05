"""Local-only wrapper for the staged Granite TinyTimeMixer checkpoint.

This optional component is deliberately separate from the benchmark adapters.  It
only loads a checkpoint that the caller has placed on disk (``local_files_only``
is always used), accepts context values, and returns the model's point forecast.
It does not fit, read targets, or provide a formal FoR37 scoring path.  Set
``SCIENCECLAW_GRANITE_MODEL`` to the checkpoint directory, or pass ``model_dir``
explicitly.  The expected r2 checkpoint has a 512-step context and a 96-step
forecast for one channel; any frequency conversion for a task must be decided in
the task adapter before this component is used.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from ._pretrained import model_path as staged_path
from ._pretrained import switched_off

__all__ = ["available", "forecast", "MODEL_ENV", "CONTEXT_LENGTH", "PREDICTION_LENGTH"]

MODEL_ENV = "SCIENCECLAW_GRANITE_MODEL"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96


def _path(model_dir=None) -> Path | None:
    raw = model_dir if model_dir is not None else os.environ.get(MODEL_ENV)
    if not raw:
        return staged_path("granite")        # a sandboxed worker inherits only SCIENCECLAW_MODELS
    return Path(raw).expanduser()


def _checkpoint_ok(path: Path | None) -> bool:
    if path is None or not path.is_dir():
        return False
    # safetensors is the staged artifact; accepting the standard PyTorch name
    # keeps this wrapper useful for an equivalent local export.
    return (path / "config.json").is_file() and any(
        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )


def available(model_dir=None) -> bool:
    """Return whether the optional local Granite component can be loaded here."""
    if switched_off() or not _checkpoint_ok(_path(model_dir)):
        return False
    try:
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("tsfm_public") is not None
    except (ImportError, ValueError):
        return False


def _check(context, model_dir=None):
    path = _path(model_dir)
    if not _checkpoint_ok(path):
        raise RuntimeError(
            f"granite: checkpoint is unavailable; set {MODEL_ENV} to a local Granite directory "
            "containing config.json and model.safetensors"
        )
    x = np.asarray(context, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3 or x.shape[1] != CONTEXT_LENGTH or x.shape[2] != 1:
        raise ValueError(f"context must have shape (n, {CONTEXT_LENGTH}) or (n, {CONTEXT_LENGTH}, 1)")
    if x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError("context must contain at least one finite series")
    return np.ascontiguousarray(x), path


def _load(path: Path, device: str):
    import torch
    # granite-tsfm exposes the model class from its models submodule; the
    # package root only exports pipeline helpers in current releases.
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    model = TinyTimeMixerForPrediction.from_pretrained(str(path), local_files_only=True)
    return model.eval().to(device), torch


def forecast(context, model_dir=None, device: str = "auto") -> np.ndarray:
    """Run one frozen local forward pass and return ``(n, 96, 1)`` float32 values.

    ``context`` contains only the past values visible to the component.  No
    future values or labels are accepted, and checkpoint loading never accesses
    the network.  This is a component interface, not a benchmark scorer.
    """
    x, path = _check(context, model_dir)
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, torch = _load(path, str(device))
    with torch.inference_mode():
        result = model(
            past_values=torch.as_tensor(x, dtype=torch.float32, device=str(device)),
            return_loss=False,
        )
    y = getattr(result, "prediction_outputs", None)
    if y is None:
        raise RuntimeError("granite: model output did not contain prediction_outputs")
    out = y.detach().float().cpu().numpy()
    if out.ndim != 3 or out.shape[1:] != (PREDICTION_LENGTH, 1):
        raise RuntimeError(f"granite: unexpected output shape {tuple(out.shape)}")
    return np.asarray(out, dtype=np.float32)
