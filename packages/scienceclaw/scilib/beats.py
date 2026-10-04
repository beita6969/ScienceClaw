"""Local-only wrapper for the pinned BEATs iter3 checkpoint.

The wrapper exposes frozen audio embeddings only.  It accepts raw waveform
values at BEATs' native 16 kHz rate, never resamples or reads task targets, and
loads both the checkpoint and the upstream ``BEATs.py`` from explicit local
paths.  It is a component interface; task adapters must choose pooling
and any split-specific protocol before formal use.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from ._pretrained import switched_off

__all__ = ["available", "encode", "MODEL_ENV", "SOURCE_ENV", "SAMPLE_RATE"]

MODEL_ENV = "SCIENCECLAW_BEATS_MODEL"
SOURCE_ENV = "SCIENCECLAW_BEATS_SOURCE"
SAMPLE_RATE = 16_000


def _model_path(model_path=None) -> Path | None:
    raw = model_path if model_path is not None else os.environ.get(MODEL_ENV)
    return Path(raw).expanduser() if raw else None


def _source_path(source_path=None) -> Path | None:
    raw = source_path if source_path is not None else os.environ.get(SOURCE_ENV)
    return Path(raw).expanduser() if raw else None


def _checkpoint_ok(path: Path | None) -> bool:
    return path is not None and path.is_file() and path.stat().st_size > 0


def _source_ok(path: Path | None) -> bool:
    return path is not None and (path / "BEATs.py").is_file()


def available(model_path=None, source_path=None) -> bool:
    if switched_off() or not _checkpoint_ok(_model_path(model_path)) or not _source_ok(_source_path(source_path)):
        return False
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
    except Exception:
        return False
    return True


def _check(waveform, sample_rate: int):
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"sample_rate must be {SAMPLE_RATE}; resampling is not implicit")
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("waveform must have shape (n, samples) or (samples,)")
    if not np.isfinite(x).all():
        raise ValueError("waveform must contain only finite values")
    model_path = _model_path()
    source_path = _source_path()
    if not _checkpoint_ok(model_path):
        raise RuntimeError(f"beats: checkpoint is unavailable; set {MODEL_ENV} to a local .pt file")
    if not _source_ok(source_path):
        raise RuntimeError(f"beats: upstream BEATs.py is unavailable; set {SOURCE_ENV} to its directory")
    return np.ascontiguousarray(x), model_path, source_path


def _load(model_path: Path, source_path: Path, device: str):
    import torch

    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("scienceclaw_beats_upstream", source_path / "BEATs.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("beats: cannot import pinned upstream BEATs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "cfg" not in checkpoint or "model" not in checkpoint:
        raise RuntimeError("beats: checkpoint must contain cfg and model")
    model = module.BEATs(module.BEATsConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().to(device), torch


def encode(waveform, sample_rate: int = SAMPLE_RATE, model_path=None, source_path=None, device: str = "auto") -> np.ndarray:
    """Return frozen BEATs embeddings with shape ``(n, frames, 768)``."""
    x = np.asarray(waveform, dtype=np.float32)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"sample_rate must be {SAMPLE_RATE}; resampling is not implicit")
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("waveform must have shape (n, samples) or (samples,)")
    if not np.isfinite(x).all():
        raise ValueError("waveform must contain only finite values")
    model_path = _model_path(model_path)
    source_path = _source_path(source_path)
    if not _checkpoint_ok(model_path):
        raise RuntimeError(f"beats: checkpoint is unavailable; set {MODEL_ENV} to a local .pt file")
    if not _source_ok(source_path):
        raise RuntimeError(f"beats: upstream BEATs.py is unavailable; set {SOURCE_ENV} to its directory")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, torch = _load(model_path, source_path, str(device))
    with torch.inference_mode():
        result = model.extract_features(torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=str(device)))
    out = result[0] if isinstance(result, tuple) else result
    out = out.detach().float().cpu().numpy()
    if out.ndim != 3 or out.shape[0] != x.shape[0] or out.shape[1] < 1 or out.shape[2] < 1:
        raise RuntimeError(f"beats: unexpected embedding shape {tuple(out.shape)}")
    return np.asarray(out, dtype=np.float32)
