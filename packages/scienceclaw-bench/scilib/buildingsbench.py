"""Local-only wrapper for the pinned BuildingsBench Gaussian-L checkpoint.

The interface consumes only visible feature arrays and returns the frozen
model's Gaussian parameters. It does not load BuildingsBench targets, fit a
model, or provide a formal FoR33 scorer. The caller must supply the pinned
upstream source and checkpoint explicitly; task-specific preprocessing remains
outside this component.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from ._pretrained import switched_off

__all__ = ["available", "forecast", "MODEL_ENV", "SOURCE_ENV", "CONTEXT_LENGTH", "PREDICTION_LENGTH"]

MODEL_ENV = "SCIENCECLAW_BUILDINGS_MODEL"
SOURCE_ENV = "SCIENCECLAW_BUILDINGS_SOURCE"
CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 24
TOTAL_LENGTH = CONTEXT_LENGTH + PREDICTION_LENGTH
FEATURES = ("latitude", "longitude", "building_type", "day_of_year", "day_of_week", "hour_of_day", "load")


def _path(value, env):
    raw = value if value is not None else os.environ.get(env)
    return Path(raw).expanduser() if raw else None


def _ok(model_path=None, source_path=None):
    model = _path(model_path, MODEL_ENV)
    source = _path(source_path, SOURCE_ENV)
    return (model is not None and model.is_file() and model.stat().st_size > 0 and source is not None
            and (source / "buildings_bench" / "models" / "transformers.py").is_file()
            and (source / "buildings_bench" / "configs" / "TransformerWithGaussian-L.toml").is_file())


def available(model_path=None, source_path=None) -> bool:
    if switched_off() or not _ok(model_path, source_path):
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _validate(features: Mapping[str, object]):
    if not isinstance(features, Mapping) or set(features) != set(FEATURES):
        raise ValueError(f"features must contain exactly {FEATURES}")
    arrays = {name: np.asarray(features[name]) for name in FEATURES}
    shape = None
    for name, value in arrays.items():
        if value.ndim != 3 or value.shape[1:] != (TOTAL_LENGTH, 1):
            raise ValueError(f"{name} must have shape (n, {TOTAL_LENGTH}, 1)")
        if shape is None:
            shape = value.shape
        elif value.shape != shape:
            raise ValueError("all feature arrays must have the same shape")
        if not np.isfinite(value.astype(np.float32, copy=False)).all():
            raise ValueError(f"{name} must contain only finite values")
    return arrays


def _load(model_path: Path, source_path: Path, device: str):
    import tomllib
    import torch

    sys.modules.setdefault("tomli", tomllib)
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    from buildings_bench.models.transformers import LoadForecastingTransformer

    cfg_path = source_path / "buildings_bench" / "configs" / "TransformerWithGaussian-L.toml"
    cfg = tomllib.loads(cfg_path.read_text())
    model = LoadForecastingTransformer(**dict(cfg["model"]))
    model.load_from_checkpoint(str(model_path))
    return model.eval().to(device), torch


def forecast(features: Mapping[str, object], model_path=None, source_path=None, device: str = "auto") -> np.ndarray:
    """Return frozen Gaussian parameters with shape ``(n, 24, 2)``."""
    arrays = _validate(features)
    model_path = _path(model_path, MODEL_ENV)
    source_path = _path(source_path, SOURCE_ENV)
    if not _ok(model_path, source_path):
        raise RuntimeError(f"buildingsbench: local checkpoint/source unavailable; set {MODEL_ENV} and {SOURCE_ENV}")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, torch = _load(model_path, source_path, str(device))
    x = {
        "latitude": torch.as_tensor(arrays["latitude"], dtype=torch.float32, device=str(device)),
        "longitude": torch.as_tensor(arrays["longitude"], dtype=torch.float32, device=str(device)),
        "building_type": torch.as_tensor(arrays["building_type"], dtype=torch.long, device=str(device)),
        "day_of_year": torch.as_tensor(arrays["day_of_year"], dtype=torch.float32, device=str(device)),
        "day_of_week": torch.as_tensor(arrays["day_of_week"], dtype=torch.float32, device=str(device)),
        "hour_of_day": torch.as_tensor(arrays["hour_of_day"], dtype=torch.float32, device=str(device)),
        "load": torch.as_tensor(arrays["load"], dtype=torch.float32, device=str(device)),
    }
    with torch.inference_mode():
        out = model(x).detach().float().cpu().numpy()
    if out.ndim != 3 or out.shape[1:] != (PREDICTION_LENGTH, 2):
        raise RuntimeError(f"buildingsbench: unexpected output shape {tuple(out.shape)}")
    return np.asarray(out, dtype=np.float32)
