"""Frozen four-source SCNet separation for FoR36.

The shipped checkpoint is the public MIMO-SCNet small 4-source model from
Sony Research.  This module deliberately keeps the model behind the same
mixture-only boundary as the Demucs wrapper: the sandbox sees only mixtures,
while the GPU worker owns the model code and checkpoint.  No task fitting or
target access happens here.

The model was trained at 44.1 kHz with source order ``vocals, bass, drums,
other``.  ScienceClaw's FoR36 contract is 22.05 kHz and orders the middle two
sources as ``drums, bass``; the wrapper performs both conversions explicitly.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from . import _remote
from ._pretrained import model_path, switched_off, torch_device

__all__ = ["MODELS", "available", "separate_pretrained"]

MODELS = ("mimo_scnet_small",)
MODEL_DIR_NAME = "scnet_mimo_small"
DEFAULT_CODE_ROOT = str(Path.home() / ".cache" / "scienceclaw" / "scnet" / "mimo-audio-separation")
_cache: dict[tuple[str, str], object] = {}


def _code_root() -> Path:
    """Source tree of the upstream repository: ``$SCIENCECLAW_SCNET_ROOT``, else the default cache location, else a ``source``
    directory next to the staged checkpoint (the only place a sandboxed worker, whose home is its work directory, can find)."""
    explicit = os.environ.get("SCIENCECLAW_SCNET_ROOT")
    if explicit:
        return Path(explicit)
    default = Path(DEFAULT_CODE_ROOT)
    if (default / "src").is_dir():
        return default
    staged = model_path(MODEL_DIR_NAME, "source")
    return staged if staged is not None else default


def _weights_dir() -> Path | None:
    explicit = os.environ.get("SCIENCECLAW_SCNET_MODEL_DIR")
    p = Path(explicit) if explicit else model_path(MODEL_DIR_NAME)
    if p is None:
        return None
    return p if (p / "config.yaml").is_file() and (p / "backbone_model.pth").is_file() else None


def _local_ok() -> bool:
    """Check only lightweight imports/files; do not initialize the network."""
    if switched_off() or _weights_dir() is None or not (_code_root() / "src" / "model").is_dir():
        return False
    return all(importlib.util.find_spec(x) is not None for x in ("torch", "torchaudio", "hydra", "omegaconf"))


def available(model: str = MODELS[0]) -> bool:
    """Whether the frozen SCNet route is staged locally or through the broker."""
    if model not in MODELS:
        return False
    return _local_ok() or _remote.enabled()


def _expose_scnet_backbone() -> None:
    """Import ``model.backbone``; when its package ``__init__`` fails (it also imports the RoFormer backbones, whose modules assert
    a CUDA device at import time, so a CPU host cannot import it), register a package holding the SCNet classes alone."""
    import importlib
    import types

    try:
        importlib.import_module("model.backbone")
        return
    except (AssertionError, ImportError):
        pass
    model_pkg = importlib.import_module("model")
    pkg = types.ModuleType("model.backbone")
    pkg.__path__ = [str(Path(model_pkg.__file__).parent / "backbone")]
    sys.modules["model.backbone"] = pkg
    scnet = importlib.import_module("model.backbone.scnet")
    pkg.SCNet, pkg.MultiSourceSCNet = scnet.SCNet, scnet.MultiSourceSCNet
    model_pkg.backbone = pkg


def _load(device: str):
    import hydra
    import torch
    from omegaconf import OmegaConf

    key = (MODELS[0], device)
    if key in _cache:
        return _cache[key]
    root = _code_root()
    src = root / "src"
    if not src.is_dir():
        raise RuntimeError(f"SCNet source tree is missing: {src}")
    # The public repository uses top-level imports (model, utils).  Keep this
    # path local to the worker and avoid mutating the caller's package layout.
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _expose_scnet_backbone()
    cfg_dir = _weights_dir()
    if cfg_dir is None:
        raise RuntimeError("SCNet checkpoint/config is not staged")
    cfg = OmegaConf.load(cfg_dir / "config.yaml")
    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(cfg_dir / "backbone_model.pth", map_location="cpu", weights_only=False)
    model.backbone_model.load_state_dict(state, strict=True)
    model.eval().to(device)
    _cache[key] = model
    return model


def _resample(x, src: int, dst: int):
    import torchaudio
    return torchaudio.functional.resample(x.contiguous(), src, dst)


def separate_pretrained(
    mixtures,
    model: str = MODELS[0],
    sample_rate: int = 22_050,
    device: str = "auto",
    iterations: int = 2,
    batch_size: int = 2,
) -> np.ndarray:
    """Separate ``(k, samples, 2)`` mixtures into FoR36's four targets."""
    x = np.asarray(mixtures, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError(f"mixtures must have shape (k, n, 2), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("mixtures must contain only finite values")
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if int(sample_rate) != sample_rate or int(sample_rate) <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if int(iterations) != iterations or int(iterations) < 1 or int(iterations) > 2:
        raise ValueError("iterations must be 1 or 2")
    if int(batch_size) != batch_size or int(batch_size) < 1:
        raise ValueError("batch_size must be a positive integer")
    if x.shape[0] == 0:
        return np.zeros((0, 4, x.shape[1], 2), np.float32)
    rate = int(sample_rate)
    if not _local_ok():
        if not _remote.enabled():
            raise RuntimeError("SCNet checkpoint/dependencies are unavailable")
        out = _remote.call(
            "scnet_pretrained",
            "separate_pretrained",
            {"mixtures": x, "model": model, "sample_rate": rate, "device": "auto",
             "iterations": int(iterations), "batch_size": int(batch_size)},
        )
        return np.asarray(out, dtype=np.float32)

    import torch

    dev = torch_device(device)
    net = _load(dev)
    out = np.zeros((x.shape[0], 4, x.shape[1], 2), dtype=np.float32)
    for start in range(0, x.shape[0], int(batch_size)):
        xb = torch.from_numpy(x[start:start + int(batch_size)].transpose(0, 2, 1).copy())
        xb = _resample(xb, rate, 44_100).to(dev)
        with torch.inference_mode():
            pred = net.inference(xb, iters=[int(iterations)])[0].detach().cpu()
        # Public SCNet order is vocals,bass,drums,other; reorder to the task
        # contract and return to the input sample rate.
        pred = _resample(pred.reshape(-1, pred.shape[2], pred.shape[3]), 44_100, rate)
        pred = pred.reshape(-1, 4, 2, pred.shape[-1]).numpy().transpose(0, 1, 3, 2)
        pred = pred[:, [0, 2, 1, 3], :x.shape[1], :]
        out[start:start + pred.shape[0]] = pred
    return out
