"""Read-only smoke wrapper for Microsoft's frozen InnerEye-HS checkpoint.

The v0.5 release is a five-model 3-D U-Net trained on ADNI (MIT-licensed model
release), not on MSD Task04.  It predicts *whole* left/right hippocampi.  FoR32
requires anterior/posterior labels in a single crop, so this module deliberately
returns a binary union only and is not exposed by the FoR32 formal adapter.  It
exists to verify that the public checkpoint can be loaded and run without
training, target reads, or an invented class mapping.

The small network definition below mirrors the released InnerEye UNet3D state
layout.  It is kept here so a smoke does not need the archived AzureML runtime.
Microsoft's source and model are MIT licensed; see the v0.5 release and the
record in ``configs/p3_asset_manifest.json``/the FoR32 report for provenance.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._pretrained import have_module, model_path, switched_off, torch_device

__all__ = ["available", "predict_binary", "checkpoint_path"]


def checkpoint_path() -> Path | None:
    """Return the staged InnerEye checkpoint, without downloading anything."""
    explicit = os.environ.get("SCIENCECLAW_INNEREYE_CHECKPOINT")
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    root = model_path("innereye")
    if root is None:
        return None
    # The release contains a nested archive.  Accept either an extracted
    # checkpoint or a caller-provided path; never extract/download here.
    hits = sorted(root.glob("**/checkpoints/last.ckpt"))
    return hits[0] if hits else None


def available() -> bool:
    return not switched_off() and have_module("torch") and checkpoint_path() is not None


def _pad_to_multiple(x: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[slice, ...]]:
    shape = np.asarray(x.shape, dtype=int)
    out_shape = np.maximum(multiple, ((shape + multiple - 1) // multiple) * multiple)
    out = np.zeros(tuple(out_shape), dtype=np.float32)
    sl = tuple(slice(0, int(n)) for n in shape)
    out[sl] = x
    return out, sl


def _normalise(x: np.ndarray) -> np.ndarray:
    """Approximate the released MRI-window transform (output range [-1, 1])."""
    a = np.asarray(x, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        raise ValueError("volume contains no finite values")
    try:
        from skimage.filters import threshold_otsu
        threshold = float(threshold_otsu(finite))
    except Exception:
        threshold = float(np.percentile(finite, 20.0))
    fg = finite[finite > threshold]
    if fg.size < 4:
        fg = finite
    q25, med, q75 = np.percentile(fg, (25.0, 50.0, 75.0))
    std = max(float((q75 - q25) / (2.0 * 0.67448975)), 1e-6)
    lo = max(med - 2.5 * std, threshold)
    hi = min(float(fg.max()), med + 2.5 * std)
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    y = np.clip(a, lo, hi)
    y[a < threshold] = lo
    return (2.0 * (y - lo) / (hi - lo) - 1.0).astype(np.float32)


def _network():
    import torch
    import torch.nn as nn

    class Basic(nn.Module):
        def __init__(self, ci, co, stride=1, *, activation=True, batchnorm=True):
            super().__init__()
            self.conv1 = nn.Conv3d(ci, co, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(co) if batchnorm else None
            self.activation = nn.ReLU(inplace=True) if activation else None

        def forward(self, x):
            x = self.conv1(x)
            if self.bn1 is not None:
                x = self.bn1(x)
            return self.activation(x) if self.activation is not None else x

    class Encode(nn.Module):
        def __init__(self, ci, co, stride=1):
            super().__init__()
            self.block1 = Basic(ci, co, stride=stride)
            self.block2 = Basic(co, co)

        def forward(self, x):
            y = self.block1(x)
            return self.block2(y) + y

    class Decode(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.upsample_block = nn.Sequential(
                nn.ConvTranspose3d(ci, co, 4, stride=2, padding=1),
                nn.BatchNorm3d(co), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.upsample_block(x)

    class Synthesis(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv1 = Basic(c, c, activation=False, batchnorm=False)
            self.conv2 = Basic(c, c, activation=False, batchnorm=False)
            self.activation_block = nn.Sequential(nn.BatchNorm3d(c), nn.ReLU(inplace=True))
            self.block2 = Basic(c, c)

        def forward(self, x, skip):
            y = self.activation_block(self.conv1(x) + self.conv2(skip))
            return self.block2(y) + y

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self._layers = nn.ModuleList([
                Encode(1, 16), Encode(16, 32, stride=2), Encode(32, 64, stride=2),
                Encode(64, 128, stride=2), Encode(128, 256, stride=2),
                Decode(256, 128), Synthesis(128), Decode(128, 64), Synthesis(64),
                Decode(64, 32), Synthesis(32), Decode(32, 16), Synthesis(16)])
            self.output_layer = nn.Conv3d(16, 3, 1)

        def forward(self, x):
            skips = []
            pending_skip = None
            for i, layer in enumerate(self._layers):
                # InnerEye's decode block upsamples first; the following
                # synthesis block then consumes the corresponding encoder skip.
                if i in (5, 7, 9, 11):
                    x = layer(x)
                    pending_skip = skips.pop()
                elif i in (6, 8, 10, 12):
                    if pending_skip is None:
                        raise RuntimeError("InnerEye decoder skip is missing")
                    x = layer(x, pending_skip)
                    pending_skip = None
                else:
                    x = layer(x)
                    if i < 4:
                        skips.append(x)
            return self.output_layer(x)

    return Net


_MODEL = None


def _load(device: str):
    global _MODEL
    import torch
    if _MODEL is None:
        p = checkpoint_path()
        if p is None:
            raise RuntimeError("InnerEye-HS checkpoint is not staged")
        net = _network()()
        raw = torch.load(str(p), map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        state = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
        _MODEL = net.eval().to(device)
    return _MODEL


def predict_binary(volumes, device: str = "auto") -> np.ndarray:
    """Predict a binary whole-hippocampus mask for each volume.

    This function intentionally has no FoR32 adapter integration: it returns
    only ``0/1`` (left/right union), never maps to anterior/posterior and never
    accesses labels or a target scorer.
    """
    x = np.asarray(volumes)
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4 or x.shape[0] < 1:
        raise ValueError("volumes must have shape (n, D, H, W) or (D, H, W)")
    if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise ValueError("volumes must contain finite numeric values")
    dev = torch_device(device)
    import torch
    net = _load(dev)
    out = []
    with torch.inference_mode():
        for v in x:
            a, crop = _pad_to_multiple(_normalise(v))
            t = torch.from_numpy(a[None, None]).to(dev)
            logits = net(t)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred = (pred > 0).astype(np.uint8)
            out.append(pred[crop])
    return np.stack(out)
