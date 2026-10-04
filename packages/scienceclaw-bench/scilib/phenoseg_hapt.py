"""Frozen HAPT hierarchical panoptic segmentation with a PhenoBench adapter port.

This is a small, dependency-light inference port of the public PRBonn HAPT
checkpoint.  It contains the released ERFNet-style encoder/decoder definition
and the checkpoint's deterministic center/offset post-processing, but no
training or target access.  HAPT predicts soil/crop/weed semantics, crop-plant
instances and crop-leaf instances; its output is converted to the same
``phenoseg`` dictionary used by the FoR30 adapter.

The checkpoint is staged under ``models/hapt/hapt_model.ckpt`` and is never
downloaded by this module.  The published checkpoint is the GrowliFlower
release (the upstream HAPT README/config names that dataset), not a
PhenoBench-trained model.  It is therefore a diagnostic-only route for the
current FoR30 adapter until a matching label/domain contract is established;
callers must disclose this provenance and must not treat it as a PhenoBench
held-out baseline.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._pretrained import have_module, model_path, switched_off, torch_device

__all__ = ["available", "checkpoint_path", "predict_panoptic"]


def checkpoint_path() -> Path | None:
    explicit = os.environ.get("SCIENCECLAW_HAPT_CHECKPOINT")
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    root = model_path("hapt")
    if root is None:
        return None
    p = root / "hapt_model.ckpt"
    return p if p.is_file() else None


def available() -> bool:
    return not switched_off() and have_module("torch") and checkpoint_path() is not None


def _norm_init(module, init_name: str) -> None:
    import torch.nn as nn

    if init_name == "he":
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif init_name == "xavier":
        nn.init.xavier_normal_(module.weight)


def _network():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class NonBottleneck1D(nn.Module):
        def __init__(self, channels: int, drop: float, dilated: int, batch_norm: bool = False):
            super().__init__()
            self.conv3x1_1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
            self.conv1x3_1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
            self.bn1 = nn.BatchNorm2d(channels, eps=1e-3) if batch_norm else None
            self.conv3x1_2 = nn.Conv2d(channels, channels, (3, 1), padding=(dilated, 0), dilation=(dilated, 1))
            self.conv1x3_2 = nn.Conv2d(channels, channels, (1, 3), padding=(0, dilated), dilation=(1, dilated))
            self.bn2 = nn.BatchNorm2d(channels, eps=1e-3) if batch_norm else None
            self.dropout = nn.Dropout2d(drop)

        def forward(self, x):
            y = F.gelu(self.conv3x1_1(x))
            y = self.conv1x3_1(y)
            if self.bn1 is not None:
                y = self.bn1(y)
            y = F.gelu(y)
            y = F.gelu(self.conv3x1_2(y))
            y = self.conv1x3_2(y)
            if self.bn2 is not None:
                y = self.bn2(y)
            if self.dropout.p:
                y = self.dropout(y)
            return F.gelu(y + x)

    class DownsamplerBlock(nn.Module):
        def __init__(self, ninput: int, noutput: int, batch_norm: bool = True):
            super().__init__()
            self.conv = nn.Conv2d(ninput, noutput - ninput, 3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
            self.bn = nn.BatchNorm2d(noutput, eps=1e-3) if batch_norm else None

        def forward(self, x):
            y = torch.cat([self.conv(x), self.pool(x)], 1)
            if self.bn is not None:
                y = self.bn(y)
            return F.gelu(y)

    class UpsamplerBlock(nn.Module):
        def __init__(self, ninput: int, noutput: int):
            super().__init__()
            self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1)
            self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

        def forward(self, x):
            return F.gelu(self.bn(self.conv(x)))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.initial_block = DownsamplerBlock(3, 16)
            layers = [DownsamplerBlock(16, 64)]
            layers.extend(NonBottleneck1D(64, 0.15, 1, batch_norm=True) for _ in range(5))
            layers.append(DownsamplerBlock(64, 128))
            for _ in range(2):
                layers.extend(NonBottleneck1D(128, 0.15, d, batch_norm=True) for d in (2, 4, 8, 16))
            # The released checkpoint includes the encoder's auxiliary semantic
            # head even though the multi-task decoder uses ``decoder_semseg``.
            # Keep it in the module so strict checkpoint loading remains a real
            # architecture check rather than silently dropping public weights.
            self.output_conv = nn.Conv2d(128, 3, 1)
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            out = [self.initial_block(x)]
            for layer in self.layers:
                out.append(layer(out[-1]))
            return out

    class SemanticDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers1 = nn.Sequential(UpsamplerBlock(128, 64), NonBottleneck1D(64, 0.15, 1, batch_norm=True))
            self.layers2 = nn.Sequential(UpsamplerBlock(64, 16), NonBottleneck1D(16, 0.15, 1, batch_norm=True))
            self.output_conv = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)

        def forward(self, enc):
            skip2, _, _, _, _, _, skip1, _, _, _, _, _, _, _, _, out = enc
            y1 = self.layers1(out) + skip1
            y2 = self.layers2(y1) + skip2
            return self.output_conv(y2), [y2, y1]

    class InstanceDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers1 = nn.Sequential(UpsamplerBlock(128, 64), NonBottleneck1D(64, 0.15, 1, batch_norm=True),
                                         NonBottleneck1D(64, 0.15, 1, batch_norm=True))
            self.layers2 = nn.Sequential(UpsamplerBlock(64, 16), NonBottleneck1D(16, 0.15, 1, batch_norm=True),
                                         NonBottleneck1D(16, 0.15, 1, batch_norm=True))
            self.output_center = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
            self.output_offset = nn.ConvTranspose2d(16, 2, 3, stride=2, padding=1, output_padding=1)

        def forward(self, inp):
            out, skip1, skip2 = inp
            y1 = self.layers1(out) + skip2
            y2 = self.layers2(y1) + skip1
            return torch.sigmoid(self.output_center(y2)), self.output_offset(y2), [y2, y1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder_semseg = SemanticDecoder()
            self.decoder_plant = InstanceDecoder()
            self.decoder_leaf = InstanceDecoder()

        def forward(self, x):
            enc = self.encoder(x)
            sem, skips = self.decoder_semseg(enc)
            pc, po, skips = self.decoder_plant((enc[-1], skips[0], skips[1]))
            lc, lo, _ = self.decoder_leaf((enc[-1], skips[0], skips[1]))
            return sem, pc, po, lc, lo

    return Net


_MODEL = None


def _load(device: str):
    global _MODEL
    import torch

    if _MODEL is None:
        p = checkpoint_path()
        if p is None:
            raise RuntimeError("HAPT checkpoint is not staged")
        net = _network()()
        raw = torch.load(str(p), map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        state = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith(("sem_loss.",))}
        net.load_state_dict(state, strict=True)
        _MODEL = net.eval().to(device)
    return _MODEL


def _instances(center: np.ndarray, offsets: np.ndarray, semantic: np.ndarray, *, nms_kernel: int,
               grouping_dist: float, threshold_fraction: float) -> np.ndarray:
    """CPU equivalent of the released ``our_instance`` post-processing."""
    from scipy import ndimage

    center = np.asarray(center, dtype=np.float32) * semantic.astype(np.float32)
    mx = float(center.max()) if center.size else 0.0
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros(semantic.shape, dtype=np.int32)
    threshold = threshold_fraction * mx
    pooled = ndimage.maximum_filter(center, size=int(nms_kernel), mode="constant")
    cy, cx = np.nonzero((center >= threshold) & (center == pooled) & (center > 0))
    if len(cy) == 0:
        return np.zeros(semantic.shape, dtype=np.int32)
    yy, xx = np.indices(semantic.shape, dtype=np.float32)
    radius = np.sqrt(np.sum(np.asarray(offsets, dtype=np.float32) ** 2, axis=0))
    dist = np.sqrt((yy[..., None] - cy[None, None, :]) ** 2 + (xx[..., None] - cx[None, None, :]) ** 2)
    crop = semantic.astype(bool)
    belongs = np.isclose(dist, radius[..., None], atol=float(grouping_dist)) & crop[..., None]
    count = belongs.sum(axis=2)
    out = np.zeros(semantic.shape, dtype=np.int32)
    unique = (count == 1)
    if unique.any():
        out[unique] = np.argmax(belongs[unique], axis=1).astype(np.int32) + 1
    overlap = (count > 1) & crop
    if overlap.any():
        out[overlap] = (np.argmin(dist[overlap], axis=1).astype(np.int32) + 1)
    return out


def _check(images) -> np.ndarray:
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[-1] != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] < 1 or x.dtype != np.uint8:
        raise ValueError("images must be a non-empty uint8 array (n, H, W, 3)")
    return np.ascontiguousarray(x)


def predict_panoptic(images, device: str = "auto") -> dict[str, np.ndarray]:
    """Run the frozen HAPT model and return PhenoBench-compatible arrays."""
    x = _check(images)
    if not available():
        raise RuntimeError("predict_panoptic: HAPT torch stack or checkpoint is unavailable")
    import torch
    import torch.nn.functional as F

    dev = torch_device(device)
    net = _load(dev)
    H, W = x.shape[1:3]
    # HAPT's released validation transform is (256, 512), with ToTensor only.
    inp = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255.0)
    inp = F.interpolate(inp, size=(256, 512), mode="bilinear", align_corners=False).to(dev)
    with torch.inference_mode():
        sem, pc, po, lc, lo = net(inp)
    sem = torch.argmax(sem, dim=1).cpu().numpy().astype(np.int16)
    pc, po = pc[:, 0].cpu().numpy(), po.cpu().numpy()
    lc, lo = lc[:, 0].cpu().numpy(), lo.cpu().numpy()
    out = {k: [] for k in ("semantics", "plant_instances", "leaf_instances")}
    for i in range(len(x)):
        si = sem[i]
        pi = _instances(pc[i], po[i], si, nms_kernel=41, grouping_dist=20.0, threshold_fraction=0.85)
        li = _instances(lc[i], lo[i], si, nms_kernel=11, grouping_dist=2.0, threshold_fraction=0.75)
        if (H, W) != si.shape:
            si = np.asarray(F.interpolate(torch.from_numpy(si[None, None].astype(np.float32)), size=(H, W), mode="nearest")[0, 0]).astype(np.int16)
            pi = np.asarray(F.interpolate(torch.from_numpy(pi[None, None].astype(np.float32)), size=(H, W), mode="nearest")[0, 0]).astype(np.int32)
            li = np.asarray(F.interpolate(torch.from_numpy(li[None, None].astype(np.float32)), size=(H, W), mode="nearest")[0, 0]).astype(np.int32)
        out["semantics"].append(si)
        out["plant_instances"].append(pi)
        out["leaf_instances"].append(li)
    return {k: np.stack(v) for k, v in out.items()}
