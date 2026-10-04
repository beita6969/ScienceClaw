"""A 3-D U-Net trained from scratch on the labelled volumes that are passed in (hippocampus crops, labels 0 background,
1 anterior, 2 posterior). No pretrained weights are involved. Training and prediction need torch and a GPU; when this
interpreter cannot run them and a remote GPU worker is configured, the calls run on the worker's GPU host and return the same
kind of values (arrays are transferred at roughly 1 MB per 5-10 s the first time the host sees them; answers are stored, so an
identical call returns the same result).

available() -> bool
    True when torch is importable here or a remote GPU worker is configured.
fit_predict(train_images, train_labels, eval_images, seed=0, **kw) -> list of uint8 arrays
    fit_unet on the labelled training volumes, then predict_unet on ``eval_images`` (any list of volumes, e.g. the dev images
    followed by the evaluation images); one label volume per image, same shape as the image, values 0/1/2. One call, one fit.
fit_unet(train_images, train_labels, seed=0, n_models=3, iters=2500, base=16, batch=4, lr=2e-3, weight_decay=1e-4,
         aug_rotate=15.0, aug_scale=0.15, aug_shift=4.0, device="auto") -> UNetModel
    Volumes are normalised per volume ("robust": (x - median) / (p95 - p5), clipped to [-3, 3]) and zero-padded to a common
    size that is a multiple of 16. ``n_models`` independent networks are trained (seeds seed, seed+1, ...). Network: 4-level
    3-D U-Net (``base`` channels at full resolution, doubling per level), 3x3x3 convolutions with instance normalisation and
    leaky ReLU, strided-convolution down-sampling, transposed-convolution up-sampling, 3-class softmax output. Loss: cross
    entropy + soft Dice over the three classes. AdamW, cosine learning-rate decay from ``lr`` to 1e-2 * lr, ``iters`` updates
    of ``batch`` volumes each. Augmentation for every sampled volume: random rotation about each axis in +-``aug_rotate``
    degrees, isotropic scaling by 1 +- ``aug_scale``, translation in +-``aug_shift`` voxels (nearest-neighbour for the
    labels), intensity scale 1 +- 0.15, offset +-0.15 and Gaussian noise (sd 0.03) on the normalised volume. No mirroring.
    Seeded and deterministic on the same hardware.
predict_unet(model, images, tta=True, device="auto") -> list of uint8 arrays
    Mean of the softmax outputs of the networks; with ``tta`` also of +-2-voxel translations along each axis (7 passes per
    network). The argmax labels go through hippo.postprocess (largest connected foreground component, holes filled, largest
    component of each label).
predict_proba(model, images, tta=True, device="auto") -> list of float32 arrays (3, *shape)
    The averaged class probabilities before post-processing.
save_model(model, path) / load_model(path) -> UNetModel
    Store / read a fitted model (with the remote worker the handle returned by fit_unet names the stored model on the host;
    save_model and load_model store and read that handle).

Measured cost on one GPU with 28 training volumes (volumes of about 40x55x40 voxels): about 44 s per network at ``iters=2500`` for
``base=16`` (default ``n_models=3`` -> about 130 s; ``iters=1000`` -> about 18 s per network); prediction under 1 s for the whole set with ``tta``.
Metric helpers of scilib.hippo (dsc, case_dsc, mean_dsc) work on the outputs.
"""
from __future__ import annotations

import time

import numpy as np

from . import _remote
from . import hippo as hp
from ._pretrained import have_module, switched_off

__all__ = ["available", "fit_unet", "predict_unet", "predict_proba", "fit_predict", "save_model", "load_model", "UNetModel"]

_MULT = 16
_CLIP = 3.0


def _local_ok() -> bool:
    return not switched_off() and have_module("torch")


def available() -> bool:
    return _local_ok() or _remote.enabled()


class UNetModel:
    def __init__(self, states: list, config: dict, n_train: int, fit_s: float):
        self.states, self.config, self.n_train, self.fit_s = states, dict(config), n_train, fit_s

    @property
    def params(self) -> dict:
        return dict(self.config)


class RemoteUNet:
    """Handle of a model that was fitted on the GPU host: ``path`` names the stored UNetModel there."""

    def __init__(self, path: str, params: dict, n_train: int, fit_s: float):
        self.path, self.params, self.n_train, self.fit_s = path, dict(params), n_train, fit_s


def save_model(model, path: str) -> None:
    if isinstance(model, RemoteUNet):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=4)
        return
    import torch
    torch.save({"states": model.states, "config": model.config, "n_train": model.n_train, "fit_s": model.fit_s}, path)


def load_model(path: str):
    import pickle
    with open(path, "rb") as f:
        head = f.read(2)
    if head[:1] == b"\x80":
        with open(path, "rb") as f:
            return pickle.load(f)
    import torch
    d = torch.load(path, map_location="cpu", weights_only=True)
    return UNetModel(d["states"], d["config"], d["n_train"], d["fit_s"])


def _device(device: str):
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _net(base: int):
    import torch
    from torch import nn

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True), nn.LeakyReLU(0.01, inplace=True),
                             nn.Conv3d(o, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True), nn.LeakyReLU(0.01, inplace=True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [base, base * 2, base * 4, base * 8, base * 8]
            self.enc = nn.ModuleList([block(1 if i == 0 else ch[i - 1], ch[i]) for i in range(5)])
            self.down = nn.ModuleList([nn.Conv3d(ch[i], ch[i], 2, stride=2) for i in range(4)])
            self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2) for i in range(4)])
            self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in range(4)])
            self.head = nn.Conv3d(base, 3, 1)

        def forward(self, x):
            skips = []
            for i in range(4):
                x = self.enc[i](x)
                skips.append(x)
                x = self.down[i](x)
            x = self.enc[4](x)
            for i in reversed(range(4)):
                x = self.up[i](x)
                x = self.dec[i](torch.cat([x, skips[i]], 1))
            return self.head(x)

    return Net()


def _prep(img) -> np.ndarray:
    return hp.normalize_intensity(np.asarray(img, dtype=np.float32), "robust").astype(np.float32)


def _pad_to(a: np.ndarray, shape) -> np.ndarray:
    out = np.zeros(shape, dtype=a.dtype)
    out[tuple(slice(0, s) for s in a.shape)] = a
    return out


def _ceil(shape) -> tuple:
    return tuple(int(-(-s // _MULT) * _MULT) for s in shape)


def _augment(x, y, rotate: float, scale: float, shift: float, gen):
    import math

    import torch
    import torch.nn.functional as F
    n = x.shape[0]
    dev = x.device
    ang = (torch.rand(n, 3, generator=gen, device=dev) * 2 - 1) * rotate * math.pi / 180
    s = 1 + (torch.rand(n, 1, generator=gen, device=dev) * 2 - 1) * scale
    cx, cy, cz = torch.cos(ang[:, 0]), torch.cos(ang[:, 1]), torch.cos(ang[:, 2])
    sx, sy, sz = torch.sin(ang[:, 0]), torch.sin(ang[:, 1]), torch.sin(ang[:, 2])
    Rx = torch.zeros(n, 3, 3, device=dev)
    Ry = torch.zeros(n, 3, 3, device=dev)
    Rz = torch.zeros(n, 3, 3, device=dev)
    Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cx; Rx[:, 1, 2] = -sx; Rx[:, 2, 1] = sx; Rx[:, 2, 2] = cx
    Ry[:, 1, 1] = 1; Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy
    Rz[:, 2, 2] = 1; Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz; Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz
    A = (Rx @ Ry @ Rz) / s.view(n, 1, 1)
    size = torch.tensor(x.shape[2:], device=dev, dtype=torch.float32)
    t = (torch.rand(n, 3, generator=gen, device=dev) * 2 - 1) * shift * 2 / size.flip(0)
    theta = torch.cat([A, t.view(n, 3, 1)], 2)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    xo = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    yo = F.grid_sample(y.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False).long()
    gain = 1 + (torch.rand(n, 1, 1, 1, 1, generator=gen, device=dev) * 2 - 1) * 0.15
    off = (torch.rand(n, 1, 1, 1, 1, generator=gen, device=dev) * 2 - 1) * 0.15
    xo = xo * gain + off + torch.randn(xo.shape, generator=gen, device=dev) * 0.03
    return xo, yo


def _train_one(X, Y, seed, iters, base, batch, lr, wd, rotate, scale, shift, dev):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    net = _net(base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 1e-2)
    n = X.shape[0]
    net.train()
    for _ in range(iters):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen, device=dev)
        xb, yb = _augment(X[idx], Y[idx], rotate, scale, shift, gen)
        logits = net(xb)
        ce = F.cross_entropy(logits, yb[:, 0])
        p = torch.softmax(logits, 1)
        oh = F.one_hot(yb[:, 0], 3).permute(0, 4, 1, 2, 3).float()
        inter = (p * oh).sum((0, 2, 3, 4))
        dice = 1 - (2 * inter + 1e-5) / (p.sum((0, 2, 3, 4)) + oh.sum((0, 2, 3, 4)) + 1e-5)
        loss = ce + dice.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def fit_unet(train_images, train_labels, seed: int = 0, n_models: int = 3, iters: int = 2500, base: int = 16, batch: int = 4,
             lr: float = 2e-3, weight_decay: float = 1e-4, aug_rotate: float = 15.0, aug_scale: float = 0.15,
             aug_shift: float = 4.0, device: str = "auto"):
    if not _local_ok() and _remote.enabled():
        r = _remote.call("hippo_unet", "fit_unet",
                         dict(train_images=[np.asarray(a) for a in train_images], train_labels=[np.asarray(a) for a in train_labels],
                              seed=seed, n_models=n_models, iters=iters, base=base, batch=batch, lr=lr, weight_decay=weight_decay,
                              aug_rotate=aug_rotate, aug_scale=aug_scale, aug_shift=aug_shift, device=device))
        return RemoteUNet(r["path"], r["params"], r["n_train"], r["fit_s"])
    if not _local_ok():
        raise RuntimeError("fit_unet: torch is not available here and no remote GPU worker is configured")
    import torch
    t0 = time.time()
    dev = _device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    imgs = [_prep(a) for a in train_images]
    labs = [np.asarray(a).astype(np.int64) for a in train_labels]
    if len(imgs) != len(labs) or not imgs:
        raise ValueError("train_images and train_labels must be non-empty lists of the same length")
    shape = _ceil(np.max([a.shape for a in imgs], axis=0))
    X = torch.from_numpy(np.stack([_pad_to(a, shape) for a in imgs])[:, None]).to(dev)
    Y = torch.from_numpy(np.stack([_pad_to(a, shape) for a in labs])[:, None]).to(dev)
    cfg = dict(seed=seed, n_models=n_models, iters=iters, base=base, batch=batch, lr=lr, weight_decay=weight_decay,
               aug_rotate=aug_rotate, aug_scale=aug_scale, aug_shift=aug_shift)
    states = [_train_one(X, Y, seed + m, iters, base, batch, lr, weight_decay, aug_rotate, aug_scale, aug_shift, dev)
              for m in range(n_models)]
    return UNetModel(states, cfg, len(imgs), time.time() - t0)


def predict_proba(model, images, tta: bool = True, device: str = "auto") -> list:
    if isinstance(model, RemoteUNet):
        return _remote.call("hippo_unet", "predict_proba", {"model": model.path, "images": [np.asarray(a) for a in images],
                                                              "tta": tta, "device": device})
    import torch
    dev = _device(device)
    nets = []
    for st in model.states:
        net = _net(int(model.config["base"])).to(dev)
        net.load_state_dict(st)
        net.eval()
        nets.append(net)
    shifts = [(0, 0, 0)] + ([tuple(s if i == a else 0 for i in range(3)) for a in range(3) for s in (-2, 2)] if tta else [])
    out = []
    with torch.no_grad():
        for img in images:
            a = _prep(img)
            shape = _ceil(a.shape)
            x = torch.from_numpy(_pad_to(a, shape))[None, None].to(dev)
            acc = 0
            for net in nets:
                for sh in shifts:
                    xs = torch.roll(x, sh, (2, 3, 4)) if any(sh) else x
                    p = torch.softmax(net(xs), 1)
                    acc = acc + (torch.roll(p, tuple(-s for s in sh), (2, 3, 4)) if any(sh) else p)
            acc = acc / (len(nets) * len(shifts))
            out.append(acc[0, :, :a.shape[0], :a.shape[1], :a.shape[2]].cpu().numpy().astype(np.float32))
    return out


def predict_unet(model, images, tta: bool = True, device: str = "auto") -> list:
    if isinstance(model, RemoteUNet):
        return _remote.call("hippo_unet", "predict_unet", {"model": model.path, "images": [np.asarray(a) for a in images],
                                                             "tta": tta, "device": device})
    return [hp.postprocess(p) for p in predict_proba(model, images, tta=tta, device=device)]


def fit_predict(train_images, train_labels, eval_images, seed: int = 0, **kw) -> list:
    if not _local_ok() and _remote.enabled():
        return _remote.call("hippo_unet", "fit_predict",
                            dict(train_images=[np.asarray(a) for a in train_images], train_labels=[np.asarray(a) for a in train_labels],
                                 eval_images=[np.asarray(a) for a in eval_images], seed=seed, **kw))
    tta = kw.pop("tta", True)
    device = kw.get("device", "auto")
    model = fit_unet(train_images, train_labels, seed=seed, **kw)
    return predict_unet(model, eval_images, tta=tta, device=device)
