"""Frozen PhenoBench Weyler hierarchical instance model.

This is an inference-only wrapper around the official PRBonn Weyler checkpoint. It
uses RGB pixels only and returns the same three arrays as ``phenoseg_m2f``.
"""
from __future__ import annotations

import numpy as np
from ._pretrained import have_module, model_path, switched_off, torch_device

WEIGHTS = "phenobench_weyler/weyler_checkpoint_0381.pth"
_CACHE = {}


def _path():
    return model_path(*WEIGHTS.split("/"))


def available() -> bool:
    return (not switched_off() and have_module("torch") and _path() is not None)


def _load(device: str = "auto"):
    import torch
    key = device
    if key in _CACHE:
        return _CACHE[key]
    from ._weyler.BranchedERFNet import BranchedERFNet
    dev = torch_device(device)
    model = BranchedERFNet(num_classes=[10, 2], batch_norm=True, instance_norm=False)
    state = torch.load(str(_path()), map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=True)
    model.eval().to(dev)
    _CACHE[key] = (model, dev)
    return model, dev


def _resize_input(x, size=None):
    import torch
    h, w = x.shape[-2:]
    size = int(size or min(1024, max(h, w)))
    if (h, w) == (size, size):
        return x, (h, w)
    y = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return y, (h, w)


def _maps(pred, out_hw):
    # Clustering is the official Weyler post-processing, with thresholds from
    # the released report configuration. It yields crop objects and leaf parts.
    from ._weyler.cluster import Cluster
    import torch
    H, W = pred.shape[-2:]
    cluster = Cluster("np", W, H, 1, 3, 11.0, 11.0, 32, 0.7, 64, 0.7, True)
    results = cluster.cluster(pred.detach().cpu().numpy())[-1]
    sem = np.zeros((H, W), dtype=np.int32)
    plants = np.zeros((H, W), dtype=np.int32)
    leaves = np.zeros((H, W), dtype=np.int32)
    # Official report uses class 0 for crop objects and parts.
    for i, obj in enumerate(results.get("objects", {}).get("0", []), 1):
        for pi in obj.get("obj_part_indicies", []):
            m = np.asarray(results["parts"]["0"][pi]["part_mask"], dtype=bool)
            plants[m] = i
            sem[m] = 1
    # Keep high-confidence part instances even if the object merge discarded
    # their object association; this preserves the hierarchy's leaf output.
    for i, part in enumerate(results.get("parts", {}).get("0", []), 1):
        m = np.asarray(part["part_mask"], dtype=bool)
        leaves[m] = i
        sem[m] = 1
    if out_hw != (H, W):
        th, tw = out_hw
        def down(a):
            t = torch.from_numpy(a[None, None].astype(np.int32)).float()
            return torch.nn.functional.interpolate(t, size=(th, tw), mode="nearest")[0, 0].numpy().astype(np.int32)
        sem, plants, leaves = down(sem), down(plants), down(leaves)
    leaves[sem != 1] = 0
    plants[sem == 0] = 0
    return sem, plants, leaves


def predict_panoptic(images, device: str = "auto", batch_size: int = 1) -> dict[str, np.ndarray]:
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[-1] != 3 or x.dtype != np.uint8 or x.shape[0] < 1:
        raise ValueError("images must be uint8 (n,H,W,3)")
    model, dev = _load(device)
    import torch
    out = {k: np.zeros((len(x), x.shape[1], x.shape[2]), dtype=np.int32)
           for k in ("semantics", "plant_instances", "leaf_instances")}
    with torch.inference_mode():
        for i in range(0, len(x), max(1, int(batch_size))):
            b = torch.from_numpy(np.ascontiguousarray(x[i:i + int(batch_size)])).permute(0, 3, 1, 2).float().to(dev) / 255.0
            b, _ = _resize_input(b)
            y = model(b)
            for j in range(y.shape[0]):
                s, p, l = _maps(y[j], (x.shape[1], x.shape[2]))
                out["semantics"][i + j], out["plant_instances"][i + j], out["leaf_instances"][i + j] = s, p, l
    return out
