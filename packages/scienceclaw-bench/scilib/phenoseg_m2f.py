"""Hierarchical panoptic segmentation of top-down field images (soil / crop / weed semantics, crop and weed plant instances, crop-leaf
instances) with the two Mask2Former checkpoints (ResNet-50 backbone, 100 queries, 3 classes: soil, crop, weed) that the authors of
PhenoBench released (GitHub organisation PRBonn, PhenoBench repositories; "plants" = panoptic segmentation, "leaves" = leaf instance
segmentation). The checkpoints were converted to the Hugging Face ``Mask2FormerForUniversalSegmentation`` format; the converted
models reproduce the released predictions (pixel agreement of the semantic map 99.8-99.96 % on three images). Running them needs torch,
transformers, the two converted checkpoints and a GPU; when this interpreter cannot run them and a remote GPU worker is configured,
the calls run on the worker's GPU host and return the same values (answers are stored, so an identical call returns the same result).
Nothing is fitted on the images that are passed in.

available() -> bool
    True when torch, transformers and both converted checkpoints are available here, or a remote GPU worker is configured.
predict_panoptic(images, plant_threshold=0.8, leaf_threshold=0.8, batch_size=4, device="auto") -> dict of arrays
    ``images``: uint8 (n, H, W, 3) RGB (any H, W; each image is resized by the models to a shorter side of 1024 and the result is
    resized back). Returns the prediction layout of ``scilib.phenoseg`` (each array (n, H, W)): ``semantics`` (int; 0 soil, 1 crop,
    2 weed), ``plant_instances`` (int; 1, 2, ... per crop or weed plant, 0 = none) and ``leaf_instances`` (int; crop leaves, 1, 2, ...,
    0 = none; only inside pixels whose semantics is crop). The "plants" model gives the first two arrays: queries whose class
    probability is >= ``plant_threshold`` are kept, every pixel goes to the kept query with the highest probability x mask
    probability (mask logits > 0), a mask is dropped when less than 80 % of it survives the overlap resolution, soil is one
    stuff segment. The "leaves" model gives the third: the same procedure with ``leaf_threshold``, crop segments only. Mask2Former
    inference on the image resized to 1024 x 1024 with the training-time input normalisation (RGB values minus 103.53 / 116.28 /
    123.675, no scaling).
assemble(plant_seg, plant_info, leaf_seg, leaf_info) -> dict    the three arrays of one image from the two models' segment maps and
    segment lists ({"id", "label_id"}); the pure-Python part of ``predict_panoptic``.
MODELS                                                           dict name -> subdirectory of the model root

Training data: both checkpoints were trained by their authors on the official PhenoBench ``train`` partition (1,407 labelled images of
sugar-beet fields, UAV captures of 2020 and 2021 dates) and early-stopped on its ``val`` partition; the ResNet-50 backbone started from
ImageNet weights. Which images of this benchmark come from the ``train`` partition is not stated here. Training used 1,407 labelled
images, many times more than the labelled images an episode provides.

Cost: an episode of 16 images of 512 x 512 took 8-13 s on one shared A100 for the two models together (measured inside a full
evaluation that also scores the result); the two models need about 20 GB of GPU memory with batch size 4; loading the two
checkpoints takes a few seconds. A remote call adds a start-up of several seconds and its answer is stored.
"""
from __future__ import annotations

import numpy as np

from . import _remote
from ._pretrained import have_module, model_path, switched_off

__all__ = ["available", "predict_panoptic", "assemble", "MODELS"]

MODELS = {"plants": "phenobench_m2f/hf_plants", "leaves": "phenobench_m2f/hf_leaves"}
_MEAN = (103.53, 116.28, 123.675)
_CACHE: dict = {}


def _dir(kind: str):
    return model_path(*MODELS[kind].split("/"))


def _local_ok() -> bool:
    if switched_off() or not (have_module("torch") and have_module("transformers")):
        return False
    return all((d := _dir(k)) is not None and any(d.glob("*.safetensors")) for k in MODELS)


def available() -> bool:
    return _local_ok() or _remote.enabled()


def _check(images, plant_threshold, leaf_threshold, batch_size):
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[3] != 3 or x.dtype != np.uint8 or x.shape[0] < 1:
        raise ValueError("images must be a uint8 array (n, H, W, 3) with n >= 1")
    for name, v in (("plant_threshold", plant_threshold), ("leaf_threshold", leaf_threshold)):
        if not 0.0 < float(v) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    return np.ascontiguousarray(x)


def assemble(plant_seg, plant_info, leaf_seg, leaf_info) -> dict:
    """semantics / plant_instances / leaf_instances (each (H, W) int) of one image.

    ``plant_seg`` / ``leaf_seg``: (H, W) segment-id maps (-1 or an id absent from the list = no segment) of the two models,
    ``*_info``: lists of {"id", "label_id"} with label_id 0 soil, 1 crop, 2 weed."""
    plant_seg, leaf_seg = np.asarray(plant_seg), np.asarray(leaf_seg)
    sem = np.zeros(plant_seg.shape, dtype=np.int64)
    pinst = np.zeros(plant_seg.shape, dtype=np.int64)
    k = 0
    for s in plant_info:
        if int(s["label_id"]) in (1, 2):
            m = plant_seg == int(s["id"])
            k += 1
            sem[m] = int(s["label_id"])
            pinst[m] = k
    linst = np.zeros(plant_seg.shape, dtype=np.int64)
    k = 0
    for s in leaf_info:
        if int(s["label_id"]) == 1:
            m = (leaf_seg == int(s["id"])) & (sem == 1)
            if m.any():
                k += 1
                linst[m] = k
    return {"semantics": sem, "plant_instances": pinst, "leaf_instances": linst}


def _load(device: str):
    key = ("models", device)
    if key not in _CACHE:
        import torch
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        proc = Mask2FormerImageProcessor(do_resize=True, size={"shortest_edge": 1024, "longest_edge": 1024}, size_divisor=32,
                                         do_rescale=False, do_normalize=True, image_mean=list(_MEAN), image_std=[1.0, 1.0, 1.0],
                                         ignore_index=255, reduce_labels=False, num_labels=3)
        nets = {k: Mask2FormerForUniversalSegmentation.from_pretrained(str(_dir(k))).eval().to(dev) for k in MODELS}
        _CACHE[key] = (proc, nets, dev)
    return _CACHE[key]


def _segments(proc, net, dev, batch, size, threshold):
    import torch
    x = proc(images=list(batch), return_tensors="pt")
    x = {k: v.to(dev) for k, v in x.items() if k in ("pixel_values", "pixel_mask")}
    with torch.inference_mode():
        out = net(**x)
    res = proc.post_process_panoptic_segmentation(out, threshold=float(threshold), mask_threshold=0.5, overlap_mask_area_threshold=0.8,
                                                  label_ids_to_fuse={0}, target_sizes=[size] * len(batch))
    return [(r["segmentation"].cpu().numpy(), r["segments_info"]) for r in res]


def _predict_local(images, plant_threshold, leaf_threshold, batch_size, device) -> dict:
    proc, nets, dev = _load(device)
    n, H, W = images.shape[:3]
    out = {k: np.zeros((n, H, W), dtype=np.int32) for k in ("semantics", "plant_instances", "leaf_instances")}
    for i in range(0, n, int(batch_size)):
        batch = images[i:i + int(batch_size)]
        pl = _segments(proc, nets["plants"], dev, batch, (H, W), plant_threshold)
        lf = _segments(proc, nets["leaves"], dev, batch, (H, W), leaf_threshold)
        for j, ((ps, pi), (ls, li)) in enumerate(zip(pl, lf)):
            r = assemble(ps, pi, ls, li)
            for k in out:
                out[k][i + j] = r[k]
    return out


def predict_panoptic(images, plant_threshold: float = 0.8, leaf_threshold: float = 0.8, batch_size: int = 4,
                     device: str = "auto") -> dict:
    x = _check(images, plant_threshold, leaf_threshold, batch_size)
    if not _local_ok() and _remote.enabled():
        res = _remote.call("phenoseg_m2f", "predict_panoptic", {"images": x, "plant_threshold": float(plant_threshold),
                                                                 "leaf_threshold": float(leaf_threshold), "batch_size": int(batch_size)})
        return {k: np.asarray(v) for k, v in res.items()}
    if _local_ok():
        return _predict_local(x, plant_threshold, leaf_threshold, batch_size, device)
    raise RuntimeError("predict_panoptic: neither local checkpoints nor a remote GPU worker are available")
