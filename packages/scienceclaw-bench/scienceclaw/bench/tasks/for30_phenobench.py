"""FoR30 Agricultural, veterinary and food sciences — PhenoBench v1.1.0 hierarchical panoptic segmentation (PQ+).

Item
    One UAV image of a sugar-beet field (official 1024x1024 RGB) with its official annotations: semantics
    (0 soil, 1 crop, 2 weed, 3 partial crop, 4 partial weed), plant instances, leaf instances (crop leaves) and the
    plant/leaf visibility maps. By default images and annotations are downsampled by 2 (512x512; RGB by 2x2 area
    mean, labels/visibility by nearest neighbour = top-left sample of every 2x2 block) to keep the tool outputs and
    the agent's arrays small on a CPU laptop; ``Adapter(scale=1)`` restores native resolution.
Delivery (data team's ``reconstructed_v2``, design ``FoR30.v2``)
    The pools are exactly the four frozen role files of the delivery catalog (``_delivery_roles``): source 32,
    validation 32, ID 64 and OOD 64 images, one row per image with the six PNG paths + sha256, the capture id
    (``lineage_group`` = the P-number of the file name) and the persistent crop-lineage ids. Source/validation come
    from the official train split, ID from the official val split, all on 05-15. The catalog names the current
    ``reconstructed_vN`` directory (``resolve_delivery``); a newer directory that the catalog does not bind is only
    reported. Role files are sha256-checked against the catalog, image/label files against the row hashes (when a
    derived array is built). Every unselected image of the official population is quarantined by the data team and is
    not accessible here.
IID / OOD
    The official split has no second dataset available locally, so OOD is a *proxy within the dataset*
    (``lineage["ood_kind"] = "proxy_within_dataset"``): acquisition-date shift 05-15 (source/val/ID) -> 06-05 (OOD;
    later growth stage) together with held-out crop lineages. Source, val, ID and OOD share no image, no capture id
    and no persistent crop id (re-verified on load, ``Adapter.verify_disjoint``).
Episodes
    Held-out splits (val/id/ood) draw ``items_per_episode`` images per episode by a seeded permutation of the role
    (item-disjoint; episodes shrink to ``pool // n`` images when ``n * items_per_episode`` exceeds the pool, recorded
    as ``lineage["items_requested"]``). src (and rep, its frozen copy) episodes hold the images of ONE source
    capture (13 / 11 / 8 images, cycled over a seeded capture order; they may repeat images across episodes).
Visible labelled data (episode-relative, from the source role only)
    ``load_train`` returns the labelled images of the source captures that hold none of the episode's images;
    ``load_dev_inputs`` + ``score_dev`` give the RGB images of the smallest remaining source capture with labels
    withheld. Val/ID/OOD annotations are never visible; a src episode never sees labels of its own capture. With
    ``Adapter(expose_source_labels=False)`` only ``load_eval_inputs`` remains (RGB only).
Metric (D_V)
    PQ+ of the official PhenoBench hierarchical task (PRBonn/phenobench @ 0edc128, ``phenobench_eval.py --task
    hierarchical``): PQ+ = (IoU_soil + IoU_weed + PQ_crop + PQ_leaf) / 4, in percent.
    * IoU: confusion matrix accumulated over the episode's images after mapping partial classes 3->1, 4->2 in
      prediction and ground truth (torchmetrics MulticlassJaccardIndex, absent class -> 0).
    * PQ_crop: per image, ground-truth plant instances with visibility <= 0.5 are removed from the GT semantics and
      predicted instances lying > 50 % inside such a partial GT instance are removed from the prediction
      (official ``filter_partial_masks``); PQ of class 1 (crop) with IoU > 0.5 matching,
      PQ = sum IoU / (TP + FP/2 + FN/2), averaged over images containing GT crops.
    * PQ_leaf: the same with leaf instances (semantics = instance > 0) and leaf visibility.
    The official code is a torch implementation; this module is an exact numpy port (validated against the
    official scorer, see docs/tasks/FoR30.md). Components are not rounded to 2 decimals as the official printout.
    The primary value is the cohort PQ+ of the episode's images; the mean of the per-image PQ+ values (the data
    team's "scorer reset for each image") is reported as ``mean_image_pq_plus``.
Reference baseline
    Excess-green vegetation index 2g-r-b (chromatic coordinates) thresholded by Otsu per image, 8-connected
    components (>= 16 px at 512^2) as plant instances and as leaf instances, all vegetation labelled crop.
    Acceptance: PQ+ > reference + ``ACCEPT_MARGIN`` points.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ..registry import DATA_ROOT
from ..task import ConstraintSpec, Episode, EvalResult, ToolSpec
from ._delivery_roles import Delivery, DeliveryError, file_sha256 as verified_sha256, resolve_delivery
from ._life_health_common import (PROTOCOL_SEED, Memo, compose_episodes, default_budget, default_cache_dir,
                                  effective_items, extract_payloads, ids_hash, make_result, rng_for, tmp_path_for,
                                  unit_constraint)

CODE = "FoR30"
DATASET_DIR = "for30-phenobench"
FOLDERS = ("images", "semantics", "plant_instances", "leaf_instances", "plant_visibility", "leaf_visibility")
LABEL_KEYS = FOLDERS[1:]
ROLE_OF_SPLIT = {"src": "source", "val": "val", "id": "id", "ood": "ood"}
NATIVE_SIZE = 1024
ACCEPT_MARGIN = 2.0                          # PQ+ points above the ExG reference
MIN_AREA_NATIVE = 64                         # reference: minimum component area at 1024^2 (scaled by 1/scale^2)
CACHE_VERSION = 2                            # bump when the derived arrays change (v1 caches carry no fingerprint)


# ----------------------------------------------------------------------------------------------------------------
# official metric (numpy port of phenobench.evaluation)
# ----------------------------------------------------------------------------------------------------------------
def convert_partial_semantics(sem: np.ndarray) -> np.ndarray:
    out = sem.copy()
    out[sem == 3] = 1
    out[sem == 4] = 2
    return out


def semantic_confusion(pred_sem: np.ndarray, gt_sem: np.ndarray) -> np.ndarray:
    """3x3 confusion matrix (rows = GT, cols = prediction) after partial-class conversion."""
    p = convert_partial_semantics(np.asarray(pred_sem).astype(np.int64)).ravel()
    g = convert_partial_semantics(np.asarray(gt_sem).astype(np.int64)).ravel()
    ok = (p >= 0) & (p < 3) & (g >= 0) & (g < 3)
    return np.bincount(g[ok] * 3 + p[ok], minlength=9).reshape(3, 3).astype(np.int64)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU in [0, 1]; zero denominator -> 0 (torchmetrics ``_safe_divide``)."""
    cm = np.asarray(cm, dtype=float)
    num = np.diag(cm)
    den = cm.sum(0) + cm.sum(1) - num
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _pair_stats(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Unique positive ids of pred/gt, their areas and pairwise intersections (flattened int arrays)."""
    pl, pa = np.unique(pred[pred > 0], return_counts=True)
    gl, ga = np.unique(gt[gt > 0], return_counts=True)
    both = (pred > 0) & (gt > 0)
    inter: dict[tuple[int, int], int] = {}
    if both.any():
        pairs, cnt = np.unique(np.stack([pred[both], gt[both]], 1), axis=0, return_counts=True)
        inter = {(int(a), int(b)): int(c) for (a, b), c in zip(pairs, cnt)}
    return pl, gl, dict(zip(pl.tolist(), pa.tolist())), {"gt_area": dict(zip(gl.tolist(), ga.tolist())), "inter": inter}


def pq_single_class(pred: np.ndarray, gt: np.ndarray) -> float:
    """``PanopticQuality.compute_pq_single_class``: greedy IoU > 0.5 matching in ascending predicted-id order."""
    pl, gl, parea, st = _pair_stats(pred, gt)
    garea, inter = st["gt_area"], st["inter"]
    by_pred: dict[int, list[tuple[int, int]]] = {}
    for (a, b), c in inter.items():
        by_pred.setdefault(a, []).append((b, c))
    gl_list = gl.tolist()
    gidx = {g: i for i, g in enumerate(gl_list)}
    matched = [False] * len(gl_list)
    tp, fp, num = 0, 0, 0.0
    for p in pl.tolist():
        best, best_i = 0.0, -1
        for g, c in sorted(by_pred.get(p, []), key=lambda t: gidx[t[0]]):
            i = gidx[g]
            if matched[i]:
                continue
            iou = c / (parea[p] + garea[g] - c)
            if iou > best:
                best, best_i = iou, i
        if best > 0.5:
            tp += 1
            num += best
            matched[best_i] = True
        else:
            fp += 1
    fn = len(gl_list) - tp
    den = tp + 0.5 * fp + 0.5 * fn
    return float(num / den) if den > 0 else float("nan")


def filter_partial_masks(pred_inst: np.ndarray, pred_sem: np.ndarray, gt_inst: np.ndarray, gt_sem: np.ndarray,
                         gt_vis: np.ndarray) -> None:
    """In-place port of the official ``filter_partial_masks`` (visibility in [0, 1])."""
    ids = np.unique(gt_inst[gt_inst > 0])
    partial = []
    for g in ids.tolist():
        m = gt_inst == g
        if float(gt_vis[m].max()) > 0.5:      # the official code reads the (single) visibility of the instance
            continue
        gt_sem[m] = 0
        partial.append(g)
    if not partial:
        return
    in_partial = np.isin(gt_inst, partial) & (pred_inst > 0)
    if not in_partial.any():
        return
    pl, pa = np.unique(pred_inst[pred_inst > 0], return_counts=True)
    area = dict(zip(pl.tolist(), pa.tolist()))
    pairs, cnt = np.unique(np.stack([pred_inst[in_partial], gt_inst[in_partial]], 1), axis=0, return_counts=True)
    drop = {int(a) for (a, _), c in zip(pairs, cnt) if c / (area[int(a)] + 1e-12) > 0.5}
    if drop:
        pred_sem[np.isin(pred_inst, list(drop))] = 0


def panoptic_image(pred_sem: np.ndarray, gt_sem: np.ndarray, pred_inst: np.ndarray, gt_inst: np.ndarray
                   ) -> dict[int, float]:
    """``PanopticQuality.compute_pq`` for one image: {class: PQ} over GT classes present (empty if none)."""
    labels = np.unique(gt_sem)
    labels = labels[labels > 0]
    out: dict[int, float] = {}
    for c in labels.tolist():
        pi = pred_inst * (pred_sem == c)
        gi = gt_inst * (gt_sem == c)
        out[int(c)] = pq_single_class(pi, gi)
    return out


def hierarchical_image_stats(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> dict:
    """All sufficient statistics of one image for PQ+ (confusion, per-class plant PQ, leaf PQ)."""
    cm = semantic_confusion(pred["semantics"], gt["semantics"])
    # plants: raw semantics (no partial conversion), plant visibility filter
    p_sem = np.asarray(pred["semantics"]).astype(np.int64).copy()
    g_sem = np.asarray(gt["semantics"]).astype(np.int64).copy()
    p_ins = np.asarray(pred["plant_instances"]).astype(np.int64)
    g_ins = np.asarray(gt["plant_instances"]).astype(np.int64)
    filter_partial_masks(p_ins, p_sem, g_ins, g_sem, np.asarray(gt["plant_visibility"], dtype=float))
    plant = panoptic_image(p_sem, g_sem, p_ins, g_ins)
    # leaves: semantics derived from instance maps, leaf visibility filter
    pl_ins = np.asarray(pred["leaf_instances"]).astype(np.int64)
    gl_ins = np.asarray(gt["leaf_instances"]).astype(np.int64)
    pl_sem = (pl_ins > 0).astype(np.int64)
    gl_sem = (gl_ins > 0).astype(np.int64)
    filter_partial_masks(pl_ins, pl_sem, gl_ins, gl_sem, np.asarray(gt["leaf_visibility"], dtype=float))
    leaf = panoptic_image(pl_sem, gl_sem, pl_ins, gl_ins)
    return {"confusion": cm, "plant_pq": plant, "leaf_pq": leaf.get(1)}


def aggregate(stats: list[dict]) -> dict:
    """Accumulate image statistics into the official hierarchical metrics (percent)."""
    cm = np.zeros((3, 3), dtype=np.int64)
    sums: dict[int, float] = {}
    cnts: dict[int, int] = {}
    leaf_sum, leaf_cnt = 0.0, 0
    for s in stats:
        cm += np.asarray(s["confusion"], dtype=np.int64)
        for c, v in s["plant_pq"].items():
            if np.isfinite(v):
                sums[int(c)] = sums.get(int(c), 0.0) + v
                cnts[int(c)] = cnts.get(int(c), 0) + 1
        if s["leaf_pq"] is not None and np.isfinite(s["leaf_pq"]):
            leaf_sum += s["leaf_pq"]
            leaf_cnt += 1
    return payload_metrics({"confusion": cm.tolist(), "plant_pq_sum": {str(k): v for k, v in sums.items()},
                            "plant_pq_count": {str(k): v for k, v in cnts.items()}, "leaf_pq_sum": leaf_sum,
                            "leaf_pq_count": leaf_cnt})


def payload_metrics(p: dict) -> dict:
    iou = iou_from_confusion(np.asarray(p["confusion"])) * 100.0
    s, c = p["plant_pq_sum"], p["plant_pq_count"]
    pq_crop = 100.0 * s["1"] / c["1"] if c.get("1") else None
    pq_weed = 100.0 * s["2"] / c["2"] if c.get("2") else None
    pq_leaf = 100.0 * p["leaf_pq_sum"] / p["leaf_pq_count"] if p["leaf_pq_count"] else None
    comps = [float(iou[0]), float(iou[2]), pq_crop, pq_leaf]
    avail = [v for v in comps if v is not None]
    return {"pq_plus": float(np.mean(avail)) if avail else None, "iou_soil": float(iou[0]), "iou_crop": float(iou[1]),
            "iou_weed": float(iou[2]), "pq_crop": pq_crop, "pq_weed_plants": pq_weed, "pq_leaf": pq_leaf,
            "pq": (pq_crop + pq_leaf) / 2 if pq_crop is not None and pq_leaf is not None else None,
            "n_components": len(avail), "summary": p}


# ----------------------------------------------------------------------------------------------------------------
# reference baseline
# ----------------------------------------------------------------------------------------------------------------
def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(x.ravel(), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])


def exg_reference(image: np.ndarray, min_area: int) -> dict[str, np.ndarray]:
    from scipy import ndimage

    rgb = image.astype(np.float64)
    s = rgb.sum(-1) + 1e-6
    r, g, b = rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s
    exg = 2 * g - r - b
    veg = exg > otsu_threshold(exg)
    lab, n = ndimage.label(veg, structure=np.ones((3, 3), dtype=int))
    if n:
        areas = np.bincount(lab.ravel())
        small = areas < min_area
        small[0] = False
        lab[small[lab]] = 0
        keep = np.unique(lab[lab > 0])
        remap = np.zeros(int(lab.max()) + 1, dtype=np.int32)
        remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
        lab = remap[lab]
    sem = (lab > 0).astype(np.int32)
    return {"semantics": sem, "plant_instances": lab.astype(np.int32), "leaf_instances": lab.astype(np.int32)}



# ----------------------------------------------------------------------------------------------------------------
# delivery: role files -> image records
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rec:
    """One image of a role file (``unit_id`` = ``<official split>/<file name>``)."""
    id: str
    name: str
    role: str                    # source | val | id | ood
    official_split: str          # train | val
    date: str                    # acquisition date prefix, e.g. 05-15
    group: str                   # capture id (P-number): images of one capture are spatially correlated
    crops: tuple[int, ...]       # persistent crop-lineage ids present in the image
    files: dict[str, Path]       # FOLDERS -> PNG
    sha256: dict[str, str]
    fingerprint: str             # hash of the six expected sha256 values (derived-cache key)


def _record(dl: Delivery, role: str, row: dict) -> Rec:
    unit = row.get("unit_id", "?")

    def bad(msg: str) -> DeliveryError:
        return DeliveryError(f"{CODE} {role}.jsonl {unit}: {msg}")

    try:
        official, _, name = str(row["unit_id"]).partition("/")
        date, _, cap = Path(name).stem.split("_")
        if row["role"] != role:
            raise bad(f"row role {row['role']!r}")
        if official != row["official_split"] or official not in ("train", "val"):
            raise bad(f"official split {row['official_split']!r} does not match the unit id")
        if cap != row["lineage_group"] or date != row["acquisition_date_prefix"]:
            raise bad("capture id / date differ from the file name")
        if [int(v) for v in row["image_size"]] != [NATIVE_SIZE, NATIVE_SIZE]:
            raise bad(f"image size {row['image_size']}")
        if list(row["input_to_agent"]) != ["images"] or not set(LABEL_KEYS) <= set(row["evaluator_only"]):
            raise bad("agent-view policy differs (input_to_agent must be images only, all labels evaluator_only)")
        if set(row["files"]) != set(FOLDERS):
            raise bad(f"files {sorted(row['files'])}")
        files = {k: dl.anchor(row["files"][k]["path"]) for k in FOLDERS}
        sha = {k: str(row["files"][k]["sha256"]).lower() for k in FOLDERS}
        crops = tuple(int(c) for c in row["crop_lineage_ids"])
    except DeliveryError:
        raise
    except (KeyError, ValueError, TypeError) as ex:
        raise bad(f"malformed row ({type(ex).__name__}: {ex})") from ex
    for k in FOLDERS:
        if files[k].name != name:
            raise bad(f"{k} file {files[k].name} is not {name}")
        if len(sha[k]) != 64:
            raise bad(f"{k}: sha256 {sha[k]!r}")
    fp = hashlib.sha256((f"v{CACHE_VERSION}|" + "|".join(f"{k}={sha[k]}" for k in FOLDERS)).encode()).hexdigest()
    return Rec(str(row["unit_id"]), name, role, official, date, cap, crops, files, sha, fp)


def check_roles(roles: dict[str, list[Rec]]) -> None:
    """Pools are non-empty and item-, capture- and crop-lineage-disjoint; OOD dates differ from the IID dates."""
    owner: dict[str, dict[Any, str]] = {"image": {}, "capture id": {}, "crop lineage id": {}}
    for split, recs in roles.items():
        if not recs:
            raise DeliveryError(f"{CODE}: role {ROLE_OF_SPLIT[split]} has no images")
        ids = [r.id for r in recs]
        if len(set(ids)) != len(ids):
            raise DeliveryError(f"{CODE}: role {ROLE_OF_SPLIT[split]} lists an image twice")
        for r in recs:
            for what, keys in (("image", [r.id]), ("capture id", [r.group]), ("crop lineage id", list(r.crops))):
                for key in keys:
                    if owner[what].setdefault(key, split) != split:
                        raise DeliveryError(f"{CODE}: {what} {key} occurs in both {owner[what][key]} and {split}")
    iid = {r.date for s in ("src", "val", "id") for r in roles[s]}
    if iid & {r.date for r in roles["ood"]}:
        raise DeliveryError(f"{CODE}: OOD capture dates overlap the IID dates")


def load_roles(dl: Delivery) -> dict[str, list[Rec]]:
    roles = {s: [_record(dl, role, row) for row in dl.rows(role)] for s, role in ROLE_OF_SPLIT.items()}
    check_roles(roles)
    return roles


# ----------------------------------------------------------------------------------------------------------------
# adapter
# ----------------------------------------------------------------------------------------------------------------
class Adapter:
    discipline = CODE
    name = "PhenoBench hierarchical panoptic segmentation"
    family = "Life & health"
    metric = "PQ+"
    direction = "max"
    task_type = "plant_phenotyping_panoptic_segmentation"

    def __init__(self, data_root: str | None = None, cache_dir: str | None = None, pool_seed: int = PROTOCOL_SEED,
                 scale: int = 2, expose_source_labels: bool = True, **_: Any) -> None:
        self.data_root = Path(data_root or os.environ.get("SCIENCECLAW_DATA_ROOT", DATA_ROOT))
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(CODE)
        self.pool_seed = int(pool_seed)                # episode composition only; the pools are the role files
        if scale not in (1, 2, 4):
            raise ValueError("scale must be 1, 2 or 4")
        self.scale = int(scale)
        self.size = NATIVE_SIZE // self.scale
        self.expose_source_labels = bool(expose_source_labels)
        self._memo = Memo()

    # ---------------------------------------------------------------- data
    def delivery(self) -> Delivery:
        return self._memo.get("delivery", lambda: resolve_delivery(CODE, self.data_root))

    def _roles(self) -> dict[str, list[Rec]]:
        return self._memo.get("roles", lambda: load_roles(self.delivery()))

    def _index(self) -> dict[str, Rec]:
        return self._memo.get("index", lambda: {r.id: r for rs in self._roles().values() for r in rs})

    def _record(self, item: str) -> Rec:
        return self._index()[item]

    def _pools(self) -> dict[str, list[str]]:
        """Image ids per split (src, val, id, ood) = the role files, sorted."""
        return {s: sorted(r.id for r in rs) for s, rs in self._roles().items()}

    def _source_groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self._roles()["src"]:
            out.setdefault(r.group, []).append(r.id)
        return {g: sorted(v) for g, v in sorted(out.items())}

    def available(self) -> tuple[bool, str]:
        try:
            import PIL  # noqa: F401
            import scipy  # noqa: F401
        except ImportError as ex:
            return False, f"{CODE}: missing python package ({ex})"
        try:
            dl = self.delivery()
            miss = dl.missing()
            if miss:
                return False, f"{CODE}: {miss[0]}" + (f" (+{len(miss) - 1} more)" if len(miss) > 1 else "")
            roles = self._roles()
        except (OSError, ValueError, KeyError) as ex:        # DeliveryError is a ValueError
            return False, f"{CODE}: {ex}"
        files = [p for rs in roles.values() for r in rs for p in r.files.values()]
        absent = [p for p in files if not p.is_file()]
        if absent:
            return False, (f"{CODE} {dl.version}: download incomplete, {len(files) - len(absent)}/{len(files)} PNG "
                           f"files present (first missing: {absent[0]})")
        return True, (f"{CODE} {dl.version}: images " + ", ".join(f"{s}={len(v)}" for s, v in roles.items())
                      + "; captures " + "/".join(str(len({r.group for r in v})) for v in roles.values()))

    def verify_disjoint(self) -> None:
        """Images, capture ids and persistent crop ids are disjoint across src/val/id/ood (raises DeliveryError)."""
        check_roles(self._roles())

    def verify_files(self, splits: tuple[str, ...] | None = None) -> int:
        """sha256 of every PNG of the given splits against the role rows (returns the number of files checked)."""
        n = 0
        for s in splits or tuple(ROLE_OF_SPLIT):
            for r in self._roles()[s]:
                self._check_hashes(r)
                n += len(FOLDERS)
        return n

    @staticmethod
    def _check_hashes(rec: Rec) -> None:
        for k in FOLDERS:
            if verified_sha256(rec.files[k]) != rec.sha256[k]:
                raise DeliveryError(f"{CODE}: {rec.files[k]} does not match the sha256 of its role row")

    def _load(self, item: str) -> dict[str, np.ndarray]:
        """Arrays of one image at the adapter scale (npz cache keyed by the row sha256 values; visibility uint8)."""
        from PIL import Image

        rec = self._record(item)
        cdir = self.cache_dir / f"scale{self.scale}"
        cfile = cdir / (item.replace("/", "__") + ".npz")
        if cfile.exists():
            try:
                with np.load(cfile) as z:
                    if "fingerprint" in z.files and str(z["fingerprint"]) == rec.fingerprint:
                        return {k: z[k] for k in FOLDERS}
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                pass                                         # unreadable cache entry: rebuild
        self._check_hashes(rec)
        s = self.scale
        out: dict[str, np.ndarray] = {}
        for k in FOLDERS:
            with Image.open(rec.files[k]) as im:
                a = np.asarray(im)
            if k == "images":
                a = a[..., :3].astype(np.float32)
                if s > 1:
                    h, w = a.shape[0] // s, a.shape[1] // s
                    a = a[: h * s, : w * s].reshape(h, s, w, s, 3).mean(axis=(1, 3))
                a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
            else:
                a = a[::s, ::s]
                a = a.astype(np.uint8) if k in ("semantics", "plant_visibility", "leaf_visibility") else a.astype(np.int32)
            out[k] = np.ascontiguousarray(a)
        cdir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_path_for(cfile, ".npz")
        np.savez_compressed(tmp, fingerprint=np.array(rec.fingerprint), **out)
        os.replace(tmp, cfile)
        return out

    def _gt(self, item: str) -> dict[str, np.ndarray]:
        a = self._load(item)
        return {"semantics": a["semantics"], "plant_instances": a["plant_instances"],
                "leaf_instances": a["leaf_instances"], "plant_visibility": a["plant_visibility"] / 255.0,
                "leaf_visibility": a["leaf_visibility"] / 255.0}

    def reference_prediction(self, items: list[str]) -> dict[str, np.ndarray]:
        min_area = max(1, MIN_AREA_NATIVE // (self.scale ** 2))
        preds = [exg_reference(self._load(i)["images"], min_area) for i in items]
        return {k: np.stack([p[k] for p in preds]) for k in ("semantics", "plant_instances", "leaf_instances")}

    @staticmethod
    def handle(item: str) -> str:
        """Opaque image handle shown to the agent (the data team's contract: opaque image id + RGB input)."""
        return "ph-" + hashlib.sha256(item.encode()).hexdigest()[:8]

    # ---------------------------------------------------------------- scoring
    def _coerce(self, y: Any, n: int) -> tuple[dict[str, np.ndarray] | None, str]:
        if not isinstance(y, dict):
            return None, "y must be a dict with keys semantics, plant_instances, leaf_instances"
        out = {}
        for k in ("semantics", "plant_instances", "leaf_instances"):
            if k not in y:
                return None, f"missing key {k!r}"
            try:
                a = np.asarray(y[k])
            except (TypeError, ValueError):
                return None, f"{k}: not an array"
            if a.shape != (n, self.size, self.size):
                return None, f"{k}: shape {a.shape}, expected {(n, self.size, self.size)}"
            if a.dtype.kind not in "iub":
                if a.dtype.kind != "f" or not np.all(np.isfinite(a)) or not np.all(a == np.rint(a)):
                    return None, f"{k}: values must be finite integers"
            a = a.astype(np.int64)
            if k == "semantics" and not np.all(np.isin(a, (0, 1, 2, 3, 4))):
                return None, "semantics: labels must be in {0,1,2} (3/4 = partial crop/weed also accepted)"
            if k != "semantics" and (a.min() < 0 or a.max() >= 2 ** 31):
                return None, f"{k}: instance ids must be in [0, 2^31)"
            out[k] = a
        return out, "ok"

    def _stats(self, items: list[str], pred: dict[str, np.ndarray]) -> list[dict]:
        return [hierarchical_image_stats({k: v[j] for k, v in pred.items()}, self._gt(i)) for j, i in enumerate(items)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """PQ+ over all images of the given episodes (sums of confusion matrices and per-image PQ values)."""
        pays = extract_payloads(per_episode, "phenobench_pq")
        if not pays:
            return None
        cm = np.zeros((3, 3), dtype=np.int64)
        s: dict[str, float] = {}
        c: dict[str, int] = {}
        ls, lc = 0.0, 0
        for p in pays:
            cm += np.asarray(p["confusion"], dtype=np.int64)
            for kk, v in p["plant_pq_sum"].items():
                s[kk] = s.get(kk, 0.0) + v
            for kk, v in p["plant_pq_count"].items():
                c[kk] = c.get(kk, 0) + v
            ls += p["leaf_pq_sum"]
            lc += p["leaf_pq_count"]
        return payload_metrics({"confusion": cm.tolist(), "plant_pq_sum": s, "plant_pq_count": c,
                                "leaf_pq_sum": ls, "leaf_pq_count": lc})["pq_plus"]

    # ---------------------------------------------------------------- episodes
    def _compose_src(self, n: int, k: int, seed: int) -> tuple[list[list[str]], bool]:
        """src / rep episodes: the images of ONE source capture each (captures cycled in a seeded order).

        Keeping an episode inside a single capture leaves the other captures free as visible labelled data
        (``_visible``). Prefix-stable in ``n``; images repeat across episodes (the source role has 32 images).
        """
        groups = self._source_groups()
        names = list(groups)
        order = [names[i] for i in rng_for(CODE, self.pool_seed, "src", seed, k, "capture-order").permutation(len(names))]
        eps = []
        for j in range(n):
            ids = groups[order[j % len(order)]]
            perm = rng_for(CODE, self.pool_seed, "src", seed, k, "items", j).permutation(len(ids))
            eps.append([ids[i] for i in perm][:k])
        flat = [i for e in eps for i in e]
        return eps, len(flat) != len(set(flat))

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        if split not in ("src", "val", "id", "ood", "rep"):
            raise ValueError(f"unknown split {split!r}")
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"{CODE} unavailable: {why}")
        ipe = int(items_per_episode)
        if ipe < 1:
            raise ValueError("items_per_episode must be >= 1")
        base = "src" if split == "rep" else split
        if base == "src":
            groups, reused = self._compose_src(n, ipe, seed)
        else:
            pool = self._pools()[base]
            k = effective_items(split, len(pool), n, ipe)
            groups, reused = compose_episodes(pool, n, k, [CODE, self.pool_seed, base, seed, ipe])
        eps = [self._episode(split, base, seed, j, items, reused) for j, items in enumerate(groups)]
        for e in eps:
            e.lineage["items_requested"] = ipe
        return eps

    def _visible(self, items: list[str]) -> tuple[list[str], list[str]]:
        """(train, dev) source-role image ids visible to an episode with evaluation ``items``.

        Whole source captures that hold none of the episode's images: the smallest is the dev capture (labels
        withheld, scored by ``score_dev``), the others are the labelled training images. Empty without
        ``expose_source_labels``.
        """
        if not self.expose_source_labels:
            return [], []
        groups = self._source_groups()
        used = {self._record(i).group for i in items}
        free = sorted((g for g in groups if g not in used), key=lambda g: (len(groups[g]), g))
        if not free:
            return [], []
        if len(free) == 1:
            return list(groups[free[0]]), []
        return [i for g in free[1:] for i in groups[g]], list(groups[free[0]])

    def _stack(self, items: list[str], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
        arrs = [self._load(i) for i in items]
        out = {}
        for k in keys:
            a = np.stack([x[k] for x in arrs])
            out[k] = a.astype(np.float32) / 255.0 if k.endswith("visibility") else a
        return out

    def _episode(self, split: str, base: str, seed: int, j: int, items: list[str], reused: bool) -> Episode:
        k, H = len(items), self.size
        train, dev = self._visible(items)
        cap_items = {self._record(i).group for i in items}
        if set(train + dev) & set(items) or {self._record(i).group for i in train + dev} & cap_items:
            raise RuntimeError(f"{CODE}: visible labelled data overlap the evaluation items (lineage leak)")
        names = [self.handle(i) for i in items]
        dates = sorted({self._record(i).date for i in items})

        def t_load_train(inputs: dict, config: dict) -> dict:
            d = self._stack(train, FOLDERS)
            d["names"] = [self.handle(i) for i in train]
            return d

        def t_load_eval(inputs: dict, config: dict) -> dict:
            return {"images": self._stack(items, ("images",))["images"], "names": list(names)}

        def t_predict_pretrained(inputs: dict, config: dict) -> dict:
            """Run the frozen PhenoBench Mask2Former pair on this episode's RGB images.

            This is deliberately a trusted, label-free convenience tool: it reads only the evaluation RGB
            arrays and calls the pinned public PRBonn checkpoints through the local/remote scilib bridge.  It
            does not fit on this episode, inspect annotations, or change the evaluator's reference or margin.
            """
            from scilib import phenoseg_m2f
            x = self._stack(items, ("images",))["images"]
            return {"y": phenoseg_m2f.predict_panoptic(x, plant_threshold=0.8, leaf_threshold=0.8, batch_size=4)}

        def t_predict_weyler(inputs: dict, config: dict) -> dict:
            """Run the frozen official PRBonn Weyler hierarchical model on RGB evaluation images."""
            from scilib import phenobench_weyler
            x = self._stack(items, ("images",))["images"]
            return {"y": phenobench_weyler.predict_panoptic(x, batch_size=1)}

        def t_predict_hapt(inputs: dict, config: dict) -> dict:
            """Run the frozen public HAPT hierarchical model on RGB evaluation images.

            HAPT was released for GrowliFlower rather than the PhenoBench training
            partition.  It is kept only as a separately auditable cross-domain
            diagnostic; it reads only evaluation RGB arrays here and must not be used
            as the final PhenoBench submission route.
            """
            from scilib import phenoseg_hapt
            x = self._stack(items, ("images",))["images"]
            return {"y": phenoseg_hapt.predict_panoptic(x)}

        def t_load_dev(inputs: dict, config: dict) -> dict:
            return {"images": self._stack(dev, ("images",))["images"], "names": [self.handle(i) for i in dev]}

        def t_score_dev(inputs: dict, config: dict) -> dict:
            pred, msg = self._coerce(inputs.get("prediction"), len(dev))
            if pred is None:
                raise ValueError(f"score_dev: {msg}")
            m = aggregate(self._stats(dev, pred))
            return {kk: (float(m[kk]) if m[kk] is not None else float("nan"))
                    for kk in ("pq_plus", "iou_soil", "iou_weed", "pq_crop", "pq_leaf")}

        def t_score_pretrained_dev(inputs: dict, config: dict) -> dict:
            """Trusted-side diagnostic for the frozen checkpoint on the visible dev images.

            Only aggregate official metrics leave the adapter.  The dev annotations are never returned, and the
            result does not participate in the held-out evaluator or acceptance decision.
            """
            from scilib import phenoseg_m2f
            x = self._stack(dev, ("images",))["images"]
            pred = phenoseg_m2f.predict_panoptic(x, plant_threshold=0.8, leaf_threshold=0.8, batch_size=4)
            m = aggregate(self._stats(dev, pred))
            return {kk: (float(m[kk]) if m[kk] is not None else float("nan"))
                    for kk in ("pq_plus", "iou_soil", "iou_weed", "pq_crop", "pq_leaf")}

        nt, nd = len(train), len(dev)
        lab = "int array (n, H, W)"
        tools = []
        if nt:
            tools.append(ToolSpec(
                "load_train", "Visible labelled training images (from other UAV captures than the evaluation "
                              "images) with all official annotations.", {},
                {"images": PortSchema("array", (nt, H, H, 3), None, "uint8", "RGB images"),
                 "semantics": PortSchema("array", (nt, H, H), "1", "int",
                                         "0 soil, 1 crop, 2 weed, 3 partial crop, 4 partial weed"),
                 "plant_instances": PortSchema("array", (nt, H, H), "1", "int", "plant instance ids (0 = none)"),
                 "leaf_instances": PortSchema("array", (nt, H, H), "1", "int", "crop-leaf instance ids (0 = none)"),
                 "plant_visibility": PortSchema("array", (nt, H, H), "1", "float",
                                                "visible fraction of the plant instance covering each pixel, 0..1"),
                 "leaf_visibility": PortSchema("array", (nt, H, H), "1", "float",
                                               "visible fraction of the leaf instance covering each pixel, 0..1"),
                 "names": PortSchema("list", (nt,), None, "str", "opaque image handles")},
                t_load_train))
        tools.append(ToolSpec(
            "load_eval_inputs", "RGB evaluation images of this episode (no annotations), in item order.", {},
            {"images": PortSchema("array", (k, H, H, 3), None, "uint8", "RGB images"),
             "names": PortSchema("list", (k,), None, "str", "opaque image handles in item order")}, t_load_eval))
        # On a GPU host the frozen PRBonn pair is exposed as a one-step, label-free route.  Keeping this
        # separate from ``load_eval_inputs`` makes use of the pretrained component auditable in the graph while
        # avoiding a prompt-only dependency on the agent remembering a long code-node recipe.  The tool is
        # omitted when the checkpoints/remote worker are unavailable, preserving the CPU-only task surface.
        # Import lazily: the base ``scilib`` package intentionally does not import optional model modules, and
        # CPU-only workers must still be able to construct the adapter without torch/transformers.
        try:
            from scilib import phenoseg_m2f
            pretrained_available = bool(phenoseg_m2f.available())
        except (ImportError, RuntimeError, OSError):
            pretrained_available = False
        if pretrained_available:
            tools.append(ToolSpec(
                "predict_pretrained",
                "Frozen PRBonn PhenoBench Mask2Former plants+leaves prediction on this episode's RGB evaluation images; "
                "uses no annotations and performs no fitting. The output is ready for submit.",
                {},
                {"y": PortSchema("dict", None, "1", "int",
                                  f"keys semantics, plant_instances, leaf_instances; integer arrays ({k},{H},{H})")},
                t_predict_pretrained))
        try:
            from scilib import phenobench_weyler
            weyler_available = bool(phenobench_weyler.available())
        except (ImportError, RuntimeError, OSError):
            weyler_available = False
        if weyler_available:
            tools.append(ToolSpec(
                "predict_weyler",
                "Frozen official PRBonn PhenoBench Weyler hierarchical instance model; RGB-only, no fitting or annotations.",
                {},
                {"y": PortSchema("dict", None, "1", "int",
                                  f"keys semantics, plant_instances, leaf_instances; integer arrays ({k},{H},{H})")},
                t_predict_weyler))
        try:
            from scilib import phenoseg_hapt
            hapt_available = bool(phenoseg_hapt.available())
        except (ImportError, RuntimeError, OSError):
            hapt_available = False
        if hapt_available:
            tools.append(ToolSpec(
                "predict_hapt",
                "Diagnostic-only public HAPT hierarchical prediction on RGB evaluation images; "
                "the checkpoint is a GrowliFlower release, not a PhenoBench-trained specialist. "
                "It uses no annotations and performs no fitting, but its output must not be submitted as the "
                "PhenoBench result.",
                {},
                {"y": PortSchema("dict", None, "1", "int",
                                  f"keys semantics, plant_instances, leaf_instances; integer arrays ({k},{H},{H})")},
                t_predict_hapt))
        if nd:
            tools.append(ToolSpec(
                "load_dev_inputs", "RGB images of the visible dev slice (annotations withheld; see score_dev).", {},
                {"images": PortSchema("array", (nd, H, H, 3), None, "uint8", "RGB images"),
                 "names": PortSchema("list", (nd,), None, "str", "opaque image handles of the dev images")},
                t_load_dev))
            tools.append(ToolSpec(
                "score_dev", "Official hierarchical metrics (PQ+, IoU soil/weed, PQ crop/leaf, percent) of a "
                             "prediction dict for the dev images (order of load_dev_inputs).",
                {"prediction": PortSchema("dict", None, "1", "int",
                                          f"keys semantics, plant_instances, leaf_instances; each {lab} for the dev images")},
                {"pq_plus": PortSchema("number", None, "%", "float", ""),
                 "iou_soil": PortSchema("number", None, "%", "float", ""),
                 "iou_weed": PortSchema("number", None, "%", "float", ""),
                 "pq_crop": PortSchema("number", None, "%", "float", ""),
                 "pq_leaf": PortSchema("number", None, "%", "float", "")},
                t_score_dev))
            if pretrained_available:
                tools.append(ToolSpec(
                    "score_pretrained_dev",
                    "Official aggregate metrics of the frozen PRBonn Mask2Former pair on the visible dev images; "
                    "annotations stay inside the trusted scorer and the result is diagnostic only.",
                    {},
                    {"pq_plus": PortSchema("number", None, "%", "float", ""),
                     "iou_soil": PortSchema("number", None, "%", "float", ""),
                     "iou_weed": PortSchema("number", None, "%", "float", ""),
                     "pq_crop": PortSchema("number", None, "%", "float", ""),
                     "pq_leaf": PortSchema("number", None, "%", "float", "")},
                    t_score_pretrained_dev))
        visible = ""
        if nt:
            visible += (f"Visible labelled training images with official annotations: {nt} images (tool load_train; "
                        "semantics 0 soil, 1 crop, 2 weed, 3 partial crop, 4 partial weed; plant instance ids; "
                        "crop-leaf instance ids; visibility = fraction of the instance inside the image), taken from "
                        "other UAV captures than the evaluation images. ")
        if nd:
            visible += (f"{nd} further RGB images (tool load_dev_inputs) can be scored with tool score_dev "
                        "(annotations withheld). ")
        objective = (
            "Hierarchical panoptic segmentation of sugar-beet field images (PhenoBench v1.1.0, UAV RGB images, "
            f"official task 'hierarchical'). Images are {H}x{H} pixels"
            + (f" (official {NATIVE_SIZE}x{NATIVE_SIZE} downsampled by {self.scale})" if self.scale > 1 else "") + ".\n"
            f"Evaluation items: {k} RGB images (tool load_eval_inputs, item order = names). {visible}\n"
            "Required output y: a dict with three integer arrays of shape "
            f"({k}, {H}, {H}) in item order: 'semantics' (0 soil, 1 crop, 2 weed; 3/4 are read as 1/2), "
            "'plant_instances' (one positive id per plant, 0 = no plant) and 'leaf_instances' (one positive id per "
            "crop leaf, 0 = no leaf).\n"
            "When the explicit predict_pretrained tool is available in a tool-on run, call it on the evaluation images "
            "before considering a CPU approximation; its `y` output is the intended pretrained route and must be "
            "submitted directly, using no annotations.\n"
            "An explicit predict_weyler tool, when available, is an official frozen hierarchical route. It reads only "
            "RGB evaluation images and may be used directly or compared as a separate diagnostic. "
            "An explicit predict_hapt tool, when available, is a diagnostic-only cross-domain route from the "
            "GrowliFlower release, not a PhenoBench specialist. It reads only RGB evaluation images and may be "
            "compared as an independent diagnostic; do not submit its output or combine it with another model.\n"
            "Score: PQ+ = (IoU_soil + IoU_weed + PQ_crop + PQ_leaf)/4 in percent, the official PhenoBench hierarchical "
            "metric: IoU from the pixel confusion matrix over all images; PQ = sum of matched IoU / (TP + FP/2 + FN/2) "
            "with a match when instance IoU > 0.5, averaged over images; ground-truth plants/leaves with visibility "
            "<= 0.5 and predictions lying mostly inside them are ignored.\n"
            + scilib.describe("phenoseg") + scilib.describe_extra("phenoseg_deep") + scilib.describe_extra("phenoseg_sam")
            + scilib.describe_extra("phenoseg_m2f"))
        constraints = [
            ConstraintSpec("output_structure",
                           f"y is a dict with semantics/plant_instances/leaf_instances arrays of shape ({k},{H},{H})",
                           lambda y, tr: self._check_structure(y, k)),
            ConstraintSpec("label_values", "integer arrays; semantics in {0..4}; instance ids >= 0",
                           lambda y, tr: (lambda r: (r[0] is not None, r[1]))(self._coerce(y, k))),
            unit_constraint("1"),
        ]
        ref_memo = Memo()

        def evaluate(y: Any, trace: Any) -> EvalResult:
            ref = ref_memo.get("ref", lambda: aggregate(self._stats(items, self.reference_prediction(items))))
            pred, msg = self._coerce(y, k)
            valid = pred is not None
            if not valid:     # no-skill fallback: everything soil, no instances
                z = np.zeros((k, H, H), dtype=np.int64)
                pred = {"semantics": z, "plant_instances": z, "leaf_instances": z}
            stats = self._stats(items, pred)
            m = aggregate(stats)
            per_image = [v for v in (aggregate([s])["pq_plus"] for s in stats) if v is not None]
            payload = dict(m["summary"], kind="phenobench_pq", items=list(items), fallback=not valid)
            metrics = {kk: m[kk] for kk in ("pq_plus", "iou_soil", "iou_crop", "iou_weed", "pq_crop", "pq_weed_plants",
                                            "pq_leaf", "pq") if m[kk] is not None}
            if per_image:
                metrics["mean_image_pq_plus"] = float(np.mean(per_image))
            metrics["reference_pq_plus"] = ref["pq_plus"]
            return make_result(m["pq_plus"], ref["pq_plus"], "max", ACCEPT_MARGIN, metrics,
                               {"pooled_payload": payload, "parse": msg, "n_components": m["n_components"],
                                "reference_desc": "ExG + Otsu + connected components, all vegetation = crop"},
                               valid)

        dl = self.delivery()
        role = ROLE_OF_SPLIT[base]
        ood = base == "ood"
        lineage = {"dataset": "PhenoBench", "version": "1.1.0", "source_url": "https://www.phenobench.org/dataset.html",
                   "archive_url": "https://www.phenobench.org/data/PhenoBench-v110.zip",
                   "license": "CC BY-SA 4.0 (official devkit README FAQ; https://www.phenobench.org/dataset.html)",
                   "data_version": dl.version, "role": role, "role_file_sha256": dl.role_sha256[role],
                   "catalog_sha256": verified_sha256(dl.catalog_path),
                   "partition_design": "FoR30.v2 (crop-lineage and capture isolated retained subset)",
                   "item_kind": "UAV RGB image with hierarchical panoptic annotations", "item_ids": list(items),
                   "capture_dates": dates, "capture_groups": sorted(cap_items),
                   "item_pool": "ood" if ood else "iid",
                   "ood_kind": "proxy_within_dataset" if ood else None,
                   "ood_rule": ("acquisition-date shift 05-15 (source/val/ID) -> 06-05 with held-out crop lineages "
                                "(no shared image, capture id or persistent crop id with source/val/ID)") if ood else None,
                   "scale": self.scale, "pool_seed": self.pool_seed, "episode_seed": int(seed), "episode_index": j,
                   "reused_items": bool(reused), "source_labels_exposed": self.expose_source_labels,
                   "visible_train_ids_hash": ids_hash(train), "n_visible_train": nt,
                   "visible_train_groups": sorted({self._record(i).group for i in train}),
                   "dev_ids_hash": ids_hash(dev), "n_dev": nd, "dev_groups": sorted({self._record(i).group for i in dev}),
                   "rebuilt_split": True, "historical_ids_recovered": False,
                   "full_upstream_component_independence": False,
                   "newer_reconstructed_dirs_on_disk": list(dl.newer_dirs),
                   "evaluator": "numpy port of PRBonn/phenobench@0edc128ef7f67c8c6577554c7d1a2e382e2ea81f hierarchical"}
        return Episode(
            id=f"{CODE}-{split}-s{seed}-e{j:02d}", discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("dict", None, "1", "int",
                                       f"keys semantics, plant_instances, leaf_instances: int arrays ({k},{H},{H})"),
            tools=tools, constraints=constraints, budget=default_budget(k, max_node_s=900, max_wall_s=3600),
            lineage=lineage,
            acceptance=f"accepted iff PQ+ > reference (ExG baseline on the same images) + {ACCEPT_MARGIN} points",
            tolerance={"rtol": 0.0, "atol": 0.0}, tags=["agriculture", "plant-phenotyping", "segmentation", "panoptic",
                                                         "uav", "image"],
            metric=self.metric, direction=self.direction, n_items=k, _evaluate=evaluate, _dev_evaluate=None)

    def _check_structure(self, y: Any, n: int) -> tuple[bool, str]:
        if not isinstance(y, dict):
            return False, "y must be a dict"
        for kk in ("semantics", "plant_instances", "leaf_instances"):
            if kk not in y:
                return False, f"missing key {kk!r}"
            shp = np.shape(y[kk])
            if tuple(shp) != (n, self.size, self.size):
                return False, f"{kk}: shape {tuple(shp)}, expected {(n, self.size, self.size)}"
        return True, "ok"

    def describe_pools(self) -> dict:
        out = {}
        for s, recs in self._roles().items():
            caps: dict[str, int] = {}
            dates: dict[str, int] = {}
            for r in recs:
                caps[r.group] = caps.get(r.group, 0) + 1
                dates[r.date] = dates.get(r.date, 0) + 1
            out[s] = {"n": len(recs), "captures": dict(sorted(caps.items())), "dates": dict(sorted(dates.items())),
                      "official_splits": sorted({r.official_split for r in recs}),
                      "crop_ids": len({c for r in recs for c in r.crops})}
        return out


if __name__ == "__main__":      # pragma: no cover
    a = Adapter()
    print(a.available())
    print(json.dumps(a.describe_pools(), indent=1))
