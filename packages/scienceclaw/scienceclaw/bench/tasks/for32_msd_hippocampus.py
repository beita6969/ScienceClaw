"""FoR32 Biomedical and clinical sciences — Medical Segmentation Decathlon Task04 Hippocampus (DSC).

Item
    One labelled 3-D T1-weighted MRI crop around one hippocampus (``imagesTr/hippocampus_XXX.nii.gz`` with its
    ``labelsTr`` mask; 0 = background, 1 = anterior, 2 = posterior; 1 mm isotropic). The official test images
    (``imagesTs``) are unlabelled and never used.
Lineage groups
    Volumes ``2k-1`` and ``2k`` are the left/right crops of the same subject (verified on the data: consecutive
    odd/even pairs share intensity encoding and distribution, 74 % of odd-start pairs vs 10 % of even-start pairs
    are near-identical). All splits are therefore made on subject groups ``(id + 1) // 2``.
IID / OOD
    No second hippocampus dataset is available locally, so OOD is a *proxy within the dataset*
    (``lineage["ood_kind"] = "proxy_within_dataset"``): the volumes are stored with three different intensity
    encodings — float32 with ~1e3 range (172 labelled volumes, IID), uint8 0-255 (55) and float32 with ~1e5 range
    (33). The two non-standard encodings (different export / scaling pipelines) form the OOD pool (64 volumes by
    subject group). Visible training data are IID-encoded only.
Splits (fixed by ``pool_seed``; subject-disjoint)
    IID: id 64, val 32, dev 8 (labels withheld; used by ``score_dev``), visible training 28, src = remaining 40.
    OOD: 64 of the 88 non-standard-encoding volumes.
Metric (D_V)
    Dice similarity coefficient per case and foreground class, DSC = 2|P∩G| / (|P| + |G|) (1 if both empty), the
    primary MSD metric (Antonelli et al., Nat Commun 2022); primary = mean over cases of the mean over the two
    classes. Secondary: normalized surface Dice at 1 mm (voxel-surface approximation, reported only).
Reference baseline
    Location-only probabilistic atlas: training masks mapped to a common normalized 48x64x48 grid, class
    frequencies averaged, argmax mapped back to each evaluation volume's grid (ignores intensities).
    Acceptance: primary > reference + ``ACCEPT_MARGIN``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

import scilib

from ...core.schema import PortSchema
from ..registry import DATA_ROOT
from ..task import ConstraintSpec, Episode, EvalResult, ToolSpec
from ._life_health_common import (PROTOCOL_SEED, Memo, allocate_groups, as_list_of_arrays, compose_episodes,
                                  default_budget, default_cache_dir, effective_items, extract_payloads, ids_hash,
                                  make_result, tmp_path_for, unit_constraint)

CODE = "FoR32"
DATASET_DIR = "for32-msd-hippocampus"
CLASSES = (1, 2)
CLASS_NAMES = {1: "anterior", 2: "posterior"}
ATLAS_GRID = (48, 64, 48)
IID_TARGETS = [("id", 64), ("val", 32), ("dev", 8), ("train", 28)]     # src = remainder
OOD_TARGET = 64
ACCEPT_MARGIN = 0.03            # absolute DSC above the location-atlas reference
NSD_TOL_MM = 1.0


# ----------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    den = int(p.sum()) + int(g.sum())
    if den == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / den)


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    if not mask.any():
        return mask.copy()
    er = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return mask & ~er


def surface_dice(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...], tol: float = NSD_TOL_MM) -> float:
    """Normalized surface Dice at tolerance ``tol`` mm (surface voxels counted, not surface-area weighted)."""
    from scipy import ndimage

    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    sp, sg = _surface(p), _surface(g)
    dg = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dp = ndimage.distance_transform_edt(~sp, sampling=spacing)
    num = int((dg[sp] <= tol).sum()) + int((dp[sg] <= tol).sum())
    return float(num / (int(sp.sum()) + int(sg.sum())))


def case_scores(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> dict:
    return {"dsc": [dice(pred == c, gt == c) for c in CLASSES],
            "nsd": [surface_dice(pred == c, gt == c, spacing) for c in CLASSES]}


# ----------------------------------------------------------------------------------------------------------------
# atlas reference
# ----------------------------------------------------------------------------------------------------------------
def _grid_index(shape_from: tuple[int, ...], shape_to: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """Nearest-neighbour index arrays mapping every voxel of ``shape_to`` to ``shape_from`` (normalized coords)."""
    axes = [np.minimum(((np.arange(t) + 0.5) * f / t).astype(int), f - 1) for f, t in zip(shape_from, shape_to)]
    return np.ix_(*axes)


def build_atlas(labels: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros((len(CLASSES) + 1, *ATLAS_GRID), dtype=float)
    for lab in labels:
        g = lab[_grid_index(lab.shape, ATLAS_GRID)]
        for c in range(len(CLASSES) + 1):
            acc[c] += (g == c)
    return acc / max(1, len(labels))


def atlas_predict(atlas: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    lab = np.argmax(atlas, axis=0).astype(np.uint8)
    return lab[_grid_index(ATLAS_GRID, shape)].copy()


# ----------------------------------------------------------------------------------------------------------------
# adapter
# ----------------------------------------------------------------------------------------------------------------
class Adapter:
    discipline = CODE
    name = "MSD Task04 Hippocampus"
    family = "Life & health"
    metric = "DSC"
    direction = "max"
    task_type = "3d_medical_image_segmentation"

    def __init__(self, data_root: str | None = None, cache_dir: str | None = None, pool_seed: int = PROTOCOL_SEED,
                 **_: Any) -> None:
        self.root = (Path(data_root or os.environ.get("SCIENCECLAW_DATA_ROOT", DATA_ROOT)) / DATASET_DIR
                     / "data" / "Task04_Hippocampus")
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir(CODE)
        self.pool_seed = int(pool_seed)
        self._memo = Memo()

    # ---------------------------------------------------------------- data
    def _cases(self) -> list[str]:
        d = self.root / "imagesTr"
        if not d.is_dir():
            return []
        out = []
        for f in sorted(d.glob("hippocampus_*.nii.gz")):
            if f.name.startswith("._"):
                continue
            if (self.root / "labelsTr" / f.name).exists():
                out.append(f.name[: -len(".nii.gz")])
        return out

    def available(self) -> tuple[bool, str]:
        try:
            import nibabel  # noqa: F401
        except ImportError:
            return False, "nibabel is not installed"
        cases = self._cases()
        if len(cases) < 200:
            return False, f"expected ~260 labelled volumes under {self.root}; found {len(cases)}"
        try:
            pools = self._pools()
        except (ValueError, OSError) as ex:
            return False, f"cannot build pools: {type(ex).__name__}: {ex}"
        return True, f"{len(cases)} labelled volumes; " + ", ".join(f"{k}={len(v)}" for k, v in pools.items())

    def _index(self) -> dict[str, dict]:
        """case -> {"subject", "dtype", "p99", "shape", "encoding"}; cached as JSON (keyed by file list)."""
        def build() -> dict[str, dict]:
            import nibabel as nib

            cases = self._cases()
            key = ids_hash([f"{c}:{(self.root / 'imagesTr' / (c + '.nii.gz')).stat().st_size}" for c in cases])
            cache = self.cache_dir / f"index_{key}.json"
            if cache.exists():
                return json.loads(cache.read_text())
            idx = {}
            for c in cases:
                img = nib.load(str(self.root / "imagesTr" / f"{c}.nii.gz"))
                a = np.asarray(img.dataobj, dtype=np.float32)
                dt = str(img.header.get_data_dtype())
                p99 = float(np.percentile(a, 99))
                enc = "uint8_0_255" if dt == "uint8" else ("float_1e5_scale" if p99 > 1e4 else "float_1e3_scale")
                num = int(re.search(r"_(\d+)$", c).group(1))
                idx[c] = {"subject": f"S{(num + 1) // 2:03d}", "dtype": dt, "p99": p99, "shape": list(a.shape),
                          "spacing": [float(z) for z in img.header.get_zooms()[:3]], "encoding": enc}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_path_for(cache)
            tmp.write_text(json.dumps(idx))
            os.replace(tmp, cache)
            return idx
        return self._memo.get("index", build)

    def _pools(self) -> dict[str, list[str]]:
        def build() -> dict[str, list[str]]:
            idx = self._index()
            iid: dict[str, list[str]] = {}
            ood: dict[str, list[str]] = {}
            for c, info in sorted(idx.items()):
                tgt = iid if info["encoding"] == "float_1e3_scale" else ood
                tgt.setdefault(info["subject"], []).append(c)
            # a subject is IID only if all its labelled volumes are IID-encoded (they always are in practice)
            mixed = set(iid) & set(ood)
            for s in mixed:
                ood[s] = sorted(ood[s] + iid.pop(s))
            pools = allocate_groups(iid, IID_TARGETS, "src", [CODE, self.pool_seed, "iid"])
            po = allocate_groups(ood, [("ood", OOD_TARGET)], "ood_unused", [CODE, self.pool_seed, "ood"])
            pools["ood"] = po["ood"]
            for name, n in IID_TARGETS + [("ood", OOD_TARGET)]:
                if len(pools[name]) != n:
                    raise ValueError(f"pool {name} has {len(pools[name])} volumes, expected {n}")
            if len(pools["src"]) < 16:
                raise ValueError(f"src pool too small ({len(pools['src'])})")
            return pools
        return self._memo.get(f"pools:{self.pool_seed}", build)

    def _volume(self, case: str) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
        def load():
            import nibabel as nib

            img = nib.load(str(self.root / "imagesTr" / f"{case}.nii.gz"))
            lab = nib.load(str(self.root / "labelsTr" / f"{case}.nii.gz"))
            a = img.get_fdata(dtype=np.float32)
            y = np.rint(np.asarray(lab.dataobj, dtype=np.float32)).astype(np.uint8)
            if a.shape != y.shape:
                raise ValueError(f"{case}: image {a.shape} vs label {y.shape}")
            return a, y, tuple(float(z) for z in img.header.get_zooms()[:3])
        return self._memo.get(f"vol:{case}", load)

    def _atlas(self) -> np.ndarray:
        return self._memo.get(f"atlas:{self.pool_seed}",
                              lambda: build_atlas([self._volume(c)[1] for c in self._pools()["train"]]))

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _coerce(y: Any, shapes: list[tuple[int, ...]]) -> tuple[list[np.ndarray] | None, str]:
        arrs = as_list_of_arrays(y, len(shapes))
        if arrs is None:
            return None, f"y must be a list of {len(shapes)} 3-D arrays"
        out = []
        for i, (a, s) in enumerate(zip(arrs, shapes)):
            if tuple(a.shape) != tuple(s):
                return None, f"volume {i}: shape {tuple(a.shape)}, expected {tuple(s)}"
            try:
                af = a.astype(float)
            except (TypeError, ValueError):
                return None, f"volume {i}: non-numeric values"
            if not np.all(np.isfinite(af)) or not np.all(np.isin(af, (0.0, 1.0, 2.0))):
                return None, f"volume {i}: labels must be in {{0,1,2}}"
            out.append(af.astype(np.uint8))
        return out, "ok"

    def _score(self, cases: list[str], preds: list[np.ndarray]) -> dict:
        per = []
        for c, p in zip(cases, preds):
            _, gt, sp = self._volume(c)
            per.append(case_scores(p, gt, sp))
        dsc = np.array([s["dsc"] for s in per])
        nsd = np.array([s["nsd"] for s in per])
        return {"dsc_cases": dsc.tolist(), "nsd_cases": nsd.tolist(), "mean_dsc": float(dsc.mean()),
                "dsc_anterior": float(dsc[:, 0].mean()), "dsc_posterior": float(dsc[:, 1].mean()),
                "mean_nsd_1mm": float(nsd.mean())}

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Mean over all cases (of all given episodes) of the per-case mean foreground DSC."""
        pays = extract_payloads(per_episode, "dsc_cases")
        vals = [float(np.mean(d)) for p in pays for d in p["dsc"]]
        return float(np.mean(vals)) if vals else None

    # ---------------------------------------------------------------- episodes
    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        if split not in ("src", "val", "id", "ood", "rep"):
            raise ValueError(f"unknown split {split!r}")
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"{CODE} unavailable: {why}")
        base = "src" if split == "rep" else split
        pool = self._pools()[base]
        k = effective_items(split, len(pool), n, int(items_per_episode))
        groups, reused = compose_episodes(pool, n, k, [CODE, self.pool_seed, base, seed, items_per_episode])
        eps = [self._episode(split, base, seed, j, cases, reused) for j, cases in enumerate(groups)]
        for e in eps:
            e.lineage["items_requested"] = int(items_per_episode)
        return eps

    def _episode(self, split: str, base: str, seed: int, j: int, cases: list[str], reused: bool) -> Episode:
        pools, idx = self._pools(), self._index()
        k = len(cases)
        shapes = [tuple(idx[c]["shape"]) for c in cases]
        train, dev = list(pools["train"]), list(pools["dev"])
        dev_shapes = [tuple(idx[c]["shape"]) for c in dev]

        def images(cs: list[str]) -> list[np.ndarray]:
            return [self._volume(c)[0].copy() for c in cs]

        def t_load_train(inputs: dict, config: dict) -> dict:
            return {"images": images(train), "labels": [self._volume(c)[1].copy() for c in train],
                    "case_ids": list(train), "spacing_mm": [list(self._volume(c)[2]) for c in train]}

        def t_load_eval(inputs: dict, config: dict) -> dict:
            return {"images": images(cases), "case_ids": list(cases),
                    "spacing_mm": [list(self._volume(c)[2]) for c in cases]}

        def t_load_dev(inputs: dict, config: dict) -> dict:
            return {"images": images(dev), "case_ids": list(dev), "spacing_mm": [list(self._volume(c)[2]) for c in dev]}

        def t_score_dev(inputs: dict, config: dict) -> dict:
            preds, msg = self._coerce(inputs.get("predictions"), dev_shapes)
            if preds is None:
                raise ValueError(f"score_dev: {msg}")
            s = self._score(dev, preds)
            return {"mean_dsc": s["mean_dsc"], "dsc_anterior": s["dsc_anterior"], "dsc_posterior": s["dsc_posterior"]}

        img_desc = "list of 3-D float32 MRI intensity arrays (arbitrary scanner units), one per case, varying shapes"
        lab_desc = "list of 3-D uint8 masks aligned with images: 0 background, 1 anterior, 2 posterior hippocampus"
        tools = [
            ToolSpec("load_train", "Visible labelled training volumes (images + masks + ids + voxel spacing).", {},
                     {"images": PortSchema("list", (len(train),), "a.u.", "float", img_desc),
                      "labels": PortSchema("list", (len(train),), "1", "int", lab_desc),
                      "case_ids": PortSchema("list", (len(train),), None, "str", "case ids"),
                      "spacing_mm": PortSchema("list", (len(train),), "mm", "float", "voxel spacing (x, y, z) per case")},
                     t_load_train),
            ToolSpec("load_eval_inputs", "Evaluation volumes of this episode (images only), in item order.", {},
                     {"images": PortSchema("list", (k,), "a.u.", "float", img_desc),
                      "case_ids": PortSchema("list", (k,), None, "str", "case ids in item order"),
                      "spacing_mm": PortSchema("list", (k,), "mm", "float", "voxel spacing per case")},
                     t_load_eval),
            ToolSpec("load_dev_inputs", "Visible dev volumes (images only; masks withheld, scored by score_dev).", {},
                     {"images": PortSchema("list", (len(dev),), "a.u.", "float", img_desc),
                      "case_ids": PortSchema("list", (len(dev),), None, "str", "dev case ids"),
                      "spacing_mm": PortSchema("list", (len(dev),), "mm", "float", "voxel spacing per case")},
                     t_load_dev),
            ToolSpec("score_dev", "Mean Dice (anterior, posterior) of predicted masks for the dev volumes "
                                  "(order of load_dev_inputs).",
                     {"predictions": PortSchema("list", (len(dev),), "1", "int", "one label volume per dev case")},
                     {"mean_dsc": PortSchema("number", None, "1", "float", "mean foreground DSC"),
                      "dsc_anterior": PortSchema("number", None, "1", "float", ""),
                      "dsc_posterior": PortSchema("number", None, "1", "float", "")},
                     t_score_dev),
        ]
        objective = (
            "Segmentation of the hippocampus in 3-D T1-weighted brain MRI (Medical Segmentation Decathlon Task04, "
            "Vanderbilt University Medical Center; 1 mm isotropic crops around one hippocampus).\n"
            f"Evaluation items: {k} volumes (tool load_eval_inputs, item order = case_ids), each a 3-D float32 array "
            "of MRI intensities in scanner units; array shapes differ between cases. Visible labelled training "
            "volumes with masks: tool load_train.\n"
            f"Required output y: a list of {k} integer arrays in item order; array i has exactly the shape of "
            "image i and labels every voxel with 0 = background, 1 = hippocampus anterior (head), 2 = hippocampus "
            "posterior (body and tail).\n"
            "Score: Dice similarity coefficient 2|P∩G|/(|P|+|G|) per case for label 1 and label 2, averaged over the "
            "two labels and then over the cases.\n"
            "No formal tool-ON episode is permitted under the current no-self-training policy: do not call "
            "scilib.hippo_unet.fit_predict or fit_unet, because those functions train on the visible labelled volumes. "
            "The formal manifest marks FoR32 blocked until a frozen, publicly documented, non-overlapping model with "
            "the exact input/output contract is installed and audited. If that model is unavailable, stop without a "
            "formal submit. A future frozen-model episode must submit its direct predictions once and finish immediately; "
            "no post-submit search or replacement is allowed. This policy text does not alter the scorer, reference, "
            "acceptance margin, or withheld labels.\n")
        constraints = [
            ConstraintSpec("output_structure", f"y is a list of {k} arrays with the shapes of the input images",
                           lambda y, tr: self._check_structure(y, shapes)),
            ConstraintSpec("label_values", "all voxels are finite integers in {0, 1, 2}",
                           lambda y, tr: (lambda r: (r[0] is not None, r[1]))(self._coerce(y, shapes))),
            unit_constraint("1"),
        ]
        ref_memo = Memo()

        def evaluate(y: Any, trace: Any) -> EvalResult:
            ref = ref_memo.get("ref", lambda: self._score(cases, [atlas_predict(self._atlas(), s) for s in shapes]))
            preds, msg = self._coerce(y, shapes)
            valid = preds is not None
            if not valid:
                preds = [np.zeros(s, dtype=np.uint8) for s in shapes]         # no-skill fallback (DSC 0)
            s = self._score(cases, preds)
            payload = {"kind": "dsc_cases", "case_ids": list(cases), "dsc": s["dsc_cases"], "fallback": not valid}
            metrics = {"mean_dsc": s["mean_dsc"], "dsc_anterior": s["dsc_anterior"], "dsc_posterior": s["dsc_posterior"],
                       "mean_nsd_1mm": s["mean_nsd_1mm"], "reference_mean_dsc": ref["mean_dsc"]}
            return make_result(s["mean_dsc"], ref["mean_dsc"], "max", ACCEPT_MARGIN, metrics,
                               {"pooled_payload": payload, "parse": msg,
                                "reference_desc": "location-only probabilistic atlas from visible training masks"},
                               valid)

        encs = sorted({idx[c]["encoding"] for c in cases})
        lineage = {"dataset": "Medical Segmentation Decathlon Task04_Hippocampus", "version": "MSD release 1.0 (04/05/2018)",
                   "source_url": "http://medicaldecathlon.com/ (archive https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar)",
                   "license": "CC-BY-SA 4.0", "item_kind": "3-D MRI volume (one hippocampus)", "item_ids": list(cases),
                   "subjects": sorted({idx[c]["subject"] for c in cases}), "encodings": encs,
                   "item_pool": "ood" if base == "ood" else "iid",
                   "ood_kind": "proxy_within_dataset" if base == "ood" else None,
                   "ood_rule": "non-standard intensity encoding (uint8 0-255 or float ~1e5 scale)" if base == "ood" else None,
                   "pool_seed": self.pool_seed, "episode_seed": int(seed), "episode_index": j, "reused_items": bool(reused),
                   "visible_train_ids_hash": ids_hash(train), "n_visible_train": len(train), "dev_ids_hash": ids_hash(dev),
                   "rebuilt_split": True, "historical_ids_recovered": False}
        return Episode(
            id=f"{CODE}-{split}-s{seed}-e{j:02d}", discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("list", (k,), "1", "int", "one 3-D label volume (0/1/2) per case, item order"),
            tools=tools, constraints=constraints, budget=default_budget(k, max_node_s=600, max_wall_s=2400),
            lineage=lineage,
            acceptance=f"accepted iff mean DSC > reference (location atlas on the same cases) + {ACCEPT_MARGIN}",
            tolerance={"rtol": 0.0, "atol": 0.0}, tags=["mri", "segmentation", "3d", "hippocampus", "medical-imaging"],
            metric=self.metric, direction=self.direction, n_items=k, _evaluate=evaluate, _dev_evaluate=None)

    @staticmethod
    def _check_structure(y: Any, shapes: list[tuple[int, ...]]) -> tuple[bool, str]:
        arrs = as_list_of_arrays(y, len(shapes))
        if arrs is None:
            return False, f"y must be a list of {len(shapes)} arrays"
        for i, (a, s) in enumerate(zip(arrs, shapes)):
            if tuple(a.shape) != tuple(s):
                return False, f"volume {i}: shape {tuple(a.shape)}, expected {tuple(s)}"
        return True, "ok"

    def describe_pools(self) -> dict:
        idx = self._index()
        out = {}
        for k, v in self._pools().items():
            enc: dict[str, int] = {}
            for c in v:
                enc[idx[c]["encoding"]] = enc.get(idx[c]["encoding"], 0) + 1
            out[k] = {"n": len(v), "subjects": len({idx[c]["subject"] for c in v}), "encodings": enc}
        return out


if __name__ == "__main__":      # pragma: no cover
    a = Adapter()
    print(a.available())
    print(json.dumps(a.describe_pools(), indent=1))
