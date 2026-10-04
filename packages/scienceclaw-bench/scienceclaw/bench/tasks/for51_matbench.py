"""FoR51 Physical sciences — Matbench v0.1 ``matbench_phonons``: last phonon-DOS peak, metric MAE in cm^-1 (min).

Data: ``matbench_phonons.json.gz`` (Matbench v0.1, 1,265 relaxed crystal structures from the Petretto et al.
2018 DFPT phonon database; target ``last phdos peak`` = frequency of the highest-frequency optical phonon DOS
peak in cm^-1), sha256 verified by the data team.

* **Item** = one crystal structure (pymatgen ``Structure`` dict, parsed here without pymatgen into lattice matrix,
  species and fractional coordinates) with its target in cm^-1. Item id = ``mb-phonons-<1-based row:04d>``
  (Matbench's naming; the dataset order is preserved).
* **Pools.** No second phonon dataset is available locally, so the OOD pool is a *lineage-disjoint chemistry
  shift inside the dataset* (``lineage["ood_kind"] = "proxy_within_dataset"``): **leave-element-out** — every
  compound containing Se or Te (any site) forms the OOD pool; these elements never occur in visible training /
  dev data or in IID items. The IID pool (all other compounds) is cut once (``partition_seed``, stratified by
  target decile) into train 62 % / dev 10 % / src 15 % / val 4 % / id 9 %.
* **Visible data.** ``load_train`` = the fixed train sub-pool (structures + targets in cm^-1), ``load_dev_inputs``
  = dev sub-pool structures whose targets are held by ``score_dev`` (MAE; visible dev signal, the episode's
  ``_dev_evaluate`` is None), ``load_eval_inputs`` = the episode's structures, ``featurize_structures`` = light
  composition / geometry descriptors (no pymatgen / matminer dependency).
* **Metric.** MAE = mean |y_pred - y_true| in cm^-1 (Matbench's primary regression metric).
* **Reference** (deterministic): 5-nearest-neighbour regression (uniform weights, Euclidean distance) on the
  standardized ``featurize_structures`` descriptors of the ``load_train`` crystals. **Acceptance:**
  ``MAE <= (1 - rel_margin) * MAE_ref`` (default ``rel_margin`` = 0.2). (A 3-feature ridge reference was
  rejected during calibration: every generic ML pipeline beat it by > 5x, so it could not discriminate.)
* **Hard constraints:** 1-D vector of length ``items``; finite; every value in the physically plausible range
  [10, 5000] cm^-1 (catches outputs in THz / eV; the required unit is declared in ``required_output``).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for34_39_40_44_51 import (
    PARTITION_SEED, Lazy, as_float_vector, c_finite, c_range, c_vector, check_split, draw_episodes, episode_id,
    episode_rng, norm_score, partition_pool, receipt_info, resolve_data_root,
)

CODE = "FoR51"
FAMILY = "Physical & Earth"
DATASET_DIR = "for51-matbench-phonons"
DATA_FILE = Path("raw") / "matbench_phonons.json.gz"
OOD_ELEMENTS = ("Se", "Te")
IID_FRACTIONS = {"train": 0.62, "dev": 0.10, "src": 0.15, "val": 0.04, "id": 0.09}
UNIT = "cm^-1"
PLAUSIBLE = (10.0, 5000.0)

# Pauling electronegativities (NaN where undefined: He, Ne, Ar).
_PAULING = {
    "H": 2.20, "He": np.nan, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Ne": np.nan, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": np.nan,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
    "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.20,
    "Gd": 1.20, "Tb": 1.10, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27, "Hf": 1.30,
    "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62,
    "Pb": 2.33, "Bi": 2.02, "Po": 2.00, "At": 2.20, "Rn": np.nan, "Fr": 0.70, "Ra": 0.90, "Ac": 1.10, "Th": 1.30,
    "Pa": 1.50, "U": 1.38, "Np": 1.36, "Pu": 1.28,
}
ELEMENT_PROPS = ("Z", "mass", "electronegativity", "covalent_radius", "group", "period", "outer_electrons")
STRUCT_FEATURES = ("n_sites", "n_elements", "volume_per_atom", "density", "packing_fraction", "min_nn_distance",
                   "mean_nn_distance")
FEATURE_NAMES = tuple(f"{p}_{s}" for p in ELEMENT_PROPS for s in ("mean", "min", "max", "std")) + STRUCT_FEATURES
_AMU_PER_A3_TO_G_CM3 = 1.66053906660


def _group_period(z: int) -> tuple[int, int]:
    """IUPAC group (lanthanides/actinides -> 3) and period of atomic number z."""
    starts = [(1, 1), (3, 2), (11, 3), (19, 4), (37, 5), (55, 6), (87, 7)]
    period, start = 1, 1
    for s, p in starts:
        if z >= s:
            period, start = p, s
    off = z - start
    if period == 1:
        return (1 if z == 1 else 18), 1
    if period in (2, 3):
        return (off + 1 if off < 2 else off + 11), period
    if period in (4, 5):
        return off + 1, period
    # periods 6 and 7 contain the f block
    if off < 2:
        return off + 1, period
    if off < 17:
        return 3, period
    return off - 13, period


class _Elements:
    """Per-element properties from RDKit's periodic table plus the Pauling table above (cached)."""

    def __init__(self) -> None:
        from rdkit import Chem
        self._pt = Chem.GetPeriodicTable()
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def props(self, sym: str) -> np.ndarray:
        with self._lock:
            if sym not in self._cache:
                z = int(self._pt.GetAtomicNumber(sym))
                g, p = _group_period(z)
                self._cache[sym] = np.array([z, self._pt.GetAtomicWeight(z), _PAULING.get(sym, np.nan),
                                             self._pt.GetRcovalent(z), g, p, self._pt.GetNOuterElecs(z)], dtype=float)
            return self._cache[sym]


_ELEMENTS: Lazy[_Elements] = Lazy(_Elements)


def parse_structure(d: dict) -> dict:
    """pymatgen Structure dict -> {"formula", "lattice" (3x3 Å), "species", "frac_coords" (n x 3)} (ordered sites)."""
    lat = [[float(v) for v in row] for row in d["lattice"]["matrix"]]
    species, frac = [], []
    for site in d["sites"]:
        sp = site["species"]
        if len(sp) != 1 or float(sp[0].get("occu", 1.0)) != 1.0:
            raise ValueError("disordered sites are not supported")
        species.append(str(sp[0]["element"]))
        frac.append([float(v) for v in site["abc"]])
    counts: dict[str, int] = {}
    for s in species:
        counts[s] = counts.get(s, 0) + 1
    formula = "".join(f"{e}{'' if c == 1 else c}" for e, c in sorted(counts.items()))
    return {"formula": formula, "lattice": lat, "species": species, "frac_coords": frac}


def structure_features(st: dict) -> np.ndarray:
    """Composition statistics of ELEMENT_PROPS + STRUCT_FEATURES for one parsed structure (FEATURE_NAMES order)."""
    el = _ELEMENTS.get()
    species = list(st["species"])
    P = np.array([el.props(s) for s in species])                     # (n_sites, n_props); site-weighted
    stats = []
    for j in range(P.shape[1]):
        v = P[:, j]
        v = v[np.isfinite(v)]
        stats.extend([v.mean(), v.min(), v.max(), v.std()] if v.size else [np.nan] * 4)
    L = np.asarray(st["lattice"], dtype=float)
    F = np.asarray(st["frac_coords"], dtype=float)
    n = len(species)
    vol = abs(float(np.linalg.det(L)))
    mass = float(P[:, 1].sum())
    rcov = P[:, 3]
    cart = F @ L
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float) @ L
    diff = cart[None, :, None, :] + shifts[None, None, :, :] - cart[:, None, None, :]   # (n, n, 27, 3)
    dist = np.linalg.norm(diff, axis=-1).reshape(n, -1)
    dist[dist < 1e-8] = np.inf                                          # self at zero shift
    nn = dist.min(axis=1)
    struct = [n, len(set(species)), vol / n, mass * _AMU_PER_A3_TO_G_CM3 / vol,
              float(np.sum(4.0 / 3.0 * np.pi * rcov ** 3) / vol), float(nn.min()), float(nn.mean())]
    return np.array(stats + struct, dtype=float)


@dataclass
class _MBData:
    ids: list[str]
    structures: dict[str, dict]
    targets: dict[str, float]
    features: dict[str, np.ndarray]
    ood: set[str]


def _load(root: Path) -> _MBData:
    with gzip.open(root / DATASET_DIR / DATA_FILE, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    cols = raw["columns"]
    si, ti = cols.index("structure"), cols.index("last phdos peak")
    ids, structs, targets, feats, ood = [], {}, {}, {}, set()
    for r, row in enumerate(raw["data"]):
        iid = f"mb-phonons-{r + 1:04d}"
        st = parse_structure(row[si])
        ids.append(iid)
        structs[iid] = st
        targets[iid] = float(row[ti])
        feats[iid] = structure_features(st)
        if any(s in OOD_ELEMENTS for s in st["species"]):
            ood.add(iid)
    return _MBData(ids, structs, targets, feats, ood)


class MatbenchPhononsAdapter:
    """TaskAdapter for FoR51 (see module docstring)."""

    discipline = CODE
    name = "matbench_phonons"
    family = FAMILY
    metric = "MAE"
    direction = "min"
    task_type = "materials_property_regression"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, rel_margin: float = 0.2,
                 budget: Budget | None = None, **_: Any) -> None:
        self.root = resolve_data_root(data_root)
        self.partition_seed = int(partition_seed)
        self.rel_margin = float(rel_margin)
        self.budget = budget
        self._data: Lazy[_MBData] = Lazy(lambda: _load(self.root))
        self._parts: Lazy[dict[str, list[str]]] = Lazy(self._partition)
        self._ref: Lazy[dict] = Lazy(self._fit_reference)

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        p = self.root / DATASET_DIR / DATA_FILE
        if not p.exists():
            return False, f"missing {p}"
        try:
            import rdkit  # noqa: F401  (periodic-table properties for the featurizer)
        except ImportError:
            return False, "RDKit is not installed (periodic table used by featurize_structures)"
        return True, f"Matbench v0.1 matbench_phonons at {p}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        data, parts = self._data.get(), self._parts.get()
        pool = parts["ood"] if split == "ood" else parts[split]
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_episodes({0: pool}, {0: items_per_episode}, int(n), rng, f"{CODE}/{split}")
        return [self._episode(data, parts, split, k, int(seed), items) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """MAE (cm^-1) over all items of all episodes; invalid outputs are scored with the reference predictions."""
        err: list[float] = []
        for p in per_episode:
            if not p:
                continue
            t = np.asarray(p.get("y_true") or [], dtype=float)
            pr = p.get("y_pred")
            if pr is None:
                pr = p.get("y_ref")
            if pr is None or len(pr) != len(t):
                raise ValueError("FoR51 pooled payload needs y_true and y_pred (or y_ref) of equal length")
            err.extend(np.abs(np.asarray(pr, dtype=float) - t).tolist())
        return float(np.mean(err)) if err else None

    # ------------------------------------------------------------ internals
    def _partition(self) -> dict[str, list[str]]:
        data = self._data.get()
        iid = [i for i in data.ids if i not in data.ood]
        y = np.array([data.targets[i] for i in iid])
        edges = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
        strata = {i: int(np.searchsorted(edges, data.targets[i], side="right")) for i in iid}
        parts = partition_pool(iid, IID_FRACTIONS, f"{CODE}|{self.partition_seed}|iid", strata)
        parts["ood"] = sorted(data.ood)
        return parts

    def _fit_reference(self) -> dict:
        data, parts = self._data.get(), self._parts.get()
        X = np.array([data.features[i] for i in parts["train"]])
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        y = np.array([data.targets[i] for i in parts["train"]])
        return {"Z": (X - mu) / sd, "y": y, "mu": mu, "sd": sd, "k": 5}

    def _ref_predict(self, data: _MBData, ids: list[str]) -> np.ndarray:
        """5-NN mean of training targets in standardized descriptor space (ties broken by training order)."""
        r = self._ref.get()
        Q = (np.array([data.features[i] for i in ids]) - r["mu"]) / r["sd"]
        d2 = ((Q[:, None, :] - r["Z"][None, :, :]) ** 2).sum(axis=-1)
        nn = np.argsort(d2, axis=1, kind="stable")[:, :r["k"]]
        return r["y"][nn].mean(axis=1)

    def _episode(self, data: _MBData, parts: dict[str, list[str]], split: str, k: int, seed: int,
                 items: list[str]) -> Episode:
        eid = episode_id(CODE, split, seed, k)
        n_items = len(items)
        y_true = np.array([data.targets[i] for i in items])
        train_ids, dev_ids = list(parts["train"]), list(parts["dev"])
        n_tr, n_dev = len(train_ids), len(dev_ids)
        y_dev = np.array([data.targets[i] for i in dev_ids])
        y_ref = self._ref_predict(data, items)
        mae_ref = float(np.mean(np.abs(y_ref - y_true)))
        dev_ref_mae = float(np.mean(np.abs(self._ref_predict(data, dev_ids) - y_dev)))

        def _structs(ids: list[str]) -> list[dict]:
            return [json.loads(json.dumps(data.structures[i])) for i in ids]      # deep copies

        def load_train(inputs: dict, config: dict) -> dict:
            return {"structures": _structs(train_ids),
                    "targets": np.array([data.targets[i] for i in train_ids], dtype=float)}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_structures": _structs(dev_ids)}

        def score_dev(inputs: dict, config: dict) -> dict:
            p, why = as_float_vector(inputs.get("dev_pred"), n_dev)
            if p is None:
                raise ValueError(f"dev_pred: {why}")
            if not np.all(np.isfinite(p)):
                raise ValueError("dev_pred contains non-finite values")
            return {"dev_mae": float(np.mean(np.abs(p - y_dev))), "dev_reference_mae": dev_ref_mae}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"structures": _structs(items)}

        def featurize(inputs: dict, config: dict) -> dict:
            sts = inputs.get("structures")
            if not isinstance(sts, (list, tuple)):
                raise ValueError("structures must be a list of structure dicts")
            X = np.array([structure_features(s) for s in sts], dtype=float).reshape(len(sts), len(FEATURE_NAMES))
            return {"X": X, "feature_names": list(FEATURE_NAMES)}

        def fit_regressor(inputs: dict, config: dict) -> dict:
            """Fit the documented log-target tree ensemble on source descriptors only."""
            Xtr = np.asarray(inputs.get("train_features"), dtype=float)
            ytr = np.asarray(inputs.get("train_targets"), dtype=float).reshape(-1)
            Xev = np.asarray(inputs.get("eval_features"), dtype=float)
            if Xtr.shape != (n_tr, len(FEATURE_NAMES)):
                raise ValueError(f"train_features must have shape {(n_tr, len(FEATURE_NAMES))}")
            if ytr.shape != (n_tr,):
                raise ValueError(f"train_targets must have shape {(n_tr,)}")
            if Xev.shape != (n_items, len(FEATURE_NAMES)):
                raise ValueError(f"eval_features must have shape {(n_items, len(FEATURE_NAMES))}")
            from scilib.matphonon_regression import cross_validate, fit_predict

            out = fit_predict(Xtr, ytr, Xev, return_info=True)
            cv = cross_validate(Xtr, ytr)
            return {"pred": out["predictions"], "cv_mae": cv["mae"], "cv_fold_mae": np.asarray(cv["fold_mae"])}

        def fit_hist_regressor(inputs: dict, config: dict) -> dict:
            """Fit the log-target histogram gradient-boosting alternative on source descriptors only."""
            Xtr = np.asarray(inputs.get("train_features"), dtype=float)
            ytr = np.asarray(inputs.get("train_targets"), dtype=float).reshape(-1)
            Xev = np.asarray(inputs.get("eval_features"), dtype=float)
            if Xtr.shape != (n_tr, len(FEATURE_NAMES)) or ytr.shape != (n_tr,) or Xev.shape != (n_items, len(FEATURE_NAMES)):
                raise ValueError("fit_hist_gradient_boosting received incompatible train/eval feature shapes")
            from scilib.matphonon_regression import fit_hist_predict

            out = fit_hist_predict(Xtr, ytr, Xev, return_info=True)
            return {"pred": out["predictions"], "provenance": out["provenance"]}

        def fit_sevennet_mlip(inputs: dict, config: dict) -> dict:
            """Use frozen SevenNet phonon descriptors, then fit only the documented visible-label regressor.

            The universal potential is used strictly as a pretrained feature extractor.  The only fitted
            component is the lightweight log-target ensemble on ``train_targets``; evaluation structures and
            their hidden targets never enter either the feature call or the fit.  ``model=chgnet`` remains an
            explicit diagnostic option, while SevenNet is the default selected from the held-out dev study.
            """
            train_st = inputs.get("train_structures")
            ytr = np.asarray(inputs.get("train_targets"), dtype=float).reshape(-1)
            eval_st = inputs.get("eval_structures")
            if not isinstance(train_st, (list, tuple)) or not isinstance(eval_st, (list, tuple)):
                raise ValueError("fit_sevennet_mlip needs train_structures and eval_structures lists")
            if len(train_st) != n_tr or ytr.shape != (n_tr,) or len(eval_st) != n_items:
                raise ValueError("fit_sevennet_mlip received incompatible train/target/eval lengths")
            cfg = dict(config or {})
            # Keep SevenNet as the documented default, while allowing a dedicated
            # engineering shard to force the alternate frozen model without
            # changing the formal objective or exposing hidden targets.
            model = str(cfg.get("model", os.environ.get("SCIENCECLAW_MLIP_MODEL", "sevennet"))).lower()
            if model not in {"sevennet", "sevennet_mf0_pbe", "sevennet_mf0_r2scan", "chgnet"}:
                raise ValueError("model must be 'sevennet', 'sevennet_mf0_pbe', 'sevennet_mf0_r2scan', or 'chgnet'")
            from scilib import matphonon_mlip
            from scilib.matphonon_regression import fit_predict

            a = matphonon_mlip.phonon_features(train_st, mesh=int(cfg.get("mesh", 8)),
                                                min_len=float(cfg.get("min_len", 7.0)),
                                                disp=float(cfg.get("disp", 0.01)), model=model)
            b = matphonon_mlip.phonon_features(eval_st, mesh=int(cfg.get("mesh", 8)),
                                                min_len=float(cfg.get("min_len", 7.0)),
                                                disp=float(cfg.get("disp", 0.01)), model=model)
            Xa = np.asarray(a["X"], dtype=float)
            Xb = np.asarray(b["X"], dtype=float)
            if Xa.shape[0] != n_tr or Xb.shape[0] != n_items or Xa.shape[1] != Xb.shape[1]:
                raise ValueError("pretrained MLIP feature shapes do not match train/eval structures")
            if not np.isfinite(Xa).all() or not np.isfinite(Xb).all():
                raise ValueError("pretrained MLIP returned non-finite features")
            Xtr = np.concatenate([np.array([structure_features(s) for s in train_st], dtype=float), Xa], axis=1)
            Xev = np.concatenate([np.array([structure_features(s) for s in eval_st], dtype=float), Xb], axis=1)
            out = fit_predict(Xtr, ytr, Xev, return_info=True)
            return {"pred": out["predictions"], "provenance": {"model": model, "feature_names": list(a["names"]),
                                                                  "pretrained": True, "fit_targets": n_tr}}

        def fit_chgnet_mlip(inputs: dict, config: dict) -> dict:
            """Explicit CHGNet alternate using the same visible-target regressor and frozen features."""
            cfg = dict(config or {})
            cfg["model"] = "chgnet"
            return fit_sevennet_mlip(inputs, cfg)

        def fit_sevennet_mf0_mlip(inputs: dict, config: dict) -> dict:
            """Explicit SevenNet-MF-0 route; modality is selected without fitting weights."""
            cfg = dict(config or {})
            # Keep the formal/default route on PBE, while allowing an explicitly
            # requested engineering probe to select the frozen R2SCAN cache.
            # This environment override is only consulted when this dedicated
            # MF-0 tool is called and never changes the formal objective.
            modality = str(os.environ.get("SCIENCECLAW_MF0_MODAL", cfg.get("modal", cfg.get("model", "pbe")))).lower()
            if modality in {"pbe", "sevennet_mf0_pbe"}:
                cfg["model"] = "sevennet_mf0_pbe"
            elif modality in {"r2scan", "r2_scan", "sevennet_mf0_r2scan"}:
                cfg["model"] = "sevennet_mf0_r2scan"
            else:
                raise ValueError("modal must be 'pbe' or 'r2scan' for SevenNet-MF-0")
            return fit_sevennet_mlip(inputs, cfg)

        st_desc = ("structure dict: formula (str), lattice (3x3 lattice vectors as rows, Å), species (element symbol "
                   "per site), frac_coords (fractional coordinates per site)")
        tools = [
            ToolSpec("load_train", f"{n_tr} training crystals: structures (list of {st_desc}) and targets = last "
                     "phonon-DOS peak frequency in cm^-1.",
                     {}, {"structures": PortSchema("list", (n_tr,), dtype="dict", description=st_desc),
                          "targets": PortSchema("array", (n_tr,), unit=UNIT, dtype="float",
                                                description="last phdos peak")},
                     load_train),
            ToolSpec("load_dev_inputs", f"{n_dev} further crystals (disjoint from load_train) whose targets are "
                     "withheld; score predictions for them with score_dev.",
                     {}, {"dev_structures": PortSchema("list", (n_dev,), dtype="dict", description=st_desc)},
                     load_dev_inputs),
            ToolSpec("score_dev", "MAE (cm^-1) of dev_pred (one prediction per dev structure, same order) against the "
                     "withheld dev targets; also returns the MAE of the adapter's reference model.",
                     {"dev_pred": PortSchema("array", (n_dev,), unit=UNIT, dtype="float")},
                     {"dev_mae": PortSchema("number", unit=UNIT), "dev_reference_mae": PortSchema("number", unit=UNIT)},
                     score_dev),
            ToolSpec("load_eval_inputs", f"The {n_items} evaluation crystals (targets hidden), in the order y must follow.",
                     {}, {"structures": PortSchema("list", (n_items,), dtype="dict", description=st_desc)},
                     load_eval_inputs),
            ToolSpec("featurize_structures", "Composition/geometry descriptors per structure: site-weighted mean/min/"
                     "max/std of element properties (atomic number, mass in amu, Pauling electronegativity, covalent "
                     "radius in Å, group, period, outer electrons), number of sites and elements, volume per atom (Å^3), "
                     "density (g/cm^3), packing fraction, min/mean nearest-neighbour distance (Å, within the 27 "
                     "neighbouring cells).",
                     {"structures": PortSchema("list", ("n",), dtype="dict")},
                     {"X": PortSchema("array", ("n", len(FEATURE_NAMES)), dtype="float"),
                      "feature_names": PortSchema("list", (len(FEATURE_NAMES),), dtype="str")},
                     featurize),
            ToolSpec("fit_log_extra_trees", f"Fit an ensemble on the {n_tr} labelled source feature rows only: the "
                     "target is fitted as log(frequency), predictions are returned in cm^-1, and 5-fold shuffled "
                     "training-only MAE is reported for diagnostics.",
                     {"train_features": PortSchema("array", (n_tr, len(FEATURE_NAMES)), dtype="float"),
                      "train_targets": PortSchema("array", (n_tr,), unit=UNIT, dtype="float"),
                      "eval_features": PortSchema("array", (n_items, len(FEATURE_NAMES)), dtype="float")},
                     {"pred": PortSchema("array", (n_items,), unit=UNIT, dtype="float"),
                      "cv_mae": PortSchema("number", unit=UNIT),
                      "cv_fold_mae": PortSchema("array", (5,), unit=UNIT, dtype="float")},
                     fit_regressor),
            ToolSpec("fit_hist_gradient_boosting", f"Fit a second CPU-only log-target gradient-boosting ensemble on the "
                     f"same {n_tr} source rows; this is a label-isolated alternative to ExtraTrees.",
                     {"train_features": PortSchema("array", (n_tr, len(FEATURE_NAMES)), dtype="float"),
                      "train_targets": PortSchema("array", (n_tr,), unit=UNIT, dtype="float"),
                      "eval_features": PortSchema("array", (n_items, len(FEATURE_NAMES)), dtype="float")},
                     {"pred": PortSchema("array", (n_items,), unit=UNIT, dtype="float"),
                      "provenance": PortSchema("dict")}, fit_hist_regressor),
            ToolSpec("fit_sevennet_mlip", f"Run a frozen SevenNet-l3i5 phonon-feature extractor on the {n_tr} visible "
                     f"training crystals and {n_items} evaluation crystals, then fit the log-target ensemble using "
                     "visible train targets only. No evaluation target is read and no universal-potential weights are "
                     "updated. Set model=chgnet only for a disclosed diagnostic comparison.",
                     {"train_structures": PortSchema("list", (n_tr,), dtype="dict", description=st_desc),
                      "train_targets": PortSchema("array", (n_tr,), unit=UNIT, dtype="float"),
                      "eval_structures": PortSchema("list", (n_items,), dtype="dict", description=st_desc)},
                     {"pred": PortSchema("array", (n_items,), unit=UNIT, dtype="float"),
                      "provenance": PortSchema("dict")}, fit_sevennet_mlip,
                     config_doc="model=sevennet (default) or chgnet; mesh=8; min_len=7.0; disp=0.01."),
            ToolSpec("fit_chgnet_mlip", f"Run a frozen CHGNet 0.3.0 phonon-feature extractor on the {n_tr} visible "
                     f"training crystals and {n_items} evaluation crystals, then fit the same visible-target log "
                     "ensemble; no evaluation target is read and no weights are updated.",
                     {"train_structures": PortSchema("list", (n_tr,), dtype="dict", description=st_desc),
                      "train_targets": PortSchema("array", (n_tr,), unit=UNIT, dtype="float"),
                      "eval_structures": PortSchema("list", (n_items,), dtype="dict", description=st_desc)},
                     {"pred": PortSchema("array", (n_items,), unit=UNIT, dtype="float"),
                      "provenance": PortSchema("dict")}, fit_chgnet_mlip,
                     config_doc="model=chgnet; mesh=8; min_len=7.0; disp=0.01."),
            ToolSpec("fit_sevennet_mf0_mlip", f"Run the frozen official SevenNet-MF-0 multi-fidelity phonon-feature "
                     f"extractor on the {n_tr} visible training crystals and {n_items} evaluation crystals, then "
                     "fit the same visible-target log ensemble. Select modal=pbe or modal=r2scan; no universal-potential "
                     "weights are updated and evaluation targets are never read.",
                     {"train_structures": PortSchema("list", (n_tr,), dtype="dict", description=st_desc),
                      "train_targets": PortSchema("array", (n_tr,), unit=UNIT, dtype="float"),
                      "eval_structures": PortSchema("list", (n_items,), dtype="dict", description=st_desc)},
                     {"pred": PortSchema("array", (n_items,), unit=UNIT, dtype="float"),
                      "provenance": PortSchema("dict")}, fit_sevennet_mf0_mlip,
                     config_doc="modal=pbe (default) or r2scan; mesh=8; min_len=7.0; disp=0.01."),
        ]
        lo, hi = PLAUSIBLE
        constraints: list[ConstraintSpec] = [
            c_vector(n_items, f"one frequency in {UNIT} per evaluation structure"),
            c_finite(n_items),
            c_range(n_items, lo, hi, "plausible_frequency_cm-1",
                    f"every value of y is a phonon frequency in {UNIT} within the plausible range [{lo:g}, {hi:g}]"),
        ]

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            payload = {"item_ids": list(items), "y_true": y_true.tolist(), "y_pred": None, "y_ref": y_ref.tolist()}
            arr, why = as_float_vector(yv, n_items)
            if arr is not None and not np.all(np.isfinite(arr)):
                arr, why = None, "non-finite values"
            base = {"reference": mae_ref, "reference_name": "5-NN regression on standardized structure descriptors",
                    "rel_margin": self.rel_margin, "n_items": n_items, "unit": UNIT}
            if arr is None:
                return EvalResult(metrics={"reference_mae": mae_ref}, primary=None, direction="min", accepted=False,
                                  details={**base, "norm_score": 0.0, "pooled_payload": payload, "invalid": why})
            mae = float(np.mean(np.abs(arr - y_true)))
            payload["y_pred"] = arr.tolist()
            return EvalResult(metrics={"mae": mae, "reference_mae": mae_ref}, primary=mae, direction="min",
                              accepted=bool(mae <= (1.0 - self.rel_margin) * mae_ref),
                              details={**base, "norm_score": norm_score(mae, mae_ref, "min"), "pooled_payload": payload})

        pool = "ood" if split == "ood" else "iid"
        objective = (
            "Physical sciences - phonon property prediction (Matbench v0.1 matbench_phonons; DFPT phonon database of "
            "Petretto et al. 2018). Each evaluation item is a relaxed inorganic crystal structure. Target: the frequency "
            f"of the highest-frequency optical phonon peak of its phonon density of states ('last phdos peak'), in {UNIT}.\n"
            f"Visible data: load_train returns {n_tr} structures with targets; load_dev_inputs returns {n_dev} further "
            "structures without targets and score_dev returns the MAE of predictions for them; load_eval_inputs returns "
            "the evaluation structures (targets hidden); featurize_structures computes composition/geometry descriptors. "
            f"Structures are dicts with formula, lattice (Å), species and frac_coords.\n"
            f"Deliverable y: a 1-D float array of length {n_items}; y[i] is the predicted last phdos peak of the i-th "
            f"structure returned by load_eval_inputs (same order), in {UNIT}. Evaluation metric: mean absolute error "
            f"in {UNIT} against the hidden DFPT values. fit_log_extra_trees trains a deterministic ExtraTrees ensemble "
            f"on the featurize_structures output in log-target space and reports shuffled training-only cross-validation "
            f"MAE; fit_hist_gradient_boosting is a second log-target CPU ensemble for cross-checking the structure "
            f"regressor. scilib.matphonon_regression provides both implementations. In a tool-on run with the frozen "
            "MLIP tools, use fit_sevennet_mlip for the existing formal baseline; use fit_sevennet_mf0_mlip with "
            "modal=pbe or r2scan for the disclosed SevenNet-MF-0 comparison; for the disclosed CHGNet comparison "
            "call the explicit fit_chgnet_mlip tool (do not rely on the SevenNet tool's default) before CPU-only "
            "approximations; its prediction is returned under the `pred` output port and must be submitted directly "
            "as y. It fits only on visible training targets and never reads evaluation targets."
            + scilib.describe("matphonon_regression")
            + scilib.describe_extra("matphonon_mlip")
        )
        lineage = {
            "dataset": "Matbench v0.1 matbench_phonons", "version": "matbench_v0.1 (metadata commit 8ddb18c7)",
            "source_url": "https://ml.materialsproject.org/projects/matbench_phonons.json.gz",
            "license": "Matbench MIT (https://github.com/materialsproject/matbench/blob/main/LICENSE); data CC-BY 4.0 "
                       "(Petretto et al., Sci. Data 5:180065)",
            "sha256": "4db551f21ec5f577e6202725f10e34dfc509aa7df3a6bdaac497da7f6dbbb9b3",
            "receipt": receipt_info(self.root / DATASET_DIR / "receipt.json"),
            "pool": pool,
            "ood_kind": "proxy_within_dataset" if pool == "ood" else None,
            "ood_shift": (f"leave-element-out: compounds containing {'/'.join(OOD_ELEMENTS)} (absent from all visible "
                          "and IID items)") if pool == "ood" else None,
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n_items, "unit": UNIT,
            "n_train": n_tr, "n_dev": n_dev,
            "train_ids_sha256": hashlib.sha256(",".join(train_ids).encode()).hexdigest(),
            "dev_ids_sha256": hashlib.sha256(",".join(dev_ids).encode()).hexdigest(),
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n_items,), unit=UNIT, dtype="float",
                                       description=f"last phdos peak per evaluation structure in {UNIT}"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=2400.0, max_node_s=600.0,
                                         max_llm_items=4 * n_items),
            lineage=lineage,
            acceptance=(f"MAE <= {1 - self.rel_margin:g} x reference MAE; reference = 5-nearest-neighbour regression "
                        "on the standardized featurize_structures descriptors of the load_train crystals"),
            tolerance={"rtol": 1e-6, "atol": 1e-6},
            tags=[CODE, "physics", "materials", "phonons", "crystal-structure", "regression", "MAE", "matbench",
                  UNIT],
            metric=self.metric, direction=self.direction, n_items=n_items,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = MatbenchPhononsAdapter

__all__ = ["Adapter", "MatbenchPhononsAdapter", "parse_structure", "structure_features", "FEATURE_NAMES"]
