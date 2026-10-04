"""FoR44 Human society — ACIC 2016 Atlantic Causal Inference Conference data challenge,
metric: response-SD normalized RMSE of the SATT estimate (min).

ACIC 2016 (Dorie, Hill, Shalit, Scott & Cervone, Statist. Sci. 34(1), 2019): 77 data-generating scenarios
(``parameters_2016``: treatment model, baseline treatment probability, overlap, response model, alignment,
effect heterogeneity) x 100 replications, all on the same real covariates (``input_2016``: 4,802 units x 58
covariates from the Collaborative Perinatal Project). Each simulated dataset has a binary treatment ``z`` and an
outcome ``y``; the truth is the pair of expected potential outcomes ``mu0``/``mu1`` per unit.

* **Item** = one simulated dataset (scenario p, replication s); the estimand is the treated-sample average effect
  ``SATT = mean_{i: z_i = 1} (mu1_i - mu0_i)`` (conditional on the sample, as in the competition's evaluation).
  Item id ``acic2016-p<pp>-r<rrr>`` (the data team's instance id).
* **Metric** (per episode and pooled): ``sqrt(mean_items(((tau_hat - SATT) / sd(y))^2))`` with ``sd(y)`` the sample
  standard deviation of the observed outcome of that dataset ("response-SD normalized RMSE"). Note: the data team's
  own baseline table uses a different target (per-subject ITE ``mu1 - mu0``, RMSE / sd(y)); this adapter keeps the
  SATT-error version, whose scale matches the paper's reported values (see docs/tasks/FoR44.md).
* **Splits = the data team's roles** (delivery ``reconstructed_v1``: 64 simulations per role, sampled from the
  77 x 100 candidate frame with seed 20260928 after assigning whole *settings* to roles: source 33, validation 19,
  evaluation 19 settings actually sampled, pairwise disjoint): ``source`` -> ``src`` (+ the visible dev sub-pool),
  ``validation`` -> ``val``, ``evaluation`` -> ``id``. The split is therefore setting-held-out (unseen data-generating
  processes for val/id) but **not** cross-dataset: every setting uses the same 4,802 covariate rows, and the data team
  states that the partition "is not cross-dataset OOD". Consequently ``ood`` is **empty by default**
  (``build_episodes("ood", ...)`` returns ``[]``; ``SplitPlan`` reports the shortfall as a warning).
* **Dev sub-pool** (visible dev signal, ``_dev_evaluate`` is None): the ``n_dev`` (default 16) source simulations
  with the smallest ``sha256(salt|id)`` rank. They are removed from ``src`` (src = source role minus dev), so dev is
  item-disjoint from every episode of every split. Dev shares settings with ``src`` (replicates of the same settings)
  but never with ``val``/``id``.
* **Optional proxy OOD** (``ood_mode="step_assignment"``, off by default): all simulations of the 32 scenarios with
  the non-smooth ``step`` treatment-assignment model (9, 17, 48-77) from *every* role form the ``ood`` pool and are
  removed from src/val/id/dev, so the assignment mechanism of OOD is never seen elsewhere
  (``lineage["ood_kind"] = "proxy_within_dataset"``). This overrides the data team's role assignment for those 78
  simulations and shrinks the other pools (src 20 after dev / val 37 / id 41 / ood 78 with the delivered data); the
  data team makes no OOD claim, so it is a documented proxy chosen by this adapter, not a delivered role.
* **Leakage caveats.** (1) Replicates of one setting share their DGP and the covariates, so simulations inside one
  split (and inside one episode) are not independent; only *settings* are disjoint across src/val/id. (2) Dev and src
  come from the same source settings. (3) Val/id settings are disjoint from src/dev, but ACIC settings are near
  neighbours (e.g. same response model, different overlap), so held-out settings measure generalization to nearby DGPs,
  not to a different dataset. (4) If the loose official/DGP file layouts are used (no role folders) the adapter
  assigns whole settings to roles by a salted hash in the data team's 39:19:19 proportions.
* **Visible data.** ``load_covariates`` (the shared 4,802 x 58 table), ``load_eval_inputs`` (z and y of the episode's
  datasets), ``load_dev_inputs`` + ``score_dev`` (z/y of dev datasets and the normalized RMSE of SATT estimates for
  them; visible dev signal).
* **Reference** (deterministic): OLS regression ``y ~ z + X`` (categoricals one-hot, numerics as is) -> coefficient of
  ``z`` (the competition documentation's own example). **Acceptance:** ``RMSE <= (1 - rel_margin) * RMSE_ref``
  (default 0.1).
* **Hard constraints:** 1-D vector of length ``items``; finite; ``|tau_hat_i| <= 5 sd(y_i)`` (effect on the outcome
  scale).

Data (2026-09-28): the data team generated 192 whole simulations with the unmodified official generator
(aciccomp2016 commit 282d2665, R 4.4.3) under ``reconstructed_v1/{source,validation,evaluation}/acic2016-pPP-rRRR/``
(``observed.csv.gz``: subject_id, z, y; ``oracle.csv.gz``: y0, y1, mu0, mu1, propensity, tau) plus
``reconstructed_v1/covariates.csv.gz``. With 16 items per episode the delivery supports src 3 (dev 16), val 4, id 4
and ood 0 episodes; ``available()`` only requires one full episode per non-empty split plus the dev sub-pool, so the
paper-scale plan (7 source rounds) raises ``PoolExhausted`` for ``src`` unless ``items_per_episode``/``rounds`` are
reduced. Also accepted (anywhere below the dataset directory except ``upstream/``): the official competition release
``<p>/zymu_<s>.csv`` (z, y0, y1, mu0, mu1) + ``x.csv``, and ``dgp_2016(..., extraInfo = TRUE)`` exports
``p<p>_s<s>.csv`` (z, y, y.0, y.1, mu.0, mu.1) + ``x.csv``.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scilib

from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import Budget, ConstraintSpec, EvalResult, Episode, ToolSpec
from ._adapter_utils_for34_39_40_44_51 import (
    PARTITION_SEED, Lazy, as_float_vector, c_finite, c_vector, check_split, draw_episodes, episode_id, episode_rng,
    hash_rank, norm_score, partition_pool, receipt_info, resolve_data_root,
)

CODE = "FoR44"
FAMILY = "Social & behavior"
DATASET_DIR = "for44-acic-2016"
EFFECT_SD_LIMIT = 5.0
DEFAULT_ITEMS = 16
# Data-team roles (reconstructed_v1/<role>/acic2016-pPP-rRRR/) -> benchmark split. 'evaluation' is the held-out role
# (whole settings unseen in source/validation); the delivery defines no OOD role, so 'ood' has no role.
ROLES = ("source", "validation", "evaluation")
ROLE_TO_SPLIT = {"source": "src", "validation": "val", "evaluation": "id"}
# Setting counts the data team assigned to the roles (39 / 19 / 19 of 77); used only to assign whole settings to
# roles for loose file layouts that carry no role folders.
ROLE_SETTING_FRACTIONS = {"source": 39, "validation": 19, "evaluation": 19}
OOD_MODES = ("none", "step_assignment")
# available(): one full episode of DEFAULT_ITEMS simulations for every non-empty split (+ the dev sub-pool). The
# paper-scale plan (7 source rounds, n_val 2, n_id 4, n_ood 4) needs more simulations than were delivered.
MIN_EPISODES = {"src": 1, "val": 1, "id": 1}

# parameters_2016 (aciccomp2016 R package, data/parameters_2016.RData): model.trt, root.trt, overlap.trt, model.rsp,
# alignment, te.hetero for scenarios 1..77.
_PARAMS = """linear .35 one-term linear .75 high|polynomial .35 one-term exponential .75 none|linear .35 one-term linear .75 none
polynomial .35 full exponential .75 high|linear .35 one-term exponential .75 high|polynomial .35 one-term linear .75 high
polynomial .35 one-term exponential .75 high|polynomial .35 one-term exponential 0 high|step .35 one-term step .75 high
linear .35 one-term exponential .25 high|polynomial .35 one-term linear .25 high|polynomial .35 one-term exponential .25 high
linear .65 one-term exponential .75 high|polynomial .65 one-term linear .75 high|polynomial .65 one-term exponential .75 high
polynomial .65 one-term exponential 0 high|step .65 one-term step .75 high|linear .65 one-term exponential .25 high
polynomial .65 one-term linear .25 high|polynomial .65 one-term exponential .25 high|polynomial .35 one-term step .25 med
polynomial .35 one-term step .25 high|polynomial .35 one-term step .75 med|polynomial .35 one-term step .75 high
polynomial .35 one-term exponential .25 med|polynomial .35 one-term exponential .75 med|polynomial .35 full step .25 med
polynomial .35 full step .25 high|polynomial .35 full step .75 med|polynomial .35 full step .75 high
polynomial .35 full exponential .25 med|polynomial .35 full exponential .25 high|polynomial .35 full exponential .75 med
polynomial .65 one-term step .25 med|polynomial .65 one-term step .25 high|polynomial .65 one-term step .75 med
polynomial .65 one-term step .75 high|polynomial .65 one-term exponential .25 med|polynomial .65 one-term exponential .75 med
polynomial .65 full step .25 med|polynomial .65 full step .25 high|polynomial .65 full step .75 med
polynomial .65 full step .75 high|polynomial .65 full exponential .25 med|polynomial .65 full exponential .25 high
polynomial .65 full exponential .75 med|polynomial .65 full exponential .75 high|step .35 one-term step .25 med
step .35 one-term step .25 high|step .35 one-term step .75 med|step .35 one-term exponential .25 med
step .35 one-term exponential .25 high|step .35 one-term exponential .75 med|step .35 one-term exponential .75 high
step .35 full step .25 med|step .35 full step .25 high|step .35 full step .75 med|step .35 full step .75 high
step .35 full exponential .25 med|step .35 full exponential .25 high|step .35 full exponential .75 med
step .35 full exponential .75 high|step .65 one-term step .25 med|step .65 one-term step .25 high
step .65 one-term step .75 med|step .65 one-term exponential .25 med|step .65 one-term exponential .25 high
step .65 one-term exponential .75 med|step .65 one-term exponential .75 high|step .65 full step .25 med
step .65 full step .25 high|step .65 full step .75 med|step .65 full step .75 high|step .65 full exponential .25 med
step .65 full exponential .25 high|step .65 full exponential .75 med|step .65 full exponential .75 high"""
PARAMETERS_2016: dict[int, dict] = {
    i + 1: dict(zip(("model.trt", "root.trt", "overlap.trt", "model.rsp", "alignment", "te.hetero"),
                    (a, float(b), c, d, float(e), f)))
    for i, (a, b, c, d, e, f) in enumerate(r.split() for r in _PARAMS.replace("\n", "|").split("|"))
}
assert len(PARAMETERS_2016) == 77
# Scenarios with the non-smooth 'step' treatment-assignment model: the candidates of the optional proxy OOD pool
# (ood_mode="step_assignment"); IID_SCENARIOS are the linear/polynomial-assignment scenarios.
OOD_SCENARIOS = tuple(p for p, v in PARAMETERS_2016.items() if v["model.trt"] == "step")
IID_SCENARIOS = tuple(p for p, v in PARAMETERS_2016.items() if v["model.trt"] != "step")

_ZYMU = re.compile(r"^zymu_(\d+)\.csv$")
_DGP = re.compile(r"^p(\d+)_s(\d+)\.csv$")
_INSTANCE = re.compile(r"^acic2016-p(\d+)-r(\d+)$")


def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _item_id(p: int, s: int) -> str:
    return f"acic2016-p{p:02d}-r{s:03d}"


def scenario_of(item: str) -> int:
    return int(_INSTANCE.match(item).group(1))


@dataclass
class _Sim:
    z: np.ndarray
    y: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray

    @property
    def satt(self) -> float:
        t = self.z == 1
        return float(np.mean(self.mu1[t] - self.mu0[t]))

    @property
    def sd_y(self) -> float:
        return float(np.std(self.y, ddof=1))


def discover(root: Path) -> dict[str, Path]:
    """item id -> simulation source (a CSV file, or a directory with observed.csv.gz + oracle.csv.gz)."""
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for d in sorted(root.rglob("acic2016-p*-r*")):
        m = _INSTANCE.match(d.name)
        if m and d.is_dir() and (d / "observed.csv.gz").exists() and (d / "oracle.csv.gz").exists():
            p, s = int(m.group(1)), int(m.group(2))
            if 1 <= p <= 77 and s >= 1:
                out.setdefault(_item_id(p, s), d)
    for f in sorted(root.rglob("*.csv")):
        if "upstream" in f.relative_to(root).parts:
            continue
        m = _ZYMU.match(f.name)
        if m and f.parent.name.isdigit():
            p, s = int(f.parent.name), int(m.group(1))
        else:
            m2 = _DGP.match(f.name)
            if not m2:
                continue
            p, s = int(m2.group(1)), int(m2.group(2))
        if 1 <= p <= 77 and s >= 1:
            out.setdefault(_item_id(p, s), f)
    return out


def read_sim(path: Path) -> _Sim:
    if path.is_dir():                      # data-team layout: observed (z, y) + oracle (mu0, mu1, ...)
        obs, orc = pd.read_csv(path / "observed.csv.gz"), pd.read_csv(path / "oracle.csv.gz")
        if "subject_id" in obs and "subject_id" in orc and not np.array_equal(obs["subject_id"], orc["subject_id"]):
            raise ValueError(f"{path}: observed/oracle subject order differs")
        df = pd.concat([obs.drop(columns=["subject_id"], errors="ignore"),
                        orc[[c for c in orc.columns if _norm_col(c) in ("mu0", "mu1")]]], axis=1)
    else:
        df = pd.read_csv(path)
    cols = {_norm_col(c): c for c in df.columns}
    need = {"z", "mu0", "mu1"}
    if not need <= set(cols):
        raise ValueError(f"{path}: needs columns z, mu0/mu.0, mu1/mu.1 (found {list(df.columns)})")
    z = df[cols["z"]].to_numpy()
    if z.dtype.kind not in "iub":
        z = np.where(pd.Series(z).astype(str).str.lower().isin(["1", "trt", "true"]), 1, 0)
    z = z.astype(int)
    if "y" in cols:
        y = df[cols["y"]].to_numpy(dtype=float)
    elif {"y0", "y1"} <= set(cols):
        y = np.where(z == 1, df[cols["y1"]].to_numpy(dtype=float), df[cols["y0"]].to_numpy(dtype=float))
    else:
        raise ValueError(f"{path}: needs y or y0/y1 columns")
    return _Sim(z, y, df[cols["mu0"]].to_numpy(dtype=float), df[cols["mu1"]].to_numpy(dtype=float))


def load_covariates(root: Path) -> pd.DataFrame | None:
    """covariates.csv.gz / x.csv (first match below root, 'upstream' excluded) or input_2016.RData via `rdata`."""
    for name in ("covariates.csv.gz", "x.csv"):
        for f in sorted(root.rglob(name)) if root.exists() else []:
            if "upstream" not in f.relative_to(root).parts:
                x = pd.read_csv(f)
                return x.drop(columns=[c for c in x.columns if str(c).startswith("Unnamed")])
    rd = root / "upstream" / "2016" / "data" / "input_2016.RData"
    if rd.exists():
        try:
            import rdata
        except ImportError:
            return None
        x = rdata.read_rda(str(rd))["input_2016"]
        return pd.DataFrame(x).reset_index(drop=True)
    return None


def design_matrix(x: pd.DataFrame) -> np.ndarray:
    """Numeric design matrix of the covariates: categoricals one-hot (first level dropped), numerics as floats."""
    cat = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
    X = pd.get_dummies(x, columns=cat, drop_first=True, dtype=float)
    return X.to_numpy(dtype=float)


def ols_effect(X: np.ndarray, z: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of z in the OLS fit y ~ 1 + z + X (minimum-norm least squares)."""
    A = np.column_stack([np.ones(len(y)), z.astype(float), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1])


def role_of(path: Path) -> str | None:
    """Data-team role of a simulation directory (``<role>/acic2016-pPP-rRRR``); None for loose file layouts."""
    return path.parent.name if path.is_dir() and path.parent.name in ROLES else None


class ACIC2016Adapter:
    """TaskAdapter for FoR44 (see module docstring)."""

    discipline = CODE
    name = "acic2016"
    family = FAMILY
    metric = "response-SD normalized RMSE"
    direction = "min"
    task_type = "causal_effect_estimation"

    def __init__(self, data_root: str | None = None, partition_seed: int = PARTITION_SEED, rel_margin: float = 0.1,
                 n_dev: int = 16, budget: Budget | None = None, ood_mode: str = "none", **_: Any) -> None:
        if ood_mode not in OOD_MODES:
            raise ValueError(f"ood_mode must be one of {OOD_MODES}, got {ood_mode!r}")
        self.root = resolve_data_root(data_root) / DATASET_DIR
        self.partition_seed = int(partition_seed)
        self.rel_margin = float(rel_margin)
        self.n_dev = int(n_dev)
        self.ood_mode = ood_mode
        self.budget = budget
        self._files: Lazy[dict[str, Path]] = Lazy(lambda: discover(self.root))
        self._x: Lazy[pd.DataFrame | None] = Lazy(lambda: load_covariates(self.root))
        self._X: Lazy[np.ndarray] = Lazy(lambda: design_matrix(self._require_x()))
        self._parts: Lazy[dict[str, list[str]]] = Lazy(self._partition)
        self._sims: dict[str, _Sim] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ protocol
    def available(self) -> tuple[bool, str]:
        if not self.root.exists():
            return False, f"missing {self.root}"
        files = self._files.get()
        if not files:
            return False, ("ACIC 2016 potential outcomes are not available locally: the aciccomp2016 package copy holds "
                           "input_2016, parameters_2016 and testData (z and y only for 20 simulations) but no mu0/mu1, "
                           "and R is not installed to run dgp_2016. Needed: counterfactual files (official "
                           "data_cf_all <p>/zymu_<s>.csv or dgp_2016(..., extraInfo=TRUE) exports p<p>_s<s>.csv) plus "
                           "x.csv; see docs/tasks/FoR44.md")
        if self._x.get() is None:
            return False, "counterfactual files found but no covariates (x.csv, or input_2016.RData + the `rdata` package)"
        try:
            parts = self._parts.get()
        except ValueError as ex:                      # delivered roles violate the setting-disjointness contract
            return False, str(ex)
        need = {s: n * DEFAULT_ITEMS for s, n in MIN_EPISODES.items()} | {"dev": self.n_dev}
        if self.ood_mode != "none":
            need["ood"] = DEFAULT_ITEMS
        sizes = {s: len(parts[s]) for s in ("dev", "src", "val", "id", "ood")}
        short = {s: (sizes[s], k) for s, k in need.items() if sizes[s] < k}
        if short:
            return False, (f"{len(files)} simulations with potential outcomes found (pools {sizes}), too few for the "
                           f"minimum plan (one episode of {DEFAULT_ITEMS} per non-empty split, dev {self.n_dev}): " +
                           ", ".join(f"{s} {have} < {k}" for s, (have, k) in short.items()) +
                           " (see docs/tasks/FoR44.md)")
        cap = {s: sizes[s] // DEFAULT_ITEMS for s in ("src", "val", "id", "ood")}
        ood_note = ("ood empty: the delivery defines no OOD role" if self.ood_mode == "none"
                    else "ood = step-assignment proxy")
        return True, (f"ACIC 2016: {len(files)} simulated datasets with potential outcomes under {self.root}; pools "
                      f"{sizes}; episodes of {DEFAULT_ITEMS} supported {cap} ({ood_note})")

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list[Episode]:
        check_split(split)
        if items_per_episode < 1:
            raise ValueError("items_per_episode must be >= 1")
        parts = self._parts.get()
        if split == "ood" and not parts["ood"]:
            return []          # no OOD role in the delivery and ood_mode="none": SplitPlan warns "returned 0 of n"
        rng = episode_rng(CODE, split, seed, self.partition_seed)
        draws = draw_episodes({0: parts[split]}, {0: items_per_episode}, int(n), rng, f"{CODE}/{split}")
        return [self._episode(split, k, int(seed), items) for k, items in enumerate(draws)]

    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """sqrt(mean over all items of ((tau_hat - SATT) / sd(y))^2); invalid outputs use the reference estimates."""
        e: list[float] = []
        for p in per_episode:
            if not p:
                continue
            est = p.get("tau_hat")
            if est is None:
                est = p.get("tau_ref")
            t, sd = p.get("satt") or [], p.get("sd_y") or []
            if est is None or not (len(est) == len(t) == len(sd)):
                raise ValueError("FoR44 pooled payload needs satt, sd_y and tau_hat (or tau_ref) of equal length")
            e.extend(((float(a) - float(b)) / float(c)) ** 2 for a, b, c in zip(est, t, sd))
        return float(np.sqrt(np.mean(e))) if e else None

    # ------------------------------------------------------------ internals
    def _require_x(self) -> pd.DataFrame:
        x = self._x.get()
        if x is None:
            raise RuntimeError("ACIC 2016 covariates not found")
        return x

    def _sim(self, item: str) -> _Sim:
        with self._lock:
            if item not in self._sims:
                sim = read_sim(self._files.get()[item])
                if sim.z.size != len(self._require_x()):
                    raise ValueError(f"{item}: {sim.z.size} units but covariates have {len(self._require_x())} rows")
                self._sims[item] = sim
            return self._sims[item]

    def _roles(self) -> dict[str, str]:
        """item -> role. Data-team layout: the role folder (items outside a role folder are ignored when any item has
        one). Loose official/DGP layouts: whole settings hashed to roles in the data team's 39:19:19 proportions."""
        files = self._files.get()
        roles = {i: r for i, f in files.items() if (r := role_of(f))}
        if roles:
            return roles
        settings = sorted({str(scenario_of(i)) for i in files})
        by_role = partition_pool(settings, ROLE_SETTING_FRACTIONS, f"{CODE}|{self.partition_seed}|role")
        role_of_setting = {p: r for r, ps in by_role.items() for p in ps}
        return {i: role_of_setting[str(scenario_of(i))] for i in files}

    def _partition(self) -> dict[str, list[str]]:
        """dev / src / val / id / ood item lists (see the module docstring). Setting-disjoint across src/val/id."""
        roles = self._roles()
        by_setting: dict[int, set[str]] = {}
        for i, r in roles.items():
            by_setting.setdefault(scenario_of(i), set()).add(r)
        mixed = {p: sorted(r) for p, r in by_setting.items() if len(r) > 1}
        if mixed:
            raise ValueError(f"FoR44 roles must be setting-disjoint but settings {sorted(mixed)[:5]} occur in several "
                             f"roles ({mixed[sorted(mixed)[0]]}); the delivered roles cannot define lineage-disjoint splits")
        step = self.ood_mode == "step_assignment"
        is_step = {i: scenario_of(i) in OOD_SCENARIOS for i in roles}
        pool = {r: sorted(i for i, ri in roles.items() if ri == r and not (step and is_step[i])) for r in ROLES}
        ranked = hash_rank(pool["source"], f"{CODE}|{self.partition_seed}|dev")
        return {"dev": ranked[: self.n_dev], "src": ranked[self.n_dev:], "val": pool["validation"],
                "id": pool["evaluation"], "ood": sorted(i for i in roles if step and is_step[i])}

    def _episode(self, split: str, k: int, seed: int, items: list[str]) -> Episode:
        eid = episode_id(CODE, split, seed, k)
        n = len(items)
        x = self._require_x()
        X = self._X.get()
        sims = [self._sim(i) for i in items]
        Z = np.stack([s.z for s in sims])
        Y = np.stack([s.y for s in sims])
        satt = np.array([s.satt for s in sims])
        sd_y = np.array([s.sd_y for s in sims])
        dev_ids = self._parts.get()["dev"][: self.n_dev]
        dev = [self._sim(i) for i in dev_ids]
        Zd, Yd = np.stack([s.z for s in dev]), np.stack([s.y for s in dev])
        dev_satt, dev_sd = np.array([s.satt for s in dev]), np.array([s.sd_y for s in dev])
        n_units, n_dev = Z.shape[1], len(dev)
        ref = Lazy(lambda: np.array([ols_effect(X, s.z, s.y) for s in sims]))

        def nrmse(est: np.ndarray, truth: np.ndarray, sd: np.ndarray) -> float:
            return float(np.sqrt(np.mean(((est - truth) / sd) ** 2)))

        dev_ref = Lazy(lambda: nrmse(np.array([ols_effect(X, s.z, s.y) for s in dev]), dev_satt, dev_sd))

        def load_covariates_tool(inputs: dict, config: dict) -> dict:
            return {"covariates": x.copy()}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"treatment": Z.copy(), "outcome": Y.copy()}

        def load_dev_inputs(inputs: dict, config: dict) -> dict:
            return {"dev_treatment": Zd.copy(), "dev_outcome": Yd.copy()}

        def score_dev(inputs: dict, config: dict) -> dict:
            est, why = as_float_vector(inputs.get("dev_estimates"), n_dev)
            if est is None:
                raise ValueError(f"dev_estimates: {why}")
            if not np.all(np.isfinite(est)):
                raise ValueError("dev_estimates contains non-finite values")
            return {"dev_normalized_rmse": nrmse(est, dev_satt, dev_sd), "dev_reference_normalized_rmse": dev_ref.get()}

        tools = [
            ToolSpec("load_covariates", f"The {n_units} x {x.shape[1]} pre-treatment covariate table shared by every "
                     "simulated dataset (row i = unit i; categorical columns as strings).",
                     {}, {"covariates": PortSchema("table", (n_units, x.shape[1]))}, load_covariates_tool),
            ToolSpec("load_eval_inputs", f"The {n} evaluation datasets: treatment[j, i] (0/1) and observed outcome[j, i] "
                     "of unit i in dataset j (rows in the order y must follow).",
                     {}, {"treatment": PortSchema("array", (n, n_units), dtype="int"),
                          "outcome": PortSchema("array", (n, n_units), dtype="float")},
                     load_eval_inputs),
            ToolSpec("load_dev_inputs", f"{n_dev} further simulated datasets (other replications of the source-pool "
                     "scenarios) whose effects are withheld; score estimates with score_dev.",
                     {}, {"dev_treatment": PortSchema("array", (n_dev, n_units), dtype="int"),
                          "dev_outcome": PortSchema("array", (n_dev, n_units), dtype="float")},
                     load_dev_inputs),
            ToolSpec("score_dev", "Response-SD normalized RMSE of dev_estimates (one SATT estimate per dev dataset) and "
                     "of the adapter's reference estimator on the same datasets.",
                     {"dev_estimates": PortSchema("array", (n_dev,), dtype="float")},
                     {"dev_normalized_rmse": PortSchema("number", unit="1"),
                      "dev_reference_normalized_rmse": PortSchema("number", unit="1")},
                     score_dev),
        ]

        def c_scale(yv: Any, trace: Trace | None) -> tuple[bool, str]:
            est, why = as_float_vector(yv, n)
            if est is None:
                return False, why
            bad = int(np.sum(~np.isfinite(est) | (np.abs(est) > EFFECT_SD_LIMIT * sd_y)))
            return bad == 0, ("all estimates within 5 sd(y)" if bad == 0 else f"{bad} estimates exceed 5 sd(y)")

        constraints: list[ConstraintSpec] = [
            c_vector(n, "one SATT estimate per evaluation dataset"), c_finite(n),
            ConstraintSpec("effect_scale", f"|y[j]| <= {EFFECT_SD_LIMIT:g} x sample SD of outcome[j] (effect on the "
                           "outcome scale)", c_scale),
        ]

        def evaluate(yv: Any, trace: Trace | None) -> EvalResult:
            tau_ref = ref.get()
            ref_err = nrmse(tau_ref, satt, sd_y)
            payload = {"item_ids": list(items), "satt": satt.tolist(), "sd_y": sd_y.tolist(), "tau_hat": None,
                       "tau_ref": tau_ref.tolist()}
            base = {"reference": ref_err, "reference_name": "OLS y ~ z + X coefficient of z",
                    "rel_margin": self.rel_margin, "n_items": n}
            est, why = as_float_vector(yv, n)
            if est is not None and not np.all(np.isfinite(est)):
                est, why = None, "non-finite values"
            if est is None:
                return EvalResult(metrics={"reference_normalized_rmse": ref_err}, primary=None, direction="min",
                                  accepted=False, details={**base, "norm_score": 0.0, "pooled_payload": payload,
                                                           "invalid": why})
            err = nrmse(est, satt, sd_y)
            payload["tau_hat"] = est.tolist()
            return EvalResult(metrics={"normalized_rmse": err, "reference_normalized_rmse": ref_err}, primary=err,
                              direction="min", accepted=bool(err <= (1.0 - self.rel_margin) * ref_err),
                              details={**base, "norm_score": norm_score(err, ref_err, "min"), "pooled_payload": payload})

        pool = "ood" if split == "ood" else "iid"
        role = next((r for r, sp in ROLE_TO_SPLIT.items() if sp == split), None)
        objective = (
            "Human society - causal effect estimation from observational data (ACIC 2016 data challenge). Each "
            f"evaluation item is one simulated observational study on the same {n_units} units: a binary treatment z "
            "and an outcome y were generated from the real pre-treatment covariates by an unknown data-generating "
            "process (treatment assignment and response surface are unknown functions of the covariates). Task: for each "
            f"of the {n} datasets estimate the sample average treatment effect on the treated, SATT = mean over treated "
            "units of E[Y(1) - Y(0) | covariates].\n"
            "Visible data: load_covariates returns the covariate table; load_eval_inputs returns treatment and outcome "
            f"matrices (one row per dataset); load_dev_inputs returns {n_dev} further datasets and score_dev returns the "
            "normalized RMSE of SATT estimates for them.\n"
            f"Deliverable y: a 1-D float array of length {n}; y[j] is the SATT estimate for dataset j (row order of "
            "load_eval_inputs), in the units of the outcome. Evaluation metric: sqrt(mean_j ((y[j] - SATT_j) / "
            "sd(outcome_j))^2).\n"
            + scilib.describe("causal")
        )
        lineage = {
            "dataset": "ACIC 2016 (aciccomp2016)", "version": "aciccomp commit 282d2665 + counterfactual files",
            "source_url": "https://github.com/vdorie/aciccomp",
            "license": "see https://github.com/vdorie/aciccomp/blob/282d26659b2d3d6fd060dde6d32feeb8f1e8ab5a/2016/DESCRIPTION",
            "receipt": receipt_info(self.root / "receipt.json"),
            "pool": pool,
            "ood_kind": "proxy_within_dataset" if pool == "ood" else None,
            "ood_shift": ("treatment-assignment DGP shift: scenarios with model.trt = step (never in src/val/id/dev; "
                          "adapter-chosen proxy, not a delivered role)" if pool == "ood" else None),
            "scenarios": sorted({scenario_of(i) for i in items}),
            "data_team_role": role,
            "ood_mode": self.ood_mode,
            "split_basis": ("data-team roles (settings disjoint across src/val/id; replicates of one setting share a "
                            "split)" if self.ood_mode == "none" else
                            "data-team roles minus step-assignment settings, which form the ood proxy pool"),
            "split": split, "split_seed": seed, "partition_seed": self.partition_seed, "index": k,
            "item_ids": list(items), "n_items": n, "dev_item_ids": list(dev_ids),
            "rebuilt_split": True, "historical_ids_recovered": False,
        }
        return Episode(
            id=eid, discipline=CODE, family=FAMILY, split=split, task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n,), dtype="float",
                                       description="SATT estimate per evaluation dataset (outcome units)"),
            tools=tools, constraints=constraints,
            budget=self.budget or Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0, max_node_s=600.0,
                                         max_llm_items=2 * n),
            lineage=lineage,
            acceptance=(f"normalized RMSE <= {1 - self.rel_margin:g} x reference; reference = OLS y ~ z + X "
                        "(coefficient of z)"),
            tolerance={"rtol": 1e-6, "atol": 1e-8},
            tags=[CODE, "social-science", "causal-inference", "treatment-effect", "observational-study", "SATT",
                  "RMSE", "acic2016"],
            metric=self.metric, direction=self.direction, n_items=n,
            _evaluate=evaluate, _dev_evaluate=None,
        )


Adapter = ACIC2016Adapter

__all__ = ["Adapter", "ACIC2016Adapter", "PARAMETERS_2016", "IID_SCENARIOS", "OOD_SCENARIOS", "discover", "read_sim",
           "ols_effect"]
