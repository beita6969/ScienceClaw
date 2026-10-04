"""FoR33 Built environment and design — BuildingsBench real-building day-ahead load forecasting.

Data (read-only): the data team's current delivery of ``<data_root>/for33-buildingsbench``, found through the
delivery catalog (``configs/data-delivery-v1/catalog.json`` entry ``FoR33`` -> ``reconstructed_v2``; see
``_delivery_roles.py``). The four role files ``configs/episode-designs/episodes/FoR33/{source,val,id,ood}.jsonl``
(sha256-checked against the catalog) define the *physical buildings* of every role and the eight forecast windows
of each building; the window arrays come from ``reconstructed_v2/{source,validation,evaluation}.npz`` and each
building's permitted historical adaptation data from ``reconstructed_v2/adaptation/<role>/<building>.npz``.

v2 delivery (27 buildings, both BuildingsBench categories in every role; balanced by category):

* ``src``  8 buildings (4 residential, 4 commercial), 64 windows;   ``val`` 3 buildings (2 + 1), 24 windows;
* ``id``  11 buildings (7 + 4), 88 windows (LCL residential; BDG-2 commercial);
* ``ood``  5 buildings (1 + 4), 40 windows: SMART residential ``HomeB`` and Electricity commercial ``MT_*`` —
  a held-out BuildingsBench *sub-dataset group* (not a separately acquired benchmark), hence
  ``ood_kind = "proxy_within_dataset"``;
* building-disjoint across roles (asserted); ``MT_070`` (reserved validation target-group building, present in
  ``validation.npz`` but in no role) is never used.

Item = one building-specific window: input 168 hourly loads (kWh), target the next 24 hourly loads. Item id =
the role file's ``window_id`` (e.g. ``residential/MAC002290/2012-11-06T16:00:00``); lineage group = building.

Episodes (fixed by the role files, one physical building = one lineage group):

* an episode holds **every building of its role** with ``m`` windows each (``m = max(1, min(M_TARGET[split],
  items_per_episode // n_buildings))``, M_TARGET = src 1 / val 2 / id 2 / ood 2; so with the default 16 items: src 8,
  val 6, id 11, ood 10 items). The official windows of one building are 24-h daily targets whose 168-h contexts
  can contain another window's target (window indices at most 7 days apart); inside an episode the windows of one
  building are therefore drawn *conflict-free* and successive episodes take *disjoint* windows of each building
  (seeded exact packing). Conflict-free = the target of one window ends at least ``MIN_DELAY_H = 144`` h
  (6 days) before the 168-h context of the other starts, in both directions: every context of an episode is visible,
  so a later window of the same building would otherwise expose the days right after an earlier target. Only the
  m >= 2 episodes (val, ood at 16 items) contain such pairs. 144 h is the largest delay that keeps the capacity
  unchanged (168 h would cut val from 4 to 3 episodes). Capacity: 8 // m episodes (src 8, val 4, id 8, ood 4 at 16
  items). ``src`` recycles windows (flagged in the lineage) beyond that; the held-out splits raise ``PoolExhausted``.
* Visible data (D_E): the episode buildings' own historical adaptation data is permitted by the data team
  ("evaluation-building history is permitted for that building's adaptation; its future targets are evaluator-only"):
  ``load_history`` returns the first 3601 hours (the delivery's ``internal_train_mask`` block) of each building; the
  following 720 hours (internal validation block) back four **dev windows** per building (168-h contexts visible in
  ``load_dev``, 24-h targets behind ``score_dev``; target days 8 days apart, so no dev target lies in another
  dev context). All history precedes every evaluation context (asserted).

Metric (lower is better) — the data team's normative convention (``metric_adapter.py`` / catalog):
per-building **CVRMSE** ``NRMSE_b = 100 * sqrt(mean_{all target hours of b}(y - ŷ)^2) / mean_{all target hours of b}(y)``
(BuildingsBench ``Metric('cvrmse')``, original kWh units), the **median over the buildings of each category**
(residential / commercial, ``np.median`` as upstream) and **balanced NRMSE (%) = 0.5 * (residential median +
commercial median)**. Reference: previous-day persistence ``ŷ[i, h] = context[i, 144 + h]`` (BuildingsBench
``CopyLastDayPersistence``). Acceptance: ``balanced NRMSE <= (1 - ACCEPT_MARGIN) * reference``. The stronger
published 7-day hour-of-day mean (``AveragePersistence``) is reported as ``avg7_balanced_nrmse_pct`` (metric only, it
does not enter the acceptance): in the BuildingsBench paper it is 0.82x of the previous-day baseline.

v1 -> v2: v1 was LCL-only residential (uniform building mean of NRMSE, 2013 grid windows, SMART OOD from CSVs,
"unselected LCL buildings" as training data); v2 adds commercial buildings, category medians, per-building
adaptation history, and takes all building/window membership from the frozen role files.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

import scilib
from ...core.schema import PortSchema
from ...core.trace import Trace
from ..task import EvalResult, Episode, ToolSpec
from ._delivery_roles import Delivery, DeliveryError, file_sha256, resolve_delivery
from ._forecast_common import (ForecastAdapterBase, PoolExhausted, PoolItem, c_declared_unit, c_finite, c_range,
                               c_shape, check_split, common_lineage, default_budget, episode_id, finish_eval,
                               invalid_eval, make_rng, to_float_array, unwrap_payloads)

CODE = "FoR33"
CONTEXT_H = 168
HORIZON_H = 24
MAX_KWH = 1300.0              # generous physical bound (largest hourly value in the delivery is ~ 0.7 MWh)
ACCEPT_MARGIN = 0.10
MIN_DELAY_H = 144             # min hours between the end of one window's target and the start of the next context
UNIT = "kWh"
M_TARGET = {"src": 1, "val": 2, "id": 2, "ood": 2}     # windows per building and episode (upper bound)
CATEGORIES = ("residential", "commercial")
DEV_TARGET_OFFSETS = (0, 192, 384, 576)                # target starts inside the internal-validation block
RESERVED_UNITS = ("MT_070",)                           # reserved validation target-group building: in no role
ROLE_OF_SPLIT = {"src": "source", "val": "val", "id": "id", "ood": "ood"}
NPZ_FILES = ("source.npz", "validation.npz", "evaluation.npz")

DATASET = "BuildingsBench (real buildings: LCL, IDEAL, Borealis, BDG-2 / SMART, Electricity; reconstructed_v2)"
VERSION = "official S3 v1.0.0 sources; data team reconstructed_v2 (seed 20260928)"
SOURCE = "https://github.com/NatLabRockies/BuildingsBench ; https://oedi-data-lake.s3.amazonaws.com/buildings-bench/v1.0.0/BuildingsBench/"
LICENSE = "https://data.openei.org/submissions/5859 (CC-BY-4.0; LCL: CC BY 4.0, SMART: UMass Trace Repository terms)"


# --------------------------------------------------------------------------------------------- data model
@dataclass(frozen=True)
class Window:
    id: str
    building: str
    category: str
    group: str                 # dataset_group of the delivery (lcl, ideal, bdg-2:bear, smart, electricity, ...)
    context: np.ndarray        # (168,) kWh
    target: np.ndarray         # (24,) kWh
    context_start: str         # ISO timestamps as in the source data
    target_start: str


@dataclass
class Building:
    id: str
    category: str
    group: str
    role: str                  # source / val / id / ood
    history: np.ndarray        # (n_train,) kWh: the internal-fitting block of the adaptation history
    history_start: str
    dev_context: np.ndarray    # (n_dev, 168)
    dev_target: np.ndarray     # (n_dev, 24)
    dev_context_start: list[str]
    dev_target_start: list[str]
    window_ids: list[str]
    adaptation_sha256: str


@dataclass
class BBData:
    delivery: Delivery
    windows: dict[str, Window]
    buildings: dict[str, Building]
    pools: dict[str, list[str]]                  # src/val/id/ood -> window ids
    split_buildings: dict[str, list[str]]
    history_hours: int


def _hour(s: str) -> np.datetime64:
    return np.datetime64(str(s).strip().replace(" ", "T"), "h")


def _hours(s: str) -> int:
    return int(_hour(s).astype("int64"))


def _iso(t: np.datetime64) -> str:
    return str(np.datetime64(t, "s"))


def reserved_units(delivery: Delivery) -> tuple[str, ...]:
    """Reserved buildings named by the frozen partition config (plus the documented constant)."""
    out = set(RESERVED_UNITS)
    cfg = (delivery.entry.get("latest_partition_config") or {}).get("path")
    if cfg:
        p = delivery.anchor(cfg)
        try:
            j = json.loads(p.read_text())
            out |= set(j.get("finite_population_limits", {}).get("reserved_validation_target_group_buildings", []))
        except (OSError, json.JSONDecodeError):
            pass
    return tuple(sorted(out))


def _role_units(delivery: Delivery) -> dict[str, list[dict]]:
    """Role rows per split key (src/val/id/ood) with the structural checks that need no array data."""
    reserved = set(reserved_units(delivery))
    rows: dict[str, list[dict]] = {}
    seen_b: dict[str, str] = {}
    seen_w: dict[str, str] = {}
    for split, role in ROLE_OF_SPLIT.items():
        rs = delivery.rows(role)
        if not rs:
            raise DeliveryError(f"{role}.jsonl has no rows")
        for r in rs:
            u = str(r["unit_id"])
            if u in reserved:
                raise DeliveryError(f"reserved building {u} occurs in role {role}")
            if u in seen_b:
                raise DeliveryError(f"building {u} occurs in roles {seen_b[u]} and {split}")
            seen_b[u] = split
            if r["category"] not in CATEGORIES:
                raise DeliveryError(f"{u}: unexpected category {r['category']!r}")
            if not r.get("windows"):
                raise DeliveryError(f"{u}: no windows")
            for w in r["windows"]:
                if w["window_id"] in seen_w:
                    raise DeliveryError(f"window {w['window_id']} occurs in roles {seen_w[w['window_id']]} and {split}")
                seen_w[w["window_id"]] = split
        rows[split] = rs
    return rows


def _load_windows(delivery: Delivery, rows: dict[str, list[dict]]) -> dict[str, Window]:
    arrays: dict[str, dict[str, Any]] = {}
    for fname in NPZ_FILES:
        with np.load(delivery.base / fname, allow_pickle=False) as d:
            ctx, tgt = d["context"].astype(float), d["target"].astype(float)
            if ctx.shape[1] != CONTEXT_H or tgt.shape[1] != HORIZON_H:
                raise DeliveryError(f"{fname}: unexpected shapes {ctx.shape} {tgt.shape}")
            for i, wid in enumerate(d["window_id"]):
                arrays[str(wid)] = {"context": ctx[i].copy(), "target": tgt[i].copy(), "building": str(d["building_id"][i]),
                                    "category": str(d["category"][i]), "group": str(d["dataset_group"][i]),
                                    "target_ts": d["target_timestamp"][i]}
    out: dict[str, Window] = {}
    for split, rs in rows.items():
        for r in rs:
            for w in r["windows"]:
                a = arrays.get(w["window_id"])
                if a is None:
                    raise DeliveryError(f"window {w['window_id']} of {r['unit_id']} not found in the delivery npz files")
                if not (a["building"] == r["unit_id"] == w["building_id"] and a["category"] == r["category"]
                        and a["group"] == r["dataset_group"]):
                    raise DeliveryError(f"{w['window_id']}: building/category/group differ between role row and npz")
                t0 = _hour(w["target_start"])
                if a["target_ts"][0].astype("datetime64[h]") != t0 or _hour(w["context_start"]) + CONTEXT_H != t0:
                    raise DeliveryError(f"{w['window_id']}: timestamps differ between role row and npz")
                if not (np.all(np.isfinite(a["context"])) and np.all(np.isfinite(a["target"]))
                        and a["context"].min() >= 0 and a["target"].min() >= 0 and a["target"].mean() > 0):
                    raise DeliveryError(f"{w['window_id']}: non-finite / negative / all-zero values")
                if max(a["context"].max(), a["target"].max()) > MAX_KWH:
                    raise DeliveryError(f"{w['window_id']}: value above the {MAX_KWH:g} kWh bound")
                out[w["window_id"]] = Window(w["window_id"], r["unit_id"], r["category"], r["dataset_group"],
                                             a["context"], a["target"], str(w["context_start"]), str(w["target_start"]))
    return out


def _load_building(delivery: Delivery, split: str, row: dict, windows: dict[str, Window]) -> Building:
    ad = row["adaptation"]
    path = delivery.anchor(ad["path"])
    if not path.is_file():
        raise DeliveryError(f"missing adaptation file {path}")
    if file_sha256(path) != ad["sha256"]:
        raise DeliveryError(f"{path.name}: sha256 differs from the role file")
    with np.load(path, allow_pickle=False) as d:
        load = d["load"].astype(float)
        ts = d["timestamp"].astype("datetime64[h]")
        mask = d["internal_train_mask"].astype(bool)
    n_train = int(mask.sum())
    if not (load.shape == ts.shape == mask.shape == (int(ad["hours"]),) and np.all(np.isfinite(load))
            and load.min() >= 0 and mask[:n_train].all() and not mask[n_train:].any()
            and np.all(np.diff(ts).astype(int) == 1)):
        raise DeliveryError(f"{path.name}: unexpected layout (hours / hourly grid / internal_train_mask prefix)")
    ctx_starts = [_hour(w["context_start"]) for w in row["windows"]]
    if not ts[-1] < min(ctx_starts):
        raise DeliveryError(f"{row['unit_id']}: adaptation history overlaps the evaluation contexts")
    n_val = load.size - n_train
    dev_c, dev_t, dev_cs, dev_ts = [], [], [], []
    for off in DEV_TARGET_OFFSETS:
        t = n_train + off
        if t + HORIZON_H > load.size or t - CONTEXT_H < 0:
            continue
        dev_c.append(load[t - CONTEXT_H:t])
        dev_t.append(load[t:t + HORIZON_H])
        dev_cs.append(_iso(ts[t - CONTEXT_H]))
        dev_ts.append(_iso(ts[t]))
    if len(dev_t) < 2 or n_val < 24:
        raise DeliveryError(f"{row['unit_id']}: internal-validation block too short for dev windows")
    dc, dt = np.stack(dev_c), np.stack(dev_t)
    if not dt.mean() > 0:
        raise DeliveryError(f"{row['unit_id']}: dev targets are all zero")
    return Building(row["unit_id"], row["category"], row["dataset_group"], split, load[:n_train].copy(), _iso(ts[0]), dc, dt,
                    dev_cs, dev_ts, [w["window_id"] for w in row["windows"]], ad["sha256"])


def load_buildingsbench(delivery: Delivery) -> BBData:
    rows = _role_units(delivery)
    windows = _load_windows(delivery, rows)
    buildings: dict[str, Building] = {}
    for split, rs in rows.items():
        for r in rs:
            buildings[r["unit_id"]] = _load_building(delivery, split, r, windows)
    hist = {b.history.size for b in buildings.values()}
    if len(hist) != 1:
        raise DeliveryError(f"buildings have different history lengths {sorted(hist)}")
    pools = {s: [w["window_id"] for r in rs for w in r["windows"]] for s, rs in rows.items()}
    return BBData(delivery=delivery, windows=windows, buildings=buildings, pools=pools,
                  split_buildings={s: [r["unit_id"] for r in rs] for s, rs in rows.items()},
                  history_hours=hist.pop())


# --------------------------------------------------------------------------------------------- metric
def nrmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-building CVRMSE (%) over all given target hours: 100 * RMSE / mean(y)."""
    y = np.asarray(y_true, float)
    m = float(np.mean(y))
    if not m > 0:
        raise ValueError("mean of the observed target is not positive (CVRMSE undefined)")
    return float(100.0 * np.sqrt(np.mean((y - np.asarray(y_pred, float)) ** 2)) / m)


def per_building_nrmse(y_true: np.ndarray, y_pred: np.ndarray, buildings: list[str]) -> dict[str, float]:
    b_arr = np.asarray(buildings)
    return {b: nrmse_pct(y_true[b_arr == b], y_pred[b_arr == b]) for b in sorted(set(buildings))}


def category_medians(per: dict[str, float], cats: dict[str, str]) -> dict[str, float]:
    """Median of the per-building values within each category present (np.median, as upstream)."""
    return {c: float(np.median([v for b, v in per.items() if cats[b] == c])) for c in CATEGORIES
            if any(cats[b] == c for b in per)}


def balanced_score(per: dict[str, float], cats: dict[str, str]) -> tuple[float, dict[str, float]]:
    """balanced = 0.5 * (residential median + commercial median) (mean of the categories present)."""
    med = category_medians(per, cats)
    if not med:
        raise ValueError("no buildings")
    return float(np.mean(list(med.values()))), med


def balanced_nrmse(y_true: np.ndarray, y_pred: np.ndarray, buildings: list[str],
                   cats: dict[str, str]) -> tuple[float, dict[str, float], dict[str, float]]:
    per = per_building_nrmse(y_true, y_pred, buildings)
    score, med = balanced_score(per, cats)
    return score, per, med


def _payload(y_true: np.ndarray, y_pred: np.ndarray, buildings: list[str], cats: list[str], ids: list[str]) -> dict:
    return {"items": [{"id": ids[i], "group": buildings[i], "cat": cats[i],
                       "sse": float(np.sum((y_true[i] - y_pred[i]) ** 2)), "sum_y": float(np.sum(y_true[i])),
                       "n": int(y_true.shape[1])} for i in range(len(ids))]}


def persistence(context: np.ndarray) -> np.ndarray:
    """Previous-day persistence (BuildingsBench CopyLastDayPersistence): ŷ[i, h] = context[i, 168 - 24 + h]."""
    return np.asarray(context, float)[:, CONTEXT_H - HORIZON_H:CONTEXT_H].copy()


def average_persistence(context: np.ndarray) -> np.ndarray:
    """BuildingsBench AveragePersistence: ŷ[i, h] = mean over the 7 days of the context of the value at hour-of-day h."""
    c = np.asarray(context, float)
    return c.reshape(len(c), CONTEXT_H // HORIZON_H, HORIZON_H).mean(axis=1)


# --------------------------------------------------------------------------------------------- window packing
def _conflict(a: PoolItem, b: PoolItem, min_delay_h: int = MIN_DELAY_H) -> bool:
    """True iff two windows of one building may not share an episode.

    A window's target must end at least ``min_delay_h`` before the other window's 168-h context starts (and vice versa);
    ``min_delay_h = 0`` is the plain "no target inside the other's context" rule.  The context of a later window
    of the same building is visible in the episode, so it carries the days right after an earlier window's target.
    """
    ca, ta = a.meta[1], a.meta[2]
    cb, tb = b.meta[1], b.meta[2]
    return ((ta < cb + CONTEXT_H and cb < ta + HORIZON_H + min_delay_h)
            or (tb < ca + CONTEXT_H and ca < tb + HORIZON_H + min_delay_h))


def min_delay_observed(items: list[PoolItem]) -> int | None:
    """Smallest hours between one target end and a later context start among windows of one building (None: no pair)."""
    best = None
    by_b: dict[str, list[PoolItem]] = {}
    for it in items:
        by_b.setdefault(it.group, []).append(it)
    for its in by_b.values():
        for a, b in combinations(its, 2):
            first, second = (a, b) if a.meta[2] <= b.meta[2] else (b, a)
            gap = second.meta[1] - (first.meta[2] + HORIZON_H)
            best = gap if best is None else min(best, gap)
    return best


def pack_groups(items: list[PoolItem], m: int, cap: int, rng: np.random.Generator,
                min_delay_h: int = MIN_DELAY_H) -> list[list[PoolItem]] | None:
    """``cap`` disjoint groups of ``m`` mutually conflict-free windows (seeded exact search), or None."""
    order = [items[j] for j in rng.permutation(len(items))]
    n = len(order)
    if cap * m > n:
        return None
    ok = [[not _conflict(order[i], order[j], min_delay_h) for j in range(n)] for i in range(n)]
    groups: list[tuple[int, ...]] = []
    used: set[int] = set()

    def rec() -> bool:
        if len(groups) == cap:
            return True
        lo = groups[-1][0] + 1 if groups else 0        # groups are unordered: enumerate them by increasing first index
        avail = [i for i in range(n) if i not in used]
        for first in [i for i in avail if i >= lo]:
            rest = [i for i in avail if i > first]
            for tail in combinations(rest, m - 1):
                combo = (first,) + tail
                if all(ok[a][b] for a, b in combinations(combo, 2)):
                    groups.append(combo)
                    used.update(combo)
                    if rec():
                        return True
                    groups.pop()
                    used.difference_update(combo)
        return False

    return [[order[i] for i in g] for g in groups] if rec() else None


# --------------------------------------------------------------------------------------------- adapter
class Adapter(ForecastAdapterBase):
    """ScienceClaw-Eval FoR33 adapter (BuildingsBench day-ahead building load forecasting, reconstructed_v2)."""

    code = CODE
    name = "BuildingsBench"
    task_type = "time_series_forecasting"

    # ----------------------------------------------------------------------------------------------- files
    def delivery(self) -> Delivery:
        return self._memo.get("delivery", lambda: resolve_delivery(CODE, self.data_root))

    def _check_files(self) -> list[str]:
        try:
            dl = self.delivery()
        except DeliveryError as ex:
            return [str(ex) if "missing" in str(ex) else f"missing or invalid delivery: {ex}"]
        missing = dl.missing()
        missing += [f"missing {dl.base / f}" for f in NPZ_FILES if not (dl.base / f).is_file()]
        if missing:
            return missing
        try:
            for split, role in ROLE_OF_SPLIT.items():
                for r in dl.rows(role):
                    if not dl.anchor(r["adaptation"]["path"]).is_file():
                        missing.append(f"missing adaptation file {dl.anchor(r['adaptation']['path'])}")
        except (DeliveryError, KeyError) as ex:
            missing.append(f"missing or invalid role file content: {ex}")
        return missing

    def data(self) -> BBData:
        return self._memo.get("data", lambda: load_buildingsbench(self.delivery()))

    def _split_pools(self) -> dict[str, list[PoolItem]]:
        dl = self.delivery()
        rows = _role_units(dl)
        return {s: [PoolItem(w["window_id"], r["unit_id"],
                             (r["category"], _hours(w["context_start"]), _hours(w["target_start"])))
                    for r in rs for w in r["windows"]] for s, rs in rows.items()}

    def available(self) -> tuple[bool, str]:
        ok, why = super().available()
        if not ok:
            return False, f"{CODE}: {why}"
        pools = self.pools()
        return True, (f"{CODE} {self.delivery().version}: windows " + ", ".join(f"{s}={len(v)}" for s, v in pools.items())
                      + "; buildings " + "/".join(str(len({it.group for it in v})) for v in pools.values()))

    def verify_disjoint(self) -> None:
        """Window ids *and* physical buildings (lineage groups) are disjoint across src/val/id/ood."""
        super().verify_disjoint()
        owner: dict[str, str] = {}
        for name, items in self.pools().items():
            for b in {it.group for it in items}:
                if owner.setdefault(b, name) != name:
                    raise ValueError(f"{CODE}: building {b} in pools {owner[b]} and {name}")

    # ----------------------------------------------------------------------------------------------- planning
    def _building_items(self, base: str) -> dict[str, list[PoolItem]]:
        out: dict[str, list[PoolItem]] = {}
        for it in self.pools()[base]:
            out.setdefault(it.group, []).append(it)
        return out

    def _per_building(self, base: str, ipe: int) -> int:
        if ipe < 1:
            raise ValueError("items_per_episode must be >= 1")
        n_b = len(self._building_items(base))
        return max(1, min(M_TARGET[base], ipe // n_b))

    def _cap(self, base: str, m: int) -> int:
        """Number of window-disjoint, conflict-free episodes (min over the buildings of the exact packing bound)."""
        def compute() -> int:
            caps = []
            for b, its in sorted(self._building_items(base).items()):
                cap = len(its) // m
                while cap > 1 and pack_groups(its, m, cap, make_rng(CODE, "cap", b, m)) is None:
                    cap -= 1
                caps.append(cap)
            return min(caps)
        return self._memo.get(("cap", base, m), compute)

    def _plan(self, base: str, m: int, seed: int, ipe: int, cycle: int) -> list[list[PoolItem]]:
        return self._memo.get(("plan", base, m, seed, ipe, cycle), lambda: self._plan_uncached(base, m, seed, ipe, cycle))

    def _plan_uncached(self, base: str, m: int, seed: int, ipe: int, cycle: int) -> list[list[PoolItem]]:
        cap = self._cap(base, m)
        packs: dict[str, list[list[PoolItem]]] = {}
        for b, its in sorted(self._building_items(base).items()):
            got = None
            for attempt in range(8):
                got = pack_groups(its, m, cap, make_rng(CODE, "pack", base, seed, ipe, cycle, b, attempt))
                if got is not None:
                    break
            if got is None:                                          # cannot happen when cap came from _cap
                raise PoolExhausted(f"{CODE}/{base}: no conflict-free packing of {b} into {cap} x {m} windows")
            packs[b] = got
        shuffle_base = int(make_rng(CODE, "shuffle", base, seed, ipe, cycle).integers(0, 2**62))
        eps: list[list[PoolItem]] = []
        for e in range(cap):
            ep = [it for b in sorted(packs) for it in packs[b][e]]
            perm = np.random.default_rng([shuffle_base, e]).permutation(len(ep))
            eps.append([ep[j] for j in perm])
        return eps

    def capacity(self, split: str, items_per_episode: int = 16) -> int:
        base = "src" if split == "rep" else split
        return self._cap(base, self._per_building(base, int(items_per_episode)))

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list:
        check_split(split)
        base = "src" if split == "rep" else split
        ipe = int(items_per_episode)
        if n <= 0:
            return []
        m = self._per_building(base, ipe)
        cap = self._cap(base, m)
        if n > cap and base != "src":
            raise PoolExhausted(f"{CODE}/{split}: requested {n} episodes but the role holds {cap} window-disjoint, "
                                f"conflict-free episodes of {m} window(s) x {len(self._building_items(base))} buildings")
        out = []
        for k in range(int(n)):
            cycle, e = divmod(k, cap)
            out.append(self._make_episode(split, k, self._plan(base, m, int(seed), ipe, cycle)[e], int(seed), cycle > 0))
        return out

    # ----------------------------------------------------------------------------------------------- episode
    def _make_episode(self, split: str, k: int, items: list[PoolItem], seed: int, reused: bool) -> Episode:
        d = self.data()
        ws = [d.windows[it.id] for it in items]
        n = len(ws)
        ids = [w.id for w in ws]
        blds = [w.building for w in ws]
        cat_of = {b: d.buildings[b].category for b in blds}
        item_cats = [cat_of[b] for b in blds]
        bids = sorted(set(blds))
        ctx = np.stack([w.context for w in ws])
        tgt = np.stack([w.target for w in ws])
        ref_pred = persistence(ctx)
        ref_score, _, _ = balanced_nrmse(tgt, ref_pred, blds, cat_of)
        ref_payload = _payload(tgt, ref_pred, blds, item_cats, ids)
        avg7_score, _, _ = balanced_nrmse(tgt, average_persistence(ctx), blds, cat_of)
        pool = "ood" if split == "ood" else "iid"

        # ------------------------------------------------------------------ D_E tools (visible data only)
        hist = np.stack([d.buildings[b].history for b in bids])
        hist_start = [d.buildings[b].history_start for b in bids]
        hist_cat = [cat_of[b] for b in bids]
        n_h = hist.shape[1]
        dev_ctx = np.concatenate([d.buildings[b].dev_context for b in bids])
        dev_tgt = np.concatenate([d.buildings[b].dev_target for b in bids])
        dev_b = [b for b in bids for _ in range(len(d.buildings[b].dev_target))]
        dev_cs = [s for b in bids for s in d.buildings[b].dev_context_start]
        dev_ts = [s for b in bids for s in d.buildings[b].dev_target_start]
        dev_cat = [cat_of[b] for b in dev_b]
        n_dev = dev_ctx.shape[0]
        dev_ref, _, _ = balanced_nrmse(dev_tgt, persistence(dev_ctx), dev_b, cat_of)
        ctx_start = [w.context_start for w in ws]
        tgt_start = [w.target_start for w in ws]

        def load_history(inputs: dict, config: dict) -> dict:
            return {"load": hist.copy(), "building_id": list(bids), "category": list(hist_cat),
                    "history_start": list(hist_start)}

        def load_dev(inputs: dict, config: dict) -> dict:
            return {"context": dev_ctx.copy(), "building_id": list(dev_b), "category": list(dev_cat),
                    "context_start": list(dev_cs), "target_start": list(dev_ts)}

        def score_dev(inputs: dict, config: dict) -> dict:
            pred = to_float_array(inputs.get("pred"))
            if pred.shape != (n_dev, HORIZON_H):
                raise ValueError(f"pred must have shape ({n_dev}, {HORIZON_H}), got {tuple(pred.shape)}")
            if not np.all(np.isfinite(pred)):
                raise ValueError("pred contains non-finite values")
            s, per, med = balanced_nrmse(dev_tgt, pred, dev_b, cat_of)
            return {"score": s, "report": {"metric": "balanced NRMSE (%) on the dev windows (lower is better)",
                                           "score": s, "reference_score": dev_ref, "per_building": per,
                                           "category_medians": med, "n_items": n_dev}}

        def load_eval_inputs(inputs: dict, config: dict) -> dict:
            return {"context": ctx.copy(), "building_id": list(blds), "category": list(item_cats),
                    "context_start": list(ctx_start), "target_start": list(tgt_start)}

        tools = [
            ToolSpec("load_history",
                     f"Permitted historical adaptation data of the {len(bids)} buildings of this episode: load[b, t] = "
                     f"kWh consumed by building_id[b] in the hour starting history_start[b] + t hours, t = 0..{n_h - 1} "
                     "(hourly, consecutive; the history ends before every evaluation context). Use it to adapt to a "
                     "building; reset any adapted state for each building.",
                     {}, {"load": PortSchema("array", (len(bids), n_h), UNIT, "float",
                                             "hourly consumption per building (internal-fitting block)"),
                          "building_id": PortSchema("list", (len(bids),), None, "str"),
                          "category": PortSchema("list", (len(bids),), None, "str", "residential / commercial"),
                          "history_start": PortSchema("list", (len(bids),), None, "str", "ISO hour of load[b, 0]")},
                     load_history),
            ToolSpec("load_dev",
                     f"Dev inputs: {n_dev} windows ({n_dev // len(bids)} per building) whose 24-hour targets lie in the "
                     "720 hours that follow the load_history block of the same buildings (contexts and format as in "
                     "load_eval_inputs plus target_start; the targets are held by score_dev; dev target days are 8 days apart).",
                     {}, {"context": PortSchema("array", (n_dev, CONTEXT_H), UNIT, "float"),
                          "building_id": PortSchema("list", (n_dev,), None, "str"),
                          "category": PortSchema("list", (n_dev,), None, "str"),
                          "context_start": PortSchema("list", (n_dev,), None, "str"),
                          "target_start": PortSchema("list", (n_dev,), None, "str")},
                     load_dev),
            ToolSpec("score_dev",
                     "Scores a forecast of the dev windows: returns their balanced NRMSE (%) (lower is better) and, "
                     "in report, the same statistic of the reference forecast.",
                     {"pred": PortSchema("array", (n_dev, HORIZON_H), UNIT, "float",
                                         "forecast for the dev windows, same layout as y")},
                     {"score": PortSchema("number", None, "%", "float"), "report": PortSchema("dict")},
                     score_dev),
            ToolSpec("load_eval_inputs",
                     f"Evaluation inputs (no targets) for the {n} items in item order: context[i] = 168 consecutive "
                     "hourly consumption values (kWh) of the item's building ending at the hour before "
                     "target_start[i]; building_id[i]; category[i]; context_start[i] / target_start[i] = ISO timestamps.",
                     {}, {"context": PortSchema("array", (n, CONTEXT_H), UNIT, "float"),
                          "building_id": PortSchema("list", (n,), None, "str"),
                          "category": PortSchema("list", (n,), None, "str"),
                          "context_start": PortSchema("list", (n,), None, "str"),
                          "target_start": PortSchema("list", (n,), None, "str")},
                     load_eval_inputs),
        ]

        # ------------------------------------------------------------------ D_V constraints + evaluator
        ctx_max = ctx.max(axis=1)

        def upper(arr: np.ndarray) -> np.ndarray:
            if arr.shape != (n, HORIZON_H):
                raise ValueError(f"shape {tuple(arr.shape)} != ({n}, {HORIZON_H})")
            return np.maximum(20.0 * ctx_max, 1.0)[:, None]

        constraints = [
            c_shape((n, HORIZON_H)),
            c_finite(),
            c_range(0.0, MAX_KWH, UNIT, upper_fn=upper,
                    upper_doc="y[i, h] <= max(20 * max(context[i]), 1 kWh) (unit/scale guard)"),
            c_declared_unit(UNIT),
        ]

        def evaluate(y: Any, trace: Trace | None) -> EvalResult:
            common = {"reference": ref_score, "direction": "min", "margin": ACCEPT_MARGIN,
                      "reference_payload": ref_payload}
            ref_m = {"reference_balanced_nrmse_pct": ref_score, "avg7_balanced_nrmse_pct": avg7_score}
            try:
                arr = to_float_array(y)
            except ValueError as ex:
                return invalid_eval(str(ex), metrics=ref_m, **common)
            if arr.shape != (n, HORIZON_H):
                return invalid_eval(f"shape {tuple(arr.shape)} != ({n}, {HORIZON_H})", metrics=ref_m, **common)
            if not np.all(np.isfinite(arr)):
                return invalid_eval("non-finite predictions", metrics=ref_m, **common)
            score, per, med = balanced_nrmse(tgt, arr, blds, cat_of)
            metrics = {"balanced_nrmse_pct": score, "pooled_nrmse_pct": nrmse_pct(tgt, arr),
                       "rmse_kwh": float(np.sqrt(np.mean((tgt - arr) ** 2))), "mae_kwh": float(np.mean(np.abs(tgt - arr))),
                       **{f"{c}_median_nrmse_pct": v for c, v in med.items()}, **ref_m}
            return finish_eval(primary=score, metrics=metrics, payload=_payload(tgt, arr, blds, item_cats, ids),
                               extra={"per_building_nrmse_pct": per, "category_median_nrmse_pct": med}, **common)

        m_per = n // len(bids)
        n_res = sum(1 for b in bids if cat_of[b] == "residential")
        objective = (
            "Built environment / building energy: day-ahead forecasting of hourly electricity consumption of real "
            f"buildings (BuildingsBench). The episode has {len(bids)} buildings ({n_res} residential, "
            f"{len(bids) - n_res} commercial) and {n} evaluation items: {m_per} window(s) per building. Input of item i "
            "(tool load_eval_inputs): context[i], the 168 consecutive hourly consumption values in kWh (energy used "
            "during each hour) ending immediately before target_start[i], plus the building id, its category and ISO "
            "timestamps. Target of item i: the 24 hourly consumption values (kWh) of the same building for the 24 hours "
            "starting at target_start[i]. Each building's own earlier history (3601 hours, tool load_history, which "
            "ends before every evaluation context) may be used to adapt to that building; per-building adapted state "
            "must be reset for every building. Dev windows whose targets lie in the following 720 hours are in "
            "load_dev/score_dev.\n"
            f"Deliverable y: a float array of shape ({n}, {HORIZON_H}) in kWh; row i is the forecast for item i in "
            "the order of load_eval_inputs, column h (0-based) is the forecast for the hour starting h hours after "
            f"target_start[i]. Values must be finite, >= 0 and <= {MAX_KWH:g}.\n"
            "Score (lower is better): balanced NRMSE (%) = 0.5 * (residential median + commercial median) where, for each "
            "building, NRMSE_b = 100 * sqrt(mean squared error over all of that building's target hours in the episode) "
            "/ (mean observed consumption of that building over the same hours) and each median is taken over the "
            "buildings of that category.\n"
            + scilib.describe("loadforecast") + scilib.describe_extra("tsfm")
        )
        dl = d.delivery
        role = ROLE_OF_SPLIT["src" if split == "rep" else split]
        lineage = common_lineage(
            dataset=DATASET, version=VERSION, source=SOURCE, license_=LICENSE, pool=pool,
            ood_kind="proxy_within_dataset" if pool == "ood" else None, items=items, seed=seed,
            partition_seed=self.partition_seed, k=k, reused=reused,
            split_rule=(f"role files of the {dl.version} delivery (physical-building-disjoint: src 8 / val 3 / id 11 / "
                        "ood 5 buildings); an episode = every building of the role with m window-disjoint, "
                        "conflict-free windows each (no target inside another item's context of the same building and "
                        f"at least {MIN_DELAY_H} h between a target's end and the other window's context start); "
                        "OOD = held-out sub-dataset group shift (SMART residential, Electricity commercial)"),
            visible=(f"load_history: {n_h} h per building of {bids} (adaptation history, internal-fitting block); "
                     f"load_dev: {n_dev} windows from the next 720 h; per-item 168-h context, category"),
            extra={"data_version": dl.version, "role": role, "role_file_sha256": dl.role_sha256[role],
                   "catalog_sha256": file_sha256(dl.catalog_path), "buildings": bids,
                   "building_categories": {b: cat_of[b] for b in bids},
                   "building_groups": {b: d.buildings[b].group for b in bids}, "windows_per_building": m_per,
                   "min_window_delay_h": MIN_DELAY_H, "min_window_delay_observed_h": min_delay_observed(items),
                   "history_hours": n_h, "dev_windows": n_dev, "context_hours": CONTEXT_H, "horizon_hours": HORIZON_H,
                   "reserved_buildings_excluded": list(reserved_units(dl)),
                   "metric_convention": "0.5*(median residential CVRMSE% + median commercial CVRMSE%)",
                   "newer_reconstructed_dirs_on_disk": list(dl.newer_dirs)})
        return Episode(
            id=episode_id(CODE, split, seed, k), discipline=CODE, family=self.family, split=split,
            task_type=self.task_type, objective=objective,
            required_output=PortSchema("array", (n, HORIZON_H), UNIT, "float",
                                       "day-ahead hourly load forecast per evaluation item"),
            tools=tools, constraints=constraints, budget=default_budget(max_node_s=300.0, max_llm_items=2 * n),
            lineage=lineage,
            acceptance=(f"accepted iff balanced NRMSE <= {1 - ACCEPT_MARGIN:.2f} x the reference balanced NRMSE on the "
                        "same items (reference recipe and value only in EvalResult.details / docs, never shown to the "
                        "policy)"),
            tolerance={"rtol": 1e-5, "atol": 1e-6},
            tags=[CODE, self.family, "time-series", "forecasting", "energy", "building-load", "hourly", "day-ahead",
                  UNIT, "BuildingsBench"],
            metric=self.metric, direction="min", n_items=n, _evaluate=evaluate, _dev_evaluate=None)

    # ----------------------------------------------------------------------------------------------
    def pooled_metric(self, per_episode: list[dict]) -> float | None:
        """Balanced NRMSE (%) over all items of the given episodes: per-building CVRMSE over all of a building's
        target hours (SSE, sum y and counts accumulated from ``pooled_payload.items``), category medians, mean."""
        acc: dict[str, list[float]] = {}
        cat: dict[str, str] = {}
        for p in unwrap_payloads(per_episode):
            for it in p.get("items", []):
                b = str(it["group"])
                a = acc.setdefault(b, [0.0, 0.0, 0.0])
                a[0] += float(it["sse"])
                a[1] += float(it["sum_y"])
                a[2] += float(it["n"])
                cat[b] = str(it["cat"])
        if not acc:
            return None
        per = {b: 100.0 * math.sqrt(sse / cnt) / (sy / cnt) for b, (sse, sy, cnt) in acc.items()}
        return balanced_score(per, cat)[0]
