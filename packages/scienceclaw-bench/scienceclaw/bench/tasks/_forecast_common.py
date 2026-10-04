"""Shared helpers of the forecasting task adapters FoR33 / FoR35 / FoR37 / FoR38 / FoR41.

Private to these five modules (the leading underscore keeps it out of the registry). It provides

* :class:`PoolItem` and **seed-independent pool partitions**: every adapter cuts its item pools into the
  disjoint sub-pools ``src`` / ``val`` / ``id`` (IID pool) and ``ood`` (OOD pool) with a fixed
  ``partition_seed`` (default 20260928, the reconstruction-policy seed). The partition never depends on the
  per-split seed passed to ``build_episodes`` (``SplitPlan`` hands a different seed to every split), so the
  splits are item-disjoint for every combination of seeds;
* **prefix-stable episode draws** (:func:`draw_episodes`): the sub-pool is ordered once per ``(split, seed)``
  and episode ``e`` takes block ``e`` of that order, so building ``n`` or ``n + 1`` episodes yields the same
  first ``n``. Optional group balancing (round-robin over e.g. buildings) or distinct groups per episode
  (e.g. one item per monitoring site). Optional **conflict rule**: a predicate that no two items of one
  episode may satisfy (e.g. initialisation times closer than a minimum spacing, so that one item's input cannot
  serve as a near-future observation of another item's target), enforced while drawing and re-checked on the result;
  ``lanes=True`` draws every episode from a single pre-built group ("lane") of the pool. Only ``src`` may reuse
  items when its sub-pool is exhausted (reuse is
  recorded in the lineage; ``SplitPlan`` reports it as a manifest warning); other splits raise
  :class:`PoolExhausted`;
* hard-constraint factories (shape, finiteness, physical range, declared unit), output coercion, the
  normalized score of DESIGN §8.6 and a uniform :func:`finish_eval` that fills ``EvalResult.details`` with
  ``reference``, ``norm_score``, ``pooled_payload`` and ``reference_payload``;
* official metric kernels: MASE (Hyndman & Koehler 2006), sMAPE (Makridakis 1993 / M4), CRPS of a normal
  predictive distribution (Gneiting & Raftery 2007), latitude-weighted RMSE (WeatherBench 2);
* the cache-directory convention ``<repo>/cache/tasks/<code>/`` (override: env ``SCIENCECLAW_TASK_CACHE``)
  and the data-root convention (constructor argument, else env ``SCIENCECLAW_DATA_ROOT``, else
  :data:`scienceclaw.bench.registry.DATA_ROOT`). Data directories are only ever read.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core.trace import Trace
from ..registry import BY_CODE, DATA_ROOT
from ..task import SPLITS, Budget, ConstraintSpec, EvalResult

PARTITION_SEED = 20260928
NORM_CLIP = 10.0
REPO_ROOT = Path(__file__).resolve().parents[3]
BUILT_SPLITS = ("src", "val", "id", "ood")


# ------------------------------------------------------------------------------------------------ paths
def resolve_data_root(data_root: str | os.PathLike | None) -> Path:
    """Constructor argument > env SCIENCECLAW_DATA_ROOT (read at call time) > registry.DATA_ROOT."""
    if data_root:
        return Path(data_root)
    return Path(os.environ.get("SCIENCECLAW_DATA_ROOT", DATA_ROOT))


def task_cache_dir(code: str, cache_root: str | os.PathLike | None = None, create: bool = True) -> Path:
    """``<repo>/cache/tasks/<code>/`` (or ``$SCIENCECLAW_TASK_CACHE/<code>`` / ``cache_root/<code>``)."""
    base = Path(cache_root) if cache_root else Path(os.environ.get("SCIENCECLAW_TASK_CACHE", REPO_ROOT / "cache" / "tasks"))
    d = base / code
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------------ hashing / rng
def stable_int(*parts: Any) -> int:
    """Deterministic 60-bit integer from the string forms of ``parts`` (sha256)."""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def make_rng(*parts: Any) -> np.random.Generator:
    """numpy PCG64 generator seeded with :func:`stable_int` of ``parts``."""
    return np.random.default_rng(stable_int(*parts))


def hash_order(ids: Sequence[str], salt: str) -> list[str]:
    """``ids`` sorted by sha256(salt|id): a seed-fixed pseudo-random order independent of input order."""
    return sorted(ids, key=lambda i: (hashlib.sha256(f"{salt}|{i}".encode()).hexdigest(), str(i)))


# ------------------------------------------------------------------------------------------------ pools
@dataclass(frozen=True)
class PoolItem:
    """One evaluation item: a stable id, its lineage group (building / series / site / country / block)."""
    id: str
    group: str
    meta: tuple = field(default=(), compare=False)


class PoolExhausted(ValueError):
    """A split's sub-pool cannot supply the requested number of item-disjoint episodes."""


def _round_robin(items: Sequence[PoolItem], rng: np.random.Generator) -> list[PoolItem]:
    groups: dict[str, list[PoolItem]] = {}
    for it in items:
        groups.setdefault(it.group, []).append(it)
    names = sorted(groups)
    names = [names[j] for j in rng.permutation(len(names))]
    lists = [[groups[g][j] for j in rng.permutation(len(groups[g]))] for g in names]
    out: list[PoolItem] = []
    k = 0
    while any(k < len(lst) for lst in lists):
        for lst in lists:
            if k < len(lst):
                out.append(lst[k])
        k += 1
    return out


Conflict = Callable[[PoolItem, PoolItem], bool]


def time_spacing_conflict(key: Callable[[PoolItem], float], min_sep: float) -> Conflict:
    """Conflict rule "closer than ``min_sep`` in ``key`` units" (e.g. init time in hours) for :func:`draw_episodes`."""
    return lambda a, b: abs(float(key(a)) - float(key(b))) < float(min_sep)


def _clashes(it: PoolItem, chosen: Sequence[PoolItem], conflict: Conflict | None) -> bool:
    return conflict is not None and any(conflict(it, o) for o in chosen)


def _distinct_group_blocks(order: Sequence[PoolItem], n_eps: int, ipe: int, distinct: bool = True,
                           conflict: Conflict | None = None) -> list[list[PoolItem]]:
    """Greedy episodes over ``order``: at most one item per group (``distinct``) and no conflicting pair.

    Stops at the first episode the conflict rule leaves unfillable and returns the complete ones only (so the
    result can be shorter than ``n_eps``; episode ``e`` never depends on how many episodes are requested).
    """
    remaining = list(order)
    eps: list[list[PoolItem]] = []
    for _ in range(n_eps):
        ep: list[PoolItem] = []
        seen: set[str] = set()
        rest: list[PoolItem] = []
        for it in remaining:
            if len(ep) < ipe and not (distinct and it.group in seen) and not _clashes(it, ep, conflict):
                ep.append(it)
                seen.add(it.group)
            else:
                rest.append(it)
        if distinct:                            # fewer groups than items per episode: fill in order
            k = 0
            while len(ep) < ipe and k < len(rest):
                if _clashes(rest[k], ep, conflict):
                    k += 1
                else:
                    ep.append(rest.pop(k))
        if len(ep) < ipe:
            break
        remaining = rest
        eps.append(ep)
    return eps


def _lane_blocks(pool: Sequence[PoolItem], ipe: int, rng: np.random.Generator) -> list[list[PoolItem]]:
    """Episodes drawn inside one group ("lane") each: ``len(lane) // ipe`` episodes per lane, in a seeded order."""
    lanes: dict[str, list[PoolItem]] = {}
    for it in pool:
        lanes.setdefault(it.group, []).append(it)
    blocks: list[list[PoolItem]] = []
    for g in sorted(lanes):
        items = [lanes[g][j] for j in rng.permutation(len(lanes[g]))]
        blocks.extend(items[c * ipe:(c + 1) * ipe] for c in range(len(items) // ipe))
    return [blocks[j] for j in rng.permutation(len(blocks))]


def draw_episodes(pool: Sequence[PoolItem], n: int, ipe: int, rng: np.random.Generator, *,
                  balance_groups: bool = False, distinct_groups: bool = False, allow_reuse: bool = False,
                  lanes: bool = False, conflict: Conflict | None = None,
                  what: str = "") -> tuple[list[list[PoolItem]], bool]:
    """Draw ``n`` episodes of ``ipe`` items from ``pool`` (prefix-stable in ``n``).

    Returns ``(episodes, reused)``; ``reused`` is True when ``allow_reuse`` had to recycle items (never within
    one episode). Without ``allow_reuse`` an over-request raises :class:`PoolExhausted`.

    ``conflict(a, b)`` marks item pairs that must never share an episode; the greedy draw skips clashing
    items and every returned episode is re-checked (a violation raises ``ValueError``). ``lanes=True`` draws each
    episode from one group of the pool (a "lane": the adapter pre-arranges items so that a whole lane is
    conflict-free; greedy selection cannot find a perfect packing in a tight pool). Lanes larger than ``ipe``
    yield several episodes, lanes smaller than ``ipe`` none.
    """
    if ipe < 1:
        raise ValueError("items_per_episode must be >= 1")
    if n <= 0:
        return [], False
    shuffle_base = int(rng.integers(0, 2**62))   # per-episode item-order shuffles (independent of n)
    pool = list(pool)
    if len(pool) < ipe:
        raise PoolExhausted(f"{what}: pool has {len(pool)} items < items_per_episode={ipe}")
    if lanes:
        sizes: dict[str, int] = {}
        for it in pool:
            sizes[it.group] = sizes.get(it.group, 0) + 1
        cap = sum(s // ipe for s in sizes.values())
        if cap == 0:
            raise PoolExhausted(f"{what}: no lane holds {ipe} items (largest lane: {max(sizes.values())}); the "
                                f"conflict-free lanes limit items_per_episode")
    else:
        cap = len(pool) // ipe
    if n > cap and not allow_reuse:
        raise PoolExhausted(f"{what}: requested {n} episodes x {ipe} items but the sub-pool holds {len(pool)} "
                            f"items (capacity {cap} item-disjoint episodes)")

    def one_cycle() -> list[list[PoolItem]]:
        if lanes:
            return _lane_blocks(pool, ipe, rng)
        if balance_groups:
            order = _round_robin(pool, rng)
        else:
            order = [pool[j] for j in rng.permutation(len(pool))]
        if distinct_groups or conflict is not None:
            return _distinct_group_blocks(order, cap, ipe, distinct=distinct_groups, conflict=conflict)
        return [order[e * ipe:(e + 1) * ipe] for e in range(cap)]

    episodes: list[list[PoolItem]] = []
    reused = False
    while len(episodes) < n:
        cycle = one_cycle()
        if not cycle or (len(episodes) + len(cycle) < n and not allow_reuse):
            raise PoolExhausted(f"{what}: requested {n} episodes x {ipe} items but only {len(episodes) + len(cycle)} "
                                f"conflict-free item-disjoint episodes fit the sub-pool of {len(pool)} items")
        if episodes:
            reused = True
        episodes.extend(cycle)
    episodes = episodes[:n]
    if conflict is not None:                       # invariant, whatever the draw mode
        for e, ep in enumerate(episodes):
            for a in range(len(ep)):
                for b in range(a + 1, len(ep)):
                    if conflict(ep[a], ep[b]):
                        raise ValueError(f"{what}: episode {e} pairs conflicting items {ep[a].id!r} and {ep[b].id!r}")
    for e, ep in enumerate(episodes):     # seeded shuffle of the item order inside each episode
        perm = np.random.default_rng([shuffle_base, e]).permutation(len(ep))
        ep[:] = [ep[j] for j in perm]
    return episodes, reused


def check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


# ------------------------------------------------------------------------------------------------ outputs
def to_float_array(y: Any) -> np.ndarray:
    """Coerce a submitted output (ndarray / nested list / pandas object) to a float ndarray.

    Raises ValueError with a precise message for dicts, strings, ragged or non-numeric content.
    """
    if y is None:
        raise ValueError("output is None")
    if isinstance(y, (str, bytes, dict)):
        raise ValueError(f"output must be a numeric array, got {type(y).__name__}")
    if hasattr(y, "to_numpy") and callable(y.to_numpy):
        y = y.to_numpy()
    try:
        arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"output is not a numeric array: {type(ex).__name__}: {ex}") from ex
    return arr


def shape_str(shape: Sequence[int]) -> str:
    return "(" + ", ".join(str(int(s)) for s in shape) + ")"


def norm_score(primary: float | None, reference: float | None, direction: str) -> float:
    """DESIGN §8.6: primary/ref ("max") or ref/primary ("min"), clipped to [0, NORM_CLIP]; 0 if undefined."""
    if primary is None or reference is None or not (math.isfinite(primary) and math.isfinite(reference)):
        return 0.0
    if direction == "max":
        v = NORM_CLIP if reference == 0 else primary / reference
    else:
        v = NORM_CLIP if primary == 0 else reference / primary
    return float(min(max(v, 0.0), NORM_CLIP))


def beats_reference(primary: float | None, reference: float, direction: str, margin: float) -> bool:
    """Acceptance: min -> primary <= (1 - margin) * reference; max -> primary >= (1 + margin) * reference."""
    if primary is None or not math.isfinite(primary) or not math.isfinite(reference):
        return False
    if direction == "min":
        return bool(primary <= (1.0 - margin) * reference)
    return bool(primary >= (1.0 + margin) * reference)


def finish_eval(*, primary: float | None, reference: float, direction: str, margin: float,
                metrics: dict[str, float], payload: Any, reference_payload: Any,
                extra: dict | None = None) -> EvalResult:
    """Uniform EvalResult with the DESIGN §8.6 details keys (h/z are filled by Episode.evaluate).

    A result without a finite primary is a *failure*: it pools the reference payload (``failed=True``) instead of
    dropping out of the pooled metric, so every held-out episode contributes exactly one payload.
    """
    ok_primary = primary is not None and math.isfinite(primary)
    details = {"reference": float(reference), "norm_score": norm_score(primary if ok_primary else None, reference,
                                                                        direction),
               "pooled_payload": payload if ok_primary else reference_payload, "reference_payload": reference_payload,
               "acceptance_margin": margin}
    if not ok_primary:
        details["failed"] = True
    if extra:
        details.update(extra)
    return EvalResult(metrics={k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))},
                      primary=float(primary) if ok_primary else None, direction=direction,
                      accepted=beats_reference(primary if ok_primary else None, reference, direction, margin),
                      details=details)


def invalid_eval(msg: str, *, reference: float, direction: str, margin: float, reference_payload: Any,
                 metrics: dict[str, float] | None = None) -> EvalResult:
    """EvalResult for a malformed output: no primary, not accepted, norm_score 0, ``failed=True`` and the *reference*
    payload as ``pooled_payload`` (a failed episode pools at reference level, never better)."""
    return finish_eval(primary=None, reference=reference, direction=direction, margin=margin,
                       metrics=dict(metrics or {}), payload=None, reference_payload=reference_payload,
                       extra={"invalid": msg})


def unwrap_payloads(per_episode: Sequence[Any]) -> list[dict]:
    """Accept pooled payload dicts, or EvalResult.details dicts that contain ``pooled_payload``; drop None."""
    out: list[dict] = []
    for p in per_episode:
        if p is None:
            continue
        if isinstance(p, dict) and "pooled_payload" in p:
            p = p["pooled_payload"]
            if p is None:
                continue
        if not isinstance(p, dict):
            raise ValueError(f"pooled payload must be a dict, got {type(p).__name__}")
        out.append(p)
    return out


# ------------------------------------------------------------------------------------------------ constraints
def c_shape(expected: tuple[int, ...], what: str = "y") -> ConstraintSpec:
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        try:
            arr = to_float_array(y)
        except ValueError as ex:
            return False, str(ex)
        if tuple(arr.shape) != tuple(expected):
            return False, f"{what} has shape {shape_str(arr.shape)}, required {shape_str(expected)}"
        return True, f"shape {shape_str(expected)}"
    return ConstraintSpec("output_shape", f"{what} is a numeric array of shape {shape_str(expected)} aligned with the "
                                          f"evaluation items", check)


def c_finite(what: str = "y") -> ConstraintSpec:
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        try:
            arr = to_float_array(y)
        except ValueError as ex:
            return False, str(ex)
        bad = int((~np.isfinite(arr)).sum())
        return bad == 0, ("all values finite" if bad == 0 else f"{bad} non-finite values")
    return ConstraintSpec("finite", f"every value of {what} is a finite real number", check)


def c_range(lo: float | None, hi: float | None, unit: str, *, name: str = "physical_range",
            upper_fn: Callable[[np.ndarray], np.ndarray] | None = None, upper_doc: str = "",
            desc: str = "") -> ConstraintSpec:
    """lo <= y <= hi elementwise (finite entries); ``upper_fn(arr)`` may give an additional per-element bound."""
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        try:
            arr = to_float_array(y)
        except ValueError as ex:
            return False, str(ex)
        fin = arr[np.isfinite(arr)]
        if lo is not None and fin.size and fin.min() < lo:
            return False, f"minimum {fin.min():.6g} {unit} < lower bound {lo:g} {unit}"
        if hi is not None and fin.size and fin.max() > hi:
            return False, f"maximum {fin.max():.6g} {unit} > upper bound {hi:g} {unit}"
        if upper_fn is not None:
            try:
                ub = upper_fn(arr)
            except ValueError as ex:
                return False, str(ex)
            viol = np.isfinite(arr) & (arr > ub)
            if viol.any():
                return False, f"{int(viol.sum())} values exceed the plausibility bound ({upper_doc})"
        return True, "within range"
    text = desc or (f"all values within [{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}] {unit}"
                    + (f" and {upper_doc}" if upper_doc else ""))
    return ConstraintSpec(name, text, check)


def c_declared_unit(required_unit: str) -> ConstraintSpec:
    """The submitted value's declared unit (if the trace records one for the submit node) equals the required unit.

    The submit port's schema *is* the required output schema, and Compat on the incoming edge already rejects a
    unit mismatch without an explicit conversion; this check additionally inspects the trace summaries.
    """
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        if trace is None:
            return True, f"no trace; unit {required_unit!r} enforced by Compat on the submit edge"
        for rec in trace.records.values():
            if getattr(rec, "kind", "") != "submit":
                continue
            for port, summ in (rec.outputs_summary or {}).items():
                u = summ.get("unit") if isinstance(summ, dict) else None
                if u is not None and u != required_unit:
                    return False, f"submit port {port!r} declares unit {u!r}, required {required_unit!r}"
        return True, f"unit {required_unit!r}"
    return ConstraintSpec("declared_unit", f"the submitted output is expressed in {required_unit!r} "
                                           "(port unit must match; conversions must be explicit)", check)


# ------------------------------------------------------------------------------------------------ metric kernels
def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, m: int) -> float:
    """Mean absolute scaled error (Hyndman & Koehler 2006; Monash archive convention).

    MASE = mean_h |y_h - ŷ_h| / ( (1/(T-m)) Σ_{t=m+1..T} |x_t - x_{t-m}| ), scale from the in-sample series x.
    """
    x = np.asarray(insample, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= m:
        raise ValueError(f"in-sample length {x.size} <= seasonal period {m}")
    scale = float(np.mean(np.abs(x[m:] - x[:-m])))
    if scale <= 0:
        raise ValueError("MASE scale is zero (constant seasonal differences)")
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))) / scale)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent (M4 definition): 200/H Σ |y - ŷ| / (|y| + |ŷ|); a 0/0 term counts as 0."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    den = np.abs(y) + np.abs(p)
    num = np.abs(y - p)
    terms = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return float(200.0 * np.mean(terms))


_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def crps_normal(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting & Raftery 2007; = scoringRules::crps_norm).

    CRPS = sigma * [ z (2 Φ(z) - 1) + 2 φ(z) - 1/sqrt(pi) ],  z = (y - mu) / sigma, sigma > 0.
    """
    from scipy.special import ndtr

    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    y = np.asarray(y, float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be > 0")
    z = (y - mu) / sigma
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * pdf - _INV_SQRT_PI)


def wb2_lat_weights(lat_deg: np.ndarray) -> np.ndarray:
    """WeatherBench 2 area weights for an equiangular grid: w ∝ sin(φ+Δ/2) - sin(φ-Δ/2), mean 1."""
    lat = np.asarray(lat_deg, float)
    d = float(np.abs(np.diff(lat)).mean())
    up = np.deg2rad(np.clip(lat + d / 2, -90.0, 90.0))
    lo = np.deg2rad(np.clip(lat - d / 2, -90.0, 90.0))
    w = np.sin(up) - np.sin(lo)
    return w / w.mean()


# ------------------------------------------------------------------------------------------------ misc
class LockedCache:
    """A tiny thread-safe memo: ``get(key, factory)`` computes ``factory()`` once per key."""

    def __init__(self) -> None:
        self._lock = threading.RLock()     # re-entrant: factories may use the memo too
        self._store: dict[Any, Any] = {}

    def get(self, key: Any, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._store:
                self._store[key] = factory()
            return self._store[key]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def registry_attrs(code: str) -> dict[str, str]:
    d = BY_CODE[code]
    return {"discipline": d.code, "family": d.family, "metric": d.metric, "direction": d.direction}


def default_budget(max_node_s: float = 300.0, max_llm_items: int = 32) -> Budget:
    """Numeric forecasting tasks: default steps/tokens, generous node timeout, small LLM-item allowance."""
    return Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=1800.0, max_node_s=max_node_s,
                  max_llm_items=max_llm_items)


def episode_id(code: str, split: str, seed: int, k: int) -> str:
    return f"{code}-{split}-s{int(seed) & 0xFFFFFFFF:08x}-e{k:02d}"


def common_lineage(*, dataset: str, version: str, source: str, license_: str, pool: str, ood_kind: str | None,
                   items: Sequence[PoolItem], seed: int, partition_seed: int, k: int, reused: bool,
                   split_rule: str, visible: str, extra: dict | None = None) -> dict:
    lin = {"dataset": dataset, "version": version, "source": source, "license": license_, "pool": pool,
           "ood_kind": ood_kind, "item_ids": [it.id for it in items], "groups": sorted({it.group for it in items}),
           "seed": int(seed), "partition_seed": int(partition_seed), "episode_index": int(k),
           "items_reused_from_earlier_episodes": bool(reused), "split_rule": split_rule, "visible_data": visible,
           "rebuilt_split": True, "historical_sample_ids_recovered": False}
    if extra:
        lin.update(extra)
    return lin


# ------------------------------------------------------------------------------------------------ adapter base
class ForecastAdapterBase:
    """Common skeleton of the five forecasting adapters.

    Subclasses set the class attributes ``code`` / ``name`` / ``task_type`` and implement

    * ``_check_files() -> list[str]``: missing/invalid inputs (empty list = data present);
    * ``_split_pools() -> dict[str, list[PoolItem]]`` with keys ``src``, ``val``, ``id``, ``ood`` (disjoint);
    * ``_draw_options(split) -> dict`` (keyword arguments of :func:`draw_episodes`);
    * ``_make_episode(split, k, items, seed, reused) -> Episode``;
    * ``pooled_metric(per_episode) -> float | None``.
    """

    code: str = ""
    name: str = ""
    task_type: str = "time_series_forecasting"
    discipline: str = ""
    family: str = ""
    metric: str = ""
    direction: str = "min"

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if cls.code:          # class-level protocol attributes straight from the registry
            for k, v in registry_attrs(cls.code).items():
                setattr(cls, k, v)

    def __init__(self, data_root: str | os.PathLike | None = None, cache_root: str | os.PathLike | None = None,
                 partition_seed: int = PARTITION_SEED, **options: Any) -> None:
        attrs = registry_attrs(self.code)
        self.discipline = attrs["discipline"]
        self.family = attrs["family"]
        self.metric = attrs["metric"]
        self.direction = attrs["direction"]
        self.data_root = resolve_data_root(data_root)
        self.cache_root = cache_root
        self.partition_seed = int(partition_seed)
        self.options = dict(options)          # accepted for forward compatibility; unused keys are ignored
        self._memo = LockedCache()

    # -------------------------------------------------------------- protocol
    def available(self) -> tuple[bool, str]:
        try:
            missing = self._check_files()
        except OSError as ex:
            return False, f"data check failed: {type(ex).__name__}: {ex}"
        if missing:
            return False, "; ".join(missing)
        try:
            pools = self.pools()
        except (OSError, ValueError, KeyError) as ex:
            return False, f"data present but could not be parsed: {type(ex).__name__}: {ex}"
        sizes = {k: len(v) for k, v in pools.items()}
        return True, f"{self.name}: pools {sizes}"

    def build_episodes(self, split: str, n: int, seed: int, items_per_episode: int = 16) -> list:
        check_split(split)
        base = "src" if split == "rep" else split
        pools = self.pools()
        rng = make_rng(self.code, "episodes", base, int(seed), int(items_per_episode))
        opts = dict(self._draw_options(base))
        episodes, reused = draw_episodes(pools[base], int(n), int(items_per_episode), rng,
                                         allow_reuse=(base == "src"), what=f"{self.code}/{split}", **opts)
        return [self._make_episode(split, k, items, int(seed), reused and k * items_per_episode >= len(pools[base]))
                for k, items in enumerate(episodes)]

    # -------------------------------------------------------------- helpers
    def pools(self) -> dict[str, list[PoolItem]]:
        pools = self._memo.get("pools", self._split_pools)
        return pools

    def capacity(self, split: str, items_per_episode: int = 16) -> int:
        """Number of item-disjoint episodes the split's sub-pool supports (per lane when ``lanes`` drawing is on)."""
        base = "src" if split == "rep" else split
        ipe = int(items_per_episode)
        if self._draw_options(base).get("lanes"):
            sizes: dict[str, int] = {}
            for it in self.pools()[base]:
                sizes[it.group] = sizes.get(it.group, 0) + 1
            return sum(s // ipe for s in sizes.values())
        return len(self.pools()[base]) // ipe

    def verify_disjoint(self) -> None:
        """Raise ValueError if two sub-pools share an item id (sanity check used by the tests)."""
        pools = self.pools()
        seen: dict[str, str] = {}
        for name, items in pools.items():
            for it in items:
                if it.id in seen and seen[it.id] != name:
                    raise ValueError(f"{self.code}: item {it.id} in pools {seen[it.id]} and {name}")
                seen[it.id] = name

    # -------------------------------------------------------------- to implement
    def _check_files(self) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _split_pools(self) -> dict[str, list[PoolItem]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _draw_options(self, split: str) -> dict:
        return {}

    def _make_episode(self, split: str, k: int, items: list[PoolItem], seed: int, reused: bool):  # pragma: no cover
        raise NotImplementedError
