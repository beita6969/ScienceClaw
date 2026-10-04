"""Shared helpers of the FoR34 / FoR39 / FoR40 / FoR44 / FoR51 task adapters.

Private to these five adapter modules (the leading underscore keeps it out of the registry). It provides:

* deterministic, seed-independent **pool partitions** (:func:`partition_pool`): every evaluation pool is cut once
  into disjoint sub-pools (``src`` / ``val`` / ``id`` and, where the visible training data comes from the same
  pool, ``train`` / ``dev``) by ranking item ids with ``sha256(salt|item_id)``. The partition depends only on
  the adapter's ``partition_seed`` and the item ids — never on the per-split seed handed to
  ``build_episodes`` — so the splits are item-disjoint for any combination of seeds (SplitPlan passes a
  different seed per split);
* seeded, prefix-stable **episode draws** (:func:`draw_episodes`): episode ``e`` takes block ``e`` of a seeded
  permutation of every stratum, so building ``n`` or ``n + 1`` episodes yields the same first ``n``;
* hard-constraint factories, the normalized score of DESIGN §8.6, a thread-safe lazy loader and the cache
  directory convention (``<repo>/cache/tasks/<code>/``; data directories are never written).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from ...core.trace import Trace
from ..task import ConstraintSpec

PARTITION_SEED = 20260928          # reconstruction-policy seed; fixes all pool partitions
NORM_CLIP = 10.0
REPO_ROOT = Path(__file__).resolve().parents[3]

T = TypeVar("T")


# ------------------------------------------------------------------------------------------------ hashing
def stable_int(*parts: Any) -> int:
    """A deterministic 60-bit integer from the string forms of ``parts`` (sha256)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:15], 16)


def hash_rank(ids: Sequence[str], salt: str) -> list[str]:
    """``ids`` ordered by ``sha256(salt|id)`` (ties impossible in practice; broken by id)."""
    return sorted(ids, key=lambda i: (hashlib.sha256(f"{salt}|{i}".encode()).hexdigest(), str(i)))


def episode_rng(code: str, split: str, seed: int, *extra: Any) -> np.random.Generator:
    return np.random.default_rng(stable_int("episode-rng", code, split, int(seed), *extra))


# ------------------------------------------------------------------------------------------------ partitions
def allocate_counts(n: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Split ``n`` into integer counts proportional to ``fractions`` (largest-remainder rounding, stable order)."""
    total = float(sum(fractions.values()))
    if total <= 0:
        raise ValueError("fractions must sum to a positive number")
    raw = {k: n * float(v) / total for k, v in fractions.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    rest = n - sum(out.values())
    order = sorted(fractions, key=lambda k: (-(raw[k] - out[k]), list(fractions).index(k)))
    for k in order[:rest]:
        out[k] += 1
    return out


def partition_pool(ids: Sequence[str], fractions: Mapping[str, float], salt: str,
                   strata: Mapping[str, Hashable] | None = None) -> dict[str, list[str]]:
    """Deterministic (stratified) partition of ``ids`` into disjoint parts named by ``fractions``' keys.

    Within each stratum (``strata[id]``; one stratum if None) ids are ranked by ``sha256(salt|id)`` and cut into
    consecutive blocks whose sizes follow ``fractions`` (largest remainder). Returned lists keep that rank order.
    """
    groups: dict[Hashable, list[str]] = {}
    for i in ids:
        groups.setdefault(strata[i] if strata is not None else 0, []).append(i)
    out: dict[str, list[str]] = {k: [] for k in fractions}
    for g in sorted(groups, key=str):
        ranked = hash_rank(groups[g], f"{salt}|{g}")
        counts = allocate_counts(len(ranked), fractions)
        pos = 0
        for k in fractions:
            out[k].extend(ranked[pos:pos + counts[k]])
            pos += counts[k]
    return out


class PoolExhausted(ValueError):
    """Raised when a split's sub-pool cannot supply the requested number of disjoint episodes."""


def draw_episodes(strata_pools: Mapping[Hashable, Sequence[str]], quotas: Mapping[Hashable, int], n: int,
                  rng: np.random.Generator, what: str = "") -> list[list[str]]:
    """Draw ``n`` item-disjoint episodes; episode ``e`` takes ``quotas[k]`` ids from every stratum ``k``.

    Each stratum is permuted once with ``rng`` and episode ``e`` takes block ``e`` of it (prefix-stable). The
    ids of an episode are returned stratum by stratum and then shuffled with the same ``rng``.
    """
    perms = {k: [strata_pools[k][j] for j in rng.permutation(len(strata_pools[k]))] if k in strata_pools else []
             for k in quotas}
    cap = min((len(perms[k]) // q) if q > 0 else n for k, q in quotas.items()) if quotas else 0
    if n > cap:
        sizes = {str(k): len(perms[k]) for k in quotas}
        raise PoolExhausted(f"{what}: requested {n} episodes but the sub-pool supports only {cap} "
                            f"(stratum sizes {sizes}, per-episode quotas { {str(k): v for k, v in quotas.items()} })")
    episodes: list[list[str]] = []
    for e in range(n):
        items: list[str] = []
        for k, q in quotas.items():
            items.extend(perms[k][e * q:(e + 1) * q])
        items = [items[j] for j in rng.permutation(len(items))]
        episodes.append(items)
    return episodes


def stratified_quotas(items: int, strata_order: Sequence[Hashable], min_each: int = 1,
                      weights: Mapping[Hashable, float] | None = None) -> dict[Hashable, int]:
    """Per-episode counts per stratum summing to ``items`` with at least ``min_each`` per stratum."""
    k = len(strata_order)
    if items < k * min_each:
        raise ValueError(f"items_per_episode={items} too small for {k} strata x {min_each}")
    w = {s: float((weights or {}).get(s, 1.0)) for s in strata_order}
    extra = allocate_counts(items - k * min_each, {str(s): w[s] for s in strata_order})
    return {s: min_each + extra[str(s)] for s in strata_order}


# ------------------------------------------------------------------------------------------------ scores
def norm_score(primary: float | None, reference: float | None, direction: str) -> float:
    """DESIGN §8.6: primary/reference ("max") or reference/primary ("min"), clipped to [0, NORM_CLIP]."""
    if primary is None or reference is None or not np.isfinite(primary) or not np.isfinite(reference):
        return 0.0
    if direction == "max":
        if reference <= 0:
            return NORM_CLIP if primary > 0 else 0.0
        v = primary / reference
    else:
        if primary <= 0:
            return NORM_CLIP
        v = reference / primary
    return float(min(max(v, 0.0), NORM_CLIP))


# ------------------------------------------------------------------------------------------------ outputs
def as_float_array(y: Any) -> tuple[np.ndarray | None, str]:
    """``y`` as a float ndarray (lists / tuples / pandas accepted); (None, reason) if not numeric."""
    if y is None:
        return None, "no output"
    if isinstance(y, (str, bytes, dict)):
        return None, f"output must be numeric array-like, got {type(y).__name__}"
    try:
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as ex:
        return None, f"output is not numeric: {type(ex).__name__}: {ex}"
    return arr, ""


def as_float_vector(y: Any, n: int | None = None) -> tuple[np.ndarray | None, str]:
    """``y`` as a 1-D float vector (column/row vectors flattened); checks the length when ``n`` is given."""
    arr, why = as_float_array(y)
    if arr is None:
        return None, why
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        return None, f"output must be 1-D, got shape {list(arr.shape)}"
    if n is not None and arr.shape[0] != n:
        return None, f"output has length {arr.shape[0]}, required {n}"
    return arr, ""


def c_vector(n: int, what: str) -> ConstraintSpec:
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        arr, why = as_float_vector(y, n)
        return (arr is not None), (why or f"1-D numeric vector of length {n}")
    return ConstraintSpec("output_shape", f"y is a 1-D numeric array of length {n} ({what}), in evaluation-item "
                                          "order", check)


def c_finite(n: int) -> ConstraintSpec:
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        arr, why = as_float_vector(y, n)
        if arr is None:
            return False, why
        bad = int((~np.isfinite(arr)).sum())
        return bad == 0, ("all values finite" if bad == 0 else f"{bad} non-finite values")
    return ConstraintSpec("finite", "every value of y is a finite real number", check)


def c_range(n: int, lo: float, hi: float, name: str, description: str) -> ConstraintSpec:
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        arr, why = as_float_vector(y, n)
        if arr is None:
            return False, why
        fin = arr[np.isfinite(arr)]
        bad = int(((fin < lo) | (fin > hi)).sum()) + int((~np.isfinite(arr)).sum())
        return bad == 0, (f"all values in [{lo:g}, {hi:g}]" if bad == 0
                          else f"{bad} values outside [{lo:g}, {hi:g}] (min {np.nanmin(arr) if fin.size else 'nan'}, "
                               f"max {np.nanmax(arr) if fin.size else 'nan'})")
    return ConstraintSpec(name, description, check)


# ------------------------------------------------------------------------------------------------ io / cache
def resolve_data_root(data_root: str | os.PathLike | None) -> Path:
    if data_root:
        return Path(data_root)
    from ..registry import DATA_ROOT
    return Path(DATA_ROOT)


def cache_dir(code: str) -> Path:
    """``$SCIENCECLAW_TASK_CACHE/<code>`` or ``<repo>/cache/tasks/<code>`` (created on demand)."""
    base = os.environ.get("SCIENCECLAW_TASK_CACHE")
    d = (Path(base) if base else REPO_ROOT / "cache" / "tasks") / code
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def receipt_info(path: Path, keys: Sequence[str] = ("version", "source", "url", "license", "license_url",
                                                        "status", "official_source")) -> dict:
    """Small provenance dict from a data team ``receipt.json`` (missing keys omitted)."""
    try:
        r = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: r[k] for k in keys if k in r and isinstance(r[k], (str, int, float, bool))}


class Lazy(Generic[T]):
    """Thread-safe, compute-once holder (``get()`` runs ``fn`` at most once; failures are re-raised each call)."""

    def __init__(self, fn: Callable[[], T]) -> None:
        self._fn = fn
        self._lock = threading.Lock()
        self._done = False
        self._value: T | None = None

    def get(self) -> T:
        if self._done:
            return self._value  # type: ignore[return-value]
        with self._lock:
            if not self._done:
                self._value = self._fn()
                self._done = True
        return self._value  # type: ignore[return-value]


def check_split(split: str, allowed: Sequence[str] = ("src", "val", "id", "ood")) -> None:
    if split == "rep":
        raise ValueError("'rep' episodes are frozen copies of src episodes made by SplitPlan; build 'src' instead")
    if split not in allowed:
        raise ValueError(f"unknown split {split!r}; expected one of {tuple(allowed)}")


def episode_id(code: str, split: str, seed: int, k: int) -> str:
    return f"{code}-{split}-{int(seed) & 0xFFFFFFFF:08x}-{k:02d}"
