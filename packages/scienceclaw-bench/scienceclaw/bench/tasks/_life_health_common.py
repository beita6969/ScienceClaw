"""Shared helpers for the Life & health adapters (FoR30 PhenoBench, FoR31 ProteinGym, FoR32 MSD Hippocampus,
FoR42 PhysioNet 2019 sepsis).

Everything here is deterministic: random choices are driven by ``numpy.random.Generator(PCG64)`` seeded from a
SHA-256 of explicit string keys, so the same keys always give the same draw on every platform.

Split policy shared by the four adapters
----------------------------------------
* *Pools* (which items may ever appear in src / val / id / ood episodes, and which items are visible training /
  dev data) are fixed by the adapter's ``pool_seed`` (default: the protocol seed 20260928), NOT by the ``seed``
  passed to ``build_episodes``. Pools are therefore item-disjoint for any combination of build seeds.
* The build ``seed`` only decides how pool items are grouped into episodes (and their order).
* If the source split needs more items than its pool holds (e.g. 7 episodes x 16 items from a small pool), items are
  reused across source episodes only (never across splits); ``lineage["reused_items"]`` is then True.
* Held-out splits (val / id / ood) never reuse an item: if the pool is too small the episodes shrink
  (``effective_items``) and ``lineage["items_requested"]`` records the requested size.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..task import Budget, ConstraintSpec, EvalResult

PROTOCOL_SEED = 20260928
REPO_ROOT = Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------------------------------------------------
# paths / hashing / rng
# ----------------------------------------------------------------------------------------------------------------
def default_cache_dir(code: str) -> Path:
    """Derived-cache directory of one discipline: ``<repo>/cache/tasks/<code>/`` (never inside the dataset root)."""
    return REPO_ROOT / "cache" / "tasks" / code


def stable_seed(*parts: Any) -> int:
    """64-bit integer seed from a SHA-256 of the string forms of ``parts``."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def rng_for(*parts: Any) -> np.random.Generator:
    """Deterministic PCG64 generator keyed by ``parts``."""
    return np.random.Generator(np.random.PCG64(stable_seed(*parts)))


def ids_hash(ids: Iterable[Any]) -> str:
    """Short order-sensitive hash of a list of ids (recorded in lineage)."""
    return hashlib.sha256("\n".join(str(i) for i in ids).encode("utf-8")).hexdigest()[:16]


def tmp_path_for(path: str | Path, suffix: str = "") -> Path:
    """Unique sibling temp path (process id + thread id) for atomic ``os.replace`` cache writes."""
    p = Path(path)
    return p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp{suffix}")


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# pool allocation and episode composition
# ----------------------------------------------------------------------------------------------------------------
def allocate_groups(groups: dict[str, list[str]], targets: Sequence[tuple[str, int]], remainder: str,
                    seed_parts: Sequence[Any]) -> dict[str, list[str]]:
    """Assign whole lineage groups to pools.

    ``groups`` maps a lineage-group id (subject, protein, plot, ...) to its item ids; groups are never split.
    Groups are visited in a seeded random order and each goes to the first pool (in ``targets`` priority order)
    whose remaining capacity can hold the whole group; groups that fit nowhere go to ``remainder``.
    Returns pool -> item ids (sorted by visiting order, then item order inside the group).
    """
    order = sorted(groups)
    perm = rng_for(*seed_parts, "allocate").permutation(len(order))
    cap = {name: int(n) for name, n in targets}
    out: dict[str, list[str]] = {name: [] for name, _ in targets}
    out.setdefault(remainder, [])
    for gi in perm:
        g = order[int(gi)]
        members = list(groups[g])
        for name, _ in targets:
            if cap[name] >= len(members):
                out[name].extend(members)
                cap[name] -= len(members)
                break
        else:
            out[remainder].extend(members)
    return out


def stratified_take(pos: list[str], neg: list[str], n_pos: int, n_neg: int, seed_parts: Sequence[Any]
                    ) -> tuple[list[str], list[str], list[str], list[str]]:
    """Seeded draw of ``n_pos`` positives and ``n_neg`` negatives; returns (taken_pos, taken_neg, rest_pos, rest_neg)."""
    if n_pos > len(pos) or n_neg > len(neg):
        raise ValueError(f"cannot take {n_pos}/{n_neg} from {len(pos)}/{len(neg)}")
    r = rng_for(*seed_parts)
    pp = [pos[i] for i in r.permutation(len(pos))]
    nn = [neg[i] for i in r.permutation(len(neg))]
    return pp[:n_pos], nn[:n_neg], pp[n_pos:], nn[n_neg:]


def _cycle_chunks(pool: Sequence[str], n: int, k: int, seed_parts: Sequence[Any]) -> tuple[list[list[str]], bool]:
    """``n`` chunks of ``k`` distinct items drawn from successive seeded permutations of ``pool``.

    Prefix-stable: chunk j does not depend on ``n``. Items repeat across chunks only once the pool is exhausted.
    """
    if k <= 0:
        return [[] for _ in range(n)], False
    if k > len(pool):
        raise ValueError(f"episode needs {k} distinct items but the pool has only {len(pool)}")
    pool = list(pool)
    cycle = 0
    r = rng_for(*seed_parts, "cycle", cycle)
    queue = [pool[i] for i in r.permutation(len(pool))]
    chunks: list[list[str]] = []
    reused = False
    for _ in range(n):
        chunk: list[str] = []
        while len(chunk) < k:
            if not queue:
                cycle += 1
                reused = True
                r = rng_for(*seed_parts, "cycle", cycle)
                fresh = [pool[i] for i in r.permutation(len(pool))]
                # items already in this chunk move to the back so the chunk stays duplicate-free
                queue = [x for x in fresh if x not in chunk] + [x for x in fresh if x in chunk]
            x = queue.pop(0)
            if x in chunk:           # can only happen right after a refill; push back and continue
                queue.append(x)
                continue
            chunk.append(x)
        chunks.append(chunk)
    return chunks, reused


def effective_items(split: str, pool_size: int, n: int, k: int) -> int:
    """Items per episode actually used.

    Held-out splits (val / id / ood) never reuse an item across their episodes: when ``n * k`` exceeds the pool,
    episodes shrink to ``pool_size // n`` items (the adapter records ``items_requested`` in lineage). src (and rep,
    its frozen copy) may reuse items across episodes and always get ``k`` items.
    """
    if split in ("src", "rep") or n * k <= pool_size:
        return k
    if pool_size < n:
        raise ValueError(f"{split}: pool of {pool_size} items cannot fill {n} episodes")
    return pool_size // n


def compose_episodes(pool: Sequence[str], n: int, k: int, seed_parts: Sequence[Any],
                     strata: dict[str, tuple[Sequence[str], int]] | None = None) -> tuple[list[list[str]], bool]:
    """Group pool items into ``n`` episodes of ``k`` items.

    Without ``strata`` items are drawn from ``pool``. With ``strata`` ({name: (items, per_episode)}; per-episode
    counts must sum to ``k``) each stratum is cycled independently and the episode is shuffled deterministically.
    Returns (episodes, reused_flag).
    """
    if strata is None:
        return _cycle_chunks(pool, n, k, seed_parts)
    if sum(c for _, c in strata.values()) != k:
        raise ValueError("stratum counts must sum to items_per_episode")
    per: dict[str, list[list[str]]] = {}
    reused = False
    for name, (items, c) in strata.items():
        per[name], ru = _cycle_chunks(items, n, c, [*seed_parts, name])
        reused = reused or ru
    episodes = []
    for j in range(n):
        ep = [x for name in strata for x in per[name][j]]
        perm = rng_for(*seed_parts, "order", j).permutation(len(ep))
        episodes.append([ep[i] for i in perm])
    return episodes, reused


# ----------------------------------------------------------------------------------------------------------------
# scoring helpers
# ----------------------------------------------------------------------------------------------------------------
def finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def norm_score(primary: float | None, reference: float | None, direction: str) -> float:
    """DESIGN 8.6: primary/reference (max) or reference/primary (min), clipped to [0, 10]; 0 when undefined.

    For "max" metrics whose reference is <= 0 (e.g. normalized utility can be 0 or negative), the ratio is not
    meaningful; we then use ``1 + (primary - reference)`` (a shift that keeps "equal to reference" = 1) and document
    this in the task cards.
    """
    p, r = finite_or_none(primary), finite_or_none(reference)
    if p is None or r is None:
        return 0.0
    if direction == "max":
        v = p / r if r > 0 else 1.0 + (p - r)
    else:
        v = r / p if p > 0 else (10.0 if r > 0 else 0.0)
    return float(min(10.0, max(0.0, v)))


def beats(primary: float | None, reference: float | None, direction: str, margin: float) -> bool:
    """Acceptance rule: primary beats the reference by more than ``margin`` (absolute, in metric units)."""
    p, r = finite_or_none(primary), finite_or_none(reference)
    if p is None or r is None:
        return False
    return p > r + margin if direction == "max" else p < r - margin


def make_result(primary: float | None, reference: float | None, direction: str, margin: float,
                metrics: dict[str, float], details: dict, valid: bool, floor: float | None = None) -> EvalResult:
    """EvalResult with the DESIGN 8.6 details keys (reference, norm_score, pooled_payload must be in ``details``).

    ``floor`` (optional) is an absolute floor on the reference *used for acceptance only* (``max`` metrics: accept iff
    primary > max(reference, floor) + margin), e.g. FoR42 requires a strictly positive clinical utility even when the
    episode's reference utility is negative. ``details['reference']`` and ``norm_score`` keep the true reference.
    """
    details = dict(details)
    details["reference"] = finite_or_none(reference)
    details["norm_score"] = norm_score(primary, reference, direction) if valid else 0.0
    details["acceptance_margin"] = margin
    ref_acc = reference
    if floor is not None:
        details["acceptance_floor"] = floor
        r = finite_or_none(reference)
        if r is not None:
            ref_acc = max(r, floor) if direction == "max" else min(r, floor)
    details.setdefault("pooled_payload", None)
    m = {k: float(v) for k, v in metrics.items() if finite_or_none(v) is not None}
    m["valid_output"] = 1.0 if valid else 0.0
    return EvalResult(metrics=m, primary=finite_or_none(primary) if valid else None, direction=direction,
                      accepted=bool(valid and beats(primary, ref_acc, direction, margin)), details=details)


def extract_payloads(per_episode: Iterable[Any], kind: str) -> list[dict]:
    """Normalize what callers hand to ``pooled_metric``.

    Accepts payload dicts themselves, dicts holding ``"pooled_payload"`` (EvalResult.details) or ``"details"``
    (EvalResult.to_dict()), and EvalResult objects. Entries without a payload of the right ``kind`` are skipped
    (e.g. episodes that produced no output at all).
    """
    out = []
    for e in per_episode:
        if e is None:
            continue
        if isinstance(e, EvalResult):
            e = e.details
        if not isinstance(e, dict):
            continue
        if "details" in e and isinstance(e["details"], dict):
            e = e["details"]
        if "pooled_payload" in e:
            e = e["pooled_payload"]
        if isinstance(e, dict) and e.get("kind") == kind:
            out.append(e)
    return out


# ----------------------------------------------------------------------------------------------------------------
# constraints
# ----------------------------------------------------------------------------------------------------------------
def unit_constraint(required_unit: str) -> ConstraintSpec:
    """Declared unit of the submitted value (if the trace records one) must equal ``required_unit``.

    The runtime records port units in ``NodeRecord.outputs_summary[port]["unit"]``. When no unit is recorded (the
    submit port inherits the required schema) the check passes with an explanatory message.
    """
    def check(y: Any, trace: Any) -> tuple[bool, str]:
        recs = getattr(trace, "records", None) or {}
        for nid, rec in recs.items():
            if getattr(rec, "kind", "") != "submit":
                continue
            for port, summ in (getattr(rec, "outputs_summary", None) or {}).items():
                unit = summ.get("unit") if isinstance(summ, dict) else None
                if unit is not None and unit != required_unit:
                    return False, f"submit port {port} declares unit {unit!r}, required {required_unit!r}"
        return True, f"unit {required_unit!r} (no conflicting declaration in trace)"
    return ConstraintSpec("declared_unit", f"declared unit of y must be {required_unit!r}", check)


def as_list_of_arrays(y: Any, n: int) -> list[np.ndarray] | None:
    """Coerce y into a list of ``n`` numpy arrays (list/tuple/object-array/2-D array rows); None if impossible."""
    if isinstance(y, np.ndarray) and y.dtype != object:
        if y.ndim >= 1 and y.shape[0] == n:
            return [np.asarray(y[i]) for i in range(n)]
        return None
    if isinstance(y, (list, tuple)) or (isinstance(y, np.ndarray) and y.dtype == object):
        seq = list(y)
        if len(seq) != n:
            return None
        try:
            return [np.asarray(v) for v in seq]
        except (TypeError, ValueError):
            return None
    return None


def default_budget(n_items: int, *, max_node_s: float, max_wall_s: float, llm_factor: int = 4) -> Budget:
    return Budget(max_steps=12, max_policy_tokens=200_000, max_wall_s=float(max_wall_s), max_node_s=float(max_node_s),
                  max_llm_items=int(llm_factor * max(1, n_items)))


class Memo:
    """Thread-safe lazily computed value keyed by a string (one lock per key)."""

    def __init__(self) -> None:
        self._vals: dict[str, Any] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, key: str, fn: Callable[[], Any]) -> Any:
        if key in self._vals:
            return self._vals[key]
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self._vals:
                self._vals[key] = fn()
            return self._vals[key]
