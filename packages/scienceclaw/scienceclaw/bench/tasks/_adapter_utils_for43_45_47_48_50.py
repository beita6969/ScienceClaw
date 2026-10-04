"""Shared helpers of the FoR43 / FoR45 / FoR47 / FoR48 / FoR50 task adapters (Humanities & law text tasks).

Private to these five adapter modules (the leading underscore keeps it out of the registry). It provides:

* **group-level, seed-independent pool partitions** (:func:`partition_groups`): every evaluation pool is cut
  once into disjoint sub-pools (``src`` / ``val`` / ``id`` …) by ranking *natural groups* (a document, a book,
  an image, a conclusion) with ``sha256(salt|group)``. The partition depends only on the adapter's
  ``partition_seed`` and the group ids — never on the per-split seed that ``SplitPlan`` hands to
  ``build_episodes`` — so splits are item- *and* group-disjoint for any combination of seeds;
* seeded, prefix-stable **stratified episode draws** (:func:`draw_stratified`): every stratum is permuted once
  with the episode RNG and consumed block by block, so building ``n`` or ``n + 1`` episodes yields the same
  first ``n`` and episodes of one split never share an item;
* the normalized score of DESIGN §8.6, a thread-safe lazy loader, the cache directory convention
  (``<repo>/cache/tasks/<code>/``; data directories are never written), receipt provenance and small
  constraint factories for list-shaped deliverables.
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_rank(ids: Sequence[str], salt: str) -> list[str]:
    """``ids`` ordered by ``sha256(salt|id)`` (ties broken by the id itself)."""
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


def partition_groups(item_group: Mapping[str, str], fractions: Mapping[str, float], salt: str,
                     item_stratum: Mapping[str, Hashable] | None = None) -> dict[str, list[str]]:
    """Deterministic group-level partition of items into disjoint parts named by ``fractions``' keys.

    Within each stratum, the *groups* (``item_group[item]``) are ranked by ``sha256(salt|stratum|group)`` and
    walked in that order; each group (with all of its items in the stratum) goes to the part whose item count
    is furthest below its target ``fraction * n_items(stratum)``. Every group lands in exactly one part, so the
    parts are group-disjoint. Items within a part keep (stratum, group-rank, item id) order.
    """
    strata: dict[Hashable, dict[str, list[str]]] = {}
    for item, g in item_group.items():
        s = item_stratum[item] if item_stratum is not None else 0
        strata.setdefault(s, {}).setdefault(g, []).append(item)
    keys = list(fractions)
    tot = float(sum(fractions.values()))
    out: dict[str, list[str]] = {k: [] for k in keys}
    for s in sorted(strata, key=str):
        groups = strata[s]
        n = sum(len(v) for v in groups.values())
        target = {k: n * float(fractions[k]) / tot for k in keys}
        have = {k: 0 for k in keys}
        for g in hash_rank(list(groups), f"{salt}|{s}"):
            items = sorted(groups[g])
            # deficit relative to target (largest first); ties broken by the order of `fractions`
            k = max(keys, key=lambda kk: (target[kk] - have[kk], -keys.index(kk)))
            out[k].extend(items)
            have[k] += len(items)
    return out


class PoolExhausted(ValueError):
    """Raised when a split's sub-pool cannot supply the requested number of disjoint episodes."""


def rotating_quotas(items: int, strata: Sequence[Hashable], e: int) -> dict[Hashable, int]:
    """Per-episode items per stratum summing to ``items``; the remainder rotates with the episode index ``e``.

    E.g. 16 items over 5 strata -> 3 each plus one extra item for stratum ``(e + j) % 5``, j < 1, so the extra
    items are spread evenly over consecutive episodes. Strata may receive 0 items when items < len(strata).
    """
    k = len(strata)
    if k == 0:
        raise ValueError("no strata")
    base, rest = divmod(int(items), k)
    q = {s: base for s in strata}
    for j in range(rest):
        q[strata[(e + j) % k]] += 1
    return q


def draw_stratified(pools: Mapping[Hashable, Sequence[str]], n: int, items: int, rng: np.random.Generator,
                    what: str = "", strata_order: Sequence[Hashable] | None = None) -> list[list[str]]:
    """Draw ``n`` item-disjoint episodes of ``items`` items, balanced over the strata of ``pools``.

    Each stratum is permuted once with ``rng``; episode ``e`` consumes the next ``rotating_quotas(...)[s]`` ids
    of every stratum ``s`` (prefix-stable: the first ``n`` episodes do not depend on ``n``). The ids of an
    episode are then shuffled with the same ``rng``. Raises :class:`PoolExhausted` if a stratum runs out.
    """
    order = list(strata_order) if strata_order is not None else sorted(pools, key=str)
    order = [s for s in order if len(pools.get(s, ())) > 0]
    if not order:
        raise PoolExhausted(f"{what}: empty pool")
    perms = {s: [pools[s][j] for j in rng.permutation(len(pools[s]))] for s in order}
    ptr = {s: 0 for s in order}
    episodes: list[list[str]] = []
    for e in range(int(n)):
        q = rotating_quotas(items, order, e)
        ids: list[str] = []
        for s in order:
            take = q[s]
            if ptr[s] + take > len(perms[s]):
                sizes = {str(k): len(v) for k, v in perms.items()}
                raise PoolExhausted(f"{what}: requested {n} episodes x {items} items but stratum {s!r} is exhausted "
                                    f"after {e} episodes (stratum sizes {sizes})")
            ids.extend(perms[s][ptr[s]:ptr[s] + take])
            ptr[s] += take
        episodes.append([ids[j] for j in rng.permutation(len(ids))])
    return episodes


def stratified_sample(ids_by_stratum: Mapping[Hashable, Sequence[str]], n: int, rng: np.random.Generator
                      ) -> list[str]:
    """Up to ``n`` ids spread over strata proportionally to their sizes (at least one per non-empty stratum)."""
    strata = [s for s in sorted(ids_by_stratum, key=str) if len(ids_by_stratum[s]) > 0]
    total = sum(len(ids_by_stratum[s]) for s in strata)
    if total == 0 or n <= 0:
        return []
    n = min(int(n), total)
    counts = allocate_counts(n, {str(s): len(ids_by_stratum[s]) for s in strata})
    if n >= len(strata):                       # guarantee one id per stratum, taken from the largest quota
        for s in strata:
            if counts[str(s)] == 0:
                donor = max(counts, key=lambda k: counts[k])
                counts[donor] -= 1
                counts[str(s)] = 1
    out: list[str] = []
    for s in strata:
        pool = list(ids_by_stratum[s])
        out.extend(pool[j] for j in rng.permutation(len(pool))[:counts[str(s)]])
    return out


def balanced_sample(ids_by_stratum: Mapping[Hashable, Sequence[str]], n: int, rng: np.random.Generator,
                    exclude: set[str] | None = None) -> list[str]:
    """Up to ``n`` ids with (as far as the strata allow) equal counts per stratum; leftovers go to larger strata.

    Each stratum is permuted once with ``rng`` (after removing ``exclude``); quotas are filled round-robin.
    """
    ex = exclude or set()
    strata = sorted(ids_by_stratum, key=str)
    perms = {s: [x for x in (list(ids_by_stratum[s])[j] for j in rng.permutation(len(ids_by_stratum[s])))
                 if x not in ex] for s in strata}
    take = {s: 0 for s in strata}
    total = 0
    while total < n:
        progressed = False
        for s in strata:
            if total < n and take[s] < len(perms[s]):
                take[s] += 1
                total += 1
                progressed = True
        if not progressed:
            break
    return [x for s in strata for x in perms[s][:take[s]]]


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
            return NORM_CLIP if reference > 0 else 1.0
        v = reference / primary
    return float(min(max(v, 0.0), NORM_CLIP))


# ------------------------------------------------------------------------------------------------ outputs
def as_list(y: Any) -> tuple[list | None, str]:
    """``y`` as a Python list (tuples / 1-D object arrays / pandas Series accepted); (None, reason) otherwise."""
    if y is None:
        return None, "no output"
    if isinstance(y, (str, bytes, dict)):
        return None, f"output must be a list, got {type(y).__name__}"
    if hasattr(y, "tolist") and not isinstance(y, list):
        try:
            y = y.tolist()
        except (TypeError, ValueError) as ex:
            return None, f"output cannot be converted to a list: {ex}"
    if isinstance(y, tuple):
        y = list(y)
    if not isinstance(y, list):
        return None, f"output must be a list, got {type(y).__name__}"
    return y, ""


def c_list_length(n: int, what: str) -> ConstraintSpec:
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        lst, why = as_list(y)
        if lst is None:
            return False, why
        return len(lst) == n, f"len(y)={len(lst)}, required {n}"
    return ConstraintSpec("output_length", f"y is a list with exactly {n} entries ({what}), in evaluation-item order",
                          check)


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


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def receipt_info(path: Path, keys: Sequence[str] = ("version", "commit", "source_url", "doi", "license",
                                                        "license_url", "status", "record_id")) -> dict:
    """Small provenance dict from a data team ``receipt.json`` (missing keys omitted)."""
    try:
        r = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, Any] = {}
    for k in keys:
        v = r.get(k)
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, dict) and "id" in v:
            out[k] = str(v["id"])
    return out


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


def ids_digest(ids: Sequence[Any]) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()
