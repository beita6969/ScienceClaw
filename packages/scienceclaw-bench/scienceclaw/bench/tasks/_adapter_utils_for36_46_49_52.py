"""Shared helpers of the FoR36 / FoR46 / FoR49 / FoR52 task adapters (private; not an adapter module).

* seed-independent **pool partitions**: ids are ranked by ``sha256(salt|id)`` with a fixed partition seed, so
  the src / val / id (and visible train / dev) sub-pools never depend on the per-split seed that SplitPlan
  passes to ``build_episodes`` -> splits are item-disjoint for every seed combination;
* **prefix-stable episode draws**: episode ``e`` takes block ``e`` of a seeded permutation (optionally
  stratified); ``cycle=True`` lets small source pools recycle items in later permutation cycles (documented per
  adapter; source episodes only);
* label-list constraints, the normalized score of DESIGN §8.6, a thread-safe lazy holder, the cache directory
  ``<repo>/cache/tasks/<code>/`` (data directories are never written) and receipt helpers.
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

PARTITION_SEED = 20260928            # reconstruction-policy seed; fixes every pool partition of these adapters
NORM_CLIP = 10.0
REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS_BUILT = ("src", "val", "id", "ood")

T = TypeVar("T")


# --------------------------------------------------------------------------------------------------- hashing
def sha_hex(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def stable_int(*parts: Any) -> int:
    """Deterministic 60-bit integer from the string forms of ``parts``."""
    return int(sha_hex(*parts)[:15], 16)


def hash_rank(ids: Sequence[str], salt: str) -> list[str]:
    """``ids`` sorted by ``sha256(salt|id)`` (ties broken by the id itself)."""
    return sorted(ids, key=lambda i: (sha_hex(salt, i), str(i)))


def episode_rng(code: str, split: str, seed: int, *extra: Any) -> np.random.Generator:
    return np.random.default_rng(stable_int("episode-rng", code, split, int(seed), *extra))


def allocate_counts(n: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Integer split of ``n`` proportional to ``fractions`` (largest remainder, stable key order)."""
    total = float(sum(fractions.values()))
    if total <= 0:
        raise ValueError("fractions must sum to a positive number")
    raw = {k: n * float(v) / total for k, v in fractions.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    keys = list(fractions)
    for k in sorted(keys, key=lambda k: (-(raw[k] - out[k]), keys.index(k)))[: n - sum(out.values())]:
        out[k] += 1
    return out


def partition_ids(ids: Sequence[str], fractions: Mapping[str, float], salt: str) -> dict[str, list[str]]:
    """Deterministic partition of ``ids`` into consecutive hash-rank blocks sized by ``fractions``."""
    ranked = hash_rank(list(ids), salt)
    counts = allocate_counts(len(ranked), fractions)
    out: dict[str, list[str]] = {}
    pos = 0
    for k in fractions:
        out[k] = ranked[pos:pos + counts[k]]
        pos += counts[k]
    return out


class PoolExhausted(ValueError):
    """A split's sub-pool cannot supply the requested number of item-disjoint episodes."""


def draw_blocks(strata: Mapping[Hashable, Sequence[str]], quotas: Mapping[Hashable, int], n: int,
                rng: np.random.Generator, what: str, cycle: bool = False) -> list[list[str]]:
    """``n`` episodes; episode ``e`` takes ``quotas[k]`` ids from block ``e`` of a seeded permutation of stratum ``k``.

    Prefix-stable (building ``n`` or ``n+1`` episodes gives the same first ``n``). Without ``cycle`` the episodes
    are item-disjoint and :class:`PoolExhausted` is raised when a stratum runs out; with ``cycle`` a fresh seeded
    permutation is started whenever a stratum is exhausted (items then recur across episodes, never inside one).
    The ids of an episode are shuffled with the same ``rng``.
    """
    for k, q in quotas.items():
        if q > 0 and len(strata.get(k, ())) < q:
            raise PoolExhausted(f"{what}: stratum {k!r} has {len(strata.get(k, ()))} ids < per-episode quota {q}")
    if not cycle:
        cap = min((len(strata.get(k, ())) // q) for k, q in quotas.items() if q > 0) if quotas else 0
        if n > cap:
            raise PoolExhausted(f"{what}: requested {n} episodes but the sub-pool supports only {cap} "
                                f"(stratum sizes { {str(k): len(strata.get(k, ())) for k in quotas} }, "
                                f"quotas { {str(k): v for k, v in quotas.items()} })")
    streams: dict[Hashable, list[str]] = {k: [] for k in quotas}
    episodes: list[list[str]] = []
    for e in range(n):
        items: list[str] = []
        for k, q in quotas.items():
            if q <= 0:
                continue
            pool = list(strata[k])
            per_cycle = len(pool) // q
            c, b = divmod(e, per_cycle)
            while len(streams[k]) < (c + 1) * per_cycle * q:
                perm = rng.permutation(len(pool))
                streams[k].extend(pool[j] for j in perm[: per_cycle * q])
            start = (c * per_cycle + b) * q
            items.extend(streams[k][start:start + q])
        items = [items[j] for j in rng.permutation(len(items))]
        episodes.append(items)
    return episodes


# --------------------------------------------------------------------------------------------------- scores
def norm_score(primary: float | None, reference: float | None, direction: str, floor: float | None = None) -> float:
    """DESIGN §8.6: primary/reference ("max") or reference/primary ("min"), clipped to [0, NORM_CLIP].

    ``floor`` (max direction) floors the reference used for the ratio, so that a reference score of ~0 does not
    turn every positive score into the clip value (documented per adapter).
    """
    if primary is None or reference is None or not np.isfinite(primary) or not np.isfinite(reference):
        return 0.0
    if direction == "max":
        ref = max(reference, floor) if floor is not None else reference
        if ref <= 0:
            return NORM_CLIP if primary > 0 else 0.0
        v = primary / ref
    else:
        if primary <= 0:
            return NORM_CLIP
        v = reference / primary
    return float(min(max(v, 0.0), NORM_CLIP))


def norm_score_db(primary_db: float | None, reference_db: float | None) -> float:
    """Normalized score for a dB metric (max): the power ratio 10^((primary - reference)/10), clipped to [0, 10].

    A dB value is a log power ratio, so primary/reference on the linear power scale is the faithful analogue of
    DESIGN §8.6 for metrics that can be zero or negative in dB.
    """
    if primary_db is None or reference_db is None or np.isnan(primary_db) or not np.isfinite(reference_db):
        return 0.0
    if np.isinf(primary_db):                     # a perfect estimate has SDR = +inf (museval convention)
        return NORM_CLIP if primary_db > 0 else 0.0
    return float(min(max(10.0 ** (min(primary_db - reference_db, 30.0) / 10.0), 0.0), NORM_CLIP))


# --------------------------------------------------------------------------------------------------- outputs
def as_str_list(y: Any, n: int | None = None) -> tuple[list[str] | None, str]:
    """``y`` as a list of python strings (list / tuple / 1-D array / pandas Series); (None, reason) otherwise."""
    if y is None:
        return None, "no output"
    if isinstance(y, (str, bytes, dict)):
        return None, f"output must be a list of strings, got {type(y).__name__}"
    if hasattr(y, "tolist") and not isinstance(y, (list, tuple)):
        try:
            y = y.tolist()
        except Exception as ex:  # noqa: BLE001 - report conversion failure as a constraint message
            return None, f"cannot convert output to a list: {type(ex).__name__}: {ex}"
    if not isinstance(y, (list, tuple)):
        return None, f"output must be a list of strings, got {type(y).__name__}"
    bad = [i for i, v in enumerate(y) if not isinstance(v, str)]
    if bad:
        return None, f"{len(bad)} entries are not strings (first at index {bad[0]}: {type(y[bad[0]]).__name__})"
    if n is not None and len(y) != n:
        return None, f"output has length {len(y)}, required {n}"
    return [str(v) for v in y], ""


def c_str_list(n: int, what: str) -> ConstraintSpec:
    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        v, why = as_str_list(y, n)
        return v is not None, (why or f"list of {n} strings")
    return ConstraintSpec("output_format", f"y is a list of exactly {n} strings ({what}), in evaluation-item order",
                          check)


def c_allowed_labels(n: int, allowed: Sequence[Sequence[str]] | Sequence[str], name: str = "allowed_labels",
                     description: str = "") -> ConstraintSpec:
    """Every y[i] is one of the allowed labels (per item when ``allowed`` is a list of lists)."""
    per_item = bool(allowed) and not isinstance(allowed[0], str)
    allowed_sets = [set(a) for a in allowed] if per_item else [set(allowed)] * n  # type: ignore[arg-type]

    def check(y: Any, trace: Trace | None) -> tuple[bool, str]:
        v, why = as_str_list(y, n)
        if v is None:
            return False, why
        bad = [i for i, s in enumerate(v) if s not in allowed_sets[i]]
        if bad:
            i = bad[0]
            return False, (f"{len(bad)} entries are not allowed labels (index {i}: {v[i]!r} not in "
                           f"{sorted(allowed_sets[i])})")
        return True, "all entries are allowed labels"
    desc = description or ("every y[i] is one of the allowed labels" + (" of item i" if per_item else
                                                                        f" {sorted(allowed_sets[0])}"))
    return ConstraintSpec(name, desc, check)


# --------------------------------------------------------------------------------------------------- io / cache
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


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def receipt_summary(path: Path) -> dict:
    """Flat provenance fields of a data-team ``receipt.json`` (scalars only; missing file -> {})."""
    try:
        r = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    keep = ("for_code", "dataset", "status", "completion_status", "source", "commit", "license", "license_url",
            "fetched_at_utc", "verified_at_utc", "completed_at_utc", "doi")
    out = {k: r[k] for k in keep if k in r and isinstance(r[k], (str, int, float, bool))}
    if isinstance(r.get("commits"), dict):
        out["commits"] = {k: v for k, v in r["commits"].items() if isinstance(v, str)}
    return out


class Lazy(Generic[T]):
    """Thread-safe compute-once holder (a failing ``fn`` is retried on the next ``get``)."""

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


def check_split(split: str) -> None:
    if split == "rep":
        raise ValueError("'rep' episodes are frozen copies of src episodes made by SplitPlan; build 'src' instead")
    if split not in SPLITS_BUILT:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS_BUILT}")


def episode_id(code: str, split: str, seed: int, k: int) -> str:
    return f"{code}-{split}-{int(seed) & 0xFFFFFFFF:08x}-{k:02d}"
