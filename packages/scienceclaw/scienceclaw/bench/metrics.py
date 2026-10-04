"""Benchmark metrics (paper Sec. 4.2 and the Sec. 6 analyses; DESIGN decision 10 and §8.6).

Conventions used throughout:

* ``scores[method][discipline]`` holds a task-native (pooled) score; ``directions[discipline]`` is ``"max"``
  (higher is better) or ``"min"`` (lower is better). Missing scores are ``None``/NaN.
* All functions are pure and deterministic (bootstrap uses an explicit seed).
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

# ------------------------------------------------------------------------------------------------------ helpers


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _check_direction(d: str) -> str:
    if d not in ("max", "min"):
        raise ValueError(f"direction must be 'max' or 'min', got {d!r}")
    return d


# ------------------------------------------------------------------------------------------------------ MacroSR


def macro_sr(z_by_discipline: Mapping[str, Iterable[int]]) -> float:
    """Eq. 4: MacroSR = (1/|D|) Σ_d (1/|I_d|) Σ_{i∈I_d} z_i, in [0, 1].

    Disciplines with no instances are excluded from the average (they carry no evidence); if no discipline has
    an instance the result is NaN.
    """
    rates = []
    for d, zs in z_by_discipline.items():
        zs = [int(bool(z)) for z in zs]
        if zs:
            rates.append(sum(zs) / len(zs))
    return float(np.mean(rates)) if rates else float("nan")


def success_rates(z_by_discipline: Mapping[str, Iterable[int]]) -> dict[str, float]:
    """Per-discipline success rate (1/|I_d|) Σ z_i (NaN for empty disciplines)."""
    out = {}
    for d, zs in z_by_discipline.items():
        zs = [int(bool(z)) for z in zs]
        out[d] = sum(zs) / len(zs) if zs else float("nan")
    return out


def group_macro(values_by_discipline: Mapping[str, float], groups: Mapping[str, str]) -> dict[str, float]:
    """Macro-average of per-discipline values within each group (e.g. family): group -> mean over its disciplines."""
    acc: dict[str, list[float]] = {}
    for d, v in values_by_discipline.items():
        g = groups.get(d)
        if g is None or not _finite(v):
            continue
        acc.setdefault(g, []).append(float(v))
    return {g: float(np.mean(vs)) for g, vs in acc.items()}


# ------------------------------------------------------------------------------------------------ pooled scores


def load_payload(path: str | Path) -> Any:
    """Load a pooled payload written by ``experiments.evaluate`` (``.json`` or ``.pkl``)."""
    p = Path(path)
    if p.suffix == ".pkl":
        with p.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - files written by this code base only
    return json.loads(p.read_text())


def _payload_of(res: Mapping[str, Any], base_dir: str | Path | None = None) -> Any:
    if res.get("pooled_payload") is not None and not isinstance(res.get("pooled_payload"), str):
        return res["pooled_payload"]
    det = res.get("details")
    if isinstance(det, Mapping) and det.get("pooled_payload") is not None:
        return det["pooled_payload"]
    path = res.get("pooled_payload_path") or (res.get("pooled_payload") if isinstance(res.get("pooled_payload"), str) else None)
    if path:
        p = Path(path)
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        return load_payload(p)
    return None


def pooled_scores(results_by_discipline: Mapping[str, list[Mapping[str, Any]]], adapters: Mapping[str, Any],
                  base_dir: str | Path | None = None) -> dict[str, float | None]:
    """Pooled task-native score per discipline: ``adapter.pooled_metric([payload_1, ..., payload_n])``.

    Each result is a dict holding the episode's pooled payload inline (``"pooled_payload"`` or
    ``details["pooled_payload"]``) or a path to it (``"pooled_payload_path"``, or a string ``"pooled_payload"``;
    relative paths resolve against ``base_dir``). Every solved held-out episode carries a payload (a
    failed episode carries the adapter's uniform failure payload, see ``Episode.evaluate``); results that still have
    none are excluded here, so callers must check :func:`pooled_coverage` / :func:`expected_coverage` before
    comparing methods (:func:`performance_index` does so through ``coverage=``). A discipline with no payload scores
    ``None``. Raises KeyError if a discipline has no adapter.
    """
    out: dict[str, float | None] = {}
    for d, results in results_by_discipline.items():
        if d not in adapters:
            raise KeyError(f"pooled_scores: no adapter for discipline {d!r}")
        payloads = [p for p in (_payload_of(r, base_dir) for r in results) if p is not None]
        if not payloads:
            out[d] = None
            continue
        v = adapters[d].pooled_metric(payloads)
        out[d] = None if v is None else float(v)
    return out


def pooled_coverage(results_by_discipline: Mapping[str, list[Mapping[str, Any]]],
                    base_dir: str | Path | None = None) -> dict[str, tuple[int, int]]:
    """discipline -> (#results with a pooled payload, #results)."""
    return {d: (sum(_payload_of(r, base_dir) is not None for r in rs), len(rs)) for d, rs in results_by_discipline.items()}


def expected_coverage(results_by_discipline: Mapping[str, list[Mapping[str, Any]]], expected: Mapping[str, int],
                      base_dir: str | Path | None = None) -> dict[str, tuple[int, int]]:
    """discipline -> (#distinct episodes with a pooled payload, #episodes the split promises).

    ``expected`` comes from the split manifest. An episode that was never solved counts as missing, and duplicate
    result lines of one episode count once, so a pooled score can be flagged incomplete instead of silently covering
    only the episodes that happened to finish. Disciplines only in ``expected`` report 0 payloads.
    """
    out: dict[str, tuple[int, int]] = {}
    for d in dict.fromkeys([*results_by_discipline, *expected]):
        rs = results_by_discipline.get(d, [])
        have = {(r.get("episode") if r.get("episode") is not None else i) for i, r in enumerate(rs)
                if _payload_of(r, base_dir) is not None}
        out[d] = (len(have), int(expected.get(d, len(rs))))
    return out


def coverage_gaps(coverage: Mapping[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Disciplines whose payload coverage is incomplete: ``{discipline: (have, want)}``."""
    return {d: (int(h), int(w)) for d, (h, w) in coverage.items() if int(h) < int(w)}


# ------------------------------------------------------------------------------------------------ PI


class PIResult(dict):
    """``{method: macro PI}`` (a plain dict, as in DESIGN §8.6) with extra attributes:

    * ``per_discipline``: ``{method: {discipline: PI_d}}``
    * ``reference``, ``normalization``
    * ``disciplines``: disciplines entering the macro average (reference score valid for the normalization)
    * ``excluded``: ``{discipline: reason}`` for disciplines dropped for every method
    * ``incomplete``: ``{method: {discipline: (have, want)}}`` for methods whose pooled-payload coverage of an
      included discipline is incomplete; their macro PI is NaN (a partial pool is not comparable)
    """

    def __init__(self, macro: dict[str, float], per_discipline: dict[str, dict[str, float]], reference: str,
                 normalization: str, disciplines: list[str], excluded: dict[str, str],
                 incomplete: dict[str, dict[str, tuple[int, int]]] | None = None) -> None:
        super().__init__(macro)
        self.per_discipline = per_discipline
        self.reference = reference
        self.normalization = normalization
        self.disciplines = disciplines
        self.excluded = excluded
        self.incomplete = incomplete or {}

    def to_dict(self) -> dict:
        return {"macro": dict(self), "per_discipline": self.per_discipline, "reference": self.reference,
                "normalization": self.normalization, "disciplines": self.disciplines, "excluded": self.excluded,
                "incomplete": {m: {d: list(v) for d, v in g.items()} for m, g in self.incomplete.items()}}


def pi_value(s: float, s_ref: float, direction: str, normalization: str = "ratio", cap: float = 1000.0) -> float:
    """PI_d of one score against the reference score (NaN if undefined).

    ``normalization="ratio"`` (default, DESIGN decision 10)::

        max:  PI_d = 100 * s / s_ref          (requires s_ref > 0)
        min:  PI_d = 100 * s_ref / s          (requires s_ref > 0, s >= 0; s == 0 -> ``cap``)

    ``normalization="linear"``::

        PI_d = 100 * (1 + sign * (s - s_ref) / |s_ref|),  sign = +1 (max), -1 (min)   (requires s_ref != 0)
        i.e. max: 100 * s / s_ref (for s_ref > 0) and min: 100 * (2 - s / s_ref) (for s_ref > 0).

    Both give the reference 100. For a lower-is-better metric they differ: halving the error gives 200 (ratio)
    but 150 (linear); doubling it gives 50 (ratio) but 0 (linear). Values are clipped to [-cap, cap].
    """
    _check_direction(direction)
    if not (_finite(s) and _finite(s_ref)):
        return float("nan")
    s, s_ref = float(s), float(s_ref)
    if normalization == "ratio":
        if s_ref <= 0:
            return float("nan")
        if direction == "max":
            v = 100.0 * s / s_ref
        else:
            if s < 0:
                return float("nan")
            v = cap if s == 0 else 100.0 * s_ref / s
    elif normalization == "linear":
        if s_ref == 0:
            return float("nan")
        sign = 1.0 if direction == "max" else -1.0
        v = 100.0 * (1.0 + sign * (s - s_ref) / abs(s_ref))
    else:
        raise ValueError(f"unknown PI normalization {normalization!r} (use 'ratio' or 'linear')")
    return float(min(max(v, -cap), cap))


def performance_index(scores: Mapping[str, Mapping[str, float | None]], reference: str,
                      directions: Mapping[str, str], normalization: str = "ratio", missing: str = "zero",
                      cap: float = 1000.0,
                      coverage: Mapping[str, Mapping[str, tuple[int, int]]] | None = None) -> PIResult:
    """Performance index against an explicit reference method (DESIGN decision 10).

    ``PI_d(A)`` is computed with :func:`pi_value` (``normalization`` "ratio" (default) or "linear") and
    ``PI(A) = mean_d PI_d(A)`` over the disciplines where the reference has a score valid for the chosen
    normalization (others are listed in ``result.excluded``). The reference therefore has PI = 100.
    A method without a (finite) score on an included discipline gets ``PI_d = 0`` when ``missing="zero"``
    (default: failure earns nothing) or is averaged over its remaining disciplines when ``missing="skip"``.

    ``coverage`` (optional, ``{method: {discipline: (have, want)}}`` from :func:`expected_coverage`) makes the index
    coverage-aware: a pooled score computed from fewer episodes than the split promises is not comparable,
    so for a method with an incomplete included discipline that discipline's PI_d and the macro PI are NaN (listed
    in ``result.incomplete``); an incomplete *reference* discipline is excluded for every method. When ``coverage``
    is given, a method or discipline missing from it counts as having no coverage.

    Returns a :class:`PIResult`: ``{method: macro PI}`` with ``.per_discipline`` = ``{method: {d: PI_d}}``.
    """
    if reference not in scores:
        raise KeyError(f"reference method {reference!r} not in scores (have {sorted(scores)})")
    if missing not in ("zero", "skip"):
        raise ValueError("missing must be 'zero' or 'skip'")
    ref = scores[reference]
    discs: list[str] = []
    excluded: dict[str, str] = {}
    all_d: list[str] = list(ref.keys()) + [d for m in scores.values() for d in m if d not in ref]
    for d in dict.fromkeys(all_d):
        if d not in directions:
            excluded[d] = "no direction"
            continue
        s_ref = ref.get(d)
        if not _finite(s_ref):
            excluded[d] = "reference score missing"
            continue
        if not math.isfinite(pi_value(float(s_ref), float(s_ref), directions[d], normalization, cap)):
            excluded[d] = f"reference score {s_ref} invalid for {normalization} normalization"
            continue
        if coverage is not None:
            h, w = coverage.get(reference, {}).get(d, (0, 1))
            if int(h) < int(w):
                excluded[d] = f"reference coverage incomplete ({int(h)}/{int(w)} episodes)"
                continue
        discs.append(d)
    per: dict[str, dict[str, float]] = {}
    macro: dict[str, float] = {}
    incomplete: dict[str, dict[str, tuple[int, int]]] = {}
    for m, sc in scores.items():
        per[m] = {}
        gaps: dict[str, tuple[int, int]] = {}
        for d in discs:
            if coverage is not None:
                h, w = coverage.get(m, {}).get(d, (0, 1))
                if int(h) < int(w):
                    gaps[d] = (int(h), int(w))
                    per[m][d] = float("nan")
                    continue
            s = sc.get(d)
            v = pi_value(s, ref[d], directions[d], normalization, cap) if _finite(s) else float("nan")
            if not math.isfinite(v) and missing == "zero":
                v = 0.0
            per[m][d] = v
        vals = [v for v in per[m].values() if math.isfinite(v)]
        macro[m] = float("nan") if gaps else (float(np.mean(vals)) if vals else float("nan"))
        if gaps:
            incomplete[m] = gaps
    return PIResult(macro, per, reference, normalization, discs, excluded, incomplete)


# ------------------------------------------------------------------------------------------------ ranks & tests


def rank_table(scores: Mapping[str, Mapping[str, float | None]], directions: Mapping[str, str]) -> dict[str, dict[str, float]]:
    """discipline -> {method: rank} (1 = best; ties share the average rank; missing scores rank last, tied)."""
    from scipy.stats import rankdata

    methods = list(scores)
    discs = [d for d in dict.fromkeys(d for m in methods for d in scores[m])
             if d in directions and any(_finite(scores[m].get(d)) for m in methods)]
    out: dict[str, dict[str, float]] = {}
    for d in discs:
        sign = -1.0 if _check_direction(directions[d]) == "max" else 1.0
        keys = [sign * float(scores[m][d]) if _finite(scores[m].get(d)) else math.inf for m in methods]
        ranks = rankdata(keys, method="average")
        out[d] = {m: float(r) for m, r in zip(methods, ranks)}
    return out


def average_ranks(scores: Mapping[str, Mapping[str, float | None]], directions: Mapping[str, str]) -> dict[str, float]:
    """Mean rank of each method over disciplines (1 = best; ties -> average rank; missing -> tied last)."""
    table = rank_table(scores, directions)
    if not table:
        return {m: float("nan") for m in scores}
    return {m: float(np.mean([table[d][m] for d in table])) for m in scores}


def friedman_test(scores: Mapping[str, Mapping[str, float | None]], directions: Mapping[str, str]) -> dict:
    """Friedman test over disciplines (blocks) and methods (Demšar 2006, JMLR 7).

    chi2_F = 12N / (k(k+1)) * [Σ_j R_j^2 - k(k+1)^2 / 4]   (R_j = average rank of method j, N blocks, k methods)
    Iman–Davenport F_F = (N-1) chi2_F / (N(k-1) - chi2_F) with (k-1, (k-1)(N-1)) d.o.f.
    Ties use average ranks; no tie correction is applied.
    """
    from scipy.stats import chi2, f as f_dist

    table = rank_table(scores, directions)
    k, n = len(scores), len(table)
    ranks = average_ranks(scores, directions)
    if k < 2 or n < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "F": float("nan"), "p_value_F": float("nan"),
                "k": k, "n": n, "avg_ranks": ranks}
    R = np.array([ranks[m] for m in scores])
    chi = 12.0 * n / (k * (k + 1)) * (float(np.sum(R ** 2)) - k * (k + 1) ** 2 / 4.0)
    p = float(chi2.sf(chi, k - 1))
    denom = n * (k - 1) - chi
    F = (n - 1) * chi / denom if denom > 0 else float("inf")
    pF = float(f_dist.sf(F, k - 1, (k - 1) * (n - 1))) if math.isfinite(F) else 0.0
    return {"statistic": float(chi), "p_value": p, "F": float(F), "p_value_F": pF, "k": k, "n": n, "avg_ranks": ranks}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference CD = q_α sqrt(k(k+1) / (6N)), q_α = studentized range quantile / sqrt(2).

    (Demšar 2006; e.g. q_0.05 = 1.960, 2.343, 2.569 for k = 2, 3, 4.) Two methods differ significantly when
    their average ranks differ by more than CD.
    """
    from scipy.stats import studentized_range

    if k < 2 or n < 1:
        raise ValueError("nemenyi_cd needs k >= 2 methods and n >= 1 blocks")
    q = float(studentized_range.ppf(1.0 - alpha, k, np.inf)) / math.sqrt(2.0)
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def sign_test(a: Mapping[str, float | None], b: Mapping[str, float | None], directions: Mapping[str, str]) -> dict:
    """Paired two-sided sign test of method a vs b over the disciplines both scored (ties dropped)."""
    from scipy.stats import binomtest

    wins = losses = ties = 0
    for d in a:
        if d not in b or d not in directions or not (_finite(a[d]) and _finite(b[d])):
            continue
        x, y = float(a[d]), float(b[d])
        if x == y:
            ties += 1
        elif (x > y) == (_check_direction(directions[d]) == "max"):
            wins += 1
        else:
            losses += 1
    n = wins + losses
    p = float(binomtest(wins, n, 0.5).pvalue) if n else 1.0
    return {"wins": wins, "losses": losses, "ties": ties, "n": n, "p_value": p}


# ------------------------------------------------------------------------------------------------ transfer


def transfer_metrics(matrix: Any, baseline: Any = None) -> dict:
    """Continual-learning transfer statistics of a square matrix ``R`` (T x T, NaN = not evaluated).

    ``R[i, j]`` = score on task j (column) after stage i (row). ``baseline[j]`` = score of the initial program
    on task j (default 0, i.e. ``R`` already holds gains over the initial program — e.g. the paper's
    source-family x target-family matrix of PI-point gains). ``G = R - baseline`` (column-wise).
    Scores must be higher-is-better (pass PI or negated errors). With 0-based indices:

    Source -> target reading (paper Fig. 5: rows = source family, columns = target family):

    * ``FWT_allcell``  = mean over all T^2 cells of G            (the paper's "all-cell convention")
    * ``FWT_offdiag``  = mean over the T(T-1) cells i != j of G  (the paper's "strictly off-diagonal mean")
    * ``NT``           = fraction of finite off-diagonal cells with G[i, j] < 0 (negative transfer)
      (+ ``n_offdiag``, ``n_offdiag_positive``, ``n_offdiag_negative``)

    Sequence-of-evaluations reading (rows = after learning tasks 0..i in order; GEM, Lopez-Paz & Ranzato 2017):

    * ``FWT_GEM``   = 1/(T-1) Σ_{j=1}^{T-1} (R[j-1, j] - b[j])     (score on task j just before learning it)
    * ``FWT_upper`` = mean_{i<j} (R[i, j] - b[j])                  (Díaz-Rodríguez et al. 2018)
    * ``BWT``       = 1/(T-1) Σ_{j=0}^{T-2} (R[T-1, j] - R[j, j])   (negative = forgetting)
    * ``forgetting`` = 1/(T-1) Σ_{j=0}^{T-2} (max_{l∈0..T-2} R[l, j] - R[T-1, j])   (Chaudhry et al. 2018, F_T)

    Undefined terms (NaN cells) are skipped; a statistic with no defined term is NaN.
    """
    R = np.asarray(matrix, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"transfer_metrics needs a square matrix, got shape {R.shape}")
    T = R.shape[0]
    b = np.zeros(T) if baseline is None else np.asarray(baseline, dtype=float).reshape(-1)
    if b.shape[0] != T:
        raise ValueError("baseline length must equal the matrix size")
    G = R - b[None, :]
    off = ~np.eye(T, dtype=bool)
    offv = G[off]
    offv = offv[np.isfinite(offv)]

    def _mean(v: np.ndarray) -> float:
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    fwt_gem = _mean(np.array([G[j - 1, j] for j in range(1, T)])) if T > 1 else float("nan")
    upper = _mean(G[np.triu_indices(T, k=1)]) if T > 1 else float("nan")
    bwt = _mean(np.array([R[T - 1, j] - R[j, j] for j in range(T - 1)])) if T > 1 else float("nan")
    forg_terms = []
    for j in range(T - 1):
        col = R[: T - 1, j]
        col = col[np.isfinite(col)]
        if col.size and np.isfinite(R[T - 1, j]):
            forg_terms.append(float(col.max() - R[T - 1, j]))
    return {
        "T": T,
        "FWT_allcell": _mean(G.reshape(-1)),
        "FWT_offdiag": float(offv.mean()) if offv.size else float("nan"),
        "FWT_GEM": fwt_gem,
        "FWT_upper": upper,
        "BWT": bwt,
        "forgetting": float(np.mean(forg_terms)) if forg_terms else float("nan"),
        "NT": float((offv < 0).mean()) if offv.size else float("nan"),
        "n_offdiag": int(offv.size),
        "n_offdiag_positive": int((offv > 0).sum()),
        "n_offdiag_negative": int((offv < 0).sum()),
    }


# ------------------------------------------------------------------------------------------------ bootstrap


def bootstrap_ci(values: Iterable[float], n: int = 2000, alpha: float = 0.05, seed: int = 0,
                 stat: Callable[[np.ndarray], float] | None = None) -> tuple[float, float]:
    """Percentile bootstrap (1-alpha) CI of ``stat`` (default: mean) over i.i.d. resamples of ``values``.

    Non-finite values are dropped. Empty input -> (nan, nan); a single value -> (v, v).
    """
    v = np.asarray([x for x in values if _finite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(int(n), v.size))
    samples = v[idx]
    stats = samples.mean(axis=1) if stat is None else np.array([stat(s) for s in samples], dtype=float)
    return float(np.quantile(stats, alpha / 2.0)), float(np.quantile(stats, 1.0 - alpha / 2.0))
