"""Tables from run receipts (paper Sec. 4.2 metrics and the Sec. 6 analyses).

``build_report(run_dir)`` reads ``eval/results.jsonl`` (+ ``splits.json``, ``candidates.jsonl``,
``usage.json``) and writes ``<run_dir>/report/``:

* ``summary.csv``        snapshot x split: MacroSR (+ bootstrap CI over disciplines), mean normalized score,
                         PI vs the explicit reference (+ CI), coverage, cost totals
* ``per_discipline.csv`` snapshot x split x discipline: success rate, pooled task-native score (+ its kind),
                         PI_d, coverage
* ``per_family.csv``     snapshot x split x family (and natural/social sphere): MacroSR and PI
* ``costs.csv``          snapshot x split: summed usage (tokens, calls, wall time ...); ``logical_tokens`` (spent +
                         cached, alias rows at their root's cost) next to ``billed_tokens`` (actually spent)
* ``promotions.csv``     evolution candidates / promotions per round (from ``candidates.jsonl``)
* ``rep_matrix_*.csv``   D_rep retention matrices (rows = snapshot A_r, columns = source round j)
* ``family_transfer.csv`` source-family x target-family PI-point gains (if family programs were evaluated)
* ``report.json`` (everything, incl. the ``usage.jsonl`` ledger sums and the provenance receipts of every phase) and
  ``report.md`` (human-readable)

Conventions (all stated in the report itself):

* PI uses :func:`bench.metrics.performance_index` with an explicit reference (default the frozen ``A_0``;
  ``pi_reference="method_final"`` = the last snapshot, as in the paper) and normalization "ratio" (default) or
  "linear"; a discipline without a pooled score for a snapshot gets PI_d = 0.
* Pooled scores come from ``adapter.pooled_metric`` over the episode payloads; if the adapter cannot be
  loaded the mean episode ``primary`` is used and flagged ``score_kind = "episode_mean"``.
* Natural sciences = families "Life & health", "Physical & Earth", "Engineering & computing"; social sciences
  and humanities = "Social & behavior", "Humanities & law" (reporting convention of this rebuild).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..bench import metrics as M
from ..bench.registry import BY_CODE, FAMILIES
from .evaluate import RESULTS, family_slug, read_results, snapshot_round

NATURAL = ("Life & health", "Physical & Earth", "Engineering & computing")
SOCIAL = ("Social & behavior", "Humanities & law")
COST_KEYS = ("policy_prompt_tokens", "policy_completion_tokens", "executor_prompt_tokens",
             "executor_completion_tokens", "total_tokens", "llm_calls", "wall_s", "policy_wall_s", "node_runs",
             "replays")


def sphere_of(family: str | None) -> str | None:
    if family in NATURAL:
        return "natural"
    if family in SOCIAL:
        return "social"
    return None


def _num(x: Any) -> float:
    try:
        return float(x) if x is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(values: Iterable[Any]) -> float:
    """Mean of the finite values (NaN if there are none; no RuntimeWarning)."""
    v = [float(x) for x in values if math.isfinite(_num(x))]
    return float(np.mean(v)) if v else float("nan")


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return "–" if not math.isfinite(float(x)) else f"{float(x):.{nd}f}"
    return str(x)


def md_table(rows: list[dict], cols: list[str] | None = None, nd: int = 3) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(_fmt(r.get(c), nd) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------------------------------- loading
def load_candidates(run_dir: Path) -> list[dict]:
    p = run_dir / "candidates.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _is_promoted(c: Mapping[str, Any]) -> bool:
    for k in ("promoted", "accepted", "admitted", "admit"):
        if k in c and isinstance(c[k], bool):
            return c[k]
    dec = str(c.get("decision", c.get("status", ""))).lower()
    return dec in ("accept", "accepted", "promote", "promoted", "admit", "admitted")


def promotion_rows(cands: list[dict]) -> list[dict]:
    by_round: dict[Any, list[dict]] = defaultdict(list)
    for c in cands:
        by_round[c.get("round", "?")].append(c)
    rows = []
    for r in sorted(by_round, key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else str(x))):
        cs = by_round[r]
        n_prom = sum(_is_promoted(c) for c in cs)
        rows.append({"round": r, "candidates": len(cs), "promoted": n_prom,
                     "rate": n_prom / len(cs) if cs else float("nan")})
    if rows:
        tot, prom = sum(r["candidates"] for r in rows), sum(r["promoted"] for r in rows)
        rows.append({"round": "total", "candidates": tot, "promoted": prom, "rate": prom / tot if tot else float("nan")})
    return rows


def _load_adapters_for(run_dir: Path, codes: Iterable[str], adapters: Mapping[str, Any] | None,
                       warnings: list[str]) -> dict[str, Any]:
    if adapters is not None:
        return dict(adapters)
    from ..bench.splits import load_adapters
    from ..config import load_config

    try:
        cfg = load_config(run_dir / "config.yaml")
    except FileNotFoundError:
        warnings.append("config.yaml missing; adapters not loaded (pooled scores fall back to episode means)")
        return {}
    cfg.bench.disciplines = sorted(set(codes))
    try:
        return load_adapters(cfg.bench, strict=False)
    except (ImportError, RuntimeError, OSError) as ex:
        warnings.append(f"adapters not loaded ({type(ex).__name__}: {ex}); pooled scores fall back to episode means")
        return {}


# --------------------------------------------------------------------------------------------------- analysis
def _group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in keys)].append(r)
    return g


def _snap_order(names: Iterable[str]) -> list[str]:
    std = sorted((n for n in names if snapshot_round(n) is not None), key=lambda n: snapshot_round(n))
    return std + sorted(n for n in names if snapshot_round(n) is None)


def expected_by_discipline(manifest: Mapping[str, Any], split: str, snapshot: str | None = None) -> dict[str, int] | None:
    """Episodes the split manifest promises per discipline for ``split`` (None if the manifest cannot tell).

    D_rep depends on the snapshot: A_r re-solves the frozen copies of the source episodes of rounds <= r (the rep
    manifest lists ``{id, round}`` only, so the discipline is looked up through the source episode with that id).
    """
    if not manifest:
        return None
    if split in ("id", "ood", "val", "src"):
        by_d = manifest.get("splits", {}).get(split)
        return {d: len(v) for d, v in by_d.items()} if by_d else None
    if split == "rep":
        r = snapshot_round(snapshot) if snapshot else None
        if r is None:
            return None
        src_d = {e["id"]: d for d, eps in manifest.get("splits", {}).get("src", {}).items() for e in eps}
        out: dict[str, int] = defaultdict(int)
        for e in manifest.get("rep", {}).get("episodes", []):
            if (e.get("round") or 0) <= r and e.get("id") in src_d:
                out[src_d[e["id"]]] += 1
        return dict(out)
    return None


def _pooled_by_discipline(rows: list[dict], adapters: Mapping[str, Any], base_dir: Path,
                          warnings: list[str], expected: Mapping[str, int] | None = None
                          ) -> dict[str, tuple[float | None, str, int, int]]:
    """discipline -> (pooled score, kind, n_with_payload, n).

    ``n`` is the number of episodes the split promises (``expected``, from the manifest) when known, else the number
    of result rows; ``n_with_payload`` counts distinct episodes with a pooled payload. ``n_with_payload < n`` means
    the pooled score covers only part of the split (LEAK-5): it must not be compared with a complete one.
    """
    out = {}
    for (d,), rs in _group(rows, "discipline").items():
        if expected is not None and d in expected:
            cov = M.expected_coverage({d: rs}, {d: expected[d]}, base_dir)[d]
        else:
            cov = M.pooled_coverage({d: rs}, base_dir)[d]
        score, kind = None, "none"
        if d in adapters:
            try:
                score = M.pooled_scores({d: rs}, adapters, base_dir)[d]
                kind = "pooled"
            except (ValueError, KeyError, TypeError, OSError) as ex:
                warnings.append(f"{d}: pooled_metric failed ({type(ex).__name__}: {ex}); using episode mean")
        if kind != "pooled":
            prim = [_num(r.get("primary")) for r in rs if math.isfinite(_num(r.get("primary")))]
            score, kind = (float(np.mean(prim)) if prim else None), "episode_mean"
        out[d] = (score, kind, cov[0], cov[1])
    return out


def rep_matrix(rows: list[dict], value: str = "z") -> tuple[np.ndarray, list[int]]:
    """R[i, j] = mean ``value`` of snapshot A_{i+1} on the rep episodes of source round j+1 (NaN if i < j).

    Averaging is macro over disciplines (per-discipline mean first) so every discipline weighs equally.
    """
    rep = [r for r in rows if r.get("split") == "rep" and snapshot_round(r["snapshot"]) and r.get("round")]
    rounds = sorted({int(r["round"]) for r in rep} | {snapshot_round(r["snapshot"]) for r in rep})
    if not rounds:
        return np.zeros((0, 0)), []
    T = max(rounds)
    R = np.full((T, T), np.nan)
    for (snap, rnd), rs in _group(rep, "snapshot", "round").items():
        i, j = snapshot_round(snap) - 1, int(rnd) - 1
        per_d = defaultdict(list)
        for r in rs:
            v = _num(r.get(value)) if value != "z" else float(int(r.get("z") or 0))
            if math.isfinite(v):
                per_d[r["discipline"]].append(v)
        if per_d:
            R[i, j] = float(np.mean([np.mean(v) for v in per_d.values()]))
    return R, list(range(1, T + 1))


# ------------------------------------------------------------------------------------------------------- cost
def logical_tokens_of(usage: Mapping[str, Any]) -> float:
    """Prompt + completion tokens of one solve irrespective of caching (spent + cached; policy and executor).

    ``policy_logical_tokens`` is written by the solver; older receipts fall back to the policy prompt/completion
    (+ cached) fields. This is the cost of the program as if nothing had been served from the response cache, so
    it is comparable between a cold run and a re-run; ``total_tokens`` (spent) is what the gateway billed.
    """
    def g(k: str) -> float:
        v = usage.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

    pol = g("policy_logical_tokens") if "policy_logical_tokens" in usage else (
        g("policy_prompt_tokens") + g("policy_completion_tokens") + g("policy_cached_prompt_tokens")
        + g("policy_cached_completion_tokens"))
    return pol + g("executor_prompt_tokens") + g("executor_completion_tokens") \
        + g("executor_cached_prompt_tokens") + g("executor_cached_completion_tokens")


def row_costs(rs: Iterable[Mapping[str, Any]]) -> tuple[dict[str, float], float, int]:
    """(summed spent usage, logical tokens, #alias rows) of result rows.

    Alias rows (M5: identical program, outcome copied from the solved snapshot) spent nothing (``usage`` empty), but
    their logical cost is the root's (``alias_usage``): the spent sum is the real bill, the logical sum the cost the
    evaluation would have without de-duplication.
    """
    spent: dict[str, float] = defaultdict(float)
    logical, n_alias = 0.0, 0
    for r in rs:
        u = r.get("usage") or {}
        for k, v in u.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                spent[k] += v
        alias = bool(r.get("alias_of"))
        n_alias += alias
        logical += logical_tokens_of((r.get("alias_usage") or {}) if alias else u)
    return dict(spent), logical, n_alias


def _ledger_report(run_dir: Path, warnings: list[str]) -> dict:
    """Sum of ``usage.jsonl`` (all phases) plus per-phase rows; warns about killed/failed segments."""
    from .provenance import read_ledger, sum_ledger

    recs = read_ledger(run_dir / "usage.jsonl")
    if not recs:
        return {}
    tot = sum_ledger(recs)
    rows = []
    for ph, d in tot["by_phase"].items():
        t = (d.get("llm") or {}).get("total") or {}
        spent = float(t.get("prompt_tokens", 0)) + float(t.get("completion_tokens", 0))
        cached = float(t.get("cached_prompt_tokens", 0)) + float(t.get("cached_completion_tokens", 0))
        rows.append({"phase": ph, "segments": d["segments"], "wall_s": d["wall_s"], "llm_calls": t.get("calls", 0),
                     "cached_calls": t.get("cached_calls", 0), "billed_tokens": spent, "cached_tokens": cached,
                     "logical_tokens": spent + cached})
    if tot["unclosed"]:
        warnings.append(f"usage ledger: {len(tot['unclosed'])} segment(s) never closed (process killed?): "
                        f"{tot['unclosed']}; their LLM spend is not recorded, totals are a lower bound")
    if tot["failed"]:
        warnings.append(f"usage ledger: {len(tot['failed'])} segment(s) ended with an error: {tot['failed']}")
    return {"total": tot, "by_phase": rows}


def _provenance_report(run_dir: Path, warnings: list[str]) -> list[dict]:
    """One row per recorded provenance receipt (run.json first segment / resumes, eval_log.jsonl phases)."""
    recs: list[tuple[str, dict]] = []
    try:
        status = json.loads((run_dir / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        status = {}
    if isinstance(status.get("provenance"), dict):
        recs.append(("evolve", status["provenance"]))
    elif isinstance(status.get("code"), dict):
        recs.append(("evolve", {"code": status["code"], "at": status.get("started")}))
    for i, rs in enumerate(status.get("resumes") or []):
        p = rs.get("provenance") if isinstance(rs, dict) else None
        recs.append((f"resume{i + 1}", p if isinstance(p, dict) else {"code": (rs or {}).get("code"),
                                                                      "at": (rs or {}).get("at")}))
    log = run_dir / "eval" / "eval_log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d.get("provenance"), dict):
                recs.append((str(d.get("kind") or d["provenance"].get("phase") or "eval"), d["provenance"]))
    rows = []
    for label, p in recs:
        code = p.get("code") or {}
        rows.append({"segment": label, "phase": p.get("phase"), "at": p.get("at"),
                     "git_commit": code.get("git_commit"), "git_dirty": code.get("git_dirty"),
                     "git_diff_sha256": code.get("git_diff_sha256"), "config_sha256": p.get("config_sha256"),
                     "split_sha256": p.get("split_sha256") or p.get("splits_json_sha256"),
                     "data_manifests": (p.get("data") or {}).get("manifests"),
                     "cache_path": (p.get("llm") or {}).get("cache_path"),
                     "eval_workers": p.get("eval_workers"), "llm_concurrency": (p.get("llm") or {}).get("concurrency"),
                     "python": code.get("python")})
        for w in p.get("warnings") or []:
            warnings.append(f"{label}: {w}")
    if not rows:
        warnings.append("no provenance receipts found in run.json / eval_log.jsonl (run predates the receipts)")
        return rows
    for key, what in (("git_commit", "code commit"), ("config_sha256", "config"), ("cache_path", "LLM cache path")):
        vals = {r[key] for r in rows if r.get(key)}
        if len(vals) > 1:
            warnings.append(f"provenance: {what} differs between phases/segments ({sorted(map(str, vals))})")
    dirty = [r["segment"] for r in rows if r.get("git_dirty")]
    if dirty:
        diffs = sorted({str(r.get("git_diff_sha256"))[:12] for r in rows if r.get("git_dirty")})
        warnings.append(f"provenance: uncommitted changes in the working tree during {dirty} (diff sha256 {diffs}); "
                        "the commit hash alone does not identify the code")
    if not any(r.get("git_commit") for r in rows):
        warnings.append("provenance: no git commit recorded")
    return rows


def build_report(run_dir: str | Path, adapters: Mapping[str, Any] | None = None, pi_reference: str = "A_0",
                 normalization: str = "ratio", n_boot: int = 2000) -> Path:
    """Compute all tables for a run; returns the path of ``report/report.md``."""
    run_dir = Path(run_dir)
    eval_dir = run_dir / "eval"
    rows = read_results(eval_dir / RESULTS)
    out_dir = run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    manifest = json.loads((run_dir / "splits.json").read_text()) if (run_dir / "splits.json").exists() else {}
    codes = sorted({r["discipline"] for r in rows})
    adapters = _load_adapters_for(run_dir, codes, adapters, warnings)
    directions: dict[str, str] = {}
    families: dict[str, str] = {}
    for r in rows:
        if r.get("direction") in ("max", "min"):
            directions.setdefault(r["discipline"], r["direction"])
        if r.get("family"):
            families.setdefault(r["discipline"], r["family"])
    for d in codes:
        if d not in directions:
            ad = adapters.get(d)
            directions[d] = getattr(ad, "direction", None) or (BY_CODE[d].direction if d in BY_CODE else "max")
            warnings.append(f"{d}: direction not in results; using {directions[d]!r}")
        if d not in families and d in BY_CODE:
            families[d] = BY_CODE[d].family

    std_rows = [r for r in rows if not r.get("family_source")]
    fam_rows = [r for r in rows if r.get("family_source")]
    splits = [s for s in ("id", "ood", "val", "rep") if any(r["split"] == s for r in std_rows)]
    snaps = _snap_order({r["snapshot"] for r in std_rows})
    ref_name = (snaps[-1] if snaps else "A_0") if pi_reference == "method_final" else pi_reference

    summary, per_disc, per_fam, costs = [], [], [], []
    pi_objs: dict[str, dict] = {}
    for split in splits:
        srows = [r for r in std_rows if r["split"] == split]
        scores: dict[str, dict[str, float | None]] = {}
        pooled_info: dict[str, dict] = {}
        cov: dict[str, dict[str, tuple[int, int]]] = {}
        for snap in snaps:
            rs = [r for r in srows if r["snapshot"] == snap]
            if not rs:
                continue
            exp_d = expected_by_discipline(manifest, split, snap)
            pooled_info[snap] = _pooled_by_discipline(rs, adapters, eval_dir, warnings, exp_d)
            scores[snap] = {d: v[0] for d, v in pooled_info[snap].items()}
            cov[snap] = {d: (v[2], v[3]) for d, v in pooled_info[snap].items()}
            for d, want in (exp_d or {}).items():        # promised but never solved: coverage 0
                cov[snap].setdefault(d, (0, int(want)))
        pi = None
        if ref_name in scores:
            # LEAK-5: coverage-aware PI -- a pooled score over fewer episodes than the split promises is not comparable
            pi = M.performance_index(scores, ref_name, directions, normalization=normalization, coverage=cov)
            pi_objs[split] = pi.to_dict()
            for d, why in pi.excluded.items():
                if "coverage" in why:
                    warnings.append(f"{split}/{d}: excluded from PI ({why})")
            for m_, gaps in pi.incomplete.items():
                warnings.append(f"{split}/{m_}: PI is NaN, pooled payload coverage incomplete on "
                                + ", ".join(f"{d} ({h}/{w})" for d, (h, w) in sorted(gaps.items())))
        elif scores:
            warnings.append(f"{split}: PI reference {ref_name!r} not evaluated on this split; PI omitted")
        for snap in scores:
            rs = [r for r in srows if r["snapshot"] == snap]
            z_by_d = {d: [int(x.get("z") or 0) for x in g] for (d,), g in _group(rs, "discipline").items()}
            sr_d = M.success_rates(z_by_d)
            norm_d = {d: _nanmean(x.get("norm_score") for x in g) for (d,), g in _group(rs, "discipline").items()}
            pi_d = pi.per_discipline.get(snap, {}) if pi is not None else {}
            gaps_s = M.coverage_gaps(cov[snap])
            pi_incomplete = pi is not None and snap in pi.incomplete
            lo, hi = M.bootstrap_ci(list(sr_d.values()), n=n_boot)
            plo, phi = (M.bootstrap_ci(list(pi_d.values()), n=n_boot) if pi_d and not pi_incomplete
                        else (float("nan"), float("nan")))
            exp_d = expected_by_discipline(manifest, split, snap)
            expected = sum(exp_d.values()) if exp_d is not None else None
            spent, logical, n_alias = row_costs(rs)
            cost = defaultdict(float, spent)
            summary.append({"snapshot": snap, "split": split, "macro_sr": M.macro_sr(z_by_d), "macro_sr_ci_lo": lo,
                            "macro_sr_ci_hi": hi,
                            "mean_norm_score": _nanmean(norm_d.values()),
                            "pi": pi.get(snap) if pi is not None else float("nan"), "pi_ci_lo": plo, "pi_ci_hi": phi,
                            "n_disciplines": len(z_by_d), "n_episodes": len(rs), "n_expected": expected,
                            "complete": not gaps_s, "incomplete_disciplines": sorted(gaps_s),
                            "n_failed": sum(1 for x in rs if x.get("failed")), "n_alias": n_alias,
                            "logical_tokens": logical, "billed_tokens": cost.get("total_tokens", 0.0),
                            "total_tokens": cost.get("total_tokens", 0.0), "llm_calls": cost.get("llm_calls", 0.0),
                            "wall_s": cost.get("wall_s", 0.0)})
            costs.append({"snapshot": snap, "split": split, "logical_tokens": logical,
                          "billed_tokens": cost.get("total_tokens", 0.0), "n_alias": n_alias,
                          **{k: cost.get(k, 0.0) for k in COST_KEYS},
                          **{k: v for k, v in cost.items() if k not in COST_KEYS}})
            for d in sorted(z_by_d, key=lambda c: (c not in BY_CODE, c)):
                sc, kind, n_pay, n = pooled_info[snap][d]
                per_disc.append({"snapshot": snap, "split": split, "discipline": d, "family": families.get(d),
                                 "direction": directions.get(d), "success_rate": sr_d[d], "pooled_score": sc,
                                 "score_kind": kind, "pi_d": pi_d.get(d, float("nan")),
                                 "mean_norm_score": norm_d.get(d), "n": n, "n_scored": n_pay,
                                 "complete": n_pay >= n})
            for grp_name, grp in (("family", families), ("sphere", {d: sphere_of(f) for d, f in families.items()})):
                sr_g = M.group_macro(sr_d, grp)
                # a group PI over a partly missing set of disciplines is not comparable either
                pi_g = M.group_macro({d: v for d, v in pi_d.items() if d not in gaps_s}, grp) if pi_d else {}
                for g_ in {grp.get(d) for d in gaps_s}:
                    pi_g.pop(g_, None)
                for g in sorted(sr_g):
                    per_fam.append({"snapshot": snap, "split": split, "group_kind": grp_name, "group": g,
                                    "macro_sr": sr_g[g], "pi": pi_g.get(g, float("nan")),
                                    "n_disciplines": sum(1 for d in sr_d if grp.get(d) == g)})

    # ---- D_rep retention (GEM-style matrix over source rounds)
    rep_stats = {}
    for value in ("z", "norm_score"):
        R, rounds = rep_matrix(std_rows, value)
        if R.size:
            rep_stats[value] = {"matrix": R.tolist(), "rounds": rounds, **{k: v for k, v in M.transfer_metrics(R).items()
                                                                          if k in ("BWT", "forgetting", "T")}}
            write_csv(out_dir / f"rep_matrix_{value}.csv",
                      [{"snapshot": f"A_{i + 1}", **{f"round_{j}": R[i, j - 1] for j in rounds}} for i in range(R.shape[0])])

    # ---- family transfer matrix (source family x target family, PI-point gain over A_0)
    fam_transfer = None
    if fam_rows:
        fam_transfer = _family_transfer(fam_rows, std_rows, adapters, eval_dir, directions, families, normalization,
                                        warnings, manifest)
        if fam_transfer:
            write_csv(out_dir / "family_transfer.csv",
                      [{"source": s, **{t: fam_transfer["matrix"][i][j] for j, t in enumerate(fam_transfer["families"])}}
                       for i, s in enumerate(fam_transfer["families"])])

    cands = load_candidates(run_dir)
    promos = promotion_rows(cands)
    usage = json.loads((run_dir / "usage.json").read_text()) if (run_dir / "usage.json").exists() else {}
    ledger = _ledger_report(run_dir, warnings)
    prov_rows = _provenance_report(run_dir, warnings)
    n_alias_all = sum(1 for r in rows if r.get("alias_of"))
    if n_alias_all:
        warnings.append(f"{n_alias_all} result row(s) are aliases of an identical program (alias_of): their outcome "
                        "is copied, `billed_tokens`/`wall_s` count them as 0 and `logical_tokens` as the root's cost")
    n_failed = sum(1 for r in rows if r.get("failed"))
    if n_failed:
        warnings.append(f"{n_failed} result row(s) are failed episodes (no usable output): they enter the pooled "
                        "score with the adapter's uniform failure payload (reference level or worse), not as gaps")
    try:
        from ..config import load_config
        rep_cfg = load_config(run_dir / "config.yaml")
    except Exception:
        rep_cfg = None
    from .provenance import provenance as _provenance
    report_prov = _provenance(rep_cfg, "report", extra={"pi_reference": ref_name, "normalization": normalization,
                                                        "n_boot": n_boot, "split_sha256": manifest.get("sha256")})

    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "per_discipline.csv", per_disc)
    write_csv(out_dir / "per_family.csv", per_fam)
    write_csv(out_dir / "costs.csv", costs)
    write_csv(out_dir / "promotions.csv", promos)
    report = {"run_dir": str(run_dir), "pi_reference": ref_name, "pi_reference_rule": pi_reference,
              "normalization": normalization, "summary": summary, "per_discipline": per_disc, "per_family": per_fam,
              "costs": costs, "promotions": promos, "rep_retention": rep_stats, "family_transfer": fam_transfer,
              "pi": pi_objs, "evolution_usage": usage, "usage_ledger": ledger, "provenance": prov_rows,
              "report_provenance": report_prov, "split_sha256": manifest.get("sha256"),
              "warnings": sorted(set(warnings))}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=_jsonable_float))
    md_path = out_dir / "report.md"
    md_path.write_text(_render_md(report))
    return md_path


def _jsonable_float(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _family_transfer(fam_rows: list[dict], std_rows: list[dict], adapters: Mapping[str, Any], eval_dir: Path,
                     directions: Mapping[str, str], families: Mapping[str, str], normalization: str,
                     warnings: list[str], manifest: Mapping[str, Any] | None = None) -> dict | None:
    split = fam_rows[0]["split"]
    ref_rows = [r for r in std_rows if r["snapshot"] == "A_0" and r["split"] == split]
    if not ref_rows:
        warnings.append(f"family transfer: A_0 not evaluated on split {split!r}; matrix omitted")
        return None
    exp_d = expected_by_discipline(manifest or {}, split, "A_0")
    cov: dict[str, dict[str, tuple[int, int]]] = {}

    def pooled(label: str, rs: list[dict]) -> dict[str, float | None]:
        info = _pooled_by_discipline(rs, adapters, eval_dir, warnings, exp_d)
        cov[label] = {d: (v[2], v[3]) for d, v in info.items()}
        for d, want in (exp_d or {}).items():
            cov[label].setdefault(d, (0, int(want)))
        return {d: v[0] for d, v in info.items()}

    scores = {"A_0": pooled("A_0", ref_rows)}
    src_fams = [f for f in FAMILIES if any(r["family_source"] == f for r in fam_rows)]
    for f in src_fams:
        rs = [r for r in fam_rows if r["family_source"] == f]
        scores[family_slug(f)] = pooled(family_slug(f), rs)
    pi = M.performance_index(scores, "A_0", directions, normalization=normalization, coverage=cov)
    for m_, gaps in pi.incomplete.items():
        warnings.append(f"family transfer/{m_}: pooled payload coverage incomplete on "
                        + ", ".join(f"{d} ({h}/{w})" for d, (h, w) in sorted(gaps.items())) + "; cells are NaN")
    tgt_fams = [f for f in FAMILIES if any(families.get(d) == f for d in pi.disciplines)]
    fams = [f for f in FAMILIES if f in src_fams and f in tgt_fams]
    Mx = np.full((len(fams), len(fams)), np.nan)
    for i, s in enumerate(fams):
        pi_t = M.group_macro(pi.per_discipline[family_slug(s)], families)
        ref_t = M.group_macro(pi.per_discipline["A_0"], families)
        gap_fams = {families.get(d) for d in pi.incomplete.get(family_slug(s), {})}
        for j, t in enumerate(fams):
            if t in pi_t and t in ref_t and t not in gap_fams:      # an incomplete target family stays NaN
                Mx[i, j] = pi_t[t] - ref_t[t]
    stats = M.transfer_metrics(Mx) if Mx.size else {}
    return {"split": split, "families": fams, "matrix": Mx.tolist(), "unit": "PI points vs A_0",
            **{k: stats.get(k) for k in ("FWT_allcell", "FWT_offdiag", "NT", "n_offdiag", "n_offdiag_positive",
                                         "n_offdiag_negative")}}


def _render_md(rep: dict) -> str:
    L = [f"# ScienceClaw rebuild — run report", "",
         f"Run: `{rep['run_dir']}`  ", f"Split manifest sha256: `{rep.get('split_sha256')}`", "",
         "> Rebuild experiment. Every number below is computed from this run's receipts "
         "(eval/results.jsonl); none is copied from, or back-filled into, the submitted paper.", "",
         "## Conventions", "",
         f"* MacroSR (Eq. 4): mean over disciplines of the per-discipline success rate of z (95% percentile "
         f"bootstrap CI over disciplines).",
         f"* PI reference: **{rep['pi_reference']}** (rule `{rep['pi_reference_rule']}`); normalization "
         f"**{rep['normalization']}**: " + ("PI_d = 100·s/s_ref (higher-better), 100·s_ref/s (lower-better)"
                                            if rep["normalization"] == "ratio" else
                                            "PI_d = 100·(1 ± (s − s_ref)/|s_ref|) (+ higher-better, − lower-better)")
         + "; PI = mean_d PI_d; a missing pooled score gives PI_d = 0. Every held-out episode contributes one "
         "payload (a failed episode: the adapter's uniform failure payload); if fewer payloads than the split "
         "promises exist for a discipline (unfinished / errored solves) the snapshot's PI is NaN, not a partial pool.",
         "* Pooled score: adapter `pooled_metric` over episode payloads (`score_kind = episode_mean` marks the "
         "fallback to the mean episode primary metric).", ""]
    L += ["## Summary (snapshot × split)", "",
          md_table(rep["summary"], ["snapshot", "split", "macro_sr", "macro_sr_ci_lo", "macro_sr_ci_hi",
                                    "mean_norm_score", "pi", "pi_ci_lo", "pi_ci_hi", "n_episodes", "n_expected",
                                    "complete", "logical_tokens", "billed_tokens"]),
          "", "`complete` = every discipline's pooled payload covers all episodes the split promises; a PI over an "
              "incomplete snapshot is NaN (LEAK-5: partial pools are not comparable).", ""]
    fam = [r for r in rep["per_family"] if r["group_kind"] == "family"]
    sph = [r for r in rep["per_family"] if r["group_kind"] == "sphere"]
    L += ["## By family", "", md_table(fam, ["snapshot", "split", "group", "macro_sr", "pi", "n_disciplines"]), "",
          "## Natural vs social sciences", "", md_table(sph, ["snapshot", "split", "group", "macro_sr", "pi"]), "",
          "## Per discipline", "",
          md_table(rep["per_discipline"], ["snapshot", "split", "discipline", "family", "direction", "success_rate",
                                           "pooled_score", "score_kind", "pi_d", "n", "n_scored"], nd=4), ""]
    L += ["## Evolution promotions (candidates.jsonl)", "", md_table(rep["promotions"]), ""]
    if rep["rep_retention"]:
        L += ["## Retention on D_rep", "",
              "R[r, j] = macro score of snapshot A_r on the source episodes of round j (j ≤ r). "
              "BWT = mean_j (R[R, j] − R[j, j]); forgetting = mean_j (max_{l<R} R[l, j] − R[R, j]).", ""]
        for k, v in rep["rep_retention"].items():
            L.append(f"* {k}: BWT = {_fmt(v.get('BWT'))}, forgetting = {_fmt(v.get('forgetting'))} (T = {v.get('T')})")
        L.append("")
    ft = rep.get("family_transfer")
    if ft:
        L += ["## Family transfer (source family → target family, PI points vs A_0)", "",
              md_table([{"source": s, **{t: ft["matrix"][i][j] for j, t in enumerate(ft["families"])}}
                        for i, s in enumerate(ft["families"])]),
              f"FWT_allcell = {_fmt(ft.get('FWT_allcell'))}; FWT_offdiag (strictly off-diagonal) = "
              f"{_fmt(ft.get('FWT_offdiag'))}; NT (negative off-diagonal fraction) = {_fmt(ft.get('NT'))} "
              f"({ft.get('n_offdiag_positive')}/{ft.get('n_offdiag')} off-diagonal cells positive).", ""]
    L += ["## Cost", "",
          "`logical_tokens` = prompt + completion tokens of the solves irrespective of the response cache (policy + "
          "executor); `billed_tokens` = tokens actually spent at the gateway (cache hits and alias rows count 0). "
          "Compare programs by `logical_tokens`; `billed_tokens` shows what this run cost.", "",
          md_table(rep["costs"], ["snapshot", "split", "logical_tokens", "billed_tokens", "llm_calls", "wall_s",
                                  "replays", "n_alias"]),
          "", f"Evolution LLM usage (sum of the evolve ledger segments): "
              f"`{json.dumps((rep.get('evolution_usage') or {}).get('llm', {}).get('total', {}), default=str)}`",
          ""]
    led = rep.get("usage_ledger") or {}
    if led:
        L += ["### Usage ledger (usage.jsonl, all phases and resume segments)", "",
              md_table(led.get("by_phase", []), ["phase", "segments", "wall_s", "llm_calls", "cached_calls",
                                                 "logical_tokens", "billed_tokens", "cached_tokens"]), ""]
    if rep.get("provenance"):
        L += ["## Provenance", "",
              "Receipts of every phase (code, config, data manifests, LLM settings); see `report.json` for the full "
              "records. A `git_dirty` phase is identified by `git_diff_sha256`, not by the commit alone.", "",
              md_table(rep["provenance"], ["segment", "phase", "at", "git_commit", "git_dirty", "config_sha256",
                                           "split_sha256", "eval_workers", "llm_concurrency", "cache_path"]), ""]
    if rep["warnings"]:
        L += ["## Warnings", ""] + [f"* {w}" for w in rep["warnings"]] + [""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------ method comparison
def compare_runs(runs: Mapping[str, str | Path], split: str = "ood", snapshot: str = "final",
                 adapters: Mapping[str, Any] | None = None, alpha: float = 0.05) -> dict:
    """Compare methods (one run each) on one split: pooled scores, average ranks, Friedman + Nemenyi CD, and
    paired sign tests of every method vs the first one. ``snapshot="final"`` uses each run's last snapshot."""
    scores: dict[str, dict[str, float | None]] = {}
    directions: dict[str, str] = {}
    warnings: list[str] = []
    incomplete: dict[str, list[str]] = {}
    for label, rd in runs.items():
        rd = Path(rd)
        rows = [r for r in read_results(rd / "eval" / RESULTS) if r["split"] == split and not r.get("family_source")]
        snaps = _snap_order({r["snapshot"] for r in rows})
        if not snaps:
            raise ValueError(f"run {rd} has no results on split {split!r}")
        snap = snaps[-1] if snapshot == "final" else snapshot
        rs = [r for r in rows if r["snapshot"] == snap]
        ads = _load_adapters_for(rd, {r["discipline"] for r in rs}, adapters, warnings)
        try:
            man = json.loads((rd / "splits.json").read_text())
        except (OSError, json.JSONDecodeError):
            man = {}
        info = _pooled_by_discipline(rs, ads, rd / "eval", warnings, expected_by_discipline(man, split, snap))
        scores[label] = {d: v[0] for d, v in info.items()}
        for d, v in info.items():
            if v[2] < v[3]:
                incomplete.setdefault(d, []).append(f"{label} ({v[2]}/{v[3]})")
        for r in rs:
            if r.get("direction") in ("max", "min"):
                directions.setdefault(r["discipline"], r["direction"])
    if incomplete:
        # LEAK-5: a discipline pooled over fewer episodes than promised is not comparable across methods
        for d, who in sorted(incomplete.items()):
            warnings.append(f"{d}: pooled payload coverage incomplete for {', '.join(who)}; discipline dropped from "
                            "the comparison")
        scores = {m: {d: s for d, s in sc.items() if d not in incomplete} for m, sc in scores.items()}
    labels = list(scores)
    fr = M.friedman_test(scores, directions)
    k, n = fr["k"], fr["n"]
    return {"split": split, "snapshot": snapshot, "scores": scores, "avg_ranks": M.average_ranks(scores, directions),
            "friedman": fr, "nemenyi_cd": M.nemenyi_cd(k, n, alpha) if k >= 2 and n >= 1 else float("nan"),
            "sign_tests": {m: M.sign_test(scores[m], scores[labels[0]], directions) for m in labels[1:]},
            "warnings": sorted(set(warnings))}
