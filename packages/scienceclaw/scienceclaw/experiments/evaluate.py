"""Frozen-snapshot evaluation on D_ID / D_OOD / D_rep (paper Sec. 4.2; DESIGN §7).

For every snapshot A_r (``programs/A_r``) and every held-out episode the program is frozen, transient state is
cleared (a fresh directory per (snapshot, split, episode) and a fresh Solver) and the episode is solved once in
Solver mode ``"eval"``. Held-out results never feed back into evolution.

Layout written under ``<run_dir>/eval/``:

* ``<snapshot>/<split>/<episode>/``      — receipts (``SolveResult.save``; the solver works in ``work/``)
* ``payloads/<snapshot>/<split>/<episode>.json`` — pooled payload for task-native pooled metrics. It holds
  hidden labels, so it is written only after the solve finished and outside every solver directory.
* ``results.jsonl`` — one line per (snapshot, split, episode): z, primary, norm_score, reference, direction,
  payload path (relative to ``eval/``), usage, ... Evaluation is resumable: keys already present are skipped.
  A solve that raises -- or that ends with an infrastructure ``stop_reason`` (``policy_error``: the LLM call itself
  failed, M2) -- is logged to ``errors.jsonl`` (not to results) so a later resume retries it. A solve that
  finished without a usable payload still contributes the adapter's uniform failure payload.
* Identical programs are solved once: snapshots whose ``AgentProgram.fingerprint()`` is equal get *alias rows*
  (``alias_of`` = the solved snapshot, copied outcome, empty ``usage``; the original spend stays in ``alias_usage``).
  Aliases wait for an in-flight solve of the same (fingerprint, split, episode) instead of re-solving it.
* ``eval_log.jsonl`` -- one record per phase with the provenance receipt (code, config, data, LLM settings, worker
  and concurrency numbers) and the LLM usage delta; the same numbers are appended to ``../usage.jsonl`` (ledger).

D_rep (retention): for snapshot A_r the frozen copies of the source episodes of rounds <= r are re-solved
(split ``"rep"``; A_0 has none). The report compares A_r on a round-j episode with A_j (the snapshot
immediately after that source task).

Family transfer (paper Fig. 5): :func:`evaluate_family_transfer` evaluates, for each source family s, the
program A_0 + {components of the final snapshot whose source episode belongs to family s}.
"""
from __future__ import annotations

import copy
import json
import pickle
import re
import shutil
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import RunConfig, load_config
from ..core.program import AgentProgram
from .provenance import UsageLedger, provenance, sha256_file
from .run_stream import SNAPSHOT_RE, snapshot_dirs

RESULTS = "results.jsonl"
ERRORS = "errors.jsonl"


# ---------------------------------------------------------------------------------------------------- helpers
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@\-]+", "_", name)


def snapshot_round(name: str) -> int | None:
    m = SNAPSHOT_RE.match(name)
    return int(m.group(1)) if m else None


def _jsonable(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def write_payload(path: Path, payload: Any) -> Path:
    """Write a pooled payload as JSON (``.json``) or, if it is not JSON-serializable, as a pickle (``.pkl``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(payload, default=_jsonable)
    except (TypeError, ValueError):
        p = path.with_suffix(".pkl")
        with p.open("wb") as f:
            pickle.dump(payload, f)
        return p
    p = path.with_suffix(".json")
    p.write_text(text)
    return p


def read_results(path: str | Path) -> list[dict]:
    """Read results.jsonl (last line wins per (snapshot, split, episode) key); tolerates a torn last line."""
    p = Path(path)
    if not p.exists():
        return []
    rows: dict[tuple, dict] = {}
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:        # interrupted write of the final line: ignore, it will be redone
                continue
            raise
        rows[(d["snapshot"], d["split"], d["episode"])] = d
    return list(rows.values())


class _JsonlWriter:
    """Thread-safe append-only JSONL writer (one flushed line per record)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, default=_jsonable, sort_keys=True)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")
            f.flush()


def _load_run(run_dir: Path, adapters: dict | None, verify: bool = True):
    from ..bench.splits import SplitPlan, load_adapters

    cfg = load_config(run_dir / "config.yaml")
    if adapters is None:
        adapters = load_adapters(cfg.bench)
    plan = SplitPlan.build(cfg.bench, adapters)
    mpath = run_dir / "splits.json"
    if verify and mpath.exists():
        plan.verify_against(SplitPlan.load_manifest(mpath))
    return cfg, adapters, plan


def _default_solver_factory(cfg: RunConfig, llm: Any) -> Callable[[], Any]:
    from ..agent.solver import Solver

    return lambda: Solver(cfg.solver, llm, cfg.evolution)


def _usage_of(res: Any) -> dict:
    u = getattr(res, "usage", None) or {}
    return {k: v for k, v in u.items() if isinstance(v, (int, float))}


INFRA_STOP_REASONS = frozenset({"policy_error"})


class InfraError(RuntimeError):
    """A solve that ended for infrastructure reasons (the policy/LLM call itself failed), not because the program
    failed the task. It is raised by :func:`_solve_job` so the job lands in ``errors.jsonl`` and a later resume
    retries it instead of recording a fake failure that would be scored as the agent's. The spend of the failed
    attempt stays visible: ``usage`` / ``wall_s`` are copied into the error record."""

    def __init__(self, msg: str, *, usage: dict | None = None, wall_s: float | None = None,
                 stop_reason: str | None = None, receipt_dir: str | None = None) -> None:
        super().__init__(msg)
        self.usage, self.wall_s, self.stop_reason, self.receipt_dir = usage or {}, wall_s, stop_reason, receipt_dir


def _failure_payload(ep: Any) -> tuple[Any, bool]:
    """The adapter's uniform failure payload for ``ep`` (``ep.evaluate(None, None)``), and whether it exists."""
    try:
        d = dict(getattr(ep.evaluate(None, None), "details", {}) or {})
    except Exception:
        return None, False
    p = d.get("pooled_payload")
    return p, p is not None


def _solve_job(job: dict, program: AgentProgram, solver_factory: Callable[[], Any], eval_dir: Path) -> dict:
    ep = job["episode"]
    snap, split = job["snapshot"], job["split"]
    ep_dir = eval_dir / _safe(snap) / split / _safe(ep.id)
    if ep_dir.exists():
        shutil.rmtree(ep_dir)          # leftovers of an interrupted attempt: start from a clean state
    ep_dir.mkdir(parents=True)
    t0 = time.time()
    solver = solver_factory()
    res = solver.solve(ep, program, "eval", ep_dir / "work")
    wall = time.time() - t0
    save_error = None
    try:
        res.save(ep_dir)
    except (OSError, TypeError, ValueError, AttributeError) as ex:
        save_error = f"{type(ex).__name__}: {ex}"
    stop_reason = getattr(res, "stop_reason", None)
    if stop_reason in INFRA_STOP_REASONS:
        raise InfraError(f"solve stopped with infrastructure stop_reason {stop_reason!r} (LLM/policy call failed); "
                         "not recorded as a result so that a resume retries it", usage=_usage_of(res), wall_s=wall,
                         stop_reason=stop_reason, receipt_dir=str(ep_dir.relative_to(eval_dir)))
    ev = res.eval
    details = dict(getattr(ev, "details", {}) or {})
    payload = details.get("pooled_payload")
    payload_source = "solve"
    if payload is None:
        # every held-out episode contributes exactly one payload, also when the solve produced no usable one
        payload, found = _failure_payload(ep)
        payload_source = "failure_fallback" if found else "none"
        details.setdefault("failed", True)
    payload_rel = None
    if payload is not None:
        p = write_payload(eval_dir / "payloads" / _safe(snap) / split / _safe(ep.id), payload)
        payload_rel = str(p.relative_to(eval_dir))
    return {
        "snapshot": snap, "split": split, "episode": ep.id, "discipline": ep.discipline, "family": ep.family,
        "round": job.get("round"), "family_source": job.get("family_source"),
        "program_version": getattr(res, "program_version", program.version),
        "program_fingerprint": program.fingerprint(),
        "z": int(getattr(ev, "z", 0) or 0), "primary": getattr(ev, "primary", None),
        "direction": ep.direction or getattr(ev, "direction", None), "metric": ep.metric,
        "norm_score": details.get("norm_score"), "reference": details.get("reference"),
        "accepted": bool(getattr(ev, "accepted", False)), "completed": bool(getattr(ev, "completed", False)),
        "hard_ok": bool(ev.hard_ok()) if hasattr(ev, "hard_ok") else None,
        "h": dict(getattr(ev, "h", {}) or {}), "reproducible": getattr(ev, "reproducible", None),
        "within_budget": getattr(ev, "within_budget", None), "stop_reason": stop_reason,
        "failed": bool(details.get("failed")), "payload_source": payload_source,
        "pooled_payload": payload_rel, "usage": _usage_of(res), "wall_s": wall,
        "receipt_dir": str(ep_dir.relative_to(eval_dir)), "save_error": save_error,
    }


def _alias_row(job: dict, program: AgentProgram, root: dict) -> dict:
    """Result row of a snapshot whose program is identical (same fingerprint) to the one that was solved.

    It copies the root row (same outcome, same payload path, same receipts) and records ``alias_of``. ``usage`` is
    empty (nothing was spent for it); the root's usage is kept as ``alias_usage`` so logical cost stays computable.
    """
    row = dict(root)
    row.update(snapshot=job["snapshot"], split=job["split"], episode=job["episode"].id,
               round=job.get("round", root.get("round")), family_source=job.get("family_source"),
               program_version=program.version, alias_of=root["snapshot"], usage={}, wall_s=0.0,
               alias_usage=dict(root.get("usage") or {}))
    return row


def run_jobs(jobs: list[dict], programs: dict[str, AgentProgram], solver_factory: Callable[[], Any], eval_dir: Path,
             workers: int = 8, progress: Callable[[str], None] | None = None, dedup: bool = True) -> dict:
    """Solve the jobs not yet in results.jsonl; returns counts {"done", "skipped", "failed", "aliased", "deferred"}.

    Program de-duplication (M5, ``dedup=True``): a job is keyed by (program fingerprint, split, episode). Only the
    first job of a key is solved; every other snapshot with an identical program gets an *alias row*
    (``alias_of`` = the solved snapshot, copied outcome, empty ``usage``) -- also across calls, using the rows
    already in ``results.jsonl``. Aliases of a job that is still in flight wait for its owner, so identical programs
    are never solved concurrently (which would bypass the response cache). If the owner fails, its aliases are not
    written (``deferred``) and the next resume retries the whole key.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    existing = read_results(eval_dir / RESULTS)
    have = {(r["snapshot"], r["split"], r["episode"]) for r in existing}
    fps = {n: p.fingerprint() for n, p in programs.items()}
    solved: dict[tuple, dict] = {}
    if dedup:
        for r in existing:
            if r.get("program_fingerprint") and not r.get("alias_of"):
                solved.setdefault((r["program_fingerprint"], r["split"], r["episode"]), r)
    out = {"done": 0, "skipped": 0, "failed": 0, "aliased": 0, "deferred": 0}
    res_w, err_w = _JsonlWriter(eval_dir / RESULTS), _JsonlWriter(eval_dir / ERRORS)
    lock = threading.Lock()
    todo: list[dict] = []
    owners: dict[tuple, dict] = {}
    waiting: dict[tuple, list[dict]] = {}
    for j in jobs:
        k3 = (j["snapshot"], j["split"], j["episode"].id)
        if k3 in have:
            out["skipped"] += 1
            continue
        key = (fps[j["snapshot"]], j["split"], j["episode"].id)
        j["_key"] = key
        if dedup and key in solved:
            res_w.write(_alias_row(j, programs[j["snapshot"]], solved[key]))
            out["aliased"] += 1
            if progress:
                progress(f"{k3[0]}/{k3[1]}/{k3[2]}: alias of {solved[key]['snapshot']}")
        elif dedup and key in owners:
            waiting.setdefault(key, []).append(j)
        else:
            owners[key] = j
            todo.append(j)

    def _one(job: dict) -> None:
        key = f"{job['snapshot']}/{job['split']}/{job['episode'].id}"
        try:
            # frozen snapshot: every solve gets its own copy, a fresh Solver and a fresh directory
            rec = _solve_job(job, copy.deepcopy(programs[job["snapshot"]]), solver_factory, eval_dir)
        except Exception as ex:  # recorded, retried on resume; the other jobs continue
            err_w.write({"snapshot": job["snapshot"], "split": job["split"], "episode": job["episode"].id,
                         "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()[-4000:],
                         "infra": isinstance(ex, InfraError), "usage": getattr(ex, "usage", None),
                         "wall_s": getattr(ex, "wall_s", None), "stop_reason": getattr(ex, "stop_reason", None),
                         "receipt_dir": getattr(ex, "receipt_dir", None), "time": time.time()})
            with lock:
                out["failed"] += 1
                out["deferred"] += len(waiting.get(job["_key"], ()))
            if progress:
                progress(f"FAILED {key}: {type(ex).__name__}: {ex}")
            return
        res_w.write(rec)
        n_alias = 0
        for aj in waiting.get(job["_key"], ()):
            res_w.write(_alias_row(aj, programs[aj["snapshot"]], rec))
            n_alias += 1
        with lock:
            out["done"] += 1
            out["aliased"] += n_alias
        if progress:
            progress(f"{key}: z={rec['z']} primary={rec['primary']}" + (f" (+{n_alias} alias)" if n_alias else ""))

    if workers <= 1:
        for j in todo:
            _one(j)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval") as pool:
            for f in as_completed([pool.submit(_one, j) for j in todo]):
                f.result()
    return out


# ------------------------------------------------------------------------------------------------ snapshots
def evaluate_snapshots(run_dir: str | Path, snapshots: str | Iterable[str] = "all",
                       splits: Iterable[str] = ("id", "ood"), llm: Any = None, workers: int = 8, *,
                       include_rep: bool = True, adapters: dict | None = None,
                       solver_factory: Callable[[], Any] | None = None,
                       progress: Callable[[str], None] | None = None) -> Path:
    """Solve every held-out episode with each frozen snapshot; returns the path of ``eval/results.jsonl``.

    ``snapshots``: "all" or names such as ``["A_0", "A_7"]``. ``splits``: subset of ("id", "ood", "val");
    ``include_rep`` adds D_rep (snapshot A_r re-solves the source episodes of rounds <= r).
    """
    run_dir = Path(run_dir)
    cfg, adapters, plan = _load_run(run_dir, adapters)
    avail = {f"A_{r}": d for r, d in snapshot_dirs(run_dir).items()}
    if snapshots == "all":
        names = list(avail)
    else:
        names = [s if s in avail else f"A_{snapshot_round(s)}" for s in ([snapshots] if isinstance(snapshots, str) else snapshots)]
    missing = [n for n in names if n not in avail]
    if missing:
        raise FileNotFoundError(f"snapshots {missing} not found under {run_dir / 'programs'} (have {list(avail)})")
    if not names:
        raise FileNotFoundError(f"no program snapshots under {run_dir / 'programs'}")
    programs = {n: AgentProgram.load(avail[n]) for n in names}
    if solver_factory is None:
        if llm is None:
            from ..llm.interface import build_chat_model
            llm = build_chat_model(cfg.llm)
        solver_factory = _default_solver_factory(cfg, llm)
    jobs: list[dict] = []
    for n in names:
        for split in splits:
            if split in ("rep", "src"):
                raise ValueError("use include_rep=True for D_rep; source episodes are not a held-out split")
            for ep in plan.split_episodes(split):
                jobs.append({"snapshot": n, "split": split, "episode": ep})
        if include_rep:
            for ep in plan.rep_until(snapshot_round(n) or 0):
                jobs.append({"snapshot": n, "split": "rep", "episode": ep, "round": ep.lineage.get("round")})
    eval_dir = run_dir / "eval"
    counts = _run_phase(run_dir, "evaluate", cfg, llm, jobs, programs, solver_factory, workers, progress,
                        {"kind": "snapshots", "snapshots": names, "splits": list(splits), "rep": include_rep})
    return eval_dir / RESULTS


def _append_log(eval_dir: Path, rec: dict) -> None:
    rec = {**rec, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "eval_log.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def eval_provenance(run_dir: Path, cfg: RunConfig, phase: str, llm: Any, programs: dict[str, AgentProgram],
                    workers: int, extra: dict | None = None) -> dict:
    """Provenance receipt of an evaluation phase incl. the concurrency settings that decide queue time.

    ``eval_workers`` solves run concurrently and share the LLM client's ``llm.concurrency`` slots: with more workers
    than slots the surplus solves queue inside the client while their wall clock (budget ``z``) keeps running, so
    the receipt records both numbers and a warning when ``eval_workers > llm.concurrency``.
    """
    conc = getattr(getattr(cfg, "llm", None), "concurrency", None)
    warnings_: list[str] = []
    if conc and workers > conc:
        warnings_.append(f"eval_workers={workers} > llm.concurrency={conc}: surplus solves wait for an LLM slot and "
                         "that queue time counts toward the episode wall budget (z); use workers <= concurrency")
    ex = {"eval_workers": workers, "llm_concurrency": conc,
          "programs": {n: p.fingerprint() for n, p in programs.items()},
          "splits_json_sha256": sha256_file(run_dir / "splits.json"), "warnings": warnings_, **(extra or {})}
    return provenance(cfg, phase, llm=llm, extra=ex)


def _run_phase(run_dir: Path, phase: str, cfg: RunConfig, llm: Any, jobs: list[dict],
               programs: dict[str, AgentProgram], solver_factory: Callable[[], Any], workers: int,
               progress: Callable[[str], None] | None, log: dict) -> dict:
    """Run ``jobs`` inside a usage-ledger segment and append provenance + usage to ``eval_log.jsonl``."""
    eval_dir = run_dir / "eval"
    prov = eval_provenance(run_dir, cfg, phase, llm, programs, workers)
    for w in prov.get("warnings", []):
        if progress:
            progress(f"WARNING: {w}")
        warnings.warn(w, RuntimeWarning, stacklevel=3)
    with UsageLedger(run_dir, phase, llm, prov, extra={"kind": log.get("kind")}) as led:
        counts = run_jobs(jobs, programs, solver_factory, eval_dir, workers, progress)
    rec = led.record or {}
    _append_log(eval_dir, {**log, "counts": counts, "provenance": prov, "usage": rec.get("llm"),
                           "wall_s": rec.get("wall_s"), "segment": rec.get("segment")})
    return counts


# ------------------------------------------------------------------------------------------ family transfer
def family_slug(family: str) -> str:
    return "F-" + re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")


def component_source_episode(comp: Any) -> str | None:
    """Source episode of an evolved Skill/Operator: provenance['episode'|'episode_id'|'source_episode'] or the
    Skill ``created`` stamp ``r{round}:e{episode}``."""
    prov = getattr(comp, "provenance", None) or {}
    for k in ("episode", "episode_id", "source_episode"):
        if prov.get(k):
            return str(prov[k])
    created = getattr(comp, "created", "") or ""
    m = re.match(r"^r\d+:e(.+)$", created)
    return m.group(1) if m else None


def family_programs(base: AgentProgram, final: AgentProgram, episode_family: dict[str, str],
                    families: Iterable[str]) -> tuple[dict[str, AgentProgram], dict]:
    """For each source family s: A_0 + the components of ``final`` whose source episode belongs to s.

    Operators referenced (as ``operator`` nodes) inside an included operator's body are included too (closure).
    Components whose source episode cannot be attributed are left out and listed in the returned info.
    """
    comp_family: dict[str, str | None] = {}
    for sid, s in final.skills.items():
        comp_family[f"skill:{sid}"] = episode_family.get(component_source_episode(s) or "")
    for oid, o in final.operators.items():
        comp_family[f"op:{oid}"] = episode_family.get(component_source_episode(o) or "")
    info = {"unattributed": sorted(k for k, v in comp_family.items() if v is None), "per_family": {}}
    out: dict[str, AgentProgram] = {}
    for fam in families:
        skills = dict(base.skills)
        ops = dict(base.operators)
        skills.update({sid: s for sid, s in final.skills.items() if comp_family[f"skill:{sid}"] == fam})
        want = [oid for oid, o in final.operators.items() if comp_family[f"op:{oid}"] == fam]
        stack = list(want)
        while stack:
            oid = stack.pop()
            if oid in ops or oid not in final.operators:
                continue
            op = final.operators[oid]
            ops[oid] = op
            stack.extend(n.ref for n in op.body.nodes.values() if n.kind == "operator" and n.ref)
        prog = AgentProgram(skills, ops, version=f"{base.version}|{family_slug(fam)}", parent=base.version)
        out[fam] = prog
        info["per_family"][fam] = {"skills": sorted(set(skills) - set(base.skills)),
                                   "operators": sorted(set(ops) - set(base.operators))}
    return out, info


def evaluate_family_transfer(run_dir: str | Path, split: str = "ood", final: str | None = None, llm: Any = None,
                             workers: int = 8, *, adapters: dict | None = None,
                             solver_factory: Callable[[], Any] | None = None,
                             progress: Callable[[str], None] | None = None) -> Path:
    """Source-family x target-discipline evaluation for the transfer matrix (paper Fig. 5).

    Evaluates each family program (snapshot label ``F-<family-slug>``, ``family_source`` = family) on
    ``split``; A_0 on the same split (the reference) must be evaluated with :func:`evaluate_snapshots`.
    """
    from ..bench.registry import FAMILIES

    run_dir = Path(run_dir)
    cfg, adapters, plan = _load_run(run_dir, adapters)
    snaps = snapshot_dirs(run_dir)
    if not snaps or 0 not in snaps:
        raise FileNotFoundError("family transfer needs programs/A_0 and a final snapshot")
    final_r = snapshot_round(final) if final else max(snaps)
    base, fin = AgentProgram.load(snaps[0]), AgentProgram.load(snaps[final_r])
    ep_family = {ep.id: ep.family for _, ep in plan.source_stream()}
    fams = [f for f in FAMILIES if f in {ep.family for _, ep in plan.source_stream()}]
    progs, info = family_programs(base, fin, ep_family, fams)
    pdir = run_dir / "programs_family"
    programs: dict[str, AgentProgram] = {}
    for fam, prog in progs.items():
        label = family_slug(fam)
        prog.save(pdir / label)
        programs[label] = prog
    (pdir / "info.json").write_text(json.dumps({"final": f"A_{final_r}", **info}, indent=1))
    if solver_factory is None:
        if llm is None:
            from ..llm.interface import build_chat_model
            llm = build_chat_model(cfg.llm)
        solver_factory = _default_solver_factory(cfg, llm)
    jobs = [{"snapshot": family_slug(fam), "split": split, "episode": ep, "family_source": fam}
            for fam in progs for ep in plan.split_episodes(split)]
    eval_dir = run_dir / "eval"
    _run_phase(run_dir, "family_transfer", cfg, llm, jobs, programs, solver_factory, workers, progress,
               {"kind": "family_transfer", "split": split, "final": f"A_{final_r}"})
    return eval_dir / RESULTS
