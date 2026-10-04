"""Stream loop of program self-evolution with the strict-improvement update (paper Eq. 2-3, Section 5.2).

For every source episode of ``plan.source_stream()`` (round-major, fixed discipline order):

    solve (mode "source") -> extract_instances (Eq. 9) -> build_bundle (Eq. 10-12 + BReplay)
    -> for each gated bundle (variants.bundles_for_variant):
         A~, omega = A.apply(B)                   (atomic application, version ids omega)
         R_src = source_replay_check(A~, omega)   (Eq. 13)
         report = gate.evaluate(A~, reuse_from=incumbent report)   (lazy re-validation)
         per_candidate:     accept iff gate.admit(report, incumbent)       (Eq. 2 + Eq. 3)
         per_round_argmax:  pool feasible candidates; at the round end accept argmax Q_val iff strictly better

The foundation model is never updated (Theta_{r+1} = Theta_0); only the program A_r = (Skills, Operators)
changes. Variant ``frozen`` solves the source stream (for comparable cost accounting) but never evolves.

Receipts written under ``run_dir``:

* ``candidates.jsonl`` - one line per gated candidate (ids, source episode, bundle summary, boundary
  replay results, R_src / Pass / Use, H_val, cost, Q_val candidate vs incumbent, accepted, reasons, timings);
* ``stream.jsonl``     - one line per processed source episode;
* ``rounds.jsonl``     - one line per finished round (per-round argmax decision, snapshot);
* ``programs/A_<r>/``  - snapshot at the end of every round (``A_0`` = program0);
* ``val_reports/<program_version>.json`` - every validation report;
* ``candidates/<cand_id>/{bundle.json,build_log.json}`` and per-episode run directories;
* ``state.json``       - commit point for idempotent restarts (see ``Evolver.run``).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..core.program import AgentProgram, Bundle
from .attribution import extract_instances
from .bundle import build_bundle, bundle_summary
from .validation import (ValidationGate, ValReport, _fresh_dir, _numeric_usage, _safe, infra_error_of, qval_key,
                         source_pass, source_replay_check, use_check)
from .variants import bundles_for_variant, check_variant

__all__ = ["Evolver", "UPDATE_SCHEDULES", "STATE_SCHEMA"]

log = logging.getLogger(__name__)

UPDATE_SCHEDULES = ("per_candidate", "per_round_argmax")
STATE_SCHEMA = 1


def _append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=str, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("ignoring corrupt line %d of %s (interrupted write)", i + 1, path)
    return out


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, default=str, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=str, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


class Evolver:
    """Program self-evolution over the source stream (resumable)."""

    def __init__(self, cfg: Any, llm: Any, plan: Any, run_dir: str | Path, *, solver: Any = None,
                 retriever_factory: Any = None, val_workers: int = 6) -> None:
        self.cfg = cfg
        self.evo = cfg.evolution
        check_variant(self.evo.variant)
        if self.evo.update_schedule not in UPDATE_SCHEDULES:
            raise ValueError(f"unknown update_schedule {self.evo.update_schedule!r}; expected one of {UPDATE_SCHEDULES}")
        self.llm = llm
        self.plan = plan
        self.run_dir = Path(run_dir)
        if solver is None:
            from ..agent.solver import Solver  # lazy: agent layer sits above core

            solver = Solver(cfg.solver, llm, cfg.evolution)
        self.solver = solver
        val = (getattr(plan, "episodes", None) or {}).get("val", {})
        self.gate = ValidationGate(self.evo, cfg.solver, solver, val, self.run_dir / "val",
                                   max_workers=val_workers, retriever_factory=retriever_factory)
        if not self.gate.val_episodes and self.evo.variant != "frozen":
            log.warning("no validation episodes: Q_val is constant, so no candidate can strictly improve")
        self.program: AgentProgram | None = None
        self.incumbent_report: ValReport | None = None
        self._sleep = time.sleep   # backoff between retries of a solve hit by a gateway outage (replaced in tests)

    # ------------------------------------------------------------------------------------ paths
    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def _snapshot_dir(self, i: int) -> Path:
        return self.run_dir / "programs" / f"A_{i}"

    def _save_report(self, rep: ValReport) -> None:
        d = self.run_dir / "val_reports"
        d.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(d / f"{_safe(rep.program_version)}.json", rep.to_dict())

    def _save_snapshot(self, i: int, program: AgentProgram, round_value: Any, extra: dict | None = None) -> None:
        d = self._snapshot_dir(i)
        program.save(d)
        _write_json_atomic(d / "snapshot.json", {"index": i, "round": round_value, **program.summary(),
                                                 **(extra or {})})

    def _run_meta(self, program0: AgentProgram) -> dict:
        return {"variant": self.evo.variant, "update_schedule": self.evo.update_schedule, "qval": self.evo.qval,
                "hval_mode": self.evo.hval_mode, "program0_fingerprint": program0.fingerprint()}

    def _commit(self, state: dict, program: AgentProgram, inc: ValReport | None) -> None:
        state["program"] = program.to_dict()
        state["incumbent_report"] = inc.to_dict() if inc is not None else None
        state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json_atomic(self.state_path, state)

    # -------------------------------------------------------------------------------- resume
    def _load_state(self, program0: AgentProgram) -> dict | None:
        if not self.state_path.exists():
            return None
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError(f"{self.state_path}: unsupported state schema {state.get('schema')!r}")
        meta = self._run_meta(program0)
        diff = {k: (state.get("meta", {}).get(k), v) for k, v in meta.items() if state.get("meta", {}).get(k) != v}
        if diff:
            raise ValueError(f"{self.run_dir} holds a different run (stored vs requested: {diff}); "
                             "use a new run_dir")
        completed = set(state.get("completed", []))
        rounds_done = {json.dumps(r) for r in state.get("rounds_done", [])}
        # drop receipts of work that was not committed (it is redone idempotently)
        stream = self.run_dir / "stream.jsonl"
        _rewrite_jsonl(stream, [r for r in _read_jsonl(stream) if r.get("key") in completed])
        cands = self.run_dir / "candidates.jsonl"
        _rewrite_jsonl(cands, [r for r in _read_jsonl(cands) if r.get("source_key") in completed])
        rounds = self.run_dir / "rounds.jsonl"
        _rewrite_jsonl(rounds, [r for r in _read_jsonl(rounds) if json.dumps(r.get("round")) in rounds_done])
        log.info("resuming %s: %d source episodes and %d rounds already committed", self.run_dir, len(completed),
                 len(rounds_done))
        return state

    # ----------------------------------------------------------------------------------- run
    def run(self, program0: AgentProgram) -> list[AgentProgram]:
        """Evolve ``program0`` over the source stream; returns the snapshots A_0..A_R.

        Idempotent restart: ``state.json`` is rewritten atomically after every source episode and every
        round end (current program, incumbent report, completed episodes, candidate counter, round pool).
        On restart with the same ``run_dir`` and configuration, committed source episodes are skipped and
        uncommitted receipts (partial candidates.jsonl / stream.jsonl / rounds.jsonl lines) are dropped.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stream = list(self.plan.source_stream())
        rounds: list[Any] = []
        by_round: dict[str, list[Any]] = {}
        for r, ep in stream:
            k = json.dumps(r)
            if k not in by_round:
                rounds.append(r)
                by_round[k] = []
            by_round[k].append(ep)
        frozen = self.evo.variant == "frozen"

        state = self._load_state(program0)
        if state is None:
            a0 = self._snapshot_dir(0)
            if (a0 / "program.json").exists() and AgentProgram.load(a0).fingerprint() != program0.fingerprint():
                raise ValueError(f"{a0} holds a different A_0; use a new run_dir")
            program = program0
            inc = None
            if not frozen:
                inc = self.gate.evaluate(program0)
                self._save_report(inc)
            self._save_snapshot(0, program0, None, {"incumbent": inc.summary() if inc else None})
            state = {"schema": STATE_SCHEMA, "meta": self._run_meta(program0), "completed": [], "rounds_done": [],
                     "cand_seq": 0, "n_accepted": 0, "pool": []}
            self._commit(state, program, inc)
        else:
            program = AgentProgram.from_dict(state["program"])
            inc = ValReport.from_dict(state["incumbent_report"]) if state.get("incumbent_report") else None
        self.program, self.incumbent_report = program, inc

        completed = set(state["completed"])
        for i, r in enumerate(rounds, start=1):
            deferred: list[Any] = []
            # An episode whose source solve was cut short by a gateway outage (after the bounded in-place
            # retries) is not committed: it is re-queued once at the end of its round, when the outage has
            # had time to clear. Only then is its (possibly still failed) entry committed.
            for last_chance, eps in ((False, by_round[json.dumps(r)]), (True, deferred)):
                for ep in list(eps):
                    key = f"{json.dumps(r)}:{ep.id}"
                    if key in completed:
                        continue
                    program, inc, entry = self._process_episode(r, i, key, ep, program, inc, state,
                                                                last_chance=last_chance)
                    _append_jsonl(self.run_dir / "stream.jsonl", entry)
                    if entry.get("requeue"):
                        deferred.append(ep)
                        continue
                    state["completed"].append(key)
                    completed.add(key)
                    self._commit(state, program, inc)
                    self.program, self.incumbent_report = program, inc
            if json.dumps(r) not in {json.dumps(x) for x in state["rounds_done"]}:
                program, inc, decision = self._finalize_round(r, i, program, inc, state)
                _append_jsonl(self.run_dir / "rounds.jsonl", decision)
                state["rounds_done"].append(r)
                self._commit(state, program, inc)
                self.program, self.incumbent_report = program, inc

        usage = getattr(self.llm, "usage", None)
        if callable(usage):
            try:
                _write_json_atomic(self.run_dir / "llm_usage_evolver.json", usage())
            except (TypeError, ValueError, OSError) as ex:
                log.warning("could not write LLM usage receipt: %s", ex)
        return [AgentProgram.load(self._snapshot_dir(i)) for i in range(len(rounds) + 1)]

    # ------------------------------------------------------------------------------ one episode
    def _solve_source(self, ep: Any, program: AgentProgram, ep_dir: Path) -> tuple[Any, int, str | None]:
        """Source solve with bounded retries (doubling backoff) when a gateway outage cut it short."""
        retries = max(0, int(getattr(self.evo, "infra_retries", 0) or 0))
        backoff = float(getattr(self.evo, "infra_backoff_s", 0.0) or 0.0)
        res, err = None, None
        for attempt in range(retries + 1):
            res = self.solver.solve(ep, program, mode="source", run_dir=str(_fresh_dir(ep_dir / "solve")))
            err = infra_error_of(res)
            if err is None:
                return res, attempt + 1, None
            log.warning("source solve of %s hit an infrastructure error (attempt %d/%d): %s", ep.id, attempt + 1,
                        retries + 1, err)
            if attempt < retries:
                self._sleep(backoff * (2 ** attempt))
        return res, retries + 1, err

    def _process_episode(self, r: Any, round_no: int, key: str, ep: Any, program: AgentProgram,
                         inc: ValReport | None, state: dict, *,
                         last_chance: bool = True) -> tuple[AgentProgram, ValReport | None, dict]:
        t0 = time.monotonic()
        ep_dir = self.run_dir / "source" / f"{_safe(str(r))}_{_safe(ep.id)}"
        entry: dict[str, Any] = {"key": key, "round": r, "round_no": round_no, "episode": ep.id,
                                 "discipline": getattr(ep, "discipline", ""), "program_version": program.version,
                                 "program_fingerprint": program.fingerprint(), "variant": self.evo.variant,
                                 "instances": [], "candidates": [], "accepted": [], "error": None, "timings": {}}
        try:
            res, attempts, infra = self._solve_source(ep, program, ep_dir)
        except Exception as ex:  # recorded in stream.jsonl; the stream continues with the next episode
            log.exception("source solve of %s failed", ep.id)
            entry["error"] = f"source solve failed: {type(ex).__name__}: {ex}"[:800]
            entry["timings"]["solve_s"] = round(time.monotonic() - t0, 3)
            entry["program_version_after"] = program.version
            return program, inc, entry
        entry["timings"]["solve_s"] = round(time.monotonic() - t0, 3)
        evs = list(getattr(res, "evidence", None) or [])
        if infra:
            entry["infra_error"], entry["solve_attempts"] = infra, attempts
            if not any(bool(e.passed) for e in evs):
                # a gateway outage, not a task failure: never recorded as an ordinary unsolved source episode
                entry["error"] = f"source solve cut short by an infrastructure error: {infra}"[:800]
                entry["requeue"] = not last_chance
                entry["program_version_after"] = program.version
                return program, inc, entry
        entry["solve"] = {"n_steps": len(getattr(res, "steps", None) or []), "n_evidence": len(evs),
                          "evidence": [{"step": e.step, "passed": bool(e.passed)} for e in evs],
                          "final_z": int(getattr(getattr(res, "eval", None), "z", 0) or 0),
                          "usage": _numeric_usage(getattr(res, "usage", {})),
                          "retrieved": getattr(res, "retrieved", {})}
        if self.evo.variant == "frozen":
            entry["program_version_after"] = program.version
            return program, inc, entry

        instances = extract_instances(res, int(self.evo.instances_per_episode))
        if not instances:
            entry["note"] = "no replay-verified success: no evolution instance"
        for j, inst in enumerate(instances):
            tb = time.monotonic()
            try:
                bundle, blog = build_bundle(inst, program, self.llm, ep, ep_dir / f"inst{j}", self.evo.variant,
                                            repeats=int(self.evo.breplay_repeats), round_idx=r)
            except Exception as ex:  # a failing abstraction must not stop the stream; it is recorded
                log.exception("bundle construction for %s instance %d failed", ep.id, j)
                entry["instances"].append({"summary": inst.summary(),
                                           "error": f"{type(ex).__name__}: {ex}"[:800]})
                continue
            parts = bundles_for_variant(bundle, self.evo.variant)
            entry["instances"].append({"summary": inst.summary(), "n_skills": len(bundle.skills),
                                       "n_operators": len(bundle.operators), "n_gated_bundles": len(parts),
                                       "bundle_s": round(time.monotonic() - tb, 3),
                                       **({} if parts else {"note": "empty bundle: no candidate"})})
            for b in parts:
                program, inc = self._gate_candidate(r, round_no, key, ep, b, blog, program, inc, state, entry)
        entry["timings"]["total_s"] = round(time.monotonic() - t0, 3)
        entry["program_version_after"] = program.version
        return program, inc, entry

    def _gate_candidate(self, r: Any, round_no: int, key: str, ep: Any, bundle: Bundle, blog: dict,
                        program: AgentProgram, inc: ValReport | None, state: dict,
                        entry: dict) -> tuple[AgentProgram, ValReport | None]:
        state["cand_seq"] = int(state.get("cand_seq", 0)) + 1
        cid = f"c{state['cand_seq']:04d}"
        cdir = self.run_dir / "candidates" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(cdir / "bundle.json", bundle.to_dict())
        _write_json_atomic(cdir / "build_log.json", blog)
        cand, omega = program.apply(bundle, new_version=f"r{round_no}-{cid}")
        rec: dict[str, Any] = {
            "cand_id": cid, "round": r, "round_no": round_no, "source_episode": ep.id, "source_key": key,
            "discipline": getattr(ep, "discipline", ""), "variant": self.evo.variant,
            "part": bundle.meta.get("part", "bundle"), "schedule": self.evo.update_schedule,
            "parent_version": program.version, "cand_version": cand.version, "omega": omega,
            "bundle": bundle_summary(bundle),
            "breplay": [{"op_id": (o.get("candidate") or {}).get("op_id"),
                         "component": (o.get("candidate") or {}).get("component"),
                         "formed": (o.get("candidate") or {}).get("formed"),
                         "reason": (o.get("candidate") or {}).get("reason"),
                         "ok": (o.get("breplay") or {}).get("ok"),
                         "error": (o.get("breplay") or {}).get("error")} for o in blog.get("operators", [])],
            "skill_log": {k: (blog.get("skills") or {}).get(k) for k in ("formed", "reason", "attributed", "leaks")},
            "R_src": None, "Pass": None, "Use": None, "use_missing": [], "H_val": None, "cost": {}, "qval": {},
            "feasible": None, "accepted": False, "reasons": {}, "timings": {}, "error": None,
        }
        entry["candidates"].append(cid)
        t0 = time.monotonic()
        try:
            ok_src, sres = source_replay_check(cand, omega, ep, self.solver, _fresh_dir(cdir / "source_replay"),
                                               retries=int(getattr(self.evo, "infra_retries", 0) or 0),
                                               backoff_s=float(getattr(self.evo, "infra_backoff_s", 0.0) or 0.0),
                                               sleep=self._sleep)
        except Exception as ex:  # recorded as a rejected candidate
            log.exception("source replay of candidate %s failed", cid)
            rec["error"] = f"source replay raised {type(ex).__name__}: {ex}"[:800]
            rec["reasons"] = {"rejected": "source replay error"}
            rec["timings"]["source_replay_s"] = round(time.monotonic() - t0, 3)
            _append_jsonl(self.run_dir / "candidates.jsonl", rec)
            return program, inc
        rec["timings"]["source_replay_s"] = round(time.monotonic() - t0, 3)
        req = bool(getattr(self.evo, "pass_requires_acceptance", True))
        rec["Pass"] = bool(source_pass(sres, req))
        rec["Use"], rec["use_missing"] = use_check(omega, sres)
        rec["R_src"] = bool(ok_src)
        rec["cost"]["source_replay"] = _numeric_usage(getattr(sres, "usage", {}))
        if not ok_src:
            rec["reasons"] = {"rejected": "R_src", "Pass": rec["Pass"], "Use": rec["Use"],
                              "use_missing": rec["use_missing"]}
            if infra_error_of(sres):
                rec["infra_error"] = infra_error_of(sres)
                rec["reasons"]["infra_error"] = rec["infra_error"]
            _append_jsonl(self.run_dir / "candidates.jsonl", rec)
            return program, inc

        if inc is None:  # only the frozen variant has no incumbent report, and it never gates
            raise RuntimeError("gating a candidate requires an incumbent validation report")
        t1 = time.monotonic()
        inc = self._refresh_incumbent(program, inc, rec)
        blocked = sorted(eid for eid, e in inc.per_episode.items() if e.get("error"))
        if blocked:
            # An incumbent entry that is still a solver / gateway error would act as a silent z = 0 baseline and
            # let any candidate that solves the episode look like an improvement: do not gate against it.
            log.warning("candidate %s not gated: incumbent %s still has errored val entries %s", cid,
                        program.version, blocked)
            rec["val_errors"] = {eid: inc.per_episode[eid]["error"] for eid in blocked}
            rec["reasons"] = {"rejected": "incumbent val entries errored", "errored_incumbent_entries": blocked}
            rec["timings"]["validation_s"] = round(time.monotonic() - t1, 3)
            _append_jsonl(self.run_dir / "candidates.jsonl", rec)
            return program, inc
        crep = self.gate.evaluate(cand, reuse_from=inc)
        rec["timings"]["validation_s"] = round(time.monotonic() - t1, 3)
        self._save_report(crep)
        rec["H_val"] = crep.h_val
        rec["cost"]["val"] = crep.cost
        rec["cost"]["val_incurred"] = crep.incurred
        rec["val_resolved"], rec["val_reused"] = crep.n_resolved, crep.n_reused
        rec["val_errors"] = {eid: e["error"] for eid, e in crep.per_episode.items() if e.get("error")}
        log.info("candidate %s: %d/%d val episodes re-solved (%d reused)", cid, crep.n_resolved,
                 len(crep.per_episode), crep.n_reused)
        rec["qval"] = {"cand": {"macro_sr": crep.macro_sr, "norm_score": crep.norm_score},
                       "inc": {"macro_sr": inc.macro_sr, "norm_score": inc.norm_score}}
        if self.evo.update_schedule == "per_candidate":
            ok, reasons = self.gate.admit(crep, inc)
            rec["feasible"], rec["accepted"], rec["reasons"] = reasons["feasible"], ok, reasons
            if ok:
                program, inc = cand, crep
                state["n_accepted"] = int(state.get("n_accepted", 0)) + 1
                entry["accepted"].append(cid)
        else:
            feas, reasons = self.gate.feasible(crep, inc)
            rec["feasible"], rec["reasons"], rec["pooled"] = feas, reasons, feas
            if feas:
                state.setdefault("pool", []).append({"cand_id": cid, "program": cand.to_dict(),
                                                     "report": crep.to_dict()})
        _append_jsonl(self.run_dir / "candidates.jsonl", rec)
        return program, inc

    def _refresh_incumbent(self, program: AgentProgram, inc: ValReport, rec: dict) -> ValReport:
        """Re-solve incumbent val episodes whose entry is a solver error (e.g. a transient API failure).

        Without this, an infrastructure error in the incumbent report (z = 0) would make any candidate look
        like an improvement on that episode. Entries without errors are reused (same program, same slice).
        """
        errored = [eid for eid, e in inc.per_episode.items() if e.get("error")]
        if not errored:
            return inc
        log.warning("incumbent %s has %d errored val entries; re-solving them before gating", program.version,
                    len(errored))
        fresh = self.gate.evaluate(program, reuse_from=inc)
        self._save_report(fresh)
        rec["incumbent_refreshed"] = {"errored": errored, "still_errored":
                                      [eid for eid, e in fresh.per_episode.items() if e.get("error")]}
        return fresh

    # ------------------------------------------------------------------------------- round end
    def _finalize_round(self, r: Any, round_no: int, program: AgentProgram, inc: ValReport | None,
                        state: dict) -> tuple[AgentProgram, ValReport | None, dict]:
        decision: dict[str, Any] = {"round": r, "round_no": round_no, "schedule": self.evo.update_schedule,
                                    "pool": [p["cand_id"] for p in state.get("pool", [])], "chosen": None,
                                    "accepted": False, "reasons": {}}
        pool = state.get("pool", [])
        if self.evo.update_schedule == "per_round_argmax" and pool and inc is not None:
            best_i = max(range(len(pool)),
                         key=lambda i: (qval_key(ValReport.from_dict(pool[i]["report"]), self.evo.qval), -i))
            best = pool[best_i]
            best_rep = ValReport.from_dict(best["report"])
            ok, reasons = self.gate.admit(best_rep, inc)
            decision.update(chosen=best["cand_id"], accepted=ok, reasons=reasons)
            if ok:
                program, inc = AgentProgram.from_dict(best["program"]), best_rep
                state["n_accepted"] = int(state.get("n_accepted", 0)) + 1
        state["pool"] = []
        self._save_snapshot(round_no, program, r, {"n_accepted_total": state.get("n_accepted", 0),
                                                   "incumbent": inc.summary() if inc else None})
        decision["snapshot"] = f"programs/A_{round_no}"
        decision["program_version"] = program.version
        decision["program_fingerprint"] = program.fingerprint()
        return program, inc, decision
