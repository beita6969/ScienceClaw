"""Evolver: the source-stream loop with R_src, the validation gate, snapshots and resumable state."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scienceclaw.config import RunConfig
from scienceclaw.core.program import AgentProgram
from scienceclaw.evolution import Evolver

from test_evolution_support import (SimulatedCrash, StubPlan, StubRetriever, StubSolver, install_fake_isolated,
                                    make_episode, make_llm)


def cfg(**evo) -> RunConfig:
    c = RunConfig()
    c.evolution.min_improved_episodes = 1     # the toy val split has one episode per discipline (see the M7 tests)
    for k, v in evo.items():
        setattr(c.evolution, k, v)
    return c


def val_split() -> dict:
    return {"D1": [make_episode("valD1", "D1", split="val")], "D2": [make_episode("valD2", "D2", split="val")]}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def run(tmp_path: Path, stream, c: RunConfig, solver: StubSolver | None = None, name: str = "run", val=None):
    solver = solver or StubSolver(c.evolution)
    ev = Evolver(c, make_llm(), StubPlan(stream, val or val_split()), tmp_path / name, solver=solver,
                 retriever_factory=StubRetriever, val_workers=4)
    return ev, solver, ev.run(AgentProgram())


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    install_fake_isolated(monkeypatch)


def test_fail_to_pass_candidate_is_accepted_then_non_improving_one_rejected(tmp_path):
    stream = [(1, make_episode("srcA", "D1")), (1, make_episode("srcB", "D1"))]
    ev, solver, snaps = run(tmp_path, stream, cfg())
    rd = tmp_path / "run"
    cands = jsonl(rd / "candidates.jsonl")
    assert [c["accepted"] for c in cands] == [True, False]
    c1, c2 = cands
    assert c1["R_src"] and c1["Pass"] and c1["Use"] and c1["feasible"]
    assert len(c1["omega"]) == 2 and c1["bundle"]["meta"]["linked"] is True
    assert [b["ok"] for b in c1["breplay"]] == [True]
    assert c1["qval"] == {"cand": {"macro_sr": 0.5, "norm_score": pytest.approx(0.6)},
                          "inc": {"macro_sr": 0.0, "norm_score": pytest.approx(0.2)}}
    assert (c1["val_resolved"], c1["val_reused"]) == (1, 1)          # lazy re-validation of D2's episode
    assert c1["reasons"]["admitted"] and c1["cost"]["source_replay"]["total_tokens"] == 100
    # the second repair revises the (attributed) Skill and Operator, but Q_val does not strictly improve
    assert c2["R_src"] and c2["feasible"] and not c2["reasons"]["improved"]
    assert c2["skill_log"]["attributed"] == [f"skill:{snaps[1].skills[next(iter(snaps[1].skills))].id}@v1"]
    assert c2["parent_version"] == c1["cand_version"] == "r1-c0001"
    # snapshots A_0 (empty) and A_1 (one Skill + one Operator), returned and on disk
    assert len(snaps) == 2 and not snaps[0].skills and not snaps[0].operators
    assert len(snaps[1].skills) == 1 and len(snaps[1].operators) == 1 and snaps[1].version == "r1-c0001"
    assert (rd / "programs" / "A_0" / "program.json").exists() and (rd / "programs" / "A_1" / "program.json").exists()
    stream_rows = jsonl(rd / "stream.jsonl")
    assert [r["key"] for r in stream_rows] == ["1:srcA", "1:srcB"]
    assert stream_rows[0]["accepted"] == ["c0001"] and stream_rows[1]["accepted"] == []
    assert stream_rows[0]["solve"]["evidence"] == [{"step": 4, "passed": False}, {"step": 9, "passed": True}]
    rounds = jsonl(rd / "rounds.jsonl")
    assert [r["snapshot"] for r in rounds] == ["programs/A_1"]
    # solver calls: 2 source solves + 2 source replays; val: 2 (A_0) + 1 + 1 (lazy)
    modes = [m for m, _, _ in solver.calls]
    assert modes.count("source") == 4 and modes.count("val") == 4
    assert (rd / "val_reports" / "A0.json").exists() and (rd / "candidates" / "c0001" / "bundle.json").exists()
    assert ev.program.version == "r1-c0001"


def test_candidate_failing_use_check_is_rejected_before_validation(tmp_path):
    c = cfg()
    solver = StubSolver(c.evolution, omit_use={"op:"})
    _, solver, snaps = run(tmp_path, [(1, make_episode("srcA", "D1"))], c, solver)
    [cand] = jsonl(tmp_path / "run" / "candidates.jsonl")
    assert cand["R_src"] is False and cand["Pass"] is True and cand["Use"] is False
    assert cand["use_missing"] and cand["use_missing"][0].startswith("op:")
    assert cand["reasons"]["rejected"] == "R_src" and cand["H_val"] is None
    assert [m for m, _, _ in solver.calls].count("val") == 2          # only the incumbent was validated
    assert not snaps[1].skills and not snaps[1].operators


def test_candidate_failing_source_replay_pass_is_rejected(tmp_path):
    c = cfg()
    _, _, snaps = run(tmp_path, [(1, make_episode("srcA", "D1"))], c, StubSolver(c.evolution, fail_source_replay=True))
    [cand] = jsonl(tmp_path / "run" / "candidates.jsonl")
    assert cand["R_src"] is False and cand["Pass"] is False and cand["accepted"] is False


def test_per_round_argmax(tmp_path):
    stream = [(1, make_episode("srcA", "D1")), (1, make_episode("srcB", "D2")), (2, make_episode("srcC", "D2"))]
    _, _, snaps = run(tmp_path, stream, cfg(update_schedule="per_round_argmax"))
    rd = tmp_path / "run"
    cands = jsonl(rd / "candidates.jsonl")
    assert [c["pooled"] for c in cands] == [True, True, True] and not any(c["accepted"] for c in cands)
    assert cands[0]["parent_version"] == cands[1]["parent_version"] == "A0"    # no update inside a round
    rounds = jsonl(rd / "rounds.jsonl")
    assert [(r["pool"], r["chosen"], r["accepted"]) for r in rounds] == [
        (["c0001", "c0002"], "c0001", True),     # tie on Q_val -> earliest candidate
        (["c0003"], "c0003", True)]
    assert rounds[0]["reasons"]["cand_macro_sr"] == 0.5 and rounds[1]["reasons"]["cand_macro_sr"] == 1.0
    assert len(snaps) == 3
    covered = [sorted({t for s in p.skills.values() for t in s.tags} & {"D1", "D2"}) for p in snaps]
    assert covered == [[], ["D1"], ["D1", "D2"]]
    # round 2 revised the attributed Skill (same id, new version) instead of adding a second one
    assert list(snaps[1].skills) == list(snaps[2].skills)
    sid = next(iter(snaps[2].skills))
    assert (snaps[1].skills[sid].version, snaps[2].skills[sid].version) == (1, 2)


def test_unlinked_variant_gates_skill_and_operator_independently(tmp_path):
    _, _, snaps = run(tmp_path, [(1, make_episode("srcA", "D1"))], cfg(variant="unlinked"))
    cands = jsonl(tmp_path / "run" / "candidates.jsonl")
    assert [(c["part"], c["accepted"]) for c in cands] == [("skills", True), ("operators", False)]
    assert len(cands[0]["omega"]) == 1 and cands[0]["omega"][0].startswith("skill:")
    assert len(snaps[1].skills) == 1 and not snaps[1].operators


def test_frozen_variant_solves_but_never_evolves(tmp_path):
    stream = [(1, make_episode("srcA", "D1")), (2, make_episode("srcB", "D1"))]
    _, solver, snaps = run(tmp_path, stream, cfg(variant="frozen"))
    rd = tmp_path / "run"
    assert jsonl(rd / "candidates.jsonl") == []
    assert [m for m, _, _ in solver.calls] == ["source", "source"]
    assert len(snaps) == 3 and len({p.fingerprint() for p in snaps}) == 1
    assert [r["key"] for r in jsonl(rd / "stream.jsonl")] == ["1:srcA", "2:srcB"]


def test_resume_after_crash_is_idempotent(tmp_path):
    stream = [(1, make_episode("srcA", "D1")), (1, make_episode("srcB", "D2")), (2, make_episode("srcC", "D1"))]
    c = cfg()
    crashing = StubSolver(c.evolution, crash_once={("source", "srcB")})
    with pytest.raises(SimulatedCrash):
        run(tmp_path, stream, c, crashing)
    rd = tmp_path / "run"
    state = json.loads((rd / "state.json").read_text())
    assert state["completed"] == ["1:srcA"] and state["cand_seq"] == 1
    # a partially written (uncommitted) receipt line must disappear on resume
    with open(rd / "candidates.jsonl", "a") as fh:
        fh.write(json.dumps({"cand_id": "c9999", "source_key": "1:srcB"}) + "\n")
    resumed_solver = StubSolver(c.evolution)
    _, resumed_solver, snaps = run(tmp_path, stream, c, resumed_solver)
    assert ("source", "srcA") not in {(m, e) for m, e, _ in resumed_solver.calls}   # committed work is skipped
    assert [r["key"] for r in jsonl(rd / "stream.jsonl")] == ["1:srcA", "1:srcB", "2:srcC"]
    cands = jsonl(rd / "candidates.jsonl")
    assert [x["cand_id"] for x in cands] == ["c0001", "c0002", "c0003"]
    # identical outcome to an uninterrupted run
    _, _, ref = run(tmp_path, stream, c, name="ref")
    ref_cands = jsonl(tmp_path / "ref" / "candidates.jsonl")
    assert [x["accepted"] for x in cands] == [x["accepted"] for x in ref_cands] == [True, True, False]
    assert [p.fingerprint() for p in snaps] == [p.fingerprint() for p in ref]
    # a finished run is a no-op when resumed again
    again_solver = StubSolver(c.evolution)
    _, again_solver, again = run(tmp_path, stream, c, again_solver)
    assert again_solver.calls == [] and [p.fingerprint() for p in again] == [p.fingerprint() for p in snaps]


def test_resume_refuses_a_different_configuration(tmp_path):
    stream = [(1, make_episode("srcA", "D1"))]
    run(tmp_path, stream, cfg())
    with pytest.raises(ValueError, match="different run"):
        run(tmp_path, stream, cfg(variant="skill_only"))


def test_source_solve_errors_are_recorded_and_the_stream_continues(tmp_path):
    class Flaky(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            if mode == "source" and episode.id == "srcA":
                raise RuntimeError("policy gateway 503")
            return super().solve(episode, program, mode, run_dir)

    c = cfg()
    stream = [(1, make_episode("srcA", "D1")), (1, make_episode("srcB", "D1"))]
    _, _, snaps = run(tmp_path, stream, c, Flaky(c.evolution))
    rows = jsonl(tmp_path / "run" / "stream.jsonl")
    assert "policy gateway 503" in rows[0]["error"] and rows[1]["accepted"] == ["c0001"]
    assert len(snaps[1].skills) == 1


def test_invalid_schedule_or_variant(tmp_path):
    with pytest.raises(ValueError):
        Evolver(cfg(update_schedule="sometimes"), make_llm(), StubPlan([], {}), tmp_path, solver=StubSolver())
    with pytest.raises(ValueError):
        Evolver(cfg(variant="magic"), make_llm(), StubPlan([], {}), tmp_path, solver=StubSolver())


def test_errored_incumbent_val_entries_are_resolved_before_gating(tmp_path):
    class FlakyVal(StubSolver):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.failed = False

        def solve(self, episode, program, mode, run_dir):
            if mode == "val" and episode.id == "valD2" and not self.failed:
                self.failed = True
                raise RuntimeError("transient 502")
            return super().solve(episode, program, mode, run_dir)

    c = cfg()
    _, solver, snaps = run(tmp_path, [(1, make_episode("srcA", "D1"))], c, FlakyVal(c.evolution))
    [cand] = jsonl(tmp_path / "run" / "candidates.jsonl")
    assert cand["incumbent_refreshed"] == {"errored": ["valD2"], "still_errored": []}
    assert cand["accepted"] and cand["qval"]["inc"]["macro_sr"] == 0.0
    assert len(snaps[1].skills) == 1


def test_candidate_is_not_gated_against_an_incumbent_that_is_still_errored(tmp_path):
    """An incumbent val entry that stays a gateway / solver error is a silent z = 0 baseline: do not gate on it."""
    class DownForA0(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            if mode == "val" and episode.id == "valD2" and program.version == "A0":
                raise RuntimeError("gateway down")
            return super().solve(episode, program, mode, run_dir)

    c = cfg()
    _, _, snaps = run(tmp_path, [(1, make_episode("srcA", "D1"))], c, DownForA0(c.evolution))
    [cand] = jsonl(tmp_path / "run" / "candidates.jsonl")
    assert cand["R_src"] is True and cand["accepted"] is False and cand["feasible"] is None
    assert cand["incumbent_refreshed"]["still_errored"] == ["valD2"]
    assert cand["reasons"]["rejected"] == "incumbent val entries errored"
    assert cand["val_errors"] == {"valD2": "RuntimeError: gateway down"}
    assert not snaps[1].skills


# ------------------------------------------------------------------ noise guard, end to end
def test_default_noise_guard_needs_two_improved_val_episodes(tmp_path):
    two_d1 = {"D1": [make_episode("valD1a", "D1", split="val"), make_episode("valD1b", "D1", split="val")],
              "D2": [make_episode("valD2", "D2", split="val")]}
    for k, val, accepted in ((2, val_split(), False), (2, two_d1, True), (1, val_split(), True)):
        c = cfg(min_improved_episodes=k)
        _, _, snaps = run(tmp_path, [(1, make_episode("srcA", "D1"))], c, val=val, name=f"g{k}-{len(val['D1'])}")
        [cand] = jsonl(tmp_path / f"g{k}-{len(val['D1'])}" / "candidates.jsonl")
        assert cand["accepted"] is accepted, (k, len(val["D1"]))
        if not accepted:
            assert cand["feasible"] and cand["reasons"]["q_gain"] and not cand["reasons"]["gain_supported"]
            assert cand["reasons"]["improved_episodes"] == ["valD1"]


# ------------------------------------------------------------------ gateway outages, end to end
class OutageSource(StubSolver):
    """Source solves of ``episodes`` fail with a gateway outage (no evidence) for their first ``n`` attempts."""

    def __init__(self, evo_cfg, episodes: dict[str, int]) -> None:
        super().__init__(evo_cfg)
        self.outages = dict(episodes)

    def solve(self, episode, program, mode, run_dir):
        res = super().solve(episode, program, mode, run_dir)
        if mode == "source" and self.outages.get(episode.id, 0) > 0 and not program.skills and not program.operators:
            self.outages[episode.id] -= 1
            res.evidence, res.eval.z, res.eval.accepted = [], 0, False
            res.infra_error = "policy call failed: gateway 503"
            res.stop_reason = "policy_error"
        return res


def _n_source(solver: StubSolver, eid: str) -> int:
    return len([c for c in solver.calls if c[0] == "source" and c[1] == eid and c[2] == "A0"])


def test_source_outage_cleared_by_the_in_place_retries_is_invisible(tmp_path):
    c = cfg(infra_retries=2, infra_backoff_s=0.0)
    solver = OutageSource(c.evolution, {"srcA": 2})
    run(tmp_path, [(1, make_episode("srcA", "D1"))], c, solver)
    [row] = jsonl(tmp_path / "run" / "stream.jsonl")
    assert not row.get("infra_error") and not row.get("requeue") and row["accepted"] == ["c0001"]
    assert _n_source(solver, "srcA") == 3


def test_source_outage_is_requeued_to_the_end_of_the_round_and_not_committed_as_unsolved(tmp_path):
    c = cfg(infra_retries=1, infra_backoff_s=0.0)
    solver = OutageSource(c.evolution, {"srcA": 2})                    # both in-place attempts fail, the re-queue works
    stream = [(1, make_episode("srcA", "D1")), (1, make_episode("srcB", "D2"))]
    run(tmp_path, stream, c, solver)
    rd = tmp_path / "run"
    rows = jsonl(rd / "stream.jsonl")
    assert [(r["episode"], bool(r.get("requeue"))) for r in rows] == [("srcA", True), ("srcB", False), ("srcA", False)]
    assert "infrastructure error" in rows[0]["error"] and rows[0]["solve_attempts"] == 2 and rows[0]["candidates"] == []
    assert rows[2]["error"] is None and rows[2]["accepted"]            # the last chance ran, on the updated program
    assert rows[2]["program_version"] != "A0" and _n_source(solver, "srcA") == 2   # srcB's Skill was already in
    state = json.loads((rd / "state.json").read_text())
    assert state["completed"].count("1:srcA") == 1 and "1:srcB" in state["completed"]


def test_persistent_source_outage_is_committed_once_with_the_infra_marker(tmp_path):
    c = cfg(infra_retries=1, infra_backoff_s=0.0)
    solver = OutageSource(c.evolution, {"srcA": 99})
    _, _, snaps = run(tmp_path, [(1, make_episode("srcA", "D1"))], c, solver)
    rows = jsonl(tmp_path / "run" / "stream.jsonl")
    assert [bool(r.get("requeue")) for r in rows] == [True, False]
    last = rows[-1]
    assert last["infra_error"] and last["error"].startswith("source solve cut short by an infrastructure error")
    assert last["candidates"] == [] and len(snaps) == 2 and not snaps[1].skills
    assert json.loads((tmp_path / "run" / "state.json").read_text())["completed"] == ["1:srcA"]


def test_source_replay_outage_is_recorded_on_the_rejected_candidate(tmp_path):
    class ReplayOutage(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            res = super().solve(episode, program, mode, run_dir)
            if mode == "source" and (program.skills or program.operators):
                res.evidence, res.infra_error = [], "all 3 LLM calls failed: HTTP 503"
                res.eval.z, res.eval.accepted = 0, False
            return res

    c = cfg(infra_retries=1, infra_backoff_s=0.0)
    solver = ReplayOutage(c.evolution)
    run(tmp_path, [(1, make_episode("srcA", "D1"))], c, solver)
    [cand] = jsonl(tmp_path / "run" / "candidates.jsonl")
    assert cand["R_src"] is False and cand["accepted"] is False
    assert cand["infra_error"].startswith("all 3 LLM calls failed") and cand["reasons"]["infra_error"]
    replays = [x for x in solver.calls if x[0] == "source" and x[2] != "A0"]
    assert len(replays) == 2                                            # one bounded retry
