"""Eq. 9: evolution instances, e-/e+ endpoints and the repair window delta."""
from __future__ import annotations

from scienceclaw.agent.solver import SolveResult, StepRecord
from scienceclaw.bench.task import EvalResult
from scienceclaw.core.graph import WorkflowGraph
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.trace import Evidence, Trace
from scienceclaw.evolution import extract_instances

from test_evolution_support import build_solve_result, make_episode, repair_steps


def test_window_with_e_minus(tmp_path):
    ep = make_episode()
    res = build_solve_result(ep, AgentProgram(), tmp_path, repair_steps(uses_at_repair=["skill:foo"]))
    assert [(e.step, e.passed) for e in res.evidence] == [(4, False), (9, True)]
    [inst] = extract_instances(res)
    assert (inst.k_minus, inst.k_plus) == (4, 9)
    assert inst.e_minus is res.evidence[0] and inst.e_plus is res.evidence[1]
    # the actions that turn the replayed failing graph (after step 4) into the passing one (after step 9)
    assert inst.delta_steps == [5, 6, 7, 8, 9]
    assert [a.type for a in inst.delta] == ["modify_node", "add_node", "remove_edge", "add_edge", "add_edge"]
    assert inst.uses_in_delta == {"skill:foo"}
    assert inst.window == (5, 9)
    assert inst.feedback_plus["dev"] == {"dev_score": 0.9}
    assert inst.feedback_minus["dev"] == {"dev_score": 0.4}
    assert inst.summary()["has_e_minus"] is True


def test_rejected_and_unparsed_steps_are_not_part_of_delta(tmp_path):
    ep = make_episode()
    # step 6: an invalid edit rejected by validation (graph unchanged); step 8: unparseable reply
    specs = repair_steps(uses_at_repair=["skill:foo"], rejected_at=6, parse_error_at=8)
    res = build_solve_result(ep, AgentProgram(), tmp_path, specs)
    assert [(e.step, e.passed) for e in res.evidence] == [(4, False), (11, True)]
    [inst] = extract_instances(res)
    assert inst.delta_steps == [5, 7, 9, 10, 11]
    assert [r["step"] for r in inst.rejected] == [6]
    assert "nope" in inst.rejected[0]["error"]
    assert inst.uses_in_delta == {"skill:foo"}


def test_first_replay_passes_no_e_minus(tmp_path):
    ep = make_episode()
    specs = repair_steps()
    specs[4] = {k: v for k, v in specs[4].items() if k != "replay"}   # no replay of the failing graph
    res = build_solve_result(ep, AgentProgram(), tmp_path, specs)
    assert [(e.step, e.passed) for e in res.evidence] == [(9, True)]
    [inst] = extract_instances(res)
    assert inst.e_minus is None and inst.k_minus is None
    assert inst.delta_steps == list(range(10))   # all actions up to and including k+
    assert inst.window == (0, 9)
    assert inst.feedback_minus is None


def _ev(step: int, passed: bool) -> Evidence:
    return Evidence(step=step, graph_dict={"nodes": [], "edges": []}, y=None, trace=Trace(), eval=EvalResult(),
                    passed=passed)


def _result(evidence: list[Evidence], n_steps: int) -> SolveResult:
    steps = [StepRecord(k, {"type": "modify_node", "payload": {"id": "a", "patch": {"config": {"k": k}}},
                            "uses": [f"skill:s{k}"]}, None, {"action_ok": True}, [f"skill:s{k}"], {}, "", None, 0.0)
             for k in range(n_steps)]
    return SolveResult(episode_id="e", mode="source", program_version="A0", final_graph=WorkflowGraph(), y=None,
                       trace=None, eval=EvalResult(), evidence=evidence, steps=steps, actions=[], retrieved={},
                       uses=set(), usage={}, run_dir="")


def test_multiple_instances_start_at_each_fail_to_pass_transition():
    # evidence (unsorted on purpose): F@1, P@2, P@3, F@4, P@5
    evs = [_ev(3, True), _ev(1, False), _ev(2, True), _ev(5, True), _ev(4, False)]
    res = _result(evs, 6)
    [first] = extract_instances(res)                       # default: the first replay-verified success
    assert (first.k_minus, first.k_plus, first.delta_steps) == (1, 2, [2])
    insts = extract_instances(res, max_instances=5)
    assert [(i.k_minus, i.k_plus) for i in insts] == [(1, 2), (4, 5)]   # P@3 continues a passing run
    assert insts[1].delta_steps == [5]
    assert insts[1].uses_in_delta == {"skill:s5"}
    assert extract_instances(res, max_instances=0) == []


def test_no_passing_evidence_gives_no_instance():
    assert extract_instances(_result([_ev(1, False), _ev(3, False)], 4)) == []
    assert extract_instances(_result([], 2)) == []
