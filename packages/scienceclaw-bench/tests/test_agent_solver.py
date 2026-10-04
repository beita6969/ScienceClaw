"""Solver tests (offline, stub executor): evidence stream, G* selection, hidden-score isolation,
orchestration restrictions, budgets, receipts and thread safety."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_agent_support import (  # noqa: E402
    ACCEPTANCE_TEXT, BUGGY_CODE, FINISH, GOOD_CODE, SECRET_METRIC, SECRET_SCORE, WRONG_CODE, ScriptedLLM,
    StubExecutor, act, add_edge, add_node, build_script, make_episode, make_solver, modify_code, step_of, stub_replay,
)

from scienceclaw.agent.solver import (  # noqa: E402
    FIXED_CODE_NODE, SolveResult, build_fixed_workflow, llm_outage_error, normalize_uses,
)
from scienceclaw.config import EvolutionConfig, SolverConfig  # noqa: E402
from scienceclaw.core.program import AgentProgram  # noqa: E402
from scienceclaw.core.skills import Skill  # noqa: E402
from scienceclaw.core.trace import NodeRecord  # noqa: E402

HIDDEN_STRINGS = (SECRET_METRIC, str(SECRET_SCORE), "0.1234567", ACCEPTANCE_TEXT, "INVISIBLE_CONSTRAINT_TEXT",
                  "hidden_rule_q", "pooled_payload")


def _assert_no_hidden(llm: ScriptedLLM) -> None:
    text = llm.all_text()
    for s in HIDDEN_STRINGS:
        assert s not in text, f"hidden evaluation content {s!r} leaked into the policy context"


def _program() -> AgentProgram:
    s = Skill(id="scale", version=1, title="Scaling workflows", body="Regression by scaling x values.",
              tags=["regression", "FoR49"])
    return AgentProgram(skills={"scale": s})


# --------------------------------------------------------------------------------------- source mode
def test_source_mode_failure_then_pass_and_final_graph(tmp_path: Path) -> None:
    llm = ScriptedLLM(build_script(BUGGY_CODE, fix=True))
    solver = make_solver(SolverConfig(), llm, EvolutionConfig())
    res = solver.solve(make_episode(), AgentProgram(), "source", tmp_path / "run")

    assert isinstance(res, SolveResult)
    assert [e.passed for e in res.evidence] == [False, True]
    fail, ok = res.evidence
    assert fail.step == 4 and ok.step == 5
    assert fail.y is None and fail.eval.completed is False and "f" in fail.trace.errors()
    assert ok.eval.reproducible is True and ok.eval.z == 1
    assert res.passed and res.z == 1 and res.final_source == "pass" and res.final_step == 5
    assert res.final_graph.nodes["f"].code == GOOD_CODE
    assert res.y == [1.25, 2.5, 3.75, 5.0]
    assert res.stop_reason == "finish"
    assert [s.evidence_idx for s in res.steps] == [None, None, None, None, 0, 1, None]
    assert res.steps[4].submit_fp and res.steps[4].y_available is False and res.steps[5].y_available is True
    assert res.usage["replays"] == 2 and res.usage["policy_prompt_tokens"] == 700
    _assert_no_hidden(llm)
    # every policy call is a fresh [system, user] context with the same system prompt
    systems = {c["messages"][0]["content"] for c in llm.calls}
    assert len(systems) == 1 and all(c["role"] == "policy" for c in llm.calls)


def test_source_mode_wrong_output_is_failure_evidence(tmp_path: Path) -> None:
    llm = ScriptedLLM(build_script(WRONG_CODE, fix=True))
    res = make_solver(SolverConfig(), llm).solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert [e.passed for e in res.evidence] == [False, True]
    assert res.evidence[0].eval.completed is True and res.evidence[0].eval.accepted is False
    assert res.evidence[0].eval.reproducible is True
    _assert_no_hidden(llm)


def test_final_is_last_verified_pass_even_if_later_edit_breaks(tmp_path: Path) -> None:
    script = build_script(GOOD_CODE, fix=False)[:-1] + [modify_code("f", BUGGY_CODE), FINISH]
    res = make_solver(SolverConfig(), ScriptedLLM(script)).solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert [e.passed for e in res.evidence] == [True, False]
    assert res.final_source == "pass" and res.final_step == 4
    assert res.final_graph.nodes["f"].code == GOOD_CODE and res.z == 1


def test_no_pass_falls_back_to_last_output(tmp_path: Path) -> None:
    res = make_solver(SolverConfig(), ScriptedLLM(build_script(WRONG_CODE, fix=False))).solve(
        make_episode(), AgentProgram(), "source", tmp_path)
    assert res.final_source == "last_output" and res.final_step == 4
    assert res.passed is False and res.z == 0 and res.y == [0.5, 1.0, 1.5, 2.0]
    assert len(res.evidence) == 1  # the final graph's evidence is reused, not replayed twice
    assert res.usage["replays"] == 1


def test_no_output_is_failure(tmp_path: Path) -> None:
    script = [add_node(id="load", kind="tool", ref="load_x"), FINISH]
    res = make_solver(SolverConfig(), ScriptedLLM(script)).solve(make_episode(), AgentProgram(), "eval", tmp_path)
    assert res.y is None and res.z == 0 and res.final_source == "none" and res.evidence == []
    assert res.eval.completed is False


def test_non_reproducible_output_does_not_pass(tmp_path: Path) -> None:
    solver = make_solver(SolverConfig(), ScriptedLLM(build_script(GOOD_CODE, fix=False)),
                         outputs_match_fn=lambda a, b, tol: False)
    res = solver.solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert len(res.evidence) == 1 and res.evidence[0].passed is False
    assert res.evidence[0].eval.reproducible is False and res.evidence[0].eval.z == 0 and res.z == 0


def test_pass_without_acceptance_when_configured(tmp_path: Path) -> None:
    solver = make_solver(SolverConfig(), ScriptedLLM(build_script(WRONG_CODE, fix=False)),
                         EvolutionConfig(pass_requires_acceptance=False))
    res = solver.solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert res.evidence[0].passed is True and res.evidence[0].eval.z == 0


def test_stop_on_first_pass(tmp_path: Path) -> None:
    script = build_script(GOOD_CODE, fix=False)[:-1] + [modify_code("f", WRONG_CODE), FINISH]
    llm = ScriptedLLM(script)
    res = make_solver(SolverConfig(stop_on_first_pass=True), llm).solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert res.stop_reason == "first_pass" and len(res.steps) == 5 and len(llm.calls) == 5


def test_replay_only_on_output_when_failed_submits_disabled(tmp_path: Path) -> None:
    solver = make_solver(SolverConfig(), ScriptedLLM(build_script(BUGGY_CODE, fix=True)), replay_failed_submits=False)
    res = solver.solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert [e.passed for e in res.evidence] == [True]


# ------------------------------------------------------------------------------------- val / eval mode
@pytest.mark.parametrize("mode", ["val", "eval"])
def test_val_eval_replay_only_final(tmp_path: Path, mode: str) -> None:
    llm = ScriptedLLM(build_script(BUGGY_CODE, fix=True))
    res = make_solver(SolverConfig(), llm).solve(make_episode(), AgentProgram(), mode, tmp_path)
    assert len(res.evidence) == 1 and res.usage["replays"] == 1
    assert res.evidence[0].step == 5 and res.passed and res.z == 1
    assert all(s.evidence_idx is None for s in res.steps)
    _assert_no_hidden(llm)


def test_policy_context_identical_across_modes(tmp_path: Path) -> None:
    """The policy cannot tell source from eval mode (hidden evaluation never feeds back)."""
    ctx = {}
    for mode in ("source", "eval"):
        llm = ScriptedLLM(build_script(BUGGY_CODE, fix=True))
        make_solver(SolverConfig(), llm).solve(make_episode(), AgentProgram(), mode, tmp_path / mode)
        ctx[mode] = [c["messages"] for c in llm.calls]
    assert ctx["source"] == ctx["eval"]


# ------------------------------------------------------------------------------------------ uses ν
def test_uses_are_restricted_to_program_and_noted(tmp_path: Path) -> None:
    script = build_script(GOOD_CODE, fix=False)
    script[1] = dict(script[1], uses=["skill:scale", "skill:nope", "op:ghost", "skill:scale@v1"])
    llm = ScriptedLLM(script)
    res = make_solver(SolverConfig(), llm).solve(make_episode(), _program(), "source", tmp_path)
    assert res.steps[1].uses == ["skill:scale"]
    assert res.steps[1].dropped_uses == ["skill:nope", "op:ghost"]
    assert res.uses == {"skill:scale"} and res.uses_versions == {"skill:scale@v1"}
    assert res.retrieved["skills"] == ["skill:scale"] and res.retrieved["slice_hash"]
    # the policy is told which ids were ignored (in the feedback of the next step)
    assert "not in your library" in llm.calls[2]["messages"][1]["content"]
    assert "skill:nope" in llm.calls[2]["messages"][1]["content"]
    # the retrieved skill is rendered in full in the system prompt
    assert "Regression by scaling x values." in llm.calls[0]["messages"][0]["content"]
    # the executor received the normalized uses
    ex = StubExecutor.instances[-1]
    assert ex.applied[1].uses == ["skill:scale"]


def _bad_tool_step(uses: list[str]) -> dict:
    return act({"type": "add_node", "node": {"id": "bad", "kind": "tool", "ref": "no_such_tool"}}, uses)


def test_uses_of_rejected_actions_are_not_counted(tmp_path: Path) -> None:
    """Use(omega) only sees components of actions that were actually applied."""
    script = [_bad_tool_step(["skill:scale"])] + build_script(GOOD_CODE, fix=False)
    res = make_solver(SolverConfig(), ScriptedLLM(script)).solve(make_episode(), _program(), "source", tmp_path)
    assert res.steps[0].feedback["action_ok"] is False and res.steps[0].uses == ["skill:scale"]   # cited, but rejected
    assert res.passed
    assert res.uses == set() and res.uses_versions == set()
    # ... and the same citation on an action that is applied counts
    ok = build_script(GOOD_CODE, fix=False)
    ok[1] = dict(ok[1], uses=["skill:scale"])
    res2 = make_solver(SolverConfig(), ScriptedLLM([_bad_tool_step(["skill:scale"])] + ok)).solve(
        make_episode(), _program(), "source", tmp_path / "ok")
    assert res2.uses == {"skill:scale"} and res2.uses_versions == {"skill:scale@v1"}


def test_uses_of_failed_parse_do_not_count(tmp_path: Path) -> None:
    script = ["not json"] + build_script(GOOD_CODE, fix=False)
    res = make_solver(SolverConfig(), ScriptedLLM(script)).solve(make_episode(), _program(), "source", tmp_path)
    assert res.uses == set()


def test_normalize_uses_variants() -> None:
    prog = _program()
    kept, dropped = normalize_uses(["scale", "skill:scale", "op:scale", "operator:x", 3], prog)
    assert kept == ["skill:scale"] and dropped == ["op:scale", "operator:x", "3"]


# ----------------------------------------------------------------------------------- parse failures
def test_parse_failure_consumes_step_and_is_fed_back(tmp_path: Path) -> None:
    script = ["this is not json", "still not json"] + build_script(GOOD_CODE, fix=False)
    llm = ScriptedLLM(script)
    res = make_solver(SolverConfig(), llm).solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert res.steps[0].parse_error and res.steps[0].action is None
    assert res.steps[0].policy_usage["attempts"] == 2 and res.steps[0].policy_usage["calls"] == 2
    # the re-ask within step 0 contains the parse error; step 1 is told the step failed
    assert "could not be parsed" in llm.calls[1]["messages"][-1]["content"]
    assert "could not be parsed" in llm.calls[2]["messages"][1]["content"]
    assert res.passed and res.usage["parse_failures"] == 1


# ------------------------------------------------------------------------------------ orchestration
def test_canvas_rejects_batch(tmp_path: Path) -> None:
    batch = act({"type": "batch", "actions": [{"type": "add_node", "node": {"id": "load", "kind": "tool", "ref": "load_x"}}]})
    llm = ScriptedLLM([batch, FINISH])
    res = make_solver(SolverConfig(), llm).solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert res.steps[0].feedback["action_ok"] is False and "single-turn" in res.steps[0].feedback["action_error"]
    assert res.final_graph.nodes == {}


def test_single_turn_one_batch_then_stop(tmp_path: Path) -> None:
    subs = [a["action"] for a in build_script(GOOD_CODE, fix=False)[:-1]]
    llm = ScriptedLLM([act({"type": "batch", "actions": subs}), FINISH])
    res = make_solver(SolverConfig(orchestration="single_turn"), llm).solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert len(llm.calls) == 1 and len(res.steps) == 1 and res.stop_reason == "single_turn"
    assert res.passed and res.z == 1
    system = llm.calls[0]["messages"][0]["content"]
    assert '"type": "batch"' in system and "at most 1" in system


def test_single_operator_restrictions(tmp_path: Path) -> None:
    script = [
        add_node(id="load", kind="tool", ref="load_x"),
        add_node(id="f", kind="code", code=GOOD_CODE, inputs={"x": {"type": "list"}}, outputs={"y": {"type": "list"}}),
        add_node(id="g", kind="code", code=GOOD_CODE, inputs={"x": {"type": "list"}}, outputs={"y": {"type": "list"}}),
        add_node(id="h", kind="llm", prompt="{x}", inputs={"items": {"type": "list"}}, outputs={"outputs": {"type": "list"}}),
        add_edge("load", "x", "f", "x"),
        add_node(id="sub", kind="submit"),
        add_edge("f", "y", "sub", "y"),
        FINISH,
    ]
    res = make_solver(SolverConfig(orchestration="single_operator"), ScriptedLLM(script)).solve(
        make_episode(), AgentProgram(), "source", tmp_path)
    fbs = [s.feedback for s in res.steps]
    assert fbs[2]["action_ok"] is False and "at most one" in fbs[2]["action_error"]
    assert fbs[3]["action_ok"] is False and "llm nodes cannot be added" in fbs[3]["action_error"]
    assert all(fbs[i]["action_ok"] for i in (0, 1, 4, 5, 6))
    assert set(res.final_graph.nodes) == {"load", "f", "sub"} and res.passed
    assert res.usage["rejected_actions"] == 2


def test_fixed_workflow(tmp_path: Path) -> None:
    ep = make_episode()
    g, code_id = build_fixed_workflow(ep)
    assert code_id == FIXED_CODE_NODE and g.validate() == []
    assert {n.kind for n in g.nodes.values()} == {"tool", "code", "submit"}
    script = [
        add_node(id="x2", kind="tool", ref="load_x"),
        modify_code("code", GOOD_CODE),
        act({"type": "modify_node", "id": "code", "patch": {"outputs": {"y": {"type": "list"}}}}),
        FINISH,
    ]
    llm = ScriptedLLM(script)
    res = make_solver(SolverConfig(orchestration="fixed_workflow"), llm).solve(ep, AgentProgram(), "source", tmp_path)
    fbs = [s.feedback for s in res.steps]
    assert fbs[0]["action_ok"] is False and "only modify_node" in fbs[0]["action_error"]
    assert fbs[1]["action_ok"] is True
    assert fbs[2]["action_ok"] is False and "'code', 'code_edit' and/or 'config'" in fbs[2]["action_error"]
    assert res.passed and res.final_graph.nodes["code"].code == GOOD_CODE
    assert "tool_load_x" in llm.calls[0]["messages"][1]["content"]   # the pre-built canvas is shown
    assert 'code node "code"' in llm.calls[0]["messages"][0]["content"]


# ------------------------------------------------------------------------------------------ budgets
def test_step_budget_comes_from_solver_config(tmp_path: Path) -> None:
    llm = ScriptedLLM(build_script(GOOD_CODE, fix=False))
    res = make_solver(SolverConfig(max_steps=3), llm).solve(make_episode(max_steps=12), AgentProgram(), "source", tmp_path)
    assert len(res.steps) == 3 and res.stop_reason == "step_budget" and res.y is None


def test_step_budget_falls_back_to_episode(tmp_path: Path) -> None:
    llm = ScriptedLLM(build_script(GOOD_CODE, fix=False))
    res = make_solver(SolverConfig(max_steps=0), llm).solve(make_episode(max_steps=3), AgentProgram(), "source", tmp_path)
    assert len(res.steps) == 3 and res.stop_reason == "step_budget"


def test_token_budget_stops(tmp_path: Path) -> None:
    llm = ScriptedLLM(build_script(GOOD_CODE, fix=False), tokens_per_call=1000)
    res = make_solver(SolverConfig(), llm).solve(make_episode(max_policy_tokens=2500), AgentProgram(), "source", tmp_path)
    assert res.stop_reason == "token_budget" and len(res.steps) == 2


def test_cached_tokens_count_for_budget_but_not_cost(tmp_path: Path) -> None:
    llm = ScriptedLLM(build_script(GOOD_CODE, fix=False), tokens_per_call=1000, cached=True)
    res = make_solver(SolverConfig(), llm).solve(make_episode(max_policy_tokens=2500), AgentProgram(), "source", tmp_path)
    assert res.stop_reason == "token_budget" and len(res.steps) == 2
    assert res.usage["policy_prompt_tokens"] == 0 and res.usage["policy_cached_prompt_tokens"] == 2000
    assert res.usage["total_tokens"] == 0 and res.usage["cached_llm_calls"] == 2


def test_policy_transport_error_stops_with_record(tmp_path: Path) -> None:
    def boom(role, messages):
        raise ConnectionError("gateway down")

    res = make_solver(SolverConfig(), ScriptedLLM(boom)).solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert res.stop_reason == "policy_error" and len(res.steps) == 1 and "gateway down" in res.steps[0].parse_error


# ------------------------------------------------------------------------------------------ receipts
def test_receipts_written_without_hidden_labels(tmp_path: Path) -> None:
    run = tmp_path / "run"
    res = make_solver(SolverConfig(), ScriptedLLM(build_script(BUGGY_CODE, fix=True))).solve(
        make_episode(), AgentProgram(), "source", run)
    for name in ("trajectory.jsonl", "final_graph.json", "eval.json", "evidence.jsonl", "usage.json", "result.json",
                 "y.pkl", "system_prompt.txt", "policy_io.jsonl"):
        assert (run / name).exists(), name
    traj = [json.loads(line) for line in (run / "trajectory.jsonl").read_text().splitlines()]
    assert len(traj) == len(res.steps) and traj[5]["evidence_idx"] == 1
    ev = json.loads((run / "eval.json").read_text())
    assert "pooled_payload" not in ev["details"] and "labels" not in ev["details"]
    assert ev["details"]["reference"] == 0.5
    blob = "".join(p.read_text() for p in run.glob("*.json*"))
    assert "5.0, 3.75" not in blob and "y_true" not in blob
    usage = json.loads((run / "usage.json").read_text())
    for key in ("policy_prompt_tokens", "policy_completion_tokens", "executor_prompt_tokens",
                "executor_completion_tokens", "total_tokens", "llm_calls", "wall_s", "policy_wall_s", "node_runs",
                "replays"):
        assert key in usage
    # the system prompt never contains hidden evaluation content
    sp = (run / "system_prompt.txt").read_text()
    assert ACCEPTANCE_TEXT not in sp and SECRET_METRIC not in sp


# ----------------------------------------------------------------------------------- thread safety
def test_concurrent_solves_are_independent(tmp_path: Path) -> None:
    scripts = {
        "good": build_script(BUGGY_CODE, fix=True),
        "bad": build_script(WRONG_CODE, fix=False),
    }

    def responder(role, messages):
        kind = "good" if "EPISODE-good" in messages[0]["content"] else "bad"
        k = step_of(messages)
        seq = scripts[kind]
        return seq[k] if k < len(seq) else FINISH

    llm = ScriptedLLM(responder)
    solver = make_solver(SolverConfig(), llm)
    jobs = [(f"e{i}", "good" if i % 2 == 0 else "bad") for i in range(8)]

    def run(job):
        eid, kind = job
        ep = make_episode(eid, objective=f"EPISODE-{kind}: predict y for each visible x.")
        return kind, solver.solve(ep, AgentProgram(), "source", tmp_path / eid)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, jobs))
    for kind, res in results:
        assert res.passed is (kind == "good")
        assert Path(res.run_dir).name == res.episode_id
        assert len(res.evidence) == (2 if kind == "good" else 1)
    assert len({r.run_dir for _, r in results}) == len(jobs)


def test_scrub_volatile() -> None:
    from scienceclaw.agent.solver import scrub_volatile

    t = '  [f] code ERROR (0.09s, cached)\n  [g] code ok (12.50s)\n  File "/x/run/exec/work/f-1a2b3c4d/node_code.py"'
    out = scrub_volatile(t, "/x/run")
    assert out == '  [f] code ERROR (cached)\n  [g] code ok\n  File "<run>/exec/work/f/node_code.py"'
    assert scrub_volatile("range=[1, 4] (3 values)") == "range=[1, 4] (3 values)"


# ---------------------------------------------------------------- infrastructure errors and usage
def test_infra_error_is_none_for_ordinary_failures(tmp_path: Path) -> None:
    res = make_solver(SolverConfig(), ScriptedLLM(build_script(BUGGY_CODE, fix=False))).solve(
        make_episode(), AgentProgram(), "source", tmp_path)
    assert not res.passed and res.infra_error is None and res.summary()["infra_error"] is None


def test_policy_transport_error_is_an_infra_error(tmp_path: Path) -> None:
    def boom(role, messages):
        raise ConnectionError("gateway down")

    res = make_solver(SolverConfig(), ScriptedLLM(boom)).solve(make_episode(), AgentProgram(), "source", tmp_path)
    assert res.stop_reason == "policy_error" and res.z == 0
    assert res.infra_error and res.infra_error.startswith("policy call failed") and "gateway down" in res.infra_error
    assert res.summary()["infra_error"] == res.infra_error
    assert any("infrastructure error" in n for n in res.notes)


def _outage_replay(msg: str):
    def replay(graph, episode, program, llm, run_dir):
        _, trace = stub_replay(graph, episode, program, llm, run_dir)
        trace.records["f"] = NodeRecord("f", "fp", "error", error=msg, kind="code")
        return None, trace
    return replay


def test_llm_outage_in_the_final_trace_is_an_infra_error(tmp_path: Path) -> None:
    res = make_solver(SolverConfig(), ScriptedLLM(build_script(GOOD_CODE, fix=False)),
                      replay_fn=_outage_replay("RuntimeError: all 3 LLM calls failed: HTTP 503")).solve(
        make_episode(), AgentProgram(), "source", tmp_path)
    assert not res.passed and res.z == 0
    assert res.infra_error and "all 3 LLM calls failed" in res.infra_error and res.infra_error.startswith("node f:")


def test_partial_llm_failures_are_not_an_outage(tmp_path: Path) -> None:
    res = make_solver(SolverConfig(), ScriptedLLM(build_script(GOOD_CODE, fix=False)),
                      replay_fn=_outage_replay("RuntimeError: 2 of 3 LLM calls failed")).solve(
        make_episode(), AgentProgram(), "source", tmp_path)
    assert not res.passed and res.infra_error is None


def test_llm_outage_error_matches_only_whole_batch_failures() -> None:
    def rec(err, status="error"):
        return {"n": NodeRecord("n", "fp", status, error=err, kind="llm")}

    assert llm_outage_error(rec("all 4 LLM calls failed: timeout"))
    assert llm_outage_error(rec("LLM call failed: HTTP 502"))
    assert llm_outage_error(rec("3 of 4 LLM calls failed")) is None
    assert llm_outage_error(rec("ValueError: bad json")) is None
    assert llm_outage_error(rec("all 4 LLM calls failed", status="ok")) is None
    assert llm_outage_error(None) is None and llm_outage_error({}) is None


def test_usage_reports_logical_tokens_spent_plus_cached(tmp_path: Path) -> None:
    class ExecUsage(StubExecutor):
        def apply(self, graph, checkpoint, action, step):
            g, ck, fb, y = super().apply(graph, checkpoint, action, step)
            fb.llm_usage = {"prompt_tokens": 10, "completion_tokens": 5, "cached_prompt_tokens": 20,
                            "cached_completion_tokens": 7, "calls": 1, "cached_calls": 1}
            return g, ck, fb, y

    llm = ScriptedLLM(build_script(GOOD_CODE, fix=False), tokens_per_call=1000, cached=True)
    res = make_solver(SolverConfig(), llm, executor_factory=ExecUsage).solve(make_episode(), AgentProgram(), "source", tmp_path)
    u = res.usage
    n_pol, n_exec = len(llm.calls), len(StubExecutor.instances[-1].applied)      # finish is not applied
    assert u["total_tokens"] == 15 * n_exec                                 # spent: executor only (policy was served from cache)
    assert u["policy_logical_tokens"] == 1500 * n_pol
    assert u["executor_logical_tokens"] == 42 * n_exec
    assert u["logical_tokens"] == u["policy_logical_tokens"] + u["executor_logical_tokens"] > u["total_tokens"]
    fresh = make_solver(SolverConfig(), ScriptedLLM(build_script(GOOD_CODE, fix=False), tokens_per_call=1000)).solve(
        make_episode(), AgentProgram(), "source", tmp_path / "fresh")
    assert fresh.usage["logical_tokens"] == fresh.usage["total_tokens"]


def test_replay_runs_hidden_probes_in_sibling_dirs_before_evaluation(tmp_path: Path) -> None:
    ep = make_episode()
    seen: list[tuple] = []

    def run_probes(y, trace, runner):
        y_p, tr_p = runner(ep, "cut")
        seen.append((y, y_p, tr_p is not None))
        return {}

    ep.run_probes = run_probes  # type: ignore[method-assign]
    res = make_solver(SolverConfig(), ScriptedLLM(build_script(GOOD_CODE, fix=False))).solve(
        ep, AgentProgram(), "source", tmp_path / "run")
    assert res.passed and seen and seen[0][0] == seen[0][1] and seen[0][2]
    dirs = sorted(p.name for p in (tmp_path / "run" / "replay").iterdir())
    assert any(d.endswith("_probe_cut") for d in dirs) and any(not d.endswith("_probe_cut") for d in dirs)
