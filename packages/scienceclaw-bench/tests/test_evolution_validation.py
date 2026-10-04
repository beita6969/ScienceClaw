"""Eq. 13 (R_src = Pass and Use) and the validation gate (Eq. 2 feasibility, Eq. 3 strict improvement)."""
from __future__ import annotations

import threading

import pytest

from scienceclaw.agent.solver import SolveResult, StepRecord
from scienceclaw.bench.task import EvalResult
from scienceclaw.config import EvolutionConfig, SolverConfig
from scienceclaw.core.graph import Node, WorkflowGraph
from scienceclaw.core.program import AgentProgram, Bundle
from scienceclaw.core.skills import Skill
from scienceclaw.core.trace import Evidence, NodeRecord, Trace
from scienceclaw.evolution import ValidationGate, ValReport, source_replay_check
from scienceclaw.evolution.validation import aggregate_report, qval_key, source_pass, use_check, used_refs

from test_evolution_support import StubRetriever, StubSolver, make_episode, val_result


def entry(disc: str, z: int, norm: float, *, hv: list[str] | None = None, integrity: list[str] | None = None,
          error: str | None = None, completed: bool = True, schema_ok: bool = True, tokens: float = 10.0,
          wall: float = 1.0, logical: float | None = None) -> dict:
    cost = {"total_tokens": tokens, "wall_s": wall}
    if logical is not None:
        cost["logical_tokens"] = logical
    return {"discipline": disc, "z": z, "norm_score": norm, "completed": completed,
            "h_ok": completed and not hv, "hard_violations": list(hv or []), "schema_ok": schema_ok,
            "integrity_violations": list(integrity or []), "error": error,
            "cost": cost, "slice_hash": "h", "reused": False, "run_dir": ""}


def report(version: str, per: dict[str, dict], mode: str = "no_regression", ref: ValReport | None = None) -> ValReport:
    return aggregate_report(version, per, mode, reference=ref)


def gate(tmp_path, **kw) -> ValidationGate:
    kw.setdefault("min_improved_episodes", 1)      # single-flip toy pairs: literal Eq. 3 unless a test says otherwise
    return ValidationGate(EvolutionConfig(**kw), SolverConfig(), None, [], tmp_path)


# --------------------------------------------------------------------------------- aggregation
def test_macro_sr_and_norm_score_are_macro_over_disciplines():
    rep = report("A", {"a1": entry("D1", 1, 1.0), "a2": entry("D1", 0, 0.2), "b1": entry("D2", 1, 0.5)})
    assert rep.macro_sr == pytest.approx((0.5 + 1.0) / 2)
    assert rep.norm_score == pytest.approx(((1.0 + 0.2) / 2 + 0.5) / 2)
    assert rep.cost == {"total_tokens": 30.0, "wall_s": 3.0} and rep.incurred == rep.cost
    assert rep.h_val == {"integrity": True, "schema": True, "no_solver_error": True, "no_new_hard_violation": True}
    assert qval_key(rep, "macrosr") == (0.75,) and qval_key(rep, "score") == (rep.norm_score,)
    assert ValReport.from_dict(rep.to_dict()) == rep


# ------------------------------------------------------------------------------------ Eq. 3
def _pair(sr_inc: tuple[int, int], sr_cand: tuple[int, int], ns_inc: float, ns_cand: float):
    """Two disciplines with one episode each; z per discipline and a common norm score."""
    inc = report("inc", {"a": entry("D1", sr_inc[0], ns_inc), "b": entry("D2", sr_inc[1], ns_inc)})
    cand = report("cand", {"a": entry("D1", sr_cand[0], ns_cand), "b": entry("D2", sr_cand[1], ns_cand)}, ref=inc)
    return cand, inc


def test_admit_macrosr_literal_rule(tmp_path):
    g = gate(tmp_path, qval="macrosr")
    cand, inc = _pair((1, 0), (1, 1), 0.5, 0.5)            # MacroSR 0.5 -> 1.0
    ok, why = g.admit(cand, inc)
    assert ok and why["delta_macro_sr"] == pytest.approx(0.5) and why["admitted"]
    cand, inc = _pair((1, 0), (0, 1), 0.2, 0.9)            # MacroSR 0.5 -> 0.5, score up: not strict in MacroSR
    ok, why = g.admit(cand, inc)
    assert not ok and why["improved"] is False and why["feasible"] is True


def test_admit_macrosr_then_score(tmp_path):
    g = gate(tmp_path, qval="macrosr_then_score", qval_eps=0.005)
    assert g.admit(*_pair((1, 0), (1, 1), 0.9, 0.1))[0]                  # MacroSR up, score down: accept
    assert g.admit(*_pair((1, 0), (0, 1), 0.500, 0.510))[0]              # tie, score +0.010 >= eps
    assert g.admit(*_pair((1, 0), (0, 1), 0.500, 0.505))[0]              # tie, score +0.005 == eps
    assert not g.admit(*_pair((1, 0), (0, 1), 0.500, 0.504))[0]          # tie, score +0.004 < eps
    assert not g.admit(*_pair((1, 1), (1, 0), 0.1, 0.9))[0]              # MacroSR down
    ok, why = g.admit(*_pair((1, 0), (0, 1), 0.5, 0.5))
    assert not ok and why["delta_norm_score"] == 0.0


def test_admit_score_mode_and_zero_eps_is_strict(tmp_path):
    g = gate(tmp_path, qval="score", qval_eps=0.005)
    assert g.admit(*_pair((1, 1), (0, 0), 0.500, 0.506))[0]              # MacroSR ignored
    assert not g.admit(*_pair((0, 0), (1, 1), 0.500, 0.504))[0]
    g0 = gate(tmp_path, qval="score", qval_eps=0.0)
    assert not g0.admit(*_pair((0, 0), (0, 0), 0.5, 0.5))[0]             # equal is not an improvement
    assert g0.admit(*_pair((0, 0), (0, 0), 0.5, 0.5001))[0]


# ------------------------------------------------------------------------------------ Eq. 2
def test_hval_no_regression(tmp_path):
    g = gate(tmp_path, qval="macrosr", hval_mode="no_regression")
    inc = report("inc", {"a": entry("D1", 0, 0.2, hv=["units"]), "b": entry("D2", 0, 0.2)})
    same = report("c1", {"a": entry("D1", 0, 0.2, hv=["units"]), "b": entry("D2", 1, 1.0)}, ref=inc)
    ok, why = g.admit(same, inc)
    assert ok and why["h_val"]["no_new_hard_violation"] and why["new_hard_violations"] == {}
    new = report("c2", {"a": entry("D1", 1, 1.0), "b": entry("D2", 0, 0.2, hv=["range"])}, ref=inc)
    assert new.h_val["no_new_hard_violation"] is False               # computed against the reference too
    ok, why = g.admit(new, inc)
    assert not ok and why["new_hard_violations"] == {"b": ["range"]} and why["improved"] is True


def test_hval_absolute(tmp_path):
    g = gate(tmp_path, qval="macrosr", hval_mode="absolute")
    inc = report("inc", {"a": entry("D1", 0, 0.2), "b": entry("D2", 0, 0.2)}, mode="absolute")
    bad = report("c", {"a": entry("D1", 1, 1.0), "b": entry("D2", 0, 0.2, hv=["units"])}, mode="absolute")
    ok, why = g.admit(bad, inc)
    assert not ok and why["h_val"]["all_hard_ok"] is False
    # an unsolved episode has no output, hence no hard constraint to violate: it only lowers MacroSR (z = 0)
    unsolved = report("c", {"a": entry("D1", 1, 1.0), "b": entry("D2", 0, 0.0, completed=False)}, mode="absolute")
    assert unsolved.h_val["all_hard_ok"] is True and unsolved.h_absolute["all_hard_ok"] is True
    assert g.admit(unsolved, inc)[0]
    # ... whereas a hard violation on a completed episode still blocks
    assert report("c", {"a": entry("D1", 1, 1.0, hv=["units"]), "b": entry("D2", 0, 0.0, completed=False)},
                  mode="absolute").h_val["all_hard_ok"] is False
    good = report("c", {"a": entry("D1", 1, 1.0), "b": entry("D2", 1, 1.0)}, mode="absolute")
    assert g.admit(good, inc)[0]


@pytest.mark.parametrize("bad, key", [
    (dict(integrity=["line 2: absolute filesystem path"]), "integrity"),
    (dict(schema_ok=False), "schema"),
    (dict(error="RuntimeError: boom"), "no_solver_error"),
])
def test_integrity_schema_and_solver_errors_block(tmp_path, bad, key):
    g = gate(tmp_path, qval="macrosr")
    inc = report("inc", {"a": entry("D1", 0, 0.2)})
    cand = report("c", {"a": entry("D1", 1, 1.0, **bad)}, ref=inc)
    ok, why = g.admit(cand, inc)
    assert not ok and why["h_val"][key] is False and why["improved"] is True


def test_budget_is_component_wise(tmp_path):
    g = gate(tmp_path, qval="macrosr", budget_tokens=100.0, budget_wall_s=10.0, budget_beta=-1.0)
    inc = report("inc", {"a": entry("D1", 0, 0.2)})
    assert g.admit(report("c", {"a": entry("D1", 1, 1.0, tokens=100.0, wall=10.0)}), inc)[0]
    ok, why = g.admit(report("c", {"a": entry("D1", 1, 1.0, tokens=101.0)}), inc)
    assert not ok and why["within_budget"] is False and why["cost_total_tokens"] == 101.0
    assert not g.admit(report("c", {"a": entry("D1", 1, 1.0, wall=10.5)}), inc)[0]


def test_unknown_modes_raise(tmp_path):
    with pytest.raises(ValueError):
        gate(tmp_path, qval="bogus")
    with pytest.raises(ValueError):
        gate(tmp_path, hval_mode="bogus")


# ------------------------------------------------------------------------------- evaluate
def _val_eps():
    return {"D1": [make_episode("v1", "D1", split="val"), make_episode("v2", "D1", split="val")],
            "D2": [make_episode("v3", "D2", split="val")]}


def test_evaluate_with_lazy_revalidation(tmp_path):
    solver = StubSolver()
    g = ValidationGate(EvolutionConfig(), SolverConfig(), solver, _val_eps(), tmp_path, max_workers=4,
                       retriever_factory=StubRetriever)
    p0 = AgentProgram()
    inc = g.evaluate(p0)
    assert sorted(c[1] for c in solver.calls) == ["v1", "v2", "v3"]
    assert inc.macro_sr == 0.0 and inc.norm_score == pytest.approx(0.2) and inc.n_resolved == 3
    cand, _ = p0.apply(Bundle([Skill("s", 1, "t", "b", ["D1"])]), new_version="cand")
    solver.calls.clear()
    rep = g.evaluate(cand, reuse_from=inc)
    assert sorted(c[1] for c in solver.calls) == ["v1", "v2"]          # D2's retrieval slice is unchanged
    assert rep.per_episode["v3"]["reused"] is True and rep.per_episode["v3"]["reused_from"] == "A0"
    assert rep.per_episode["v1"]["reused"] is False
    assert rep.macro_sr == pytest.approx(0.5) and rep.norm_score == pytest.approx((1.0 + 0.2) / 2)
    assert (rep.n_resolved, rep.n_reused) == (2, 1)
    assert rep.cost["total_tokens"] == 30.0 and rep.incurred["total_tokens"] == 20.0
    assert list(rep.per_episode) == ["v1", "v2", "v3"]                  # deterministic order
    ok, why = g.admit(rep, inc)
    assert ok, why
    # an unchanged program re-uses everything
    solver.calls.clear()
    again = g.evaluate(cand, reuse_from=rep)
    assert solver.calls == [] and again.macro_sr == rep.macro_sr and again.n_reused == 3


def test_evaluate_without_lazy_revalidation_resolves_all(tmp_path):
    solver = StubSolver()
    g = ValidationGate(EvolutionConfig(lazy_revalidation=False), SolverConfig(), solver, _val_eps(), tmp_path,
                       retriever_factory=StubRetriever)
    inc = g.evaluate(AgentProgram())
    solver.calls.clear()
    g.evaluate(AgentProgram(), reuse_from=inc)
    assert len(solver.calls) == 3


def test_evaluate_runs_concurrently_and_records_solver_errors(tmp_path):
    barrier = threading.Barrier(3, timeout=10)

    class Concurrent(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            barrier.wait()                       # deadlocks (timeout) unless the 3 solves run concurrently
            if episode.id == "v3":
                raise RuntimeError("executor crashed")
            return super().solve(episode, program, mode, run_dir)

    g = ValidationGate(EvolutionConfig(), SolverConfig(), Concurrent(), _val_eps(), tmp_path, max_workers=4,
                       retriever_factory=StubRetriever)
    rep = g.evaluate(AgentProgram())
    assert rep.per_episode["v3"]["error"].startswith("RuntimeError: executor crashed")
    assert rep.h_val["no_solver_error"] is False and rep.per_episode["v3"]["z"] == 0


def test_evaluate_scans_final_graph_for_integrity_violations(tmp_path):
    class Leaky(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            code = "def run(inputs, config):\n    return {'o': open('/etc/passwd').read()}\n"
            return val_result(episode, program, str(run_dir), z=1, norm=1.0, code_nodes={"n1": code})

    g = ValidationGate(EvolutionConfig(), SolverConfig(), Leaky(), {"D1": _val_eps()["D1"]}, tmp_path,
                       retriever_factory=StubRetriever)
    rep = g.evaluate(AgentProgram())
    assert rep.h_val["integrity"] is False
    assert "absolute filesystem path" in rep.per_episode["v1"]["integrity_violations"][0]


def test_evaluate_with_default_retriever(tmp_path):
    solver = StubSolver()
    g = ValidationGate(EvolutionConfig(), SolverConfig(), solver, _val_eps(), tmp_path)
    prog = AgentProgram(skills={"s": Skill("s", 1, "vector transformation", "Applicability: vector", ["D1"])})
    inc = g.evaluate(prog)
    solver.calls.clear()
    rep = g.evaluate(prog, reuse_from=inc)
    assert solver.calls == [] and rep.n_reused == 3 and rep.macro_sr == inc.macro_sr


def test_schema_and_hard_violations_from_eval(tmp_path):
    class Violating(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            return val_result(episode, program, str(run_dir), z=0, norm=0.3,
                              h={"output_schema": False, "finite_output": True})

    g = ValidationGate(EvolutionConfig(), SolverConfig(), Violating(), {"D1": _val_eps()["D1"]}, tmp_path,
                       retriever_factory=StubRetriever)
    rep = g.evaluate(AgentProgram())
    e = rep.per_episode["v1"]
    assert e["hard_violations"] == ["output_schema"] and e["schema_ok"] is False and e["norm_score"] == 0.3
    assert rep.h_val["schema"] is False


# ------------------------------------------------------------------------------ Eq. 13 R_src
def test_source_replay_check_pass_and_use(tmp_path):
    ep = make_episode("src1", "D1")
    cand, omega = AgentProgram().apply(Bundle([Skill("s", 1, "t", "b", ["D1"])]), new_version="c1")
    ok, res = source_replay_check(cand, omega, ep, StubSolver(), tmp_path / "a")
    assert ok and source_pass(res) and use_check(omega, res) == (True, [])
    ok, res = source_replay_check(cand, omega, ep, StubSolver(omit_use={"skill:"}), tmp_path / "b")
    assert not ok and source_pass(res) and use_check(omega, res) == (False, ["skill:s@v1"])
    ok, res = source_replay_check(cand, omega, ep, StubSolver(fail_source_replay=True), tmp_path / "c")
    # the solver claims the use, but the source run never produced passing evidence: nothing was used in e_src
    assert not ok and not source_pass(res) and use_check(omega, res) == (False, ["skill:s@v1"])


def _src_result(*, op_status: str | None = "ok", passed: bool = True, op_ref: str = "op:abc", ev_step: int = 5,
                steps: list[tuple[int, bool, list[str]]] | None = None, final_has_op: bool = True) -> SolveResult:
    """A source solve with one (possibly passing) evidence whose graph holds an operator node ``o``.

    ``steps``: (step, action_ok, uses) triples.
    """
    g = WorkflowGraph({"o": Node("o", "operator", ref=op_ref)})
    trace = Trace()
    if op_status is not None:
        trace.records["o"] = NodeRecord(node_id="o", fingerprint="f", status=op_status, kind="operator")
        trace.order.append("o")
    ev = Evidence(step=ev_step, graph_dict=g.to_dict(), y=[1.0], trace=trace, eval=EvalResult(), passed=passed,
                  graph_fp="f")
    recs = [StepRecord(k, {}, None, {"action_ok": ok}, list(uses), {}, "f", None, 0.0)
            for k, ok, uses in (steps or [(1, True, [])])]
    return SolveResult(episode_id="e", mode="source", program_version="p",
                       final_graph=g if final_has_op else WorkflowGraph(), y=None, trace=trace, eval=EvalResult(),
                       evidence=[ev], steps=recs, actions=[], retrieved={}, uses={"skill:claimed@v9", "op:claimed@v9"},
                       usage={}, run_dir="")


def test_operator_use_requires_an_ok_operator_node_in_the_passing_evidence():
    res = _src_result()
    assert used_refs(res) == {"op:abc"}                                # version suffixes are stripped
    assert use_check(["op:abc@v2"], res) == (True, [])
    assert use_check(["op:zzz@v1"], res) == (False, ["op:zzz@v1"])
    # the operator node failed / never ran in e_src
    assert used_refs(_src_result(op_status="error")) == set()
    assert used_refs(_src_result(op_status="pending")) == set()
    assert used_refs(_src_result(op_status=None)) == set()
    # the solver's own bookkeeping (result.uses) does not count: only the evidence does
    assert use_check(["op:claimed@v9", "skill:claimed@v9"], res)[0] is False


def test_use_is_judged_on_the_passing_evidence_not_on_a_failing_or_absent_one():
    failing = _src_result(passed=False, final_has_op=False)   # the operator only exists in a failing replay
    assert not source_pass(failing) and used_refs(failing) == set()
    other = _src_result(final_has_op=False)         # e_src has no operator node in the final graph, but the passing evidence does
    assert used_refs(other) == {"op:abc"}


def test_skill_use_counts_only_on_effective_steps_up_to_e_src():
    steps = [(1, True, ["skill:ok@v1"]),            # applied
             (2, False, ["skill:rejected@v1"]),     # the action was rejected / failed: no use
             (5, True, ["skill:at_src@v1"]),        # the step of e_src itself
             (6, True, ["skill:later@v1"])]         # after the passing replay: not part of e_src
    res = _src_result(ev_step=5, steps=steps)
    assert used_refs(res) == {"op:abc", "skill:ok", "skill:at_src"}
    assert use_check(["skill:ok@v1", "skill:rejected@v1", "skill:later@v1"], res) == (
        False, ["skill:rejected@v1", "skill:later@v1"])


def test_source_replay_check_retries_on_infrastructure_errors(tmp_path):
    ep = make_episode("src1", "D1")
    cand, omega = AgentProgram().apply(Bundle([Skill("s", 1, "t", "b", ["D1"])]), new_version="c1")

    class Outage(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            res = super().solve(episode, program, mode, run_dir)
            if len([c for c in self.calls if c[0] == "source"]) <= 2:
                res.infra_error = "policy call failed: gateway 503"
            return res

    slept: list[float] = []
    solver = Outage()
    ok, res = source_replay_check(cand, omega, ep, solver, tmp_path / "a", retries=2, backoff_s=3.0, sleep=slept.append)
    assert ok and res.infra_error is None and slept == [3.0, 6.0]
    assert len([c for c in solver.calls if c[0] == "source"]) == 3
    # bounded: with too few retries the outage is still visible on the returned result
    solver = Outage()
    ok, res = source_replay_check(cand, omega, ep, solver, tmp_path / "b", retries=1, backoff_s=0.0, sleep=slept.append)
    assert res.infra_error and len([c for c in solver.calls if c[0] == "source"]) == 2


def test_existing_integrity_or_schema_failures_do_not_block_in_no_regression_mode(tmp_path):
    bad = dict(hv=["output_shape"], schema_ok=False, integrity=["n1: line 2: absolute path"])
    inc = report("inc", {"a": entry("D1", 0, 0.2, **bad), "b": entry("D2", 0, 0.2)})
    # the candidate improves D2 and repeats (e.g. via a reused entry) the incumbent's failures on D1
    cand = report("c", {"a": entry("D1", 0, 0.2, **bad), "b": entry("D2", 1, 1.0)}, ref=inc)
    ok, why = gate(tmp_path, qval="macrosr").admit(cand, inc)
    assert ok and why["h_val"]["integrity"] and why["h_val"]["schema"]
    assert why["h_absolute"] == {"integrity": False, "schema": False, "all_hard_ok": False}
    assert cand.h_absolute["schema"] is False                      # the literal status stays in the report
    ok, why = gate(tmp_path, qval="macrosr", hval_mode="absolute").admit(
        report("c", cand.per_episode, mode="absolute"), inc)
    assert not ok and why["h_val"]["integrity"] is False and why["h_val"]["schema"] is False
    assert why["h_val"]["all_hard_ok"] is False
    # a new schema failure on another episode is blocking in both modes
    worse = report("c2", {"a": entry("D1", 1, 1.0),
                          "b": entry("D2", 1, 1.0, hv=["output_shape"], schema_ok=False)}, ref=inc)
    ok, why = gate(tmp_path, qval="macrosr").admit(worse, inc)
    assert not ok and why["new_schema_failures"] == ["b"] and why["new_hard_violations"] == {"b": ["output_shape"]}


def test_schema_constraint_names_of_adapters():
    from scienceclaw.evolution.validation import SCHEMA_CONSTRAINT_PATTERN as P

    for name in ("output_schema", "output_shape", "output_structure", "output_length", "declared_unit"):
        assert P.search(name), name
    for name in ("finite", "finite_values", "label_values", "effect_scale", "query_budget"):
        assert not P.search(name), name


# ------------------------------------------------------------------ budget (fidelity F3, cost M3, F5/m3)
def test_budget_scales_with_the_number_of_val_episodes(tmp_path):
    g = gate(tmp_path, budget_tokens_per_val_episode=100.0, budget_wall_s_per_val_episode=5.0)
    assert g.budget_for(1)["tokens"] == 100.0 and g.budget_for(6)["tokens"] == 600.0
    assert g.budget_for(6)["wall_s"] == 30.0
    # an explicit absolute budget wins over the per-episode one; 0 / 0 means unbounded
    assert gate(tmp_path, budget_tokens=250.0, budget_tokens_per_val_episode=100.0).budget_for(6)["tokens"] == 250.0
    off = gate(tmp_path, budget_tokens_per_val_episode=0.0, budget_wall_s_per_val_episode=0.0).budget_for(6)
    assert off["tokens"] == float("inf") and off["wall_s"] == float("inf")
    # the default is per-episode (the old flat 2M tokens made the gate vacuous at |D_val| in the hundreds)
    assert EvolutionConfig().budget_tokens == 0.0 and EvolutionConfig().budget_tokens_per_val_episode > 0


def test_relative_budget_rule_and_absolute_rule_must_both_hold(tmp_path):
    g = gate(tmp_path, qval="macrosr", budget_tokens_per_val_episode=100.0, budget_beta=0.5)
    inc = report("inc", {"a": entry("D1", 0, 0.2, tokens=40.0), "b": entry("D2", 0, 0.2, tokens=40.0)})   # C = 80
    def cand(tok: float, n: float = 1.0):
        return report("c", {"a": entry("D1", 1, 1.0, tokens=tok), "b": entry("D2", 0, 0.2, tokens=tok)}, ref=inc)
    ok, why = g.admit(cand(60.0), inc)                       # 120 <= 1.5 * 80 and <= 200
    assert ok and why["budget_tokens_rel"] == pytest.approx(120.0) and why["budget_tokens"] == 200.0
    ok, why = g.admit(cand(61.0), inc)                       # 122 > 120: inside the absolute budget, not the relative
    assert not ok and why["within_budget_abs"] and not why["within_budget_rel"]
    assert why["budget_violated"] == ["tokens_rel"]
    tight = gate(tmp_path, qval="macrosr", budget_tokens_per_val_episode=50.0, budget_beta=5.0)   # absolute 100
    ok, why = tight.admit(cand(60.0), inc)                   # 120 > 100 although far inside 6 x 80
    assert not ok and not why["within_budget_abs"] and why["within_budget_rel"]
    # beta < 0 switches the relative rule off
    assert gate(tmp_path, qval="macrosr", budget_tokens_per_val_episode=100.0, budget_beta=-1.0).admit(
        cand(99.0), inc)[0]


def test_budget_counts_logical_tokens_not_cache_hits(tmp_path):
    g = gate(tmp_path, qval="macrosr", budget_tokens_per_val_episode=100.0, budget_beta=-1.0)
    inc = report("inc", {"a": entry("D1", 0, 0.2)})
    cheap_but_cached = report("c", {"a": entry("D1", 1, 1.0, tokens=5.0, logical=150.0)}, ref=inc)   # 5 spent, 150 logical
    ok, why = g.admit(cheap_but_cached, inc)
    assert not ok and why["cost_tokens"] == 150.0 and why["cost_total_tokens"] == 5.0
    assert g.admit(report("c", {"a": entry("D1", 1, 1.0, tokens=5.0, logical=100.0)}, ref=inc), inc)[0]


# ------------------------------------------------------------------------ noise guard (cost M7)
def _four(z_inc: tuple[int, int, int, int], z_cand: tuple[int, int, int, int]):
    ids = ("a1", "a2", "b1", "b2")
    discs = ("D1", "D1", "D2", "D2")
    inc = report("inc", {i: entry(d, z, 1.0 if z else 0.2) for i, d, z in zip(ids, discs, z_inc)})
    cand = report("cand", {i: entry(d, z, 1.0 if z else 0.2) for i, d, z in zip(ids, discs, z_cand)}, ref=inc)
    return cand, inc


def test_single_episode_flip_is_not_enough_at_the_default_and_is_with_the_literal_switch(tmp_path):
    assert EvolutionConfig().min_improved_episodes == 2                       # default
    default = ValidationGate(EvolutionConfig(qval="macrosr"), SolverConfig(), None, [], tmp_path)
    cand, inc = _four((0, 0, 0, 0), (1, 0, 0, 0))
    ok, why = default.admit(cand, inc)
    assert not ok and why["q_gain"] is True and why["gain_supported"] is False and why["improved"] is False
    assert why["improved_episodes"] == ["a1"] and why["feasible"] is True
    literal = gate(tmp_path, qval="macrosr", min_improved_episodes=1)         # the paper's literal Eq. 3
    assert literal.admit(cand, inc)[0]


def test_two_independent_improvements_are_admitted(tmp_path):
    g = gate(tmp_path, qval="macrosr", min_improved_episodes=2)
    cand, inc = _four((0, 0, 0, 0), (1, 0, 1, 0))
    ok, why = g.admit(cand, inc)
    assert ok and why["improved_episodes"] == ["a1", "b1"] and why["regressed_episodes"] == []


def test_regression_cap_of_the_noise_guard(tmp_path):
    cand, inc = _four((1, 1, 0, 0), (0, 1, 1, 1))              # a1 regressed, b1 and b2 improved: MacroSR 0.5 -> 0.75
    ok, why = gate(tmp_path, qval="macrosr", min_improved_episodes=2, max_regressed_episodes=1).admit(cand, inc)
    assert ok and why["regressed_episodes"] == ["a1"]
    ok, why = gate(tmp_path, qval="macrosr", min_improved_episodes=2, max_regressed_episodes=0).admit(cand, inc)
    assert not ok and why["q_gain"] and not why["gain_supported"]
    ok, _ = gate(tmp_path, qval="macrosr", min_improved_episodes=2, max_regressed_episodes=-1).admit(cand, inc)
    assert ok                                                   # -1 = unlimited
    # min_improved_episodes = 1 is the literal rule and ignores the regression cap
    assert gate(tmp_path, qval="macrosr", min_improved_episodes=1, max_regressed_episodes=0).admit(cand, inc)[0]


def test_noise_guard_uses_the_score_when_z_ties(tmp_path):
    g = gate(tmp_path, qval="macrosr_then_score", qval_eps=0.05, min_improved_episodes=2)
    inc = report("inc", {"a1": entry("D1", 0, 0.2), "a2": entry("D1", 0, 0.2), "b1": entry("D2", 0, 0.2)})
    one = report("c", {"a1": entry("D1", 0, 0.6), "a2": entry("D1", 0, 0.2), "b1": entry("D2", 0, 0.2)}, ref=inc)
    two = report("c", {"a1": entry("D1", 0, 0.6), "a2": entry("D1", 0, 0.5), "b1": entry("D2", 0, 0.2)}, ref=inc)
    assert not g.admit(one, inc)[0] and g.admit(two, inc)[0]


# ------------------------------------------------------------------ gateway outages (cost M2)
def _outage_solver(outages: dict[str, int]):
    """Val solves of the given episode ids report an infrastructure error for their first N attempts."""
    class Outage(StubSolver):
        def solve(self, episode, program, mode, run_dir):
            res = super().solve(episode, program, mode, run_dir)
            if mode == "val" and outages.get(episode.id, 0) > 0:
                outages[episode.id] -= 1
                res.infra_error = "policy call failed: gateway 503"
                res.eval.z, res.eval.accepted = 0, False
            return res
    return Outage()


def test_val_outage_is_retried_then_becomes_an_error_entry_not_a_silent_z0(tmp_path):
    solver = _outage_solver({"v1": 99})
    g = ValidationGate(EvolutionConfig(infra_retries=2, infra_backoff_s=5.0), SolverConfig(), solver, _val_eps(),
                       tmp_path, retriever_factory=StubRetriever)
    slept: list[float] = []
    g._sleep = slept.append
    rep = g.evaluate(AgentProgram())
    e = rep.per_episode["v1"]
    assert e["error"].startswith("infrastructure error after 3 attempt(s)") and e["completed"] is False
    assert rep.h_val["no_solver_error"] is False                       # a candidate with such an entry is infeasible
    assert slept == [5.0, 10.0]                                        # doubling backoff, bounded retries
    assert len([c for c in solver.calls if c[1] == "v1"]) == 3
    assert rep.per_episode["v2"]["error"] is None and rep.per_episode["v3"]["error"] is None
    # an errored entry is never reused: the next evaluation of the same program re-solves it
    solver.calls.clear()
    solver_ok = _outage_solver({})
    g.solver = solver_ok
    again = g.evaluate(AgentProgram(), reuse_from=rep)
    assert [c[1] for c in solver_ok.calls] == ["v1"] and again.per_episode["v1"]["error"] is None
    assert again.per_episode["v1"]["reused"] is False and again.per_episode["v2"]["reused"] is True


def test_val_outage_that_clears_within_the_retries_is_transparent(tmp_path):
    solver = _outage_solver({"v1": 2})
    g = ValidationGate(EvolutionConfig(infra_retries=2, infra_backoff_s=0.0), SolverConfig(), solver, _val_eps(),
                       tmp_path, retriever_factory=StubRetriever)
    g._sleep = lambda s: None
    rep = g.evaluate(AgentProgram())
    assert rep.per_episode["v1"]["error"] is None and rep.h_val["no_solver_error"] is True
    assert len([c for c in solver.calls if c[1] == "v1"]) == 3
