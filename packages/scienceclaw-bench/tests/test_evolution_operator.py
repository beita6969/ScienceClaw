"""Eq. 12: operator candidates (boundary ports, contract, provenance) and boundary replay."""
from __future__ import annotations

import pytest

from scienceclaw.core.graph import Node, WorkflowGraph
from scienceclaw.core.operators import check_contract_entries
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.schema import PortSchema
from scienceclaw.evolution import boundary_replay, extract_instances, make_operator_candidate, split_edits
from scienceclaw.evolution.operator_abstraction import load_recorded_value, operator_candidate_with_log
from scienceclaw.llm.fake import FakeLLM
from scienceclaw.runtime.values import save_value

import test_evolution_support as S
from test_evolution_support import build_solve_result, install_fake_isolated, make_episode, patch_responder


@pytest.fixture(params=["fake", "real"])
def isolated(request, monkeypatch):
    """Boundary replay through the in-process fake, and through the real runtime when it exists."""
    if request.param == "fake":
        install_fake_isolated(monkeypatch)
    elif not S.real_isolated_available():
        pytest.skip("runtime run_operator_isolated is not implemented yet")
    return request.param


def _setup(tmp_path, llm=None):
    ep = make_episode()
    prog = AgentProgram()
    [inst] = extract_instances(build_solve_result(ep, prog, tmp_path / "solve"))
    _, [comp] = split_edits(inst, prog)
    llm = llm or FakeLLM(patch_responder())
    op = make_operator_candidate(inst, comp, prog, llm, ep)
    return ep, prog, inst, comp, op, llm


def test_boundary_ports_contract_and_provenance(tmp_path):
    ep, prog, inst, comp, op, llm = _setup(tmp_path)
    assert comp == {"prep", "post"}
    assert op.id.startswith("scale-then-offset-") and op.name == "scale_then_offset"
    assert op.description.startswith("Scales a numeric list")
    # one input per incoming boundary edge (load.table -> prep.x), one output per leaving port (post.z -> sub.y)
    assert op.inputs == {"prep__x": PortSchema("list")} and op.input_map == {"prep__x": [("prep", "x")]}
    assert op.outputs == {"post__z": PortSchema("list")} and op.output_map == {"post__z": ("post", "z")}
    assert set(op.body.nodes) == {"prep", "post"} and len(op.body.edges) == 1
    assert op.body.nodes["prep"].config == {"scale": 2.0}
    pre = {(c["port"], c["check"]) for c in op.contract.pre}
    assert pre == {("prep__x", "type"), ("prep__x", "shape"), ("prep__x", "finite"), ("prep__x", "nonempty")}
    post = {c["check"]: c.get("value") for c in op.contract.post}
    assert "finite" in post and post["len_eq_port"] == "prep__x"
    assert "range" not in post                               # observed ranges are not a post-condition by default
    assert op.contract.applicability == {"disciplines": ["FoR99"], "task_types": ["toy_regression"],
                                         "input_types": ["list"]}
    prov = op.provenance
    assert (prov["episode"], prov["k_plus"], prov["k_minus"], prov["program_version"]) == ("ep1", 9, 4, "A0")
    assert prov["node_ids"] == ["post", "prep"]
    assert "FoR99" in op.tags and "toy_regression" in op.tags
    # kappa holds on the recorded boundary values of tau+
    rec = inst.e_plus.trace.records
    values = {"prep__x": load_recorded_value(rec["prep"].input_refs["x"]),
              "post__z": load_recorded_value(rec["post"].output_refs["z"])}
    schemas = {**op.inputs, **op.outputs}
    assert check_contract_entries(op.contract.pre + op.contract.post, values, schemas) == []
    assert len(llm.calls) == 1 and llm.calls[0]["tag"] == "evo.operator_doc"


def test_same_body_gives_same_id_and_fallback_name(tmp_path):
    _, _, _, _, op1, _ = _setup(tmp_path / "a")
    _, _, _, _, op2, _ = _setup(tmp_path / "b", FakeLLM(patch_responder(op_reply="no json here")))
    assert op2.name == "toy_regression_code_block"
    assert op1.id.split("-")[-1] == op2.id.split("-")[-1]         # hash of the body only
    _, _, _, _, op3, _ = _setup(tmp_path / "c", FakeLLM([RuntimeError("down")]))
    assert op3 is not None and op3.name == "toy_regression_code_block"


def test_boundary_replay_passes(tmp_path, isolated):
    ep, prog, inst, comp, op, llm = _setup(tmp_path)
    S.ISOLATED_CALLS.clear()
    ok, det = boundary_replay(op, inst, comp, prog, llm, ep, tmp_path / "br", repeats=3)
    assert ok, det
    assert len(det["runs"]) == 3 and all(r["ok"] for r in det["runs"])
    assert len({r["dir"] for r in det["runs"]}) == 3              # fresh directory per repetition
    if isolated == "fake":
        assert [c["inputs"] for c in S.ISOLATED_CALLS] == [["prep__x"]] * 3


def test_boundary_replay_detects_output_mismatch(tmp_path, isolated):
    ep, prog, inst, comp, op, llm = _setup(tmp_path)
    save_value([0.0, 0.0, 0.0, 0.0], inst.e_plus.trace.records["post"].output_refs["z"])
    ok, det = boundary_replay(op, inst, comp, prog, llm, ep, tmp_path / "br")
    assert not ok and det["runs"][0]["mismatched"] == ["post__z: differs beyond tolerance"]


def test_boundary_replay_detects_tampered_body(tmp_path, isolated):
    ep, prog, inst, comp, op, llm = _setup(tmp_path)
    op.body.nodes["post"].code = S.POST_CODE.replace("v + 1.0", "v + 2.0")
    ok, det = boundary_replay(op, inst, comp, prog, llm, ep, tmp_path / "br")
    assert not ok


def test_boundary_replay_within_tolerance(tmp_path, isolated):
    ep, prog, inst, comp, op, llm = _setup(tmp_path)
    save_value([3.0, 5.0, 7.0, 9.0 + 1e-9], inst.e_plus.trace.records["post"].output_refs["z"])
    ok, _ = boundary_replay(op, inst, comp, prog, llm, ep, tmp_path / "br")
    assert ok


def test_boundary_replay_requires_recorded_values(tmp_path):
    ep, prog, inst, comp, op, llm = _setup(tmp_path)
    inst.e_plus.trace.records["prep"].input_refs.clear()
    ok, det = boundary_replay(op, inst, comp, prog, llm, ep, tmp_path / "br")
    assert not ok and "inputs=['prep__x']" in det["error"] and det["runs"] == []


def test_component_without_outputs_or_with_failed_nodes_is_skipped(tmp_path):
    ep, prog, inst, comp, op, llm = _setup(tmp_path)
    g = WorkflowGraph.from_dict(inst.e_plus.graph_dict)
    g.nodes["dead"] = Node("dead", "code", code="def run(i, c):\n    return {'o': 1}\n", outputs={"o": PortSchema()})
    inst.e_plus.graph_dict = g.to_dict()
    inst.e_plus.trace.records["dead"] = S.NodeRecord("dead", "", "ok")
    op_dead, log = operator_candidate_with_log(inst, {"dead"}, prog, llm, ep)
    assert op_dead is None and "no boundary outputs" in log["reason"]
    inst.e_plus.trace.records["post"].status = "error"
    op_err, log = operator_candidate_with_log(inst, comp, prog, llm, ep)
    assert op_err is None and "post" in log["reason"]
    op_none, log = operator_candidate_with_log(inst, {"ghost"}, prog, llm, ep)
    assert op_none is None and "not in G+" in log["reason"]


def test_range_postcondition_opt_in(tmp_path, monkeypatch):
    import scienceclaw.evolution.operator_abstraction as oa
    monkeypatch.setattr(oa, "RANGE_IN_CONTRACT", True)
    ep, prog, inst, comp, op, llm = _setup(tmp_path)
    post = {c["check"]: c.get("value") for c in op.contract.post}
    lo, hi = post["range"]                                   # recorded z = [3, 5, 7, 9], 10 % slack of the span
    assert lo == pytest.approx(3 - 0.6) and hi == pytest.approx(9 + 0.6)
