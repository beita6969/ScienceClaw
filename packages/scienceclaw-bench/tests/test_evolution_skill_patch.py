"""Eq. 11: Patch_Theta0 Skill candidates."""
from __future__ import annotations

import json

from scienceclaw.core.program import AgentProgram, Bundle
from scienceclaw.core.skills import Skill
from scienceclaw.evolution import extract_instances, make_skill_candidates, split_edits
from scienceclaw.evolution.skill_patch import SKILL_SECTIONS, skill_candidates_with_log
from scienceclaw.llm.fake import FakeLLM

from test_evolution_support import (HIDDEN_MSG, HIDDEN_PRIMARY_FAIL, SKILL_REPLY, build_solve_result, make_episode,
                                    patch_responder, repair_steps)

EP_ID = "episode-XYZ-42"


def _setup(tmp_path, program: AgentProgram, uses: list[str] | None = None, specs=None):
    ep = make_episode(EP_ID)
    res = build_solve_result(ep, program, tmp_path, specs or repair_steps(uses_at_repair=uses))
    [inst] = extract_instances(res)
    control, _ = split_edits(inst, program)
    return ep, inst, control


def test_new_skill_from_reproduced_repair(tmp_path):
    prog = AgentProgram()
    ep, inst, control = _setup(tmp_path, prog)
    llm = FakeLLM(patch_responder())
    skills = make_skill_candidates(inst, control, prog, llm, ep)
    assert len(skills) == 1
    s = skills[0]
    assert s.id.startswith("rescale-then-offset-vector-outputs-") and s.version == 1
    for sec in SKILL_SECTIONS:
        assert f"{sec}:" in s.body
    assert "1. Set the scale" in s.body and "- Submitting the unscaled" in s.body
    assert s.tags[:2] == ["vector", "rescale"] and "FoR99" in s.tags and "toy_regression" in s.tags
    assert s.provenance["episode"] == EP_ID and (s.provenance["k_minus"], s.provenance["k_plus"]) == (4, 9)
    # exactly one patch call; its prompt holds the control edits and e-/e+ summaries but no hidden information
    assert len(llm.calls) == 1 and llm.calls[0]["role"] == "patch"
    prompt = "\n".join(m["content"] for m in llm.calls[0]["messages"])
    assert "Control edits of the repair" in prompt
    assert "[step 5] modify_node prep fields=['config']" in prompt
    assert "[step 7] remove_edge prep.y -> sub.y" in prompt
    assert '"hidden_evaluation": "FAIL"' in prompt and '"hidden_evaluation": "PASS"' in prompt
    assert HIDDEN_MSG not in prompt and str(HIDDEN_PRIMARY_FAIL) not in prompt
    assert "secret_check" not in prompt and EP_ID not in prompt
    # deterministic: same inputs -> same id
    again = make_skill_candidates(inst, control, prog, FakeLLM(patch_responder()), ep)
    assert again[0].id == s.id and again[0].body == s.body


def test_attributed_skill_is_revised_in_place(tmp_path):
    old = Skill("rescale", 2, "Rescale outputs", "Applicability: x\nProcedure:\n1. y\nPitfalls:\n- z", ["vector"])
    prog = AgentProgram(skills={"rescale": old})
    ep, inst, control = _setup(tmp_path, prog, uses=["skill:rescale"])
    assert inst.uses_in_delta == {"skill:rescale"}
    reply = {"skills": [{**SKILL_REPLY["skills"][0], "revises": "rescale"}]}
    llm = FakeLLM(patch_responder(skill_reply=reply))
    [s] = make_skill_candidates(inst, control, prog, llm, ep)
    assert s.id == "rescale" and s.version == 2
    assert s.tags[0] == "vector" and "FoR99" in s.tags          # union of the old and the new tags
    assert s.provenance["revised_from"] == "skill:rescale@v2"
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "Rescale outputs" in prompt and "id: rescale" in prompt
    new_prog, omega = prog.apply(Bundle([s], []))
    assert omega == ["skill:rescale@v3"] and new_prog.skills["rescale"].title == s.title


def test_revision_target_enforced_when_model_omits_it(tmp_path):
    old = Skill("rescale", 1, "Rescale outputs", "body", ["vector"])
    prog = AgentProgram(skills={"rescale": old})
    ep, inst, control = _setup(tmp_path, prog, uses=["skill:rescale@v1"])
    two = {"skills": [SKILL_REPLY["skills"][0], {**SKILL_REPLY["skills"][0], "title": "Extra"}]}
    skills = make_skill_candidates(inst, control, prog, FakeLLM(patch_responder(skill_reply=two)), ep)
    assert [s.id for s in skills] == ["rescale"]      # only the attributed skill may change


def test_no_skill_without_e_minus_or_control_edits(tmp_path):
    prog = AgentProgram()
    specs = repair_steps()
    specs[4] = {k: v for k, v in specs[4].items() if k != "replay"}
    ep, inst, control = _setup(tmp_path / "a", prog, specs=specs)
    assert inst.e_minus is None and control
    llm = FakeLLM(patch_responder())
    skills, log = skill_candidates_with_log(inst, control, prog, llm, ep)
    assert skills == [] and "no replay-verified failure" in log["reason"] and llm.calls == []
    ep, inst, control = _setup(tmp_path / "b", prog)
    skills, log = skill_candidates_with_log(inst, [], prog, llm, ep)
    assert skills == [] and log["reason"] == "no control edits in delta" and llm.calls == []


def test_patch_model_failures_yield_no_candidate(tmp_path):
    prog = AgentProgram()
    ep, inst, control = _setup(tmp_path, prog)
    skills, log = skill_candidates_with_log(inst, control, prog, FakeLLM(patch_responder(skill_reply="not json")), ep)
    assert skills == [] and "unparseable" in log["reason"]
    boom = FakeLLM([RuntimeError("gateway down")])
    skills, log = skill_candidates_with_log(inst, control, prog, boom, ep)
    assert skills == [] and "gateway down" in log["llm"]["error"]
    empty = FakeLLM(patch_responder(skill_reply={"skills": [{"title": "", "procedure": []}]}))
    assert make_skill_candidates(inst, control, prog, empty, ep) == []


def test_instance_specific_tokens_are_scrubbed(tmp_path):
    prog = AgentProgram()
    ep, inst, control = _setup(tmp_path, prog)
    leaky = {"skills": [{**SKILL_REPLY["skills"][0],
                         "pitfalls": [f"Episode {EP_ID} reads /Users/someone/data/items.csv directly.",
                                      f"Item {EP_ID}-item-0001 is special."]}]}
    skills, log = skill_candidates_with_log(inst, control, prog, FakeLLM(patch_responder(skill_reply=leaky)), ep)
    body = skills[0].body
    assert EP_ID not in body and "/Users/someone" not in body
    assert "<instance>" in body and "<path>" in body
    assert log["leaks"]


def test_body_string_output_is_accepted(tmp_path):
    prog = AgentProgram()
    ep, inst, control = _setup(tmp_path, prog)
    reply = {"skills": [{"title": "T", "body": "Applicability: a\nProcedure:\n1. b\nPitfalls:\n- c", "tags": "x"}]}
    [s] = make_skill_candidates(inst, control, prog, FakeLLM(patch_responder(skill_reply=json.dumps(reply))), ep)
    assert s.body.startswith("Applicability: a") and s.tags == ["FoR99", "toy_regression"]


def test_revision_keeps_linked_operator_section(tmp_path):
    body = "Applicability: x\nProcedure:\n1. y\nPitfalls:\n- z\nLinked operators:\n- op:abc-1 - abc: op:abc-1() -> ()"
    prog = AgentProgram(skills={"rescale": Skill("rescale", 1, "Rescale outputs", body, ["vector"])})
    ep, inst, control = _setup(tmp_path, prog, uses=["skill:rescale"])
    [s] = make_skill_candidates(inst, control, prog, FakeLLM(patch_responder()), ep)
    assert s.id == "rescale" and s.body.count("Linked operators:") == 1 and "- op:abc-1" in s.body
    assert "Set the scale" in s.body
