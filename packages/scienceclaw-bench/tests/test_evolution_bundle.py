"""Bundles B_i and the ablation variants."""
from __future__ import annotations

import json
import sys
import types

import pytest

from scienceclaw.core.program import AgentProgram, Bundle
from scienceclaw.core.skills import Skill
from scienceclaw.evolution import build_bundle, bundles_for_variant, extract_instances
from scienceclaw.evolution.bundle import LINK_HEADER, link_skills_to_operators
from scienceclaw.evolution.variants import VARIANTS, wants_components
from scienceclaw.llm.fake import FakeLLM

import test_evolution_support as S
from test_evolution_support import build_solve_result, install_fake_isolated, make_episode, patch_responder


def _inst(tmp_path):
    ep = make_episode()
    [inst] = extract_instances(build_solve_result(ep, AgentProgram(), tmp_path / "solve"))
    return ep, inst


def _kinds(llm: FakeLLM) -> list[str]:
    return [c["tag"] for c in llm.calls]


def test_full_bundle_links_skill_and_operator(tmp_path, monkeypatch):
    install_fake_isolated(monkeypatch)
    ep, inst = _inst(tmp_path)
    llm = FakeLLM(patch_responder())
    b, log = build_bundle(inst, AgentProgram(), llm, ep, tmp_path / "b", "full", repeats=2, round_idx=3)
    assert len(b.skills) == 1 and len(b.operators) == 1
    s, o = b.skills[0], b.operators[0]
    assert LINK_HEADER in s.body and f"- {o.ref} - {o.name}:" in s.body
    assert o.provenance["linked_skills"] == [s.ref]
    assert s.created == "r3:eep1"
    assert b.meta["linked"] is True and b.meta["skill_ids"] == [s.id] and b.meta["operator_ids"] == [o.id]
    assert log["operators"][0]["admitted_to_bundle"] is True and len(log["operators"][0]["breplay"]["runs"]) == 2
    assert sorted(_kinds(llm)) == ["evo.operator_doc", "evo.skill_patch"]
    assert bundles_for_variant(b, "full") == [b]
    prog, omega = AgentProgram().apply(b)
    assert sorted(omega) == sorted([s.version_id, o.version_id])
    json.dumps(log, default=str)   # the build log is JSON-serializable


def test_skill_only_and_operator_only(tmp_path, monkeypatch):
    install_fake_isolated(monkeypatch)
    ep, inst = _inst(tmp_path)
    llm = FakeLLM(patch_responder())
    b, _ = build_bundle(inst, AgentProgram(), llm, ep, tmp_path / "s", "skill_only")
    assert len(b.skills) == 1 and b.operators == [] and _kinds(llm) == ["evo.skill_patch"]
    assert LINK_HEADER not in b.skills[0].body
    [only] = bundles_for_variant(b, "skill_only")
    assert only.operators == [] and only.meta["part"] == "skills"
    llm = FakeLLM(patch_responder())
    b, _ = build_bundle(inst, AgentProgram(), llm, ep, tmp_path / "o", "operator_only")
    assert b.skills == [] and len(b.operators) == 1 and _kinds(llm) == ["evo.operator_doc"]
    assert "linked_skills" not in b.operators[0].provenance


def test_unlinked_gives_two_independent_bundles(tmp_path, monkeypatch):
    install_fake_isolated(monkeypatch)
    ep, inst = _inst(tmp_path)
    b, _ = build_bundle(inst, AgentProgram(), FakeLLM(patch_responder()), ep, tmp_path / "u", "unlinked")
    assert LINK_HEADER not in b.skills[0].body and "linked_skills" not in b.operators[0].provenance
    parts = bundles_for_variant(b, "unlinked")
    assert [(len(p.skills), len(p.operators)) for p in parts] == [(1, 0), (0, 1)]
    assert [p.meta["part"] for p in parts] == ["skills", "operators"]
    assert all(p.meta["linked"] is False for p in parts)


def test_workflow_only_stores_whole_graph_as_one_skill(tmp_path):
    ep, inst = _inst(tmp_path)
    llm = FakeLLM(patch_responder())
    b, _ = build_bundle(inst, AgentProgram(), llm, ep, tmp_path / "w", "workflow_only")
    assert llm.calls == [] and b.operators == [] and len(b.skills) == 1
    s = b.skills[0]
    assert s.id.startswith("workflow-toy-regression-") and "workflow_exemplar" in s.tags
    graph_json = s.body.split("```json\n", 1)[1].split("\n```", 1)[0]
    g = json.loads(graph_json)
    assert {n["id"] for n in g["nodes"]} == {"load", "prep", "post", "sub"}
    assert "origin" not in g["nodes"][0]
    assert bundles_for_variant(b, "workflow_only") == [b]


def test_frozen_and_empty_bundles(tmp_path):
    ep, inst = _inst(tmp_path)
    llm = FakeLLM(patch_responder())
    b, _ = build_bundle(inst, AgentProgram(), llm, ep, tmp_path / "f", "frozen")
    assert b.is_empty() and llm.calls == [] and bundles_for_variant(b, "frozen") == []
    assert bundles_for_variant(Bundle(), "full") == [] and bundles_for_variant(Bundle(), "unlinked") == []
    with pytest.raises(ValueError):
        build_bundle(inst, AgentProgram(), llm, ep, tmp_path, "bogus")
    assert wants_components("full") == (True, True) and set(VARIANTS) >= {"frozen", "full", "unlinked"}


def test_operator_failing_boundary_replay_is_excluded(tmp_path, monkeypatch):
    def wrong(op, inputs, program, llm, run_dir, episode=None):
        outs, rec = S.fake_run_operator_isolated(op, inputs, program, llm, run_dir, episode)
        return {k: [x + 0.5 for x in v] for k, v in outs.items()}, rec

    mod = types.ModuleType("scienceclaw.runtime.replay")
    mod.run_operator_isolated = wrong
    monkeypatch.setitem(sys.modules, "scienceclaw.runtime.replay", mod)
    ep, inst = _inst(tmp_path)
    b, log = build_bundle(inst, AgentProgram(), FakeLLM(patch_responder()), ep, tmp_path / "x", "full")
    assert b.operators == [] and len(b.skills) == 1
    assert LINK_HEADER not in b.skills[0].body
    assert log["operators"][0]["admitted_to_bundle"] is False
    assert log["operators"][0]["breplay"]["ok"] is False


def test_link_section_is_merged_not_duplicated(tmp_path, monkeypatch):
    install_fake_isolated(monkeypatch)
    ep, inst = _inst(tmp_path)
    b, _ = build_bundle(inst, AgentProgram(), FakeLLM(patch_responder()), ep, tmp_path / "m", "full")
    s, o = b.skills[0], b.operators[0]
    old = Skill("s", 1, "t", s.body.replace(o.ref, "op:older-1"), [])
    link_skills_to_operators([old], [o])
    assert old.body.count(LINK_HEADER) == 1
    assert "- op:older-1" in old.body and f"- {o.ref}" in old.body
