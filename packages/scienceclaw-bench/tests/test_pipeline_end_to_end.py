"""End-to-end pipeline check (synthetic adapter + scripted fake policy).

One run (2 rounds) is shared by all tests of this module. The assertions check that every stage of the
method leaves a receipt: a source episode whose evidence holds a replay-verified failure followed by a
replay-verified success (Eq. 8-9), an evolution instance, a linked Skill+Operator bundle (Eq. 10-12), a passing
boundary replay, source-task replay + Use check (Eq. 13), validation-gate decisions (Eq. 2-3, both an accept
and a reject), program snapshots A_0..A_R, a held-out evaluation and a report. No network access is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_support import run_pipeline


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("pipeline")
    info = run_pipeline(out, rounds=2, workers=1)
    info["dir"] = Path(info["run_dir"])
    return info


def test_source_episode_has_replay_verified_failure_then_success(pipeline):
    run = pipeline["dir"]
    solves = sorted((run / "source").glob("*/solve"))
    assert len(solves) == 2                                   # one source episode per round
    ev = _jsonl(solves[0] / "evidence.jsonl")
    passed = [e["passed"] for e in ev]
    assert True in passed
    k_plus = passed.index(True)
    assert k_plus >= 1 and passed[k_plus - 1] is False        # e- precedes e+
    assert all(e["reproducible"] for e in ev)                 # every evidence item is a clean replay
    assert ev[k_plus - 1]["step"] < ev[k_plus]["step"]
    # receipts never carry hidden labels
    assert "pooled_payload" not in (solves[0] / "evidence.jsonl").read_text()


def test_evolution_instance_and_linked_bundle(pipeline):
    run = pipeline["dir"]
    stream = _jsonl(run / "stream.jsonl")
    inst = stream[0]["instances"][0]["summary"]
    assert inst["has_e_minus"] and inst["k_minus"] < inst["k_plus"]
    assert inst["delta_steps"] and inst["delta_steps"][-1] == inst["k_plus"]
    assert "remove_node" in inst["delta_types"] and "add_node" in inst["delta_types"]

    cands = _jsonl(run / "candidates.jsonl")
    first = cands[0]
    meta = first["bundle"]["meta"]
    assert meta["variant"] == "full" and meta["linked"] is True
    assert len(meta["skill_ids"]) >= 1 and len(meta["operator_ids"]) >= 1
    # boundary replay of every operator in the bundle passed
    assert first["breplay"] and all(b["ok"] for b in first["breplay"])
    bundle = json.loads((run / "candidates" / first["cand_id"] / "bundle.json").read_text())
    assert bundle


def test_source_replay_use_check_and_gate_decisions(pipeline):
    run = pipeline["dir"]
    cands = _jsonl(run / "candidates.jsonl")
    for c in cands:
        assert c["R_src"] is True and c["Pass"] is True and c["Use"] is True
        assert (run / "candidates" / c["cand_id"] / "source_replay" / "evidence.jsonl").exists()
        for key in ("feasible", "improved", "h_val", "cand_macro_sr", "inc_macro_sr"):
            assert key in c["reasons"]
    decisions = [c["accepted"] for c in cands]
    assert True in decisions and False in decisions        # one promotion, one non-improving candidate rejected
    acc = next(c for c in cands if c["accepted"])
    assert acc["reasons"]["cand_macro_sr"] > acc["reasons"]["inc_macro_sr"]
    assert (run / "val_reports" / "A0.json").exists()


def test_program_snapshots_contain_linked_components(pipeline):
    run = pipeline["dir"]
    assert pipeline["snapshots"] == ["A_0", "A_1", "A_2"]
    a0 = json.loads((run / "programs" / "A_0" / "program.json").read_text())
    assert not a0["skills"] and not a0["operators"]           # empty initial library
    a1 = json.loads((run / "programs" / "A_1" / "program.json").read_text())
    assert len(a1["skills"]) >= 1 and len(a1["operators"]) >= 1
    md = (run / "programs" / "A_1" / "program.md").read_text()
    assert "Linked operators:" in md


def test_held_out_evaluation_and_report(pipeline):
    run = pipeline["dir"]
    res = _jsonl(run / "eval" / "results.jsonl")
    assert len(res) == pipeline["n_results"] == 3 * (2 + 2) + 1 + 2
    assert {r["split"] for r in res} >= {"id", "ood", "rep"}
    assert (run / "report" / "report.md").exists() and (run / "report" / "report.json").exists()
    summ = {(r["snapshot"], r["split"]): r["macro_sr"] for r in pipeline["summary"]}
    assert summ[("A_0", "id")] + summ[("A_0", "ood")] < summ[("A_2", "id")] + summ[("A_2", "ood")]
    total = next(p for p in pipeline["promotions"] if p["round"] == "total")
    assert total["promoted"] >= 1
