"""FoR49 SMT-LIB QF_NIA adapter: sanitizer, splits, leakage, z3 tool, evaluator (reference / oracle), constraints."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scienceclaw.bench.tasks import for49_smt as m


# ------------------------------------------------------------------------------------------------ offline units
def test_sanitizer_removes_metadata_and_comments():
    src = ('(set-info :smt-lib-version 2.6)\n(set-logic QF_NIA)\n(set-info :source |\nGenerated; by (x)\n|)\n'
           '; a comment (with parens\n(set-info :status unsat)\n(declare-fun x () Int)\n'
           '(assert (> (* x x) 3)) ; trailing\n(check-sat)\n(get-model)\n(exit)\n')
    q = m.sanitize_query(src)
    assert ":status" not in q and "set-info" not in q and "comment" not in q and "get-model" not in q
    assert q.splitlines() == ["(set-logic QF_NIA)", "(declare-fun x () Int)", "(assert (> (* x x) 3))",
                              "(check-sat)", "(exit)"]
    with pytest.raises(ValueError):
        m.split_commands("(assert (> x 1)")


def test_lineage_groups():
    g = m.lineage_group("non-incremental/QF_NIA/20170427-VeryMax/CInteger/Stroeder_15__GCD2.c__p23106_terminationG_0.smt2")
    assert g == "20170427-VeryMax/CInteger/Stroeder_15__GCD2.c"
    assert m.lineage_group("non-incremental/QF_NIA/calypto/problem-000001.cvc.2.smt2") == \
        m.lineage_group("non-incremental/QF_NIA/calypto/problem-000001.cvc.1.smt2")
    assert m.lineage_group("non-incremental/QF_NIA/20220315-MathProblems/STC_0772.smt2") == "20220315-MathProblems/STC"


Z3 = m.find_z3()


@pytest.mark.skipif(Z3 is None, reason="z3 binary not available")
def test_z3_runner_statuses_and_memo():
    r = m.Z3Runner(Z3)
    sat = "(set-logic QF_NIA)(declare-fun x () Int)(assert (= (* x x) 49))(check-sat)"
    unsat = "(set-logic QF_NIA)(declare-fun x () Int)(assert (= (* x x) 2))(check-sat)"
    bad = "(set-logic QF_NIA)(assert (> y 1))(check-sat)"
    a = r.check(sat, 5, 512, None, ())
    b = r.check(unsat, 5, 512, None, ())
    c = r.check(bad, 5, 512, None, ())
    assert (a["status"], b["status"], c["status"]) == ("sat", "unsat", "unknown")
    assert c["reason"].startswith("error")
    assert r.check(sat, 5, 512, None, ())["memoized"] is True


@pytest.mark.skipif(Z3 is None, reason="z3 binary not available")
def test_z3_runner_reports_rejected_parameter_not_memout():
    """A parameter z3 rejects is reported with z3's own message (its 'legal parameters' listing mentions
    max_memory and used to be misread as a memory-limit stop)."""
    r = m.Z3Runner(Z3)
    sat = "(set-logic QF_NIA)(declare-fun x () Int)(assert (= (* x x) 49))(check-sat)"
    for bad in ("nlsat.nl=true", "smt.arith.ineq_fast=true"):
        res = r.check(sat, 5, 512, None, (bad,))
        assert res["status"] == "unknown"
        assert res["reason"].startswith("error: unknown parameter"), res["reason"]
        assert "memout" not in res["reason"]
    assert r.check(sat, 5, 512, None, ("smt.arith.solver=2",))["status"] == "sat"
    assert r.check(sat, 5, 1, None, ())["reason"] == "memout"


# ------------------------------------------------------------------------------------------------ real data
ADAPTER = m.Adapter()
OK, WHY = ADAPTER.available()
needs_data = pytest.mark.skipif(not OK, reason=f"FoR49 data unavailable: {WHY}")
COUNTS = {"src": 7, "val": 2, "id": 4, "ood": 4}


@pytest.fixture(scope="module")
def episodes():
    return {s: ADAPTER.build_episodes(s, n, seed=900 + i) for i, (s, n) in enumerate(COUNTS.items())}


@needs_data
def test_splits_disjoint_composition_determinism(episodes):
    data = ADAPTER._data.get()
    sets = {s: [i for e in eps for i in e.lineage["item_ids"]] for s, eps in episodes.items()}
    for s, ids in sets.items():
        assert len(ids) == len(set(ids)) == 16 * COUNTS[s]
    for a in sets:
        for b in sets:
            if a < b:
                assert not set(sets[a]) & set(sets[b])
    groups = {s: {data.units[u].group for e in eps for u in e.lineage["lineage_units"]} for s, eps in episodes.items()}
    for a in ("src", "val", "id"):
        for b in ("src", "val", "id", "ood"):
            if a < b:
                assert not groups[a] & groups[b]                    # lineage-group disjoint
    visible = {i for e in episodes["id"] for i in e.lineage["train_item_ids"] + e.lineage["dev_item_ids"]}
    assert not visible & {i for ids in sets.values() for i in ids}
    for e in episodes["ood"]:
        assert e.lineage["ood_kind"] == "proxy_within_dataset"
        assert not set(e.lineage["families"]) & set(m.IID_FAMILIES)
    for e in episodes["id"]:
        assert e.lineage["tiers"].count("hard") == 12 and 6 <= e.lineage["n_sat"] <= 10
    again = ADAPTER.build_episodes("id", 4, seed=902)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["id"]]


@needs_data
def test_tools_do_not_leak_status(episodes):
    e = episodes["ood"][0]
    q = e.tool("load_eval_inputs").fn({}, {})["queries"]
    assert len(q) == 16 and all("(check-sat)" in x for x in q)
    blob = json.dumps(q) + json.dumps(e.tool("load_dev_inputs").fn({}, {}))
    assert ":status" not in blob and "set-info" not in blob
    for iid in e.lineage["item_ids"]:
        assert iid not in blob and iid.rsplit("/", 1)[-1] not in blob
    tr = e.tool("load_train").fn({}, {})
    assert len(tr["queries"]) == len(tr["status"]) == ADAPTER.n_train
    assert set(tr["status"]) <= {"sat", "unsat"}


@needs_data
def test_reference_oracle_constraints_and_score_dev(episodes):
    data = ADAPTER._data.get()
    e = episodes["val"][0]
    truth = [data.units[u].status for u in e.lineage["lineage_units"]]
    ov = e.evaluate(truth, None)
    assert ov.primary == 1.0 and ov.accepted and ov.hard_ok() and ov.z == 1
    ref = ov.details["reference"]
    assert ref == pytest.approx(truth.count("sat") / 16)            # training majority is 'sat'
    rv = e.evaluate(["sat"] * 16, None)
    assert rv.primary == pytest.approx(ref) and not rv.accepted
    uv = e.evaluate(["unknown"] * 16, None)
    assert uv.primary == 0.0 and uv.hard_ok()
    h, _ = e.check_constraints(["maybe"] * 16, None)
    assert not h["allowed_labels"]
    h, _ = e.check_constraints(["sat"] * 15, None)
    assert not h["output_format"]
    assert e.evaluate(["sat"] * 15, None).primary is None
    dev = e.tool("load_dev_inputs").fn({}, {})["dev_queries"]
    sd = e.tool("score_dev").fn({"dev_pred": ["sat"] * len(dev)}, {})
    assert sd["dev_accuracy"] == pytest.approx(sd["dev_reference_accuracy"])
    pay = [ov.details["pooled_payload"], {**rv.details["pooled_payload"], "y_pred": None}]
    assert ADAPTER.pooled_metric(pay) == pytest.approx((16 + 16 * ref) / 32)


@needs_data
def test_z3_tool_on_easy_items(episodes):
    data = ADAPTER._data.get()
    e = episodes["id"][0]
    units = [data.units[u] for u in e.lineage["lineage_units"]]
    easy = [i for i, u in enumerate(units) if u.tier == "easy"][:4]
    q = e.tool("load_eval_inputs").fn({}, {})["queries"]
    out = e.tool("z3_check").fn({"queries": [q[i] for i in easy]}, {"query_timeout_s": 10, "workers": 4})
    assert out["status"] == [units[i].status for i in easy]
    with pytest.raises(ValueError):
        e.tool("z3_check").fn({"queries": q[:1]}, {"params": ["bad param"]})


@needs_data
def test_objective_documents_the_domain_library(episodes):
    ob = episodes["id"][0].objective
    assert "scilib.logic" in ob and "decide(" in ob and "fallback" in ob
    assert "call z3_check" in ob and "wire z3_check.status directly to submit.y and finish" in ob
    assert "latest status output directly to submit.y" in ob
    assert "query_timeout_s=120" in ob
    # policy-visible text stays factual: no reference/acceptance hints
    for w in ("reference", "acceptance", "baseline", "margin"):
        assert w not in ob.lower().split("library `scilib.logic`")[1]


def test_status_baseline_diagnostics_split_majority_and_pure_z3_by_tier():
    d = m.status_baseline_diagnostics(
        ["sat", "unsat", "sat", "unsat"],
        ["sat", "sat", "unknown", "unsat"],
        ["sat", "sat", "sat", "sat"],
        ["sat", "unsat", "unknown", "sat"],
        ["hard", "hard", "easy", "easy"],
    )
    assert d["diagnostic_only"] is True
    assert d["tool_shortcut_not_capability_evidence"] is True
    assert d["majority_guess_accuracy"] == pytest.approx(0.5)
    assert d["submitted_accuracy"] == pytest.approx(0.5)
    assert d["pure_z3_accuracy"] == pytest.approx(0.5)
    assert d["pure_z3_decided"] == 3 and d["pure_z3_unknown"] == 1
    assert d["by_tier"]["hard"]["pure_z3_accuracy"] == pytest.approx(1.0)
    assert d["by_tier"]["easy"]["majority_guess_accuracy"] == pytest.approx(0.5)


def test_pooled_diagnostics_keeps_solver_shortcut_separate_from_primary():
    payloads = [
        {"y_true": ["sat", "unsat", "sat", "unsat"],
         "y_pred": ["sat", "sat", "unknown", "unsat"],
         "y_ref": ["sat"] * 4,
         "pure_z3": ["sat", "unsat", "unknown", "sat"],
         "tiers": ["hard", "hard", "easy", "easy"]},
        {"y_true": ["unsat", "unsat"], "y_pred": ["unsat", "sat"], "y_ref": ["sat", "sat"],
         "pure_z3": ["unknown", "unsat"], "tiers": ["hard", "easy"]},
    ]
    d = ADAPTER.pooled_diagnostics(payloads)
    assert d["diagnostic_only"] is True and d["tool_shortcut_not_capability_evidence"] is True
    assert d["n_episodes"] == 2 and d["n_items"] == 6
    assert d["pooled_accuracy"] == pytest.approx(3 / 6)
    assert d["pooled_reference_accuracy"] == pytest.approx(2 / 6)
    assert d["pure_z3_accuracy"] == pytest.approx(3 / 6)
    assert d["pure_z3_decided"] == 4 and d["pure_z3_unknown"] == 2
    assert d["by_tier"]["hard"]["n"] == 3
    assert d["by_tier"]["easy"]["pure_z3_decided"] == 2


def test_full_split_reference_is_deterministic_and_diagnostic_only():
    """The full-pool majority summary is separate from episode scoring and does not invoke Z3."""
    units = {
        "a": m._Unit("a", "a.smt2", "g1", "f1", "sat", "hard", Path("a")),
        "b": m._Unit("b", "b.smt2", "g2", "f1", "sat", "easy", Path("b")),
        "c": m._Unit("c", "c.smt2", "g3", "f2", "sat", "hard", Path("c")),
        "d": m._Unit("d", "d.smt2", "g4", "f2", "unsat", "easy", Path("d")),
    }
    strata = [(tier, status) for tier in ("hard", "easy") for status in ("sat", "unsat")]
    pools = {split: {key: [] for key in strata} for split in ("train", "dev", "src", "val", "id", "ood")}
    pools["id"][("hard", "sat")] = ["a", "c"]
    pools["id"][("easy", "sat")] = ["b"]
    pools["id"][("easy", "unsat")] = ["d"]
    data = m._SmtData(units, pools, {}, 4, 0, {})
    adapter = m.Adapter()
    adapter._data = m.Lazy(lambda: data)
    out = adapter.full_split_reference("id")
    assert out["diagnostic_only"] and out["tool_shortcut_not_capability_evidence"]
    assert out["n_items"] == 4 and out["status_counts"] == {"sat": 3, "unsat": 1}
    assert out["reference_status"] == "sat" and out["reference_accuracy"] == pytest.approx(0.75)
    assert out["tier_counts"] == {"easy": 2, "hard": 2}
    assert out["tier_reference_accuracy"] == {"easy": pytest.approx(0.5), "hard": pytest.approx(1.0)}
    assert out["pure_z3_known"] == 2 and out["pure_z3_unknown"] == 2
    assert out["pure_z3_decided_lower_bound"] == 2 and out["pure_z3_decided_upper_bound"] == 4
    assert out["pure_z3_accuracy_lower_bound"] == pytest.approx(0.5)
    assert out["pure_z3_accuracy_upper_bound"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        adapter.full_split_reference("rep")
