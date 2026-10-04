"""FoR52 Psych-201 adapter: splits, leakage, evaluator (reference / oracle), constraints."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scienceclaw.bench.tasks import for52_psych201 as m

ADAPTER = m.Adapter()
OK, WHY = ADAPTER.available()
pytestmark = pytest.mark.skipif(not OK, reason=f"FoR52 data unavailable: {WHY}")

COUNTS = {"src": 7, "val": 2, "id": 4, "ood": 4}


@pytest.fixture(scope="module")
def episodes():
    return {s: ADAPTER.build_episodes(s, n, seed=500 + i) for i, (s, n) in enumerate(COUNTS.items())}


def _ids(eps):
    return [i for e in eps for i in e.lineage["item_ids"]]


def test_splits_disjoint_and_studies(episodes):
    data = ADAPTER._data.get()
    sets = {s: set(_ids(eps)) for s, eps in episodes.items()}
    for s, eps in episodes.items():
        assert len(eps) == COUNTS[s] and all(e.n_items == 16 for e in eps)
        assert len(_ids(eps)) == len(sets[s])                      # item-disjoint episodes
    for a in sets:
        for b in sets:
            if a < b:
                assert not (sets[a] & sets[b])
    visible = set(data.pools["train"]) | set(data.pools["dev"])
    assert not visible & set().union(*sets.values())
    ood_studies = {data.sessions[u].study for u in sets["ood"]}
    iid_studies = {data.sessions[u].study for s in ("src", "val", "id") for u in sets[s]}
    assert not ood_studies & iid_studies and ood_studies <= set(data.ood_studies)
    assert all(e.lineage["ood_kind"] == "proxy_within_dataset" for e in episodes["ood"])
    groups = [data.sessions[u].group_id for u in set().union(*sets.values()) | visible]
    assert len(groups) == len(set(groups))                          # one unit per participant group
    assert not any(data.sessions[u].flagged for u in visible)


def test_determinism():
    a = ADAPTER.build_episodes("val", 2, seed=3)
    b = ADAPTER.build_episodes("val", 2, seed=3)
    assert [e.lineage["item_ids"] for e in a] == [e.lineage["item_ids"] for e in b]
    assert a[0].tool("load_train").fn({}, {}) == b[0].tool("load_train").fn({}, {})
    c = ADAPTER.build_episodes("val", 1, seed=3)
    assert c[0].lineage["item_ids"] == a[0].lineage["item_ids"]


def test_no_label_leakage(episodes):
    data = ADAPTER._data.get()
    for s in ("id", "ood"):
        e = episodes[s][0]
        items = e.tool("load_eval_inputs").fn({}, {})["items"]
        assert len(items) == e.n_items
        for uid, it in zip(e.lineage["item_ids"], items):
            ses = data.sessions[uid]
            assert it["history"] == ses.text[:ses.target_start]
            assert not it["history"].endswith("<<") and ses.text[ses.target_start:].startswith(f"<<{ses.target}>>")
            assert set(it) == {"study", "history", "options", "options_text"}
        train = e.tool("load_train").fn({}, {})["sessions"]
        blob = json.dumps(train)
        for uid in e.lineage["item_ids"]:
            assert data.sessions[uid].history[-2000:] not in blob     # eval sessions never in training data
        dev = e.tool("load_dev_inputs").fn({}, {})["dev_items"]
        assert all("target" not in d for d in dev)


def test_reference_and_oracle(episodes):
    data = ADAPTER._data.get()
    for s in ("src", "ood"):
        e = episodes[s][0]
        items = e.tool("load_eval_inputs").fn({}, {})["items"]
        ref = [m.reference_prediction(x["history"], x["options"]) for x in items]
        rv = e.evaluate(ref, None)
        assert rv.hard_ok() and rv.primary == pytest.approx(rv.details["reference"]) and not rv.accepted
        oracle = [data.sessions[u].target for u in e.lineage["item_ids"]]
        ov = e.evaluate(oracle, None)
        assert ov.primary == 1.0 and ov.hard_ok()
        assert ov.accepted == (1.0 >= rv.details["reference"] + ADAPTER.accept_margin)
        assert ov.details["norm_score"] == pytest.approx(min(10.0, 1.0 / rv.details["reference"]))
        assert ov.details["pooled_payload"]["y_pred"] == oracle


def test_constraints_and_invalid(episodes):
    e = episodes["val"][1]
    items = e.tool("load_eval_inputs").fn({}, {})["items"]
    y = [x["options"][0] for x in items]
    h, _ = e.check_constraints(y, None)
    assert all(h.values())
    bad = list(y)
    bad[3] = "not-a-key"
    h, msgs = e.check_constraints(bad, None)
    assert h["output_format"] and not h["allowed_labels"] and "index 3" in msgs["allowed_labels"]
    h, _ = e.check_constraints(y[:5], None)
    assert not h["output_format"]
    ev = e.evaluate(y[:5], None)
    assert ev.z == 0 and ev.primary == 0.0 and ev.details["pooled_payload"]["y_pred"] is None


def test_score_dev_and_pooled(episodes):
    e = episodes["id"][0]
    dev = e.tool("load_dev_inputs").fn({}, {})["dev_items"]
    out = e.tool("score_dev").fn({"dev_pred": [m.reference_prediction(d["history"], d["options"]) for d in dev]}, {})
    assert out["dev_accuracy"] == pytest.approx(out["dev_reference_accuracy"])
    with pytest.raises(ValueError):
        e.tool("score_dev").fn({"dev_pred": ["A"]}, {})
    pay = [{"y_true": ["A", "B"], "y_pred": ["A", "A"]}, {"y_true": ["C", "C"], "y_pred": None}]
    assert ADAPTER.pooled_metric(pay) == pytest.approx(0.25)


def test_objective_documents_the_domain_library(episodes):
    e = episodes["src"][0]
    assert "scilib.psych" in e.objective and "fit_predict" in e.objective
    # the language-model entry points and the llm-node shape are documented, and the prompt lists fit one llm node
    assert "build_prompts" in e.objective and "parse_answers" in e.objective and "llm_answers" in e.objective
    from scilib import psych
    items = e.tool("load_eval_inputs").fn({}, {})["items"]
    dev = e.tool("load_dev_inputs").fn({}, {})["dev_items"]
    prompts = psych.build_prompts(list(items) + list(dev))
    assert len(prompts) == len(items) + len(dev) <= e.budget.max_llm_items and max(map(len, prompts)) < 32000
    # the dev tools say which items dev_pred refers to
    assert "load_dev_inputs" in e.tool("score_dev").description and "load_dev_inputs" in e.objective


def test_fixed_visible_route_is_deterministic_and_rejects_targets():
    """The fixed qlearn route consumes only the two visible tool payloads."""
    e = ADAPTER.build_episodes("val", 1, seed=17)[0]
    train = e.tool("load_train").fn({}, {})["sessions"]
    items = e.tool("load_eval_inputs").fn({}, {})["items"]
    tool = e.tool("psych_fixed_predict")
    a = tool.fn({"train_sessions": train, "eval_items": items}, {})
    b = tool.fn({"train_sessions": train, "eval_items": items}, {})
    assert a == b and len(a["y"]) == e.n_items
    assert all(y in item["options"] for y, item in zip(a["y"], items))
    assert a["provenance"]["method"] == m.PSYCH_TOOL_METHOD == "qlearn"
    assert a["provenance"]["seed"] == m.PSYCH_TOOL_SEED == 0
    with pytest.raises(ValueError, match="forbidden"):
        bad = dict(items[0], target="hidden")
        tool.fn({"train_sessions": train, "eval_items": [bad] + items[1:]}, {})


def test_objective_requires_fixed_output_to_be_submitted_directly():
    e = ADAPTER.build_episodes("id", 1, seed=17)[0]
    assert "psych_fixed_predict" in e.objective
    assert "Wire the tool's y directly to submit.y and finish immediately" in e.objective


def test_adaptive_visible_route_is_label_free_and_submit_ready():
    e = ADAPTER.build_episodes("id", 1, seed=17)[0]
    train = e.tool("load_train").fn({}, {})["sessions"]
    items = e.tool("load_eval_inputs").fn({}, {})["items"]
    out = e.tool("psych_adaptive_predict").fn({"train_sessions": train, "eval_items": items}, {})
    assert len(out["y"]) == e.n_items
    assert all(pred in item["options"] for pred, item in zip(out["y"], items))
    info = out["provenance"]
    assert info["methods"] == ["qlearn", "personal", "wsls"]
    assert info["selected_method"] in info["methods"]
    assert set(info["cv_accuracy"]) == set(info["methods"])
    assert info["data"] == "visible load_train sessions only"


def test_domain_adaptive_route_switches_only_on_visible_study_ids():
    """The mixed-domain route uses qlearn for seen studies and pooled gbdt otherwise."""
    e = ADAPTER.build_episodes("ood", 1, seed=17)[0]
    train = e.tool("load_train").fn({}, {})["sessions"]
    items = e.tool("load_eval_inputs").fn({}, {})["items"]
    tool = e.tool("psych_domain_adaptive_predict")
    out = tool.fn({"train_sessions": train, "eval_items": items}, {})
    assert len(out["y"]) == e.n_items
    assert all(pred in item["options"] for pred, item in zip(out["y"], items))
    info = out["provenance"]
    assert info["seen_method"] == "qlearn" and info["unseen_method"] == "gbdt"
    assert info["n_seen_items"] + info["n_unseen_items"] == e.n_items
    assert info["data"] == "visible load_train sessions and label-free load_eval_inputs items"


def test_dev_slice_size_and_llm_budget(episodes):
    """The dev slice is the whole IID dev pool (48); the LLM-item budget stays at 3 x (16 eval + 16)."""
    for e in [x for eps in episodes.values() for x in eps]:
        dev = e.tool("load_dev_inputs").fn({}, {})["dev_items"]
        assert len(dev) == m.N_DEV_ITEMS == 48
        assert len({d["history"] for d in dev}) == 48
        assert e.budget.max_llm_items == 96


def test_option_parser_and_reference():
    txt = ("There are two slot machines, labeled Q and Y.\nYou press <<Q>> and get 3 points.\n"
           "You press <<Y>> and get 1 points.\nYou press <<Q>> and get 2 points.\nYou press <<Y>>")
    marks = m.parse_markers(txt, "wilson2014humans")
    assert [mk["options"] for mk in marks] == [["Q", "Y"]] * 4 and all(mk["valid"] for mk in marks)
    hist = txt[: marks[3]["start"]]
    assert m.reference_prediction(hist, ["Q", "Y"]) == "Q"
    assert m.reference_prediction("You press <<Q>> <<Y>>", ["Q", "Y"]) == "Y"     # tie -> most recent
    dyn = "Intro.\nYou can choose between option K and option W. You press <<W>>."
    assert m.parse_markers(dyn, "anystudy")[0]["options"] == ["K", "W"]


def test_visible_study_reference_and_pooled_diagnostics():
    """The stronger reference uses only visible study responses and remains diagnostic-only."""
    train = [
        SimpleNamespace(study="s", responses=[{"response": "B"}, {"response": "B"}, {"response": "A"}]),
        SimpleNamespace(study="s", responses=[{"response": "B"}]),
        SimpleNamespace(study="other", responses=[{"response": "A"}, {"response": "A"}]),
    ]
    assert m.visible_study_prediction(train, "s", ["A", "B"], "") == "B"
    # No usable row falls back to the original history baseline, without looking at a target.
    assert m.visible_study_prediction(train, "missing", ["A", "B"], "You press <<B>>") == "B"

    payloads = [
        {"y_true": ["A", "B"], "y_pred": ["A", "A"], "y_ref": ["A", "B"],
         "y_study_ref": ["A", "A"]},
        {"y_true": ["C", "C"], "y_pred": None, "y_ref": ["C", "D"],
         "y_study_ref": ["C", "C"]},
    ]
    d = ADAPTER.pooled_diagnostics(payloads)
    assert d["pooled_accuracy"] == pytest.approx(0.25)
    assert d["pooled_reference_accuracy"] == pytest.approx(0.75)
    assert d["pooled_study_reference_accuracy"] == pytest.approx(0.75)
    assert d["n_items"] == 4 and d["diagnostic_only"]
    # The official pooled primary is unchanged and still scores an invalid episode as all-wrong.
    assert ADAPTER.pooled_metric(payloads) == pytest.approx(0.25)


def test_full_split_reference_uses_only_visible_training_pool():
    """Full-pool references are trusted-side diagnostics, separate from 16-item episode scoring."""
    for split in ("id", "ood"):
        out = ADAPTER.full_split_reference(split)
        assert out["split"] == split and out["diagnostic_only"]
        assert out["n_items"] == len(ADAPTER._data.get().pools[split])
        assert out["n_studies"] >= 1 and out["visible_train_sessions"] == len(ADAPTER._data.get().pools["train"])
        assert out["visible_train_studies"] == len(ADAPTER._data.get().iid_studies)
        assert 0.0 <= out["participant_history_accuracy"] <= 1.0
        assert 0.0 <= out["study_conditioned_accuracy"] <= 1.0
