"""FoR43 HIPE-OCRepair-2026 adapter tests (skipped when the data are not available)."""
from __future__ import annotations

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.tasks import for43_hipe as m
from scienceclaw.bench.tasks._adapter_utils_for43_45_47_48_50 import PoolExhausted

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]
SEED = 20260928


def test_norm_and_weights():
    assert m.hipe_norm("Straße,  ist\ngut¬\nes!") == "strasse ist gutes"
    assert m.official_weight("dta19-l1/de") == pytest.approx(1 / 3)
    assert m.official_weight("icdar2017/en") == 1.0
    score, per = m.weighted_cmer_micro([(9, 1, 0, 0), (10, 0, 0, 0), (8, 0, 2, 0)],
                                       ["dta19-l0/de", "dta19-l1/de", "overproof-combined/en"])
    # per test set: 0.1, 0.0, 0.2 ; weights 1/3, 1/3, 1
    assert per["dta19-l0/de"] == pytest.approx(0.1)
    assert score == pytest.approx((0.1 / 3 + 0.0 / 3 + 0.2) / (1 / 3 + 1 / 3 + 1))


@pytest.fixture(scope="module")
def adapter():
    a = m.Adapter()
    ok, why = a.available()
    if not ok:
        pytest.skip(why)
    return a


@pytest.fixture(scope="module")
def episodes(adapter):
    return {s: adapter.build_episodes(s, n, split_seed(SEED, m.CODE, s)) for s, n in SPLITS}


def _items(eps):
    return [i for e in eps for i in e.lineage["item_ids"]]


def test_splits_disjoint_and_deterministic(adapter, episodes):
    items = {s: _items(eps) for s, eps in episodes.items()}
    for s, lst in items.items():
        assert len(lst) == len(set(lst)), s
        assert all(e.n_items == 16 for e in episodes[s])
    for a, b in combinations(items, 2):
        assert not set(items[a]) & set(items[b]), (a, b)
    groups = {s: {g for e in eps for g in e.lineage["groups"]} for s, eps in episodes.items()}
    for a, b in combinations(("src", "val", "id"), 2):
        assert not groups[a] & groups[b], (a, b)       # document-disjoint IID splits
    again = m.Adapter().build_episodes("id", 4, split_seed(SEED, m.CODE, "id"))
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["id"]]
    assert [e.id for e in again] == [e.id for e in episodes["id"]]
    ood_sets = {ts for e in episodes["ood"] for ts in e.lineage["test_sets"]}
    assert ood_sets <= set(m.OOD_TEST_SETS) and len(ood_sets) == 4
    iid_sets = {ts for e in episodes["src"] for ts in e.lineage["test_sets"]}
    assert iid_sets == set(m.IID_TEST_SETS)
    assert all(e.lineage["ood_kind"] == "cross_corpus_within_benchmark" for e in episodes["ood"])
    with pytest.raises(PoolExhausted):
        adapter.build_episodes("val", 50, 1)


def test_visible_data_disjoint_and_no_label_leak(adapter, episodes):
    units = adapter._units.get()
    for s, eps in episodes.items():
        for e in eps[:2]:
            ev = [units[u] for u in e.lineage["item_ids"]]
            ev_groups = {u.group for u in ev}
            out = {t.name: t.fn({}, {}) for t in e.tools if t.name in ("load_train", "load_dev_inputs", "load_eval_inputs")}
            blob = json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, default=str)
            for u in ev:
                if u.gt != u.ocr:
                    assert u.gt not in blob, (s, u.uid)
            assert all("gt_text" not in r and "quality_report" not in r for r in out["load_eval_inputs"]["items"])
            assert all("gt_text" not in r for r in out["load_dev_inputs"]["dev_items"])
            tr_texts = {r["ocr_text"] for r in out["load_train"]["train"]}
            assert not tr_texts & {u.ocr for u in ev}
            dev_ids = set(e.lineage["dev_item_ids"])
            assert not dev_ids & set(e.lineage["item_ids"])
            assert not {units[d].group for d in dev_ids} & ev_groups
            items = out["load_eval_inputs"]["items"]
            assert [r["ocr_text"] for r in items] == [u.ocr for u in ev]


def test_evaluator_reference_oracle_and_constraints(adapter, episodes):
    units = adapter._units.get()
    e = episodes["id"][0]
    items = e.tool("load_eval_inputs").fn({}, {})["items"]
    ref = e.evaluate([r["ocr_text"] for r in items], None)
    assert ref.completed and ref.hard_ok()
    assert ref.primary == pytest.approx(ref.details["reference"])
    assert ref.details["norm_score"] == pytest.approx(1.0)
    assert not ref.accepted and ref.z == 0
    gold = [units[u].gt for u in e.lineage["item_ids"]]
    orc = e.evaluate(gold, None)
    assert orc.primary == pytest.approx(0.0) and orc.accepted and orc.z == 1
    assert orc.details["norm_score"] == pytest.approx(10.0)
    # malformed outputs fire the hard constraints
    bad_len = e.evaluate(gold[:-1], None)
    assert not bad_len.h["output_length"] and bad_len.primary is None and bad_len.z == 0
    bad_type = e.evaluate(gold[:-1] + [None], None)
    assert not bad_type.h["strings"] and bad_type.z == 0
    trunc = e.evaluate([g[:5] for g in gold], None)
    assert not trunc.h["length_plausible"] and trunc.z == 0
    assert e.evaluate(None, None).z == 0
    # pooled metric over episodes: reference payloads give the pooled reference
    pay = [e2.evaluate([r["ocr_text"] for r in e2.tool("load_eval_inputs").fn({}, {})["items"]], None)
           .details["pooled_payload"] for e2 in episodes["id"]]
    pooled = adapter.pooled_metric(pay)
    assert 0.0 < pooled < 0.2
    assert adapter.pooled_metric([orc.details["pooled_payload"]]) == pytest.approx(0.0)


def test_score_dev_and_text_mer(adapter, episodes):
    e = episodes["ood"][0]
    dev = e.tool("load_dev_inputs").fn({}, {})["dev_items"]
    r = e.tool("score_dev").fn({"dev_predictions": [d["ocr_text"] for d in dev]}, {})
    assert r["dev_weighted_cmer_micro"] == pytest.approx(r["dev_reference_weighted_cmer_micro"])
    with pytest.raises(ValueError):
        e.tool("score_dev").fn({"dev_predictions": ["x"]}, {})
    t = e.tool("text_mer").fn({"texts_a": ["abc def", "xyz"], "texts_b": ["abc def", "xyq"]}, {})
    assert t["mer"][0] == 0.0 and t["mer"][1] > 0


def _official_eval_module(adapter):
    path = adapter.root / m.DATASET_DIR / "scorer_upstream" / "hipe_ocrepair_scorer" / "ocrepair_eval.py"
    if not path.is_file():
        pytest.skip("official HIPE-OCRepair scorer source not available")
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                 # never write __pycache__ into the read-only data directory
    try:
        spec = importlib.util.spec_from_file_location("_hipe_official_eval", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)               # type: ignore[union-attr]
    finally:
        sys.dont_write_bytecode = old
    return mod


def test_matches_official_scorer(adapter, episodes):
    off = _official_eval_module(adapter)
    units = adapter._units.get()
    e = episodes["src"][0]
    ev = [units[u] for u in e.lineage["item_ids"]]
    rng = np.random.default_rng(0)
    preds = []
    for u in ev:                                    # perturb the OCR a little to get non-trivial counts
        chars = list(u.ocr)
        for j in rng.choice(len(chars), size=max(1, len(chars) // 50), replace=False):
            chars[j] = "e"
        preds.append("".join(chars))
    for u in ev:
        assert off.norm(u.gt) == m.hipe_norm(u.gt)
    res = e.evaluate(preds, None)
    recs = [{"document_metadata": {"primary_dataset_name": u.test_set, "document_id": u.uid},
             "ground_truth": {"transcription_unit": u.gt}, "ocr_hypothesis": {"transcription_unit": u.ocr},
             "ocr_postcorrection_output": {"transcription_unit": p}} for u, p in zip(ev, preds)]
    out = off.Evaluation(recs).score_over_datasets(normalize=True)
    per_off = {k: v["cmer_micro"][0] for k, v in out["fold_scores"].items()}
    assert set(per_off) == set(res.details["per_test_set"])
    for k, v in per_off.items():
        assert res.details["per_test_set"][k] == pytest.approx(v, abs=1e-12)
    w = {k: m.official_weight(k) for k in per_off}
    assert res.primary == pytest.approx(sum(w[k] * per_off[k] for k in w) / sum(w.values()), abs=1e-12)


def _run_pipeline(e, tmp_path, tool_ports, code_src, code_in, code_out):
    """tool nodes -> one code node -> submit through the real Executor (Eq. 7), then a clean replay (Eq. 8)."""
    from scienceclaw.core.actions import Action
    from scienceclaw.core.graph import WorkflowGraph
    from scienceclaw.core.program import AgentProgram
    from scienceclaw.runtime.executor import Executor
    from scienceclaw.runtime.replay import replay
    from scienceclaw.runtime.values import outputs_match
    prog = AgentProgram()
    ex = Executor(e, prog, None, tmp_path / "run")
    g, cp = WorkflowGraph(), ex.new_checkpoint()
    acts = [Action("add_node", {"node": {"id": f"t{j}", "kind": "tool", "ref": name}})
            for j, (name, _, _) in enumerate(tool_ports)]
    acts.append(Action("add_node", {"node": {"id": "c", "kind": "code", "code": code_src, "inputs": code_in,
                                             "outputs": code_out}}))
    acts += [Action("add_edge", {"edge": {"src": f"t{j}", "src_port": sp, "dst": "c", "dst_port": dp}})
             for j, (_, sp, dp) in enumerate(tool_ports)]
    acts += [Action("add_node", {"node": {"id": "s", "kind": "submit"}}),
             Action("add_edge", {"edge": {"src": "c", "src_port": next(iter(code_out)), "dst": "s", "dst_port": "y"}})]
    fb = None
    y = None
    for k, a in enumerate(acts):
        g, cp, fb, y = ex.apply(g, cp, a, k)
        assert fb.action_ok, fb.action_error
    assert fb.submit_ready, fb.render()
    assert all(v[0] for v in fb.visible_constraints.values()), fb.visible_constraints
    y2, _ = replay(g, e, prog, None, tmp_path / "replay")
    assert outputs_match(y, y2, e.tolerance)
    return y


def test_executor_pipeline_copy_baseline(adapter, episodes, tmp_path):
    e = episodes["val"][0]
    code = "def run(inputs, config):\n    return {'y': [it['ocr_text'] for it in inputs['items']]}\n"
    y = _run_pipeline(e, tmp_path, [("load_eval_inputs", "items", "items")], code, {"items": {"type": "list"}},
                      {"y": {"type": "list", "shape": ["n"]}})
    res = e.evaluate(y, None)
    assert res.z == 0 and res.primary == pytest.approx(res.details["reference"])


def test_task_card_exists():
    card = Path(__file__).resolve().parents[1] / "docs" / "tasks" / "FoR43.md"
    assert card.is_file()


def test_objective_documents_the_domain_library(adapter, episodes):
    ep = episodes["val"][0]
    assert "scilib.ocrfix" in ep.objective and "plan(" in ep.objective and "merge(" in ep.objective
    assert "reference" not in ep.objective.lower()     # F5: the reference method is not named


def test_ocrfix_plan_llm_merge_pipeline_through_the_executor(adapter, episodes, tmp_path):
    """The documented recipe: tools -> code (plan) -> llm node (prompt "{prompt}", dict items) -> code (merge) -> submit."""
    from scienceclaw.core.actions import Action
    from scienceclaw.core.graph import WorkflowGraph
    from scienceclaw.core.program import AgentProgram
    from scienceclaw.llm import fake
    from scienceclaw.runtime.executor import Executor
    e = episodes["val"][0]

    def respond(role, messages):                                    # echoes the chunk (a reply that changes nothing)
        text = messages[-1]["content"]
        return text.split("OCR text to correct:\n", 1)[1].rsplit("\n\nCorrected text:", 1)[0]

    llm = fake.FakeLLM(respond)
    plan_code = ("from scilib.ocrfix import plan\n\ndef run(inputs, config):\n"
                 "    p = plan(inputs['dev_items'] + inputs['items'], inputs['train'])\n"
                 "    return {'plan': p, 'llm_items': p['items']}\n")
    merge_code = ("from scilib.ocrfix import merge\n\ndef run(inputs, config):\n"
                  "    r = merge(inputs['plan'], inputs['outputs'])\n    return {'y': r['y'][16:], 'report': r['report']}\n")
    ex = Executor(e, AgentProgram(), llm, tmp_path / "run")
    g, cp = WorkflowGraph(), ex.new_checkpoint()
    any_ = {"type": "any"}
    acts = [Action("add_node", {"node": {"id": t, "kind": "tool", "ref": t}})
            for t in ("load_train", "load_dev_inputs", "load_eval_inputs")]
    acts += [Action("add_node", {"node": {"id": "plan", "kind": "code", "code": plan_code,
                                          "inputs": {"train": any_, "dev_items": any_, "items": any_},
                                          "outputs": {"plan": any_, "llm_items": {"type": "list"}}}}),
             Action("add_node", {"node": {"id": "llm", "kind": "llm", "prompt": "{prompt}", "config": {"parse": "text"},
                                          "inputs": {"items": {"type": "list"}}, "outputs": {"outputs": {"type": "list"}}}}),
             Action("add_node", {"node": {"id": "merge", "kind": "code", "code": merge_code,
                                          "inputs": {"plan": any_, "outputs": any_},
                                          "outputs": {"y": {"type": "list", "shape": ["n"]}, "report": any_}}}),
             Action("add_node", {"node": {"id": "s", "kind": "submit"}})]
    edges = [("load_train", "train", "plan", "train"), ("load_dev_inputs", "dev_items", "plan", "dev_items"),
             ("load_eval_inputs", "items", "plan", "items"), ("plan", "llm_items", "llm", "items"),
             ("plan", "plan", "merge", "plan"), ("llm", "outputs", "merge", "outputs"), ("merge", "y", "s", "y")]
    acts += [Action("add_edge", {"edge": {"src": a, "src_port": b, "dst": c, "dst_port": d}}) for a, b, c, d in edges]
    fb, y = None, None
    for k, a in enumerate(acts):
        g, cp, fb, y = ex.apply(g, cp, a, k)
        assert fb.action_ok, fb.action_error
    assert fb.submit_ready, fb.render()
    assert all(v[0] for v in fb.visible_constraints.values()), fb.visible_constraints
    assert len(llm.calls) >= 10 and "dev_items" not in llm.calls[0]["messages"][-1]["content"]
    res = e.evaluate(y, None)
    assert res.primary is not None and res.primary <= res.details["reference"] + 1e-12    # echo replies: only the hyphen rule acts
