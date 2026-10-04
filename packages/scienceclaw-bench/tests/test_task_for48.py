"""FoR48 ContractNLI adapter tests (skipped when the data are not available)."""
from __future__ import annotations

import ast
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.tasks import for48_contractnli as m

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]
SEED = 20260928


def test_metric_building_blocks():
    y = np.array([0, 1, 0, 1])
    assert m.average_precision(y, np.array([0.1, 0.9, 0.2, 0.8])) == pytest.approx(1.0)
    assert m.average_precision(np.array([1, 0]), np.array([0.1, 0.9])) == pytest.approx(0.5)
    assert m.precision_at_recall(y, np.array([0.1, 0.9, 0.2, 0.8]), 0.8) == pytest.approx(1.0)
    assert np.isnan(m.precision_at_recall(np.zeros(3, int), np.ones(3), 0.8))
    s = m.overlap_scores("Receiving Party shall not reverse engineer", ["reverse engineering is prohibited", "hello"])
    assert s[0] > s[1] == 0.0


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
    for a, b in combinations(items, 2):
        assert not set(items[a]) & set(items[b]), (a, b)
    docs = {s: {d for e in eps for d in e.lineage["documents"]} for s, eps in episodes.items()}
    for a, b in combinations(docs, 2):
        assert not docs[a] & docs[b], (a, b)                  # contract-disjoint splits
    data = adapter._data.get()
    assert all(data.docs[d].doc_type in m.OOD_TYPES for d in docs["ood"])
    assert all(data.docs[d].doc_type in m.IID_TYPES for s in ("src", "val", "id") for d in docs[s])
    assert all(data.docs[d].official_split in ("dev", "test") for s in docs for d in docs[s])
    again = m.Adapter().build_episodes("src", 7, split_seed(SEED, m.CODE, "src"))
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["src"]]


def test_visible_data_disjoint_and_no_label_leak(adapter, episodes):
    data = adapter._data.get()
    for s, eps in episodes.items():
        for e in eps[:2]:
            out = {t.name: t.fn({}, {}) for t in e.tools
                   if t.name in ("load_train", "load_dev_inputs", "load_eval_inputs", "load_hypotheses")}
            ev_docs = set(e.lineage["documents"])
            tr_docs = {d["doc_id"] for d in out["load_train"]["train_documents"]}
            assert not tr_docs & ev_docs
            assert all(data.docs[d].official_split == "train" for d in tr_docs)
            assert all(a["doc_id"] in tr_docs for a in out["load_train"]["train_annotations"])
            dev_docs = {d["doc_id"] for d in out["load_dev_inputs"]["dev_documents"]}
            assert not dev_docs & (tr_docs | ev_docs)
            item_keys = {"index", "doc_id", "hypothesis_key", "hypothesis", "n_spans"}
            doc_keys = {"doc_id", "text", "spans", "span_texts"}
            assert all(set(r) == item_keys for r in out["load_eval_inputs"]["items"])
            assert all(set(r) == item_keys for r in out["load_dev_inputs"]["dev_items"])
            assert all(set(d) == doc_keys for d in out["load_eval_inputs"]["documents"])
            assert all(set(d) == doc_keys for d in out["load_dev_inputs"]["dev_documents"])
            blob = json.dumps([out["load_eval_inputs"]["items"], out["load_dev_inputs"]["dev_items"]])
            assert "Entailment" not in blob and "Contradiction" not in blob
            assert '"evidence_spans"' not in blob and '"label"' not in blob
            items = out["load_eval_inputs"]["items"]
            assert len(items) == e.n_items
            assert [f"contractnli/{it['doc_id']}/{it['hypothesis_key']}" for it in items] == e.lineage["item_ids"]


def _gold_y(adapter, e):
    data = adapter._data.get()
    y = []
    for item in e.lineage["item_ids"]:
        _, d, h = item.split("/")
        doc = data.docs[d]
        a = doc.annotations[h]
        sc = np.zeros(len(doc.spans))
        sc[a["spans"]] = 1.0
        y.append({"label": a["choice"], "span_scores": sc.tolist()})
    return y


def test_evaluator_reference_oracle_and_constraints(adapter, episodes):
    data = adapter._data.get()
    e = episodes["id"][0]
    gold = _gold_y(adapter, e)
    orc = e.evaluate(gold, None)
    assert orc.primary == pytest.approx(1.0) and orc.metrics["nli_binary_accuracy"] == 1.0
    assert orc.metrics["p_at_r80"] == pytest.approx(1.0) and orc.accepted and orc.z == 1
    ref = orc.details["reference"]
    assert 0.05 < ref < 0.8
    # reference predictions reproduce the reference score
    tr = e.tool("load_train").fn({}, {})
    maj = {}
    for a in tr["train_annotations"]:
        if a["label"] in m.LABELS:
            maj.setdefault(a["hypothesis_key"], []).append(a["label"])
    ref_y = []
    for item in e.lineage["item_ids"]:
        _, d, h = item.split("/")
        doc = data.docs[d]
        labs = maj.get(h, [])
        lab = "Contradiction" if labs.count("Contradiction") > labs.count("Entailment") else "Entailment"
        ref_y.append({"label": lab, "span_scores": m.overlap_scores(data.hypotheses[h]["hypothesis"],
                                                                    doc.span_texts()).tolist()})
    rr = e.evaluate(ref_y, None)
    assert rr.primary == pytest.approx(ref) and rr.details["norm_score"] == pytest.approx(1.0) and rr.z == 0
    # malformed outputs
    assert not e.evaluate(gold[:-1], None).h["output_length"]
    wrong_n = [dict(g) for g in gold]
    wrong_n[0] = {"label": "Entailment", "span_scores": gold[0]["span_scores"][:-1]}
    r = e.evaluate(wrong_n, None)
    assert not r.h["span_scores_valid"] and r.primary is None and r.z == 0
    out_of_range = [dict(g) for g in gold]
    out_of_range[1] = {"label": gold[1]["label"], "span_scores": [2.0] * len(gold[1]["span_scores"])}
    r = e.evaluate(out_of_range, None)
    assert not r.h["span_scores_valid"] and r.z == 0
    bad_label = [dict(g) for g in gold]
    bad_label[2] = {"label": "NotMentioned", "span_scores": gold[2]["span_scores"]}
    r = e.evaluate(bad_label, None)
    assert not r.h["labels_valid"] and r.z == 0 and r.primary == pytest.approx(1.0)
    pooled = adapter.pooled_metric([orc.details["pooled_payload"], rr.details["pooled_payload"]])
    assert pooled == pytest.approx(0.5 * (1.0 + ref))


def test_score_dev(adapter, episodes):
    e = episodes["ood"][0]
    dev = e.tool("load_dev_inputs").fn({}, {})
    n = len(dev["dev_items"])
    assert 0 < n <= adapter.max_dev_items
    r = e.tool("score_dev").fn({"dev_predictions": [{"label": "Entailment", "span_scores": [0.5] * it["n_spans"]}
                                                    for it in dev["dev_items"]]}, {})
    assert 0.0 <= r["dev_map"] <= 1.0 and r["dev_reference_map"] > r["dev_map"]
    with pytest.raises(ValueError):
        e.tool("score_dev").fn({"dev_predictions": [{"label": "Entailment", "span_scores": [0.5]}] * n}, {})


def test_precision_at_recall_matches_official(adapter):
    path = adapter.root / m.DATASET_DIR / "reconstructed_v1" / "evaluator" / "upstream" / "contract_nli" / "evaluation.py"
    if not path.is_file():
        pytest.skip("official ContractNLI evaluation.py not available")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "precision_at_recall")
    import sklearn.metrics
    ns: dict = {"np": np, "sklearn": sklearn}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)   # official source, no I/O
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(3, 60))
        y = (rng.random(n) < 0.2).astype(int)
        y[int(rng.integers(0, n))] = 1
        p = np.round(rng.random(n), 2)
        assert m.precision_at_recall(y, p, 0.8) == pytest.approx(ns["precision_at_recall"](y, p, 0.8))


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


def test_executor_pipeline_uniform_scores(adapter, episodes, tmp_path):
    e = episodes["val"][1]
    code = ("def run(inputs, config):\n"
            "    return {'y': [{'label': 'Entailment', 'span_scores': [0.5] * it['n_spans']} for it in inputs['items']]}\n")
    y = _run_pipeline(e, tmp_path, [("load_eval_inputs", "items", "items")], code, {"items": {"type": "list"}},
                      {"y": {"type": "list", "shape": ["n"]}})
    res = e.evaluate(y, None)
    assert res.hard_ok() and res.primary is not None and res.z == 0


def test_task_card_exists():
    assert (Path(__file__).resolve().parents[1] / "docs" / "tasks" / "FoR48.md").is_file()


def test_objective_braces_and_hypothesis_port_text(adapter, episodes):
    e = episodes["val"][0]
    assert "[...]} for items[i]" in e.objective and "}}" not in e.objective
    port = e.tool("load_hypotheses").outputs["hypotheses"]
    assert "short_description" in port.description
    h = e.tool("load_hypotheses").fn({}, {})["hypotheses"]
    assert all(set(v) == {"hypothesis", "short_description"} for v in h.values())


def test_objective_documents_the_domain_library(adapter, episodes):
    ep = episodes["val"][0]
    assert "scilib.contracts" in ep.objective and "fit_predict" in ep.objective
    assert "}}" not in ep.objective


def test_pooled_p_at_r80_and_majority_label_baseline(adapter, episodes):
    e = episodes["id"][0]
    gold = _gold_y(adapter, e)
    orc = e.evaluate(gold, None)
    assert orc.metrics["p_at_r80_pooled"] == pytest.approx(1.0)
    data = adapter._data.get()
    tr = e.tool("load_train").fn({}, {})
    maj = {}
    for a in tr["train_annotations"]:
        if a["label"] in m.LABELS:
            maj.setdefault(a["hypothesis_key"], []).append(a["label"])
    n_ok = 0
    for item, g in zip(e.lineage["item_ids"], gold):
        _, d, h = item.split("/")
        labs = maj.get(h, [])
        lab = "Contradiction" if labs.count("Contradiction") > labs.count("Entailment") else "Entailment"
        n_ok += int(lab == g["label"])
    assert orc.metrics["majority_label_accuracy"] == pytest.approx(n_ok / len(gold))
    # pooled definition: one precision-recall curve over every span of every pair
    rng = np.random.default_rng(0)
    noisy = [{"label": g["label"], "span_scores": rng.random(len(g["span_scores"])).tolist()} for g in gold]
    r = e.evaluate(noisy, None)
    flat_g = np.concatenate([np.asarray(g["span_scores"]) for g in gold])
    flat_s = np.concatenate([np.asarray(p["span_scores"]) for p in noisy])
    assert r.metrics["p_at_r80_pooled"] == pytest.approx(m.precision_at_recall(flat_g, flat_s, 0.8))
