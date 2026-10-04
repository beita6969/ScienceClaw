"""FoR50 SemEval-2023 Task 4 ValueEval adapter tests (skipped when the data are not available)."""
from __future__ import annotations

import csv
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.tasks import for50_valueeval as m

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]
SEED = 20260928


def test_official_f1_definition():
    # 2 categories with support: P = (1/2 + 3/4)/2 = 0.625, R = (1 + 1/2)/2 = 0.75 -> F1 = 15/22 (data-team fixture
    # values: harmonic mean of macro P and macro R, not the mean of per-category F1)
    t = np.zeros((4, 20), int)
    p = np.zeros((4, 20), int)
    t[0, 0] = 1
    p[[0, 1], 0] = 1                        # category 0: TP 1, FP 1 -> P 1/2, R 1
    t[:, 1] = [1, 1, 1, 1]
    p[:, 1] = [1, 1, 0, 0]                  # category 1: TP 2 of 2 predicted -> P 1, R 1/2
    p[2, 5] = 1                             # category 5 has no gold positive -> skipped (official rule)
    r = m.official_f1(t, p)
    P, R = (0.5 + 1.0) / 2, (1.0 + 0.5) / 2
    assert r["precision"] == pytest.approx(P) and r["recall"] == pytest.approx(R)
    assert r["f1"] == pytest.approx(2 * P * R / (P + R)) and r["n_categories"] == 2
    assert r["mean_category_f1"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert m.official_f1(t, np.zeros_like(t))["f1"] == 0.0          # zero guard
    assert m.official_f1(t, t)["f1"] == pytest.approx(1.0)
    assert np.isnan(m.official_f1(np.zeros((2, 20), int), np.ones((2, 20), int))["f1"])


def test_pooled_diagnostics_separates_slice_and_pooled_metric():
    a = m.Adapter()
    t1 = np.zeros((4, 20), int); t1[:, 0] = [1, 1, 0, 0]
    t2 = np.zeros((4, 20), int); t2[:, 0] = [1, 0, 1, 0]; t2[:, 1] = [0, 1, 0, 1]
    p1 = t1.copy(); p1[3, 0] = 1
    p2 = np.ones_like(t2)
    d = a.pooled_diagnostics([{"y_true": t1.tolist(), "y_pred": p1.tolist()},
                              {"y_true": t2.tolist(), "y_pred": p2.tolist()}])
    pooled = m.official_f1(np.vstack([t1, t2]), np.vstack([p1, p2]))["f1"]
    assert d["diagnostic_only"] is True and d["n_episodes"] == 2 and d["n_items"] == 8
    assert d["pooled_f1"] == pytest.approx(pooled)
    assert d["n_scored_categories"] == 2
    assert d["mean_episode_f1"] != pytest.approx(d["pooled_f1"])


def test_pooled_diagnostics_counts_nan_episode_without_using_it_in_the_mean():
    """An episode with no supported category remains in the audit item/episode count."""
    a = m.Adapter()
    empty_t = np.zeros((2, 20), dtype=int)
    supported_t = np.zeros((2, 20), dtype=int)
    supported_t[0, 0] = 1
    d = a.pooled_diagnostics([
        {"y_true": empty_t.tolist(), "y_pred": np.ones_like(empty_t).tolist()},
        {"y_true": supported_t.tolist(), "y_pred": supported_t.tolist()},
    ])
    assert d["n_episodes"] == 2 and d["n_valid_episode_f1"] == 1
    assert d["n_items"] == 4 and d["pooled_f1"] == pytest.approx(0.5)


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
    data = adapter._data.get()
    items = {s: _items(eps) for s, eps in episodes.items()}
    for s, lst in items.items():
        assert len(lst) == len(set(lst)), s
        assert all(e.n_items == 16 for e in episodes[s])
    for a, b in combinations(items, 2):
        assert not set(items[a]) & set(items[b]), (a, b)
    concl = {s: {data.args[u].conclusion for u in items[s]} for s in ("src", "val")}
    assert not concl["src"] & concl["val"]                       # conclusion-disjoint (official grouping)
    assert {data.args[u].source for u in items["src"] + items["val"]} == {"validation"}
    assert {data.args[u].source for u in items["id"]} == {"test"}
    assert {data.args[u].source for u in items["ood"]} == {"test-nahjalbalagha"}
    again = m.Adapter().build_episodes("id", 4, split_seed(SEED, m.CODE, "id"))
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["id"]]


def test_visible_data_disjoint_and_no_label_leak(adapter, episodes):
    data = adapter._data.get()
    for s, eps in episodes.items():
        for e in eps[:2]:
            tr = e.tool("load_train").fn({}, {})
            ev = e.tool("load_eval_inputs").fn({}, {})
            dv = e.tool("load_dev_inputs").fn({}, {})
            assert list(ev["arguments"].columns) == ["conclusion", "stance", "premise"]
            assert list(dv["dev_arguments"].columns) == ["conclusion", "stance", "premise"]
            ev_ids = {data.args[u].arg_id for u in e.lineage["item_ids"]}
            assert not set(tr["arguments"]["argument_id"]) & ev_ids
            assert all(data.args[f"valueeval/{a}"].source == "training" for a in tr["arguments"]["argument_id"])
            assert tr["labels"].shape == (len(tr["arguments"]), 20)
            assert list(ev["arguments"]["premise"]) == [data.args[u].premise for u in e.lineage["item_ids"]]
            assert not set(e.lineage["dev_item_ids"]) & set(e.lineage["item_ids"])


def test_evaluator_reference_oracle_and_constraints(adapter, episodes):
    data = adapter._data.get()
    e = episodes["ood"][2]
    gold = np.asarray([data.args[u].labels for u in e.lineage["item_ids"]], dtype=int)
    orc = e.evaluate(gold, None)
    assert orc.primary == pytest.approx(1.0) and orc.accepted and orc.z == 1
    ref = e.evaluate(np.ones_like(gold), None)
    assert ref.primary == pytest.approx(ref.details["reference"]) and not ref.accepted and ref.z == 0
    assert ref.details["norm_score"] == pytest.approx(1.0)
    r = e.evaluate(gold[:-1], None)
    assert not r.h["output_shape"] and r.primary is None and r.z == 0
    bad = gold.astype(float)
    bad[0, 0] = 0.5
    r = e.evaluate(bad, None)
    assert not r.h["binary"] and r.primary is None and r.z == 0
    assert not e.evaluate("nope", None).h["output_shape"]
    assert e.evaluate(gold.tolist(), None).primary == pytest.approx(1.0)
    pooled = adapter.pooled_metric([orc.details["pooled_payload"], r.details["pooled_payload"]])
    both_t = np.vstack([gold, gold])
    assert pooled == pytest.approx(m.official_f1(both_t, np.vstack([gold, np.ones_like(gold)]))["f1"])


def test_score_dev_and_taxonomy(adapter, episodes):
    e = episodes["src"][0]
    dv = e.tool("load_dev_inputs").fn({}, {})["dev_arguments"]
    r = e.tool("score_dev").fn({"dev_predictions": np.ones((len(dv), 20), int)}, {})
    assert r["dev_f1"] == pytest.approx(r["dev_reference_f1"])
    with pytest.raises(ValueError):
        e.tool("score_dev").fn({"dev_predictions": np.ones((len(dv), 19), int)}, {})
    tax = e.tool("load_value_taxonomy").fn({}, {})
    assert list(tax["taxonomy"]) == list(m.VALUES)


def test_fixed_fit_predict_tool_uses_visible_tables_and_fixed_parameters(adapter, episodes, monkeypatch):
    """The explicit specialist route receives only visible tables and cannot drift its fit configuration."""
    from scilib import valueeval

    e = episodes["id"][0]
    tool = e.tool("fit_predict")
    assert tool is not None
    train = e.tool("load_train").fn({}, {})
    ev = e.tool("load_eval_inputs").fn({}, {})
    seen = {}

    class FakePred(list):
        oof_f1 = 0.123

    def fake_fit_predict(train_df, labels, targets, **kwargs):
        seen["train_columns"] = list(train_df.columns)
        seen["label_shape"] = np.asarray(labels).shape
        seen["target_columns"] = list(targets[0].columns)
        seen["kwargs"] = kwargs
        return FakePred([np.zeros((len(targets[0]), len(m.VALUES)), dtype=int)])

    monkeypatch.setattr(valueeval, "fit_predict", fake_fit_predict)
    out = tool.fn({"train_arguments": train["arguments"], "train_labels": train["labels"],
                   "eval_arguments": ev["arguments"]}, {})
    assert set(out) == {"y", "provenance"}
    assert out["y"].shape == (e.n_items, len(m.VALUES))
    assert seen["train_columns"] == ["argument_id", "conclusion", "stance", "premise"]
    assert seen["target_columns"] == ["conclusion", "stance", "premise"]
    assert seen["label_shape"] == (len(train["arguments"]), len(m.VALUES))
    assert {k: seen["kwargs"][k] for k in ("n_folds", "C", "k", "seed", "n_jobs", "decision")} == {
        "n_folds": m.VALUEEVAL_TOOL_FOLDS, "C": m.VALUEEVAL_TOOL_C, "k": m.VALUEEVAL_TOOL_K,
        "seed": m.VALUEEVAL_TOOL_SEED, "n_jobs": m.VALUEEVAL_TOOL_N_JOBS,
        "decision": m.VALUEEVAL_TOOL_DECISION,
    }
    assert np.array_equal(seen["kwargs"]["groups"], train["arguments"]["conclusion"].astype(str).to_numpy())
    assert out["provenance"]["label_source"] == "visible load_train.labels only"
    with pytest.raises(ValueError, match="forbidden"):
        tool.fn({"train_arguments": train["arguments"], "train_labels": train["labels"],
                 "eval_arguments": ev["arguments"].assign(labels=0)}, {})
    with pytest.raises(ValueError, match="0 and 1"):
        tool.fn({"train_arguments": train["arguments"], "train_labels": train["labels"].astype(float) + 0.25,
                 "eval_arguments": ev["arguments"]}, {})
    def bad_fit_predict(*args, **kwargs):
        return FakePred([np.full((len(ev["arguments"]), len(m.VALUES)), 0.5)])
    monkeypatch.setattr(valueeval, "fit_predict", bad_fit_predict)
    with pytest.raises(ValueError, match="predictions.*0 and 1"):
        tool.fn({"train_arguments": train["arguments"], "train_labels": train["labels"],
                 "eval_arguments": ev["arguments"]}, {})


def test_full_split_diagnostics_is_trusted_side_and_reuses_fixed_route(adapter, monkeypatch):
    """The full-pool audit returns aggregates only and does not alter episode scoring."""
    calls = {}

    def fake_route(train_table, train_labels, eval_table, n_values):
        calls["n_train"] = len(train_table)
        calls["n_eval"] = len(eval_table)
        calls["labels"] = np.asarray(train_labels).shape
        return np.zeros((len(eval_table), n_values), dtype=int), {"method": "fake-fixed-route", "seed": 0}

    monkeypatch.setattr(m, "_fixed_fit_predict", fake_route)
    out = adapter.full_split_diagnostics("id")
    assert out["diagnostic_only"] is True and out["split"] == "id"
    assert out["n_items"] == len(adapter._data.get().eval_split["id"]["all"])
    assert out["n_train"] == calls["n_train"]
    assert calls["n_eval"] == out["n_items"] and calls["labels"][1] == len(m.VALUES)
    assert set(out) >= {"f1", "precision", "recall", "reference_f1", "tool", "partition_seed",
                        "train_ids_sha256", "split_ids_sha256", "receipt"}
    assert len(out["train_ids_sha256"]) == 64 and len(out["split_ids_sha256"]) == 64
    assert "y_true" not in out and "predictions" not in out and "item_ids" not in out


def test_full_split_driver_has_label_free_cli_surface():
    driver = Path(__file__).resolve().parents[1] / "scripts" / "f50" / "full_split_valueeval.py"
    assert driver.is_file()
    r = subprocess.run([sys.executable, str(driver), "--help"], check=True, capture_output=True, text=True)
    assert "complete split" in r.stdout and "--data-root" in r.stdout


def test_matches_official_evaluator_script(adapter, episodes, tmp_path):
    """Run the unchanged official evaluator (touche-code, 2023-08-13) in a subprocess on one episode."""
    script = adapter.root / m.DATASET_DIR / "reconstructed_v1" / "evaluator" / "upstream" / "evaluator.py"
    if not script.is_file():
        pytest.skip("official ValueEval evaluator.py not available")
    data = adapter._data.get()
    e = episodes["id"][0]
    ids = [data.args[u].arg_id for u in e.lineage["item_ids"]]
    gold = np.asarray([data.args[u].labels for u in e.lineage["item_ids"]], dtype=int)
    rng = np.random.default_rng(5)
    pred = (rng.random(gold.shape) < 0.3).astype(int) | gold * (rng.random(gold.shape) < 0.6)
    truth_dir, run_dir, out_dir = tmp_path / "truth", tmp_path / "run", tmp_path / "out"
    truth_dir.mkdir()
    run_dir.mkdir()
    for d, mat in ((truth_dir, gold), (run_dir, pred)):
        with open(d / "labels-x.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["Argument ID", *m.VALUES])
            for aid, row in zip(ids, mat):
                w.writerow([aid, *[str(int(v)) for v in row]])
    subprocess.run([sys.executable, "-B", str(script), "-i", str(truth_dir), "-r", str(run_dir), "-o", str(out_dir)],
                   check=True, capture_output=True, timeout=60)
    text = (out_dir / "evaluation.prototext").read_text()
    f1 = float(text.split('key: "F1"')[1].split('value: "')[1].split('"')[0])
    assert e.evaluate(pred, None).primary == pytest.approx(f1, abs=1e-12)


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


def test_executor_pipeline_all_ones(adapter, episodes, tmp_path):
    e = episodes["val"][0]
    code = ("import numpy as np\n\n"
            "def run(inputs, config):\n"
            "    return {'y': np.ones((len(inputs['arguments']), len(inputs['value_names'])), dtype=int)}\n")
    y = _run_pipeline(e, tmp_path, [("load_eval_inputs", "arguments", "arguments"),
                                    ("load_eval_inputs", "value_names", "value_names")], code,
                      {"arguments": {"type": "table"}, "value_names": {"type": "list"}},
                      {"y": {"type": "array", "shape": ["n", 20]}})
    res = e.evaluate(y, None)
    assert res.hard_ok() and res.primary == pytest.approx(res.details["reference"]) and res.z == 0


def test_executor_pipeline_fit_predict_direct_to_submit(adapter, episodes, tmp_path, monkeypatch):
    """The specialist ToolSpec wiring carries visible tables into y and preserves output lineage on replay."""
    from scienceclaw.core.actions import Action
    from scienceclaw.core.graph import WorkflowGraph
    from scienceclaw.core.program import AgentProgram
    from scienceclaw.runtime.executor import Executor
    from scienceclaw.runtime.replay import replay
    from scienceclaw.runtime.values import outputs_match

    e = episodes["val"][0]

    def fake_route(train_table, train_labels, eval_table, n_values):
        return np.zeros((len(eval_table), n_values), dtype=int), {"method": "fake", "seed": 0}

    monkeypatch.setattr(m, "_fixed_fit_predict", fake_route)
    prog = AgentProgram()
    ex = Executor(e, prog, None, tmp_path / "run-fit-tool")
    g, cp = WorkflowGraph(), ex.new_checkpoint()
    actions = [
        Action("add_node", {"node": {"id": "tr", "kind": "tool", "ref": "load_train"}}),
        Action("add_node", {"node": {"id": "ev", "kind": "tool", "ref": "load_eval_inputs"}}),
        Action("add_node", {"node": {"id": "fp", "kind": "tool", "ref": "fit_predict"}}),
        Action("add_edge", {"edge": {"src": "tr", "src_port": "arguments", "dst": "fp",
                                         "dst_port": "train_arguments"}}),
        Action("add_edge", {"edge": {"src": "tr", "src_port": "labels", "dst": "fp",
                                         "dst_port": "train_labels"}}),
        Action("add_edge", {"edge": {"src": "ev", "src_port": "arguments", "dst": "fp",
                                         "dst_port": "eval_arguments"}}),
        Action("add_node", {"node": {"id": "s", "kind": "submit"}}),
        Action("add_edge", {"edge": {"src": "fp", "src_port": "y", "dst": "s", "dst_port": "y"}}),
    ]
    fb = None
    y = None
    for k, action in enumerate(actions):
        g, cp, fb, y = ex.apply(g, cp, action, k)
        assert fb.action_ok, fb.action_error
    assert fb.submit_ready and fb.records["fp"].status == "ok" and fb.records["s"].status == "ok"
    assert fb.records["fp"].output_refs["y"] == fb.records["s"].input_refs["y"]
    replay_y, replay_trace = replay(g, e, prog, None, tmp_path / "replay-fit-tool")
    assert outputs_match(y, replay_y, e.tolerance)
    assert replay_trace.records["fp"].status == "ok"


def test_task_card_exists():
    assert (Path(__file__).resolve().parents[1] / "docs" / "tasks" / "FoR50.md").is_file()


def test_dev_slice_is_held_out_by_conclusion(adapter, episodes):
    """Like the real evaluation, dev conclusions never occur among the visible training arguments (no near-duplicate leak)."""
    data = adapter._data.get()
    for s, eps in episodes.items():
        for e in eps[:2]:
            tr = e.tool("load_train").fn({}, {})["arguments"]
            dv = e.tool("load_dev_inputs").fn({}, {})["dev_arguments"]
            assert len(dv) == adapter.n_dev
            assert not set(tr["conclusion"]) & set(dv["conclusion"]), (s, e.id)
            assert len(set(dv["conclusion"])) >= 4                      # several conclusions, not one topic
    assert "reference predictor's F1" in episodes["id"][0].tool("score_dev").description


def test_objective_documents_the_domain_library(adapter, episodes):
    ep = episodes["val"][0]
    assert "scilib.valueeval" in ep.objective and "fit_predict" in ep.objective
