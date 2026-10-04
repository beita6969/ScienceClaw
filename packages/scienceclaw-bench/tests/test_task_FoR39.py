"""FoR39 Eedi Task 4 adapter: splits, masks, query tool + receipt budget, evaluator (real data; skipped if absent)."""
from __future__ import annotations

import itertools
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_agent_support import FINISH, act, add_edge, add_node, step_of  # noqa: E402

from scienceclaw.agent.solver import Solver
from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.tasks.for39_eedi import N_Q, QUERY_BUDGET, Adapter, check_query_budget, solve_ledger_roots
from scienceclaw.config import EvolutionConfig, SolverConfig
from scienceclaw.core.program import AgentProgram
from scienceclaw.core.trace import NodeRecord, Trace
from scienceclaw.llm import fake
from scienceclaw.runtime.values import save_value

SPLITS = {"src": 7, "val": 2, "id": 4, "ood": 4}


@pytest.fixture(scope="module")
def adapter():
    a = Adapter()
    ok, why = a.available()
    if not ok:
        pytest.skip(why)
    return a


@pytest.fixture(scope="module")
def plan(adapter):
    return {s: adapter.build_episodes(s, n, split_seed(20260928, "FoR39", s)) for s, n in SPLITS.items()}


def _tools(ep):
    return {t.name: t for t in ep.tools}


_LAYOUT_COUNTER = itertools.count()


def _layout(tmp_path):
    """A solver-shaped solve directory: <solve>/exec (session) and <solve>/replay/k000 (Trace.run_dir)."""
    solve = tmp_path / f"solve{next(_LAYOUT_COUNTER)}"
    (solve / "exec" / "values").mkdir(parents=True)
    rd = solve / "replay" / "k000"
    (rd / "values").mkdir(parents=True)
    return solve, rd


def _trace(tmp_path, receipts, session=(), nested=()):
    """Trace whose tool nodes carry ``receipts``; ``session`` receipts are persisted only in the session directory
    (nodes the policy removed later), ``nested`` ones inside an operator body of the replay (not in trace.records)."""
    solve, rd = _layout(tmp_path)
    recs = {}
    for k, r in enumerate(receipts):
        p = rd / "values" / f"fp{k}" / "receipt.pkl"
        save_value(r, p)
        recs[f"q{k}"] = NodeRecord(f"q{k}", f"fp{k}", "ok", outputs_summary={"receipt": {}},
                                   output_refs={"receipt": str(p)}, kind="tool")
    for k, r in enumerate(session):
        save_value(r, solve / "exec" / "values" / f"sfp{k}" / "receipt.pkl")
    for k, r in enumerate(nested):
        save_value(r, rd / "values" / "opfp" / "body" / f"nfp{k}" / "receipt.pkl")
    return Trace(records=recs, run_dir=str(rd))


def _truth(adapter, ep):
    data, setup = adapter._data.get(), adapter._setup.get()
    mats = {"public": data.public, "private": data.private}
    rows = [setup["where"][i] for i in ep.lineage["item_ids"]]
    correct = np.stack([mats[t].correct[r] for t, r in rows])
    targets = np.stack([mats[t].targets[r, m] for (t, r), m in zip(rows, ep.lineage["masks"])])
    return correct, targets


def test_splits_disjoint_masks_balanced(adapter, plan):
    data = adapter._data.get()
    pub = {f"eedi-u{u}" for u in data.public.users}
    priv = {f"eedi-u{u}" for u in data.private.users}
    train = {f"eedi-u{u}" for u in data.train.users}
    assert not (pub & priv) and not (pub & train) and not (priv & train)
    owner = {}
    for s, eps in plan.items():
        assert len(eps) == SPLITS[s]
        for ep in eps:
            ids = ep.lineage["item_ids"]
            assert ep.n_items == len(ids) == 16 == len(set(ids))
            assert set(ids) <= (priv if s == "ood" else pub)
            assert all(0 <= m < 10 for m in ep.lineage["masks"])
            for i in ids:
                assert owner.setdefault(i, s) == s
    masks = [adapter._setup.get()["mask_of"][i] for i in pub]
    assert max(np.bincount(masks)) - min(np.bincount(masks)) <= 1


def test_deterministic(plan):
    again = Adapter().build_episodes("val", 2, split_seed(20260928, "FoR39", "val"))
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in plan["val"]]


def test_eval_tools_hide_answers_and_query_rules(adapter, plan):
    ep = plan["id"][0]
    t = _tools(ep)
    correct, targets = _truth(adapter, ep)
    ev = t["load_eval_inputs"].fn({}, {})
    assert set(ev) == {"can_query", "targets", "student_meta"}
    assert ev["can_query"].dtype == bool and ev["targets"].dtype == bool
    assert np.array_equal(ev["targets"], targets) and not (ev["can_query"] & ev["targets"]).any()
    assert np.array_equal(ev["can_query"] | ev["targets"], correct >= 0)
    tr = t["load_train"].fn({}, {})
    dev_users = set(adapter._data.get().train.users[adapter._setup.get()["dev_rows"]].tolist())
    assert tr["answers"].shape == (adapter._data.get().train.users.size - adapter.n_dev, N_Q)
    assert len(tr["student_meta"]) == tr["answers"].shape[0] and dev_users
    # a target cell can never be queried
    s, q = np.argwhere(targets)[0]
    sel = np.full((16, 1), -1)
    sel[s, 0] = q
    with pytest.raises(ValueError, match="not queryable"):
        t["query_answers"].fn({"selections": sel}, {})
    with pytest.raises(ValueError):
        t["query_answers"].fn({"selections": np.full((16, QUERY_BUDGET + 1), -1)}, {})
    q0 = int(np.flatnonzero(ev["can_query"][0])[0])
    dup = np.full((16, 2), -1)
    dup[0] = [q0, q0]
    with pytest.raises(ValueError, match="duplicate"):
        t["query_answers"].fn({"selections": dup}, {})
    ok = np.full((16, 1), -1)
    ok[0, 0] = q0
    out = t["query_answers"].fn({"selections": ok}, {})
    assert out["revealed"][0, q0] == correct[0, q0] and (out["revealed"] >= 0).sum() == 1
    assert not (out["revealed"][targets] >= 0).any()
    assert json.loads(out["receipt"])["cells"] == [[0, q0]]


def _select(can_query, k, offset=0):
    sel = np.full((can_query.shape[0], k), -1)
    for i in range(can_query.shape[0]):
        c = np.flatnonzero(can_query[i])[offset:offset + k]
        sel[i, :len(c)] = c
    return sel


def test_query_budget_constraint_uses_trace_receipts(adapter, plan, tmp_path):
    ep = plan["src"][0]
    t = _tools(ep)
    cq = t["load_eval_inputs"].fn({}, {})["can_query"]
    correct, targets = _truth(adapter, ep)
    oracle = np.where(targets, correct, -1)
    r1 = t["query_answers"].fn({"selections": _select(cq, 10)}, {})["receipt"]
    r1b = t["query_answers"].fn({"selections": _select(cq, 5)}, {})["receipt"]     # overlaps r1 -> still 10
    r2 = t["query_answers"].fn({"selections": _select(cq, 10, offset=10)}, {})["receipt"]
    assert ep.evaluate(oracle, _trace(tmp_path, [r1, r1b])).h["query_budget"] is True
    over = ep.evaluate(oracle, _trace(tmp_path, [r1, r2]))
    assert over.h["query_budget"] is False and over.z == 0
    forged = json.loads(r2)
    forged["cells"] = forged["cells"][:1]
    assert ep.evaluate(oracle, _trace(tmp_path, [json.dumps(forged)])).h["query_budget"] is False
    assert ep.evaluate(oracle, None).h["query_budget"] is False                      # fail closed without a trace
    dev_t = t["query_dev_answers"].fn({"selections": _select(t["load_dev_inputs"].fn({}, {})["dev_can_query"], 10)}, {})
    assert "receipt" not in dev_t
    solve, rd = _layout(tmp_path)
    missing = Trace(records={"q": NodeRecord("q", "fp", "ok", outputs_summary={"receipt": {}}, kind="tool")}, run_dir=str(rd))
    assert ep.evaluate(oracle, missing).h["query_budget"] is False


def test_evaluator_reference_oracle_pooled(adapter, plan, tmp_path):
    pays = []
    for ep in plan["val"] + plan["ood"][:2]:
        correct, targets = _truth(adapter, ep)
        empty = _trace(tmp_path, [])
        ref_pred = np.repeat(adapter._setup.get()["majority"][None, :], 16, axis=0).astype(int)
        r = ep.evaluate(ref_pred, empty)
        assert r.primary == pytest.approx(r.details["reference"]) and not r.accepted and r.hard_ok()
        o = ep.evaluate(np.where(targets, correct, -1), empty)
        assert o.primary == pytest.approx(1.0) and o.accepted and o.z == 1
        pays.append(o.details["pooled_payload"])
    assert adapter.pooled_metric(pays) == pytest.approx(1.0)
    ep = plan["val"][0]
    bad = ep.evaluate(np.zeros((16, 5)), _trace(tmp_path, []))
    assert bad.primary is None and bad.h["output_shape"] is False
    assert adapter.pooled_metric([bad.details["pooled_payload"]]) == pytest.approx(bad.details["reference"])
    _, targets = _truth(adapter, ep)
    holes = np.where(targets, 1, -1)
    holes[np.argwhere(targets)[0][0], np.argwhere(targets)[0][1]] = -1
    assert ep.evaluate(holes, _trace(tmp_path, [])).h["output_shape"] is False
    assert ep.evaluate(np.full((16, N_Q), 2), _trace(tmp_path, [])).h["output_shape"] is False


def test_score_dev(plan):
    t = _tools(plan["src"][1])
    dv = t["load_dev_inputs"].fn({}, {})
    n_dev = dv["dev_targets"].shape[0]
    out = t["score_dev"].fn({"predictions": np.ones((n_dev, N_Q), dtype=int)}, {})
    assert 0.0 < out["dev_accuracy"] < 1.0 and 0.0 < out["dev_reference_accuracy"] < 1.0
    with pytest.raises(ValueError):
        t["score_dev"].fn({"predictions": np.full((n_dev, N_Q), -1)}, {})


# ------------------------------------------------------------------------------------------- per-solve ledger
def _receipt(t, cq, k, offset=0):
    return t["query_answers"].fn({"selections": _select(cq, k, offset)}, {})["receipt"]


def _oracle(adapter, ep):
    correct, targets = _truth(adapter, ep)
    return np.where(targets, correct, -1)


def _budget(ep, y, trace):
    return ep.evaluate(y, trace).h["query_budget"]


def test_query_budget_ledger_counts_removed_session_queries(adapter, plan, tmp_path):
    """Answers revealed by a node the policy deleted before submitting still count."""
    ep = plan["src"][0]
    t, y = _tools(ep), _oracle(adapter, ep)
    cq = t["load_eval_inputs"].fn({}, {})["can_query"]
    r1, r2 = _receipt(t, cq, 10), _receipt(t, cq, 10, offset=10)
    # the final trace holds only r2 (10 cells); the deleted session node had revealed r1 (10 other cells)
    res = ep.evaluate(y, _trace(tmp_path, [r2], session=[r1]))
    assert res.h["query_budget"] is False and res.z == 0
    assert "budget" in ep.check_constraints(y, _trace(tmp_path, [r2], session=[r1]))[1]["query_budget"]
    # the same cells in the session and in the final trace: still 10 distinct cells per student
    assert _budget(ep, y, _trace(tmp_path, [r1], session=[r1])) is True
    # a partial overlap that stays within 10 distinct cells per student is fine, too
    assert _budget(ep, y, _trace(tmp_path, [r1], session=[_receipt(t, cq, 4)])) is True
    # session queries are charged even when no query node is left in the final trace
    assert _budget(ep, y, _trace(tmp_path, [], session=[r1, r2])) is False
    assert _budget(ep, y, _trace(tmp_path, [], session=[r1])) is True


def test_query_budget_ledger_sees_operator_bodies(adapter, plan, tmp_path):
    """A query_answers node inside an operator body has no top-level record; its receipt is found on disk."""
    ep = plan["src"][0]
    t, y = _tools(ep), _oracle(adapter, ep)
    cq = t["load_eval_inputs"].fn({}, {})["can_query"]
    r1, r2 = _receipt(t, cq, 10), _receipt(t, cq, 10, offset=10)
    assert _budget(ep, y, _trace(tmp_path, [r1], nested=[r1])) is True
    assert _budget(ep, y, _trace(tmp_path, [r1], nested=[r2])) is False
    assert _budget(ep, y, _trace(tmp_path, [], nested=[r1, r2])) is False


def test_query_budget_ledger_ignores_foreign_files_and_fails_closed(adapter, plan, tmp_path):
    ep, other = plan["src"][0], plan["src"][1]
    t, t_other = _tools(ep), _tools(other)
    y = _oracle(adapter, ep)
    cq = t["load_eval_inputs"].fn({}, {})["can_query"]
    r1 = _receipt(t, cq, 10)
    foreign = _receipt(t_other, t_other["load_eval_inputs"].fn({}, {})["can_query"], 10)   # another episode's receipt
    for extra in (foreign, "not a receipt", '{"kind": "other"}', np.arange(3), {"receipt": 1}):
        assert _budget(ep, y, _trace(tmp_path, [r1], session=[extra])) is True, type(extra)
    # a tampered eval receipt anywhere in the ledger fails closed
    forged = json.loads(_receipt(t, cq, 10, offset=10))
    forged["cells"] = forged["cells"][:1]
    assert _budget(ep, y, _trace(tmp_path, [r1], session=[json.dumps(forged)])) is False
    # a truncated ledger file fails closed
    tr = _trace(tmp_path, [r1])
    bad = Path(tr.run_dir).parent.parent / "exec" / "values" / "x" / "receipt.pkl"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(pickle.dumps("abcdefgh")[:6])
    assert _budget(ep, y, tr) is False


def test_query_budget_needs_a_locatable_ledger(adapter, plan, tmp_path):
    ep = plan["src"][0]
    y = _oracle(adapter, ep)
    good = _trace(tmp_path, [])
    assert _budget(ep, y, good) is True
    roots, why = solve_ledger_roots(good)
    assert why == "" and [p.name for p in roots] == ["k000", "exec"]
    assert _budget(ep, y, Trace(records={})) is False                                   # no run_dir at all
    assert _budget(ep, y, Trace(records={}, run_dir=str(tmp_path / "elsewhere" / "k000"))) is False   # not <solve>/replay/<k>
    solve, rd = _layout(tmp_path)
    shutil.rmtree(solve / "exec")
    assert _budget(ep, y, Trace(records={}, run_dir=str(rd))) is False                  # session directory missing


def test_query_answers_texts_describe_the_session_ledger(plan):
    ep = plan["src"][0]
    assert "later remove or replace" in _tools(ep)["query_answers"].description
    assert "removed or replaced" in {c.name: c for c in ep.constraints}["query_budget"].description


_SEL = ("def run(inputs, config):\n    import numpy as np\n    cq = np.asarray(inputs['cq'])\n"
        "    sel = np.full((cq.shape[0], 10), -1)\n    for i in range(cq.shape[0]):\n"
        "        c = np.flatnonzero(cq[i])[{off}:{off} + 10]\n        sel[i, :len(c)] = c\n    return {{'sel': sel}}\n")
_PRED = ("def run(inputs, config):\n    import numpy as np\n    return {'y': np.where(np.asarray(inputs['targets']), 1, -1).astype(int)}\n")


def _script(swap: bool) -> list[dict]:
    """Policy: query 10 answers per student, then (swap=True) delete that query node and query 10 *other* answers."""
    def sel_node(nid, off):
        return add_node(id=nid, kind="code", code=_SEL.format(off=off), inputs={"cq": {"type": "array"}},
                        outputs={"sel": {"type": "array"}})

    s = [add_node(id="load", kind="tool", ref="load_eval_inputs"), sel_node("sel1", 0),
         add_node(id="q1", kind="tool", ref="query_answers"), add_edge("load", "can_query", "sel1", "cq"),
         add_edge("sel1", "sel", "q1", "selections"),
         add_node(id="pred", kind="code", code=_PRED, inputs={"targets": {"type": "array"}, "rev": {"type": "array"}},
                  outputs={"y": {"type": "array"}}),
         add_edge("load", "targets", "pred", "targets"), add_edge("q1", "revealed", "pred", "rev"),
         add_node(id="sub", kind="submit"), add_edge("pred", "y", "sub", "y")]                # replay k009: 10 cells
    if swap:
        s += [act({"type": "remove_node", "id": "q1"}), act({"type": "remove_node", "id": "sel1"}), sel_node("sel2", 10),
              add_node(id="q2", kind="tool", ref="query_answers"), add_edge("load", "can_query", "sel2", "cq"),
              add_edge("sel2", "sel", "q2", "selections"), add_edge("q2", "revealed", "pred", "rev")]   # replay k016
    return s + [FINISH]


def _solve(ep, script, tmp_path):
    llm = fake.FakeLLM(lambda role, messages: json.dumps(script[step_of(messages)] if 0 <= step_of(messages) < len(script)
                                                          else FINISH))
    return Solver(SolverConfig(max_steps=24), llm, EvolutionConfig()).solve(ep, AgentProgram(), "source", tmp_path / "run")


def test_real_solver_charges_deleted_session_queries(adapter, plan, tmp_path):
    """End to end with the real Solver / executor / replay: layout contract between solver.py and the ledger."""
    swap = _solve(plan["val"][0], _script(True), tmp_path)
    budgets = [e.eval.h["query_budget"] for e in swap.evidence]
    assert budgets == [True, False], [e.summary() for e in swap.evidence]      # k009 fine, the swapped graph is over budget
    last = swap.evidence[-1]
    assert last.eval.z == 0 and [c for c, v in last.eval.h.items() if v is False] == ["query_budget"]
    assert last.trace.records["q2"].status == "ok" and "q1" not in last.trace.records   # the final trace shows 10 cells only
    assert swap.z == 0 or swap.final_step != last.step
    honest = _solve(plan["val"][1], _script(False), tmp_path / "h")
    assert [e.eval.h["query_budget"] for e in honest.evidence] == [True]


def test_objective_documents_the_domain_library(plan):
    from test_adapter_visible_text import scan_acceptance, scan_recipe, visible_text

    ep = plan["val"][0]
    for name in ("scilib.adaptive", "select_queries", "predict", "fit_item_curves", "merge_revealed"):
        assert name in ep.objective
    text = visible_text(ep)
    assert not scan_recipe(text) and not scan_acceptance(text)
    assert "reference" not in ep.objective.lower() and "accepted" not in ep.objective.lower()


def test_domain_library_round_trip_through_the_tools(adapter, plan, tmp_path):
    """The library's documented array conventions match what the tools return; two rounds stay within the budget."""
    from scilib import adaptive as ad

    ep = plan["val"][0]
    t = _tools(ep)
    correct, targets = _truth(adapter, ep)
    answers = t["load_train"].fn({}, {})["answers"]
    cq = t["load_eval_inputs"].fn({}, {})["can_query"]
    model = ad.fit_item_curves(answers, iterations=8)
    sel1 = ad.select_queries(model, cq, [], 4)
    out1 = t["query_answers"].fn({"selections": sel1}, {})
    assert set(np.unique(out1["revealed"])) <= {-1, 0, 1} and (out1["revealed"] >= 0).sum() == (sel1 >= 0).sum()
    sel2 = ad.select_queries(model, cq, [out1["revealed"]], 6)
    out2 = t["query_answers"].fn({"selections": sel2}, {})
    assert not ((out2["revealed"] >= 0) & (out1["revealed"] >= 0)).any()          # a call returns only its own cells
    merged = ad.merge_revealed([out1["revealed"], out2["revealed"]])
    assert (merged >= 0).sum(axis=1).max() <= QUERY_BUDGET
    assert np.array_equal(merged[merged >= 0], correct[merged >= 0])
    y = ad.predict(model, [out1["revealed"], out2["revealed"]], targets)
    assert y.shape == (16, N_Q) and (y[targets] >= 0).all()
    res = ep.evaluate(y, _trace(tmp_path, [out1["receipt"], out2["receipt"]]))
    assert res.h["query_budget"] is True and res.h["output_shape"] is True and res.primary is not None
