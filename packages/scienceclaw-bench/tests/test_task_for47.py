"""FoR47 CoNLL-2018 UD parsing adapter tests (skipped when the data are not available)."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.tasks import for47_ud as m

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]
SEED = 20260928

_SAMPLE = """# sent_id = s1
# text = Il va au marché.
1\tIl\til\tPRON\t_\t_\t2\tnsubj\t_\t_
2\tva\taller\tVERB\t_\t_\t0\troot\t_\t_
3-4\tau\t_\t_\t_\t_\t_\t_\t_\t_
3\tà\tà\tADP\t_\t_\t5\tcase\t_\t_
4\tle\tle\tDET\t_\t_\t5\tdet\t_\t_
5\tmarché\tmarché\tNOUN\t_\t_\t2\tobl:mod\t_\tSpaceAfter=No
6\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

"""


def test_conllu_reader_and_tree_rules():
    s = m.parse_conllu(_SAMPLE)[0]
    assert s.forms == ("Il", "va", "à", "le", "marché", ".") and s.mwts == ((3, 4, "au"),)
    assert m.parse_conllu(m.to_conllu([s]))[0] == s
    assert m.tree_issues(list(s.heads), list(s.deprels)) == []
    assert any("cycle" in i for i in m.tree_issues([2, 1, 0], ["nsubj", "obj", "root"]))
    assert any("roots" in i for i in m.tree_issues([0, 0], ["root", "root"]))
    assert any("outside" in i for i in m.tree_issues([3, 0], ["dep", "root"]))
    assert any("relations" in i for i in m.tree_issues([0, 1], ["root", "foo"]))
    # LAS ignores relation subtypes
    las, uas = m.attachment_counts(s, [2, 0, 5, 5, 2, 2], ["nsubj", "root", "case", "det", "obl", "punct"])
    assert (las, uas) == (6, 6)
    h, r, why = m.coerce_parse([[2, "nsubj"], [0, "root"]], 2)
    assert h == [2, 0] and r == ["nsubj", "root"]
    assert m.coerce_parse({"head": [1.5, 0], "deprel": ["x", "root"]}, 2)[0] is None


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
    assert all("/UD_French-GSD/dev/" in i for i in items["src"] + items["val"])
    assert all("/UD_French-GSD/test/" in i for i in items["id"])
    assert all("/UD_French-PUD/test/" in i for i in items["ood"])
    again = m.Adapter().build_episodes("val", 2, split_seed(SEED, m.CODE, "val"))
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["val"]]


def test_visible_data_disjoint_and_no_label_leak(adapter, episodes):
    data = adapter._data.get()
    eval_forms = {data.sents[u].forms for eps in episodes.values() for e in eps for u in e.lineage["item_ids"]}
    for s, eps in episodes.items():
        for e in eps[:2]:
            out = {t.name: t.fn({}, {}) for t in e.tools if t.name in ("load_train", "load_dev_inputs", "load_eval_inputs")}
            for sent in out["load_eval_inputs"]["sentences"] + out["load_dev_inputs"]["dev_sentences"]:
                assert all(set(w) == {"id", "form"} for w in sent["words"])
                assert "sent_id" not in sent
            tr_forms = {tuple(w["form"] for w in t["words"]) for t in out["load_train"]["train"]}
            assert not tr_forms & eval_forms
            assert "train" in e.lineage["train_source"]
            assert not set(e.lineage["dev_item_ids"]) & set(e.lineage["item_ids"])
            ev = [data.sents[u] for u in e.lineage["item_ids"]]
            assert [tuple(w["form"] for w in x["words"]) for x in out["load_eval_inputs"]["sentences"]] == \
                [x.forms for x in ev]
            blob = json.dumps(out["load_eval_inputs"], ensure_ascii=False)
            assert '"head"' not in blob and '"deprel"' not in blob


def _gold_y(adapter, e):
    data = adapter._data.get()
    return [{"head": list(data.sents[u].heads), "deprel": list(data.sents[u].deprels)} for u in e.lineage["item_ids"]]


def test_evaluator_reference_oracle_and_constraints(adapter, episodes):
    e = episodes["ood"][0]
    gold = _gold_y(adapter, e)
    orc = e.evaluate(gold, None)
    assert orc.primary == pytest.approx(1.0) and orc.metrics["uas"] == pytest.approx(1.0) and orc.z == 1
    ref = orc.details["reference"]
    assert 0.1 < ref < 0.5
    # the reference baseline itself scores exactly the reference
    tr = [m.Sentence(**{**{f.name: None for f in dataclasses.fields(m.Sentence)}, "uid": t["sent_id"],
                        "sent_id": t["sent_id"], "text": t["text"], "forms": tuple(w["form"] for w in t["words"]),
                        "deprels": tuple(w["deprel"] for w in t["words"]), "mwts": ()})
          for t in e.tool("load_train").fn({}, {})["train"]]
    table, fb = m.form_relation_table(tr)
    data = adapter._data.get()
    ref_y = [dict(zip(("head", "deprel"), m.chain_baseline(data.sents[u], table, fb))) for u in e.lineage["item_ids"]]
    rr = e.evaluate(ref_y, None)
    assert rr.hard_ok() and rr.primary == pytest.approx(ref) and not rr.accepted and rr.z == 0
    # malformed outputs
    assert not e.evaluate(gold[:-1], None).h["output_length"]
    bad_fmt = gold[:-1] + [{"head": gold[-1]["head"][:-1], "deprel": gold[-1]["deprel"]}]
    r = e.evaluate(bad_fmt, None)
    assert not r.h["parse_format"] and r.primary is None and r.z == 0
    two_roots = [dict(g) for g in gold]
    two_roots[0] = {"head": [0] * len(gold[0]["head"]), "deprel": ["root"] * len(gold[0]["head"])}
    r = e.evaluate(two_roots, None)
    assert not r.h["valid_tree"] and r.z == 0 and r.primary is not None
    bad_rel = [dict(g) for g in gold]
    bad_rel[1] = {"head": gold[1]["head"], "deprel": ["nsubjj"] * len(gold[1]["head"])}
    assert not e.evaluate(bad_rel, None).h["ud_relations"]
    # pooled LAS = words-weighted
    pooled = adapter.pooled_metric([orc.details["pooled_payload"], rr.details["pooled_payload"]])
    assert pooled == pytest.approx(0.5 * (1.0 + ref))


def test_tools(adapter, episodes, monkeypatch):
    e = episodes["src"][0]
    dev = e.tool("load_dev_inputs").fn({}, {})["dev_sentences"]
    chain = [{"head": [i + 1 if i < d["n_words"] else 0 for i in range(1, d["n_words"] + 1)],
              "deprel": ["dep"] * (d["n_words"] - 1) + ["root"]} for d in dev]
    r = e.tool("score_dev").fn({"dev_parses": chain}, {})
    assert 0.0 <= r["dev_las"] <= r["dev_uas"] <= 1.0 and r["dev_reference_las"] > r["dev_las"]
    chk = e.tool("check_trees").fn({"parses": [{"head": [0, 0], "deprel": ["root", "root"]}, chain[0]]}, {})
    assert chk["valid"].tolist() == [False, True]
    rc = e.tool("read_conllu").fn({"conllu": _SAMPLE}, {})["sentences"]
    assert rc[0]["words"][4]["deprel"] == "obl:mod"
    text = e.tool("load_train").fn({}, {})["train_conllu"]
    assert len(m.parse_conllu(text)) == e.lineage["n_train"]

    # The frozen parser must be exposed as an explicit task tool, so a formal episode can
    # record a tool node instead of hiding the call inside an arbitrary code node.
    assert "`parses` output port" in e.objective and "directly as y" in e.objective
    assert "finish immediately" in e.objective and "later rewire" in e.objective
    from scilib import udparse_pretrained as pre
    monkeypatch.setattr(pre, "available", lambda: True)
    monkeypatch.setattr(pre, "parse_gold_tokens", lambda sents: [
        {"head": [i + 1 if i + 1 < len(s["words"]) else 0 for i in range(len(s["words"]))],
         "deprel": ["dep"] * (len(s["words"]) - 1) + ["root"]}
        for s in sents
    ])
    sentences = e.tool("load_eval_inputs").fn({}, {})["sentences"][:2]
    out = e.tool("pretrained_parse").fn({"sentences": sentences}, {})
    assert len(out["parses"]) == 2
    assert all(len(p["head"]) == len(s["words"]) for p, s in zip(out["parses"], sentences))


def _official_module(adapter):
    path = adapter.root / m.DATASET_DIR / "reconstructed_v1" / "evaluator" / "upstream" / "conll18_ud_eval.py"
    if not path.is_file():
        pytest.skip("official conll18_ud_eval.py not available")
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("_conll18_official", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)            # type: ignore[union-attr]
    finally:
        sys.dont_write_bytecode = old
    return mod


def test_matches_official_conll18_evaluator(adapter, episodes, tmp_path):
    off = _official_module(adapter)
    data = adapter._data.get()
    e = episodes["id"][0]
    gold = [data.sents[u] for u in e.lineage["item_ids"]]
    rng = np.random.default_rng(3)
    sys_sents, y = [], []
    for s in gold:                                # corrupt relations and heads, keeping valid trees
        heads, rels = list(s.heads), list(s.deprels)
        for j in range(s.n):
            if heads[j] != 0 and rng.random() < 0.33:
                rels[j] = "dep" if rels[j] != "dep" else "obj"
            if heads[j] != 0 and rng.random() < 0.2:
                cand = heads[j] + 1
                trial = heads.copy()
                trial[j] = cand
                if 1 <= cand <= s.n and not m.tree_issues(trial, rels):
                    heads = trial
        y.append({"head": heads, "deprel": rels})
        sys_sents.append(dataclasses.replace(s, heads=tuple(heads), deprels=tuple(rels)))
    gp, sp = tmp_path / "gold.conllu", tmp_path / "sys.conllu"
    gp.write_text(m.to_conllu(gold), encoding="utf-8")
    sp.write_text(m.to_conllu(sys_sents), encoding="utf-8")
    ev = off.evaluate(off.load_conllu_file(str(gp)), off.load_conllu_file(str(sp)))
    res = e.evaluate(y, None)
    assert res.primary == pytest.approx(ev["LAS"].f1, abs=1e-12)
    assert res.metrics["uas"] == pytest.approx(ev["UAS"].f1, abs=1e-12)
    assert res.primary < 1.0


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


def test_executor_pipeline_chain_parser(adapter, episodes, tmp_path):
    e = episodes["id"][1]
    code = ("def run(inputs, config):\n"
            "    out = []\n"
            "    for s in inputs['sentences']:\n"
            "        n = s['n_words']\n"
            "        out.append({'head': [i + 1 if i < n else 0 for i in range(1, n + 1)],\n"
            "                    'deprel': ['dep'] * (n - 1) + ['root']})\n"
            "    return {'y': out}\n")
    y = _run_pipeline(e, tmp_path, [("load_eval_inputs", "sentences", "sentences")], code,
                      {"sentences": {"type": "list"}}, {"y": {"type": "list", "shape": ["n"]}})
    res = e.evaluate(y, None)
    assert res.hard_ok() and res.primary is not None and res.z == 0


def test_objective_documents_the_domain_library(adapter, episodes):
    obj = episodes["src"][0].objective
    assert "scilib.udparse" in obj and "fit_predict" in obj and "las_uas" in obj
    # factual interface text only: no reference value, margin or recommendation is spelled out
    low = obj.lower()
    assert "0.15" not in obj and "recommend" not in low and "best recipe" not in low


def test_task_card_exists():
    assert (Path(__file__).resolve().parents[1] / "docs" / "tasks" / "FoR47.md").is_file()
