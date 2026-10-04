"""FoR45 AmericasNLP 2026 captioning adapter tests (skipped when the data are not available)."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from scienceclaw.bench.splits import split_seed
from scienceclaw.bench.tasks import for45_americasnlp as m

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]
SEED = 20260928


def test_chrf_and_descriptors():
    assert m.chrf_pp("the cat sat", "the cat sat") == pytest.approx(100.0)
    assert m.chrf_pp("", "abc") == 0.0
    from sacrebleu.metrics import CHRF
    assert m.chrf_pp("a cat sat on", "the cat sat") == pytest.approx(
        CHRF(word_order=2).sentence_score("a cat sat on", ["the cat sat"]).score)
    imgs = np.random.default_rng(0).integers(0, 256, size=(3, 64, 64, 3), dtype=np.uint8)
    X, names = m.image_descriptors(imgs, "all")
    assert X.shape == (3, len(names)) and np.all(np.isfinite(X))
    with pytest.raises(ValueError):
        m.image_descriptors(imgs[..., 0], "all")
    med = m.medoid_captions(["aa bb", "aa bb cc", "zz"], ["x", "x", "y"])
    assert med["y"] == "zz" and med["x"] in ("aa bb", "aa bb cc")


def test_image_knn_captions_uses_only_visible_same_language_rows():
    """The retrieval route is deterministic and cannot read an evaluation caption."""
    rng = np.random.default_rng(4)
    train_images = rng.integers(0, 256, size=(3, 64, 64, 3), dtype=np.uint8)
    train = [
        {"iso_lang": "x", "caption": "x nearest", "has_image": True},
        {"iso_lang": "y", "caption": "y nearest", "has_image": True},
        {"iso_lang": "x", "caption": "x caption only", "has_image": False},
    ]
    items = [{"iso_lang": "x"}, {"iso_lang": "y"}, {"iso_lang": "z"}]
    eval_images = np.stack([train_images[0], train_images[1], train_images[2]])
    got = m.image_knn_captions(train, train_images, items, eval_images, k=1, kind="gray_thumb")
    assert got[:2] == ["x nearest", "y nearest"]
    # No visible row for z: the helper has an explicit empty fallback and never inspects eval labels.
    assert got[2] == ""
    with pytest.raises(ValueError):
        m.image_knn_captions(train, train_images, items, eval_images, k=0)


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
        assert all(e.n_items == adapter.max_items for e in episodes[s])
        assert all(e.lineage["items_per_episode_capped"] for e in episodes[s])
    for a, b in combinations(items, 2):
        assert not set(items[a]) & set(items[b]), (a, b)
    img = {s: {g for e in eps for g in e.lineage["image_groups"]} for s, eps in episodes.items()}
    for a, b in combinations(img, 2):
        assert not img[a] & img[b], (a, b)
    iid_langs = {lang for s in ("src", "val", "id") for e in episodes[s] for lang in e.lineage["languages"]}
    ood_langs = {lang for e in episodes["ood"] for lang in e.lineage["languages"]}
    assert iid_langs == {"bzd", "grn", "yua"} and ood_langs == {"nlv", "hch"}
    again = m.Adapter().build_episodes("ood", 4, split_seed(SEED, m.CODE, "ood"))
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["ood"]]


def test_visible_data_disjoint_and_no_label_leak(adapter, episodes):
    rows = adapter._data.get().rows
    for s, eps in episodes.items():
        for e in eps[:2]:
            ev_caps = {rows[u].caption for u in e.lineage["item_ids"]}
            out = {t.name: t.fn({}, {}) for t in e.tools if t.name in ("load_train", "load_dev_inputs", "load_eval_inputs")}
            train_caps = {r["caption"] for r in out["load_train"]["train"]}
            assert not train_caps & ev_caps
            blob = json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith("images")} for k, v in out.items()},
                              ensure_ascii=False)
            for c in ev_caps:
                assert c not in blob, (s, c)
            for r in out["load_eval_inputs"]["items"] + out["load_dev_inputs"]["dev_items"]:
                assert "caption" not in r and "id" not in r
            assert out["load_eval_inputs"]["images"].shape == (e.n_items, 64, 64, 3)
            assert not set(e.lineage["dev_item_ids"]) & set(e.lineage["item_ids"])
            # the evaluation thumbnails are not among the visible training thumbnails
            tr_imgs = out["load_train"]["train_images"]
            for im in out["load_eval_inputs"]["images"]:
                assert not np.any(np.all(tr_imgs == im[None], axis=(1, 2, 3)))


def test_evaluator_reference_oracle_and_constraints(adapter, episodes):
    rows = adapter._data.get().rows
    e = episodes["id"][1]
    gold = [rows[u].caption for u in e.lineage["item_ids"]]
    orc = e.evaluate(gold, None)
    assert orc.primary == pytest.approx(100.0) and orc.accepted and orc.z == 1
    ref = orc.details["reference"]
    assert 0.0 < ref < 60.0
    # the reference predictions (per-language medoid captions) score exactly the reference
    tr = e.tool("load_train").fn({}, {})["train"]
    items = e.tool("load_eval_inputs").fn({}, {})["items"]
    lang_dir = {r.iso: r.lang_dir for r in rows.values()}
    med = m.medoid_captions([r["caption"] for r in tr], [lang_dir[r["iso_lang"]] for r in tr])
    ref_y = [med[lang_dir[it["iso_lang"]]] for it in items]
    rr = e.evaluate(ref_y, None)
    assert rr.primary == pytest.approx(ref) and rr.details["norm_score"] == pytest.approx(1.0)
    assert not rr.accepted and rr.z == 0
    bad = e.evaluate(gold[:-1], None)
    assert not bad.h["output_length"] and bad.primary is None
    empty = e.evaluate(gold[:-1] + [""], None)
    assert not empty.h["captions_valid"] and empty.z == 0
    nonstr = e.evaluate(gold[:-1] + [3], None)
    assert not nonstr.h["captions_valid"] and nonstr.primary is None
    pooled = adapter.pooled_metric([orc.details["pooled_payload"], bad.details["pooled_payload"]])
    assert pooled == pytest.approx((100.0 + ref) / 2)


def test_image_blind_diagnostic_does_not_change_score(adapter, episodes):
    rows = adapter._data.get().rows
    e = episodes["id"][1]
    gold = [rows[u].caption for u in e.lineage["item_ids"]]
    items = e.tool("load_eval_inputs").fn({}, {})["items"]
    d = e.evaluate(gold, None).details["diagnostics"]
    assert d["n_distinct_captions"] == len(set(c.strip() for c in gold)) and not d["image_blind"]
    const = {}
    y = [const.setdefault(it["iso_lang"], f"word{len(const)} tokens for {it['iso_lang']}") for it in items]
    r = e.evaluate(y, None)
    dd = r.details["diagnostics"]
    assert dd["image_blind"] and dd["n_distinct_captions"] == len(const)
    assert all(v == 1 for v in dd["distinct_captions_per_language"].values())
    assert dd["distinct_caption_ratio"] == pytest.approx(len(const) / len(items))
    assert r.primary is not None and r.details["norm_score"] is not None
    knn = e.tool("image_knn_reference").fn({}, {})
    assert len(knn["captions"]) == e.n_items and all(isinstance(x, str) for x in knn["captions"])
    assert 0.0 <= dd["image_knn_reference_mean_chrf_pp"] <= 100.0
    assert dd["image_knn_reference_k"] == 1 and dd["image_knn_reference_feature"] == "all"


def test_full_split_reference_is_pooled_and_diagnostic_only(adapter):
    """Full-pool references use the complete visible partition without changing episode scoring."""
    for split in ("src", "val", "id", "ood"):
        out = adapter.full_split_reference(split)
        assert out["split"] == split and out["diagnostic_only"]
        assert out["n_items"] == sum(len(v) for v in adapter._data.get().eval_split[split].values())
        assert out["n_items"] > 0 and out["n_train"] > 0
        assert 0.0 <= out["medoid_reference_chrf_pp"] <= 100.0
        assert 0.0 <= out["image_knn_reference_chrf_pp"] <= 100.0
        assert set(out["per_language"]) == set(adapter._data.get().eval_split[split])
        assert all(v["n_items"] > 0 for v in out["per_language"].values())


def test_full_split_reference_cli_has_no_label_output_surface():
    driver = Path(__file__).resolve().parents[1] / "scripts" / "f45_full_split_reference.py"
    text = driver.read_text(encoding="utf-8")
    assert "full_split_reference" in text and "target labels" in text


def test_clip_medoid_alias_forces_route_parameters(adapter, episodes, monkeypatch):
    """The controlled comparison tool cannot silently fall back to nearest/k=1."""
    e = episodes["id"][0]
    calls = []

    def fake_retrieve(*args, **kwargs):
        calls.append(kwargs)
        return ["visible caption"] * e.n_items

    monkeypatch.setattr(m.clip_retrieval, "retrieve_captions", fake_retrieve)
    monkeypatch.setattr(m.clip_retrieval, "provenance", lambda model: {"model": model, "frozen": True})
    tool = e.tool("clip_knn_reference_medoid")
    assert tool is not None
    out = tool.fn({}, {"model": "open_clip_vit_b32", "k": 1, "selection": "nearest"})
    assert out["k"] == 3 and out["selection"] == "caption_medoid"
    assert calls and calls[0]["k"] == 3 and calls[0]["selection"] == "caption_medoid"
    # The retrieval helper looks up fallbacks by policy-visible ISO code, not
    # the adapter's internal language-directory name.
    train = e.tool("load_train").fn({}, {})["train"]
    assert set(calls[0]["fallback"]) == {r["iso_lang"] for r in train}


def test_score_dev_and_features(adapter, episodes):
    e = episodes["src"][0]
    dev = e.tool("load_dev_inputs").fn({}, {})
    n = len(dev["dev_items"])
    assert n > 0
    r = e.tool("score_dev").fn({"dev_captions": ["x"] * n}, {})
    assert 0.0 <= r["dev_mean_chrf_pp"] <= 100.0 and r["dev_reference_mean_chrf_pp"] > 0
    with pytest.raises(ValueError):
        e.tool("score_dev").fn({"dev_captions": ["x"] * (n + 1)}, {})
    f = e.tool("image_features").fn({"images": dev["dev_images"]}, {"kind": "color_hist"})
    assert f["X"].shape == (n, 64)


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


def test_executor_pipeline_language_caption(adapter, episodes, tmp_path):
    e = episodes["ood"][0]
    code = ("def run(inputs, config):\n"
            "    first = {}\n"
            "    for r in inputs['train']:\n"
            "        first.setdefault(r['iso_lang'], r['caption'])\n"
            "    return {'y': [first[it['iso_lang']] for it in inputs['items']]}\n")
    y = _run_pipeline(e, tmp_path, [("load_train", "train", "train"), ("load_eval_inputs", "items", "items")], code,
                      {"train": {"type": "list"}, "items": {"type": "list"}}, {"y": {"type": "list", "shape": ["n"]}})
    res = e.evaluate(y, None)
    assert res.completed and res.hard_ok() and res.primary is not None and res.primary > 0


def test_objective_documents_the_domain_library_and_the_visible_counts(adapter, episodes):
    for split in ("src", "ood"):
        e = episodes[split][0]
        rows = e.tool("load_train").fn({}, {})["train"]
        assert "scilib.captions" in e.objective and "fit_predict" in e.objective and "loo_score" in e.objective
        assert f"has_image is true for {sum(r['has_image'] for r in rows)} of the {len(rows)} rows" in e.objective
        for iso in {r["iso_lang"] for r in rows}:
            got = sum(r["has_image"] for r in rows if r["iso_lang"] == iso)
            assert f"{iso} {got}/{sum(r['iso_lang'] == iso for r in rows)}" in e.objective
        assert f"covers only {e.lineage['n_dev']} items" in e.objective and "noisy" in e.tool("score_dev").description
        assert "clip_retrieval_status" in e.objective and "wire its captions directly to submit.y" in e.objective
        assert "medoid" not in e.tool("load_train").description.lower() and "margin" not in e.objective.lower()


def test_task_card_exists():
    assert (Path(__file__).resolve().parents[1] / "docs" / "tasks" / "FoR45.md").is_file()
