"""FoR30 PhenoBench hierarchical panoptic segmentation adapter tests (data team's ``reconstructed_v2`` role files).

Skipped when the data are not available. The optional cross-check against the pinned official scorer runs only when
``SCIENCECLAW_OFFICIAL_SCORERS=1`` and the data team's ``environments/vision`` interpreter (torch + torchmetrics
0.10.3 + phenobench@0edc128) exists.
"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from scilib import phenobench_weyler, phenoseg_hapt, phenoseg_m2f
from scienceclaw.bench.tasks import for30_phenobench as m
from scienceclaw.bench.tasks._delivery_roles import DeliveryError

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]
ROLE_IMAGES = {"src": 32, "val": 32, "id": 64, "ood": 64}
ROLE_CAPTURES = {"src": 3, "val": 2, "id": 5, "ood": 3}
VISION_PY = Path("/Users/admin/Datasets/ScienceClaw-rebuild-20260928/environments/vision/bin/python")


# ----------------------------------------------------------------------------------------------------------------
# metric (no data needed)
# ----------------------------------------------------------------------------------------------------------------
def test_pq_single_class_known_answer():
    gt = np.zeros((10, 10), int)
    gt[0:5, 0:5] = 1              # 25 px
    gt[6:10, 6:10] = 2            # 16 px
    pred = np.zeros((10, 10), int)
    pred[0:5, 0:4] = 7            # IoU 20/25 = 0.8 with gt 1
    pred[6:8, 0:2] = 9            # false positive
    assert m.pq_single_class(pred, gt) == pytest.approx(0.8 / (1 + 0.5 + 0.5))
    assert m.pq_single_class(gt, gt) == pytest.approx(1.0)


def test_filter_partial_masks_and_iou():
    gt_inst = np.zeros((6, 6), int)
    gt_inst[0:3, 0:3] = 1
    gt_inst[3:6, 3:6] = 2
    vis = np.where(gt_inst == 2, 0.2, 1.0)
    gt_sem = (gt_inst > 0).astype(int)
    pred_inst = gt_inst.copy()
    pred_sem = gt_sem.copy()
    m.filter_partial_masks(pred_inst, pred_sem, gt_inst, gt_sem, vis)
    assert gt_sem[4, 4] == 0 and pred_sem[4, 4] == 0 and gt_sem[0, 0] == 1 and pred_sem[0, 0] == 1
    cm = m.semantic_confusion(np.array([0, 3, 4, 1]), np.array([0, 1, 2, 1]))
    assert cm.tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    assert m.iou_from_confusion(np.zeros((3, 3))).tolist() == [0.0, 0.0, 0.0]


def _rec(id_: str, role: str, group: str, crops: tuple[int, ...], date: str = "05-15") -> m.Rec:
    return m.Rec(id_, id_.split("/")[-1], role, "train", date, group, crops, {}, {}, "fp")


def test_check_roles_rejects_lineage_overlap():
    ok = {"src": [_rec("train/a", "source", "P1", (1,))], "val": [_rec("train/b", "val", "P2", (2,))],
          "id": [_rec("val/c", "id", "P3", (3,))], "ood": [_rec("val/d", "ood", "P4", (4,), "06-05")]}
    m.check_roles(ok)
    for what, split, rec in (("image", "val", _rec("train/a", "val", "P2", (2,))),
                             ("capture id", "id", _rec("val/c", "id", "P1", (3,))),
                             ("crop lineage id", "ood", _rec("val/d", "ood", "P4", (1, 4), "06-05")),
                             ("dates", "ood", _rec("val/d", "ood", "P4", (4,), "05-15"))):
        bad = dict(ok, **{split: [rec]})
        with pytest.raises(DeliveryError, match=what if what != "dates" else "dates"):
            m.check_roles(bad)
    with pytest.raises(DeliveryError, match="no images"):
        m.check_roles(dict(ok, val=[]))


# ----------------------------------------------------------------------------------------------------------------
# adapter on the delivered data
# ----------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def adapter():
    a = m.Adapter()
    ok, why = a.available()
    if not ok:
        pytest.skip(why)
    if a.delivery().version != "reconstructed_v2":
        pytest.skip(f"tests are pinned to reconstructed_v2 (catalog names {a.delivery().version})")
    return a


@pytest.fixture(scope="module")
def episodes(adapter):
    return {s: adapter.build_episodes(s, n, seed=5) for s, n in SPLITS}


def test_pools_are_the_role_files(adapter):
    ok, why = adapter.available()
    assert ok and "reconstructed_v2" in why and "src=32" in why and "captures 3/2/5/3" in why
    pools = adapter._pools()
    assert {k: len(v) for k, v in pools.items()} == ROLE_IMAGES
    roles = adapter._roles()
    assert {k: len({r.group for r in v}) for k, v in roles.items()} == ROLE_CAPTURES
    assert {r.date for r in roles["ood"]} == {"06-05"} and {r.date for s in ("src", "val", "id") for r in roles[s]} == {"05-15"}
    assert {r.official_split for r in roles["src"] + roles["val"]} == {"train"}
    assert {r.official_split for r in roles["id"]} == {"val"}
    dl = adapter.delivery()
    assert dl.base.name == "reconstructed_v2" and dl.base.is_dir()
    for split, role in m.ROLE_OF_SPLIT.items():                    # exactly the frozen role rows
        assert sorted(r["unit_id"] for r in dl.rows(role)) == pools[split]
    adapter.verify_disjoint()
    # image, capture and persistent-crop overlaps are all zero between any two roles
    for a, b in combinations(roles, 2):
        assert not {r.id for r in roles[a]} & {r.id for r in roles[b]}
        assert not {r.group for r in roles[a]} & {r.group for r in roles[b]}
        assert not {c for r in roles[a] for c in r.crops} & {c for r in roles[b] for c in r.crops}
    desc = adapter.describe_pools()
    assert desc["src"]["captures"] == {"P0030692": 11, "P0030855": 8, "P0030947": 13}


def test_role_files_and_pngs_match_their_sha256(adapter):
    assert adapter.verify_files() == 6 * sum(ROLE_IMAGES.values())
    rec = adapter._roles()["src"][0]
    bad = dataclasses.replace(rec, sha256={**rec.sha256, "images": "0" * 64})
    with pytest.raises(DeliveryError, match="sha256"):
        adapter._check_hashes(bad)


def test_splits_disjoint_and_deterministic(adapter, episodes):
    items = {s: {i for e in eps for i in e.lineage["item_ids"]} for s, eps in episodes.items()}
    caps = {s: {adapter._record(i).group for i in v} for s, v in items.items()}
    for a, b in combinations(items, 2):
        assert not items[a] & items[b], (a, b)
        assert not caps[a] & caps[b], (a, b)                    # captures never straddle pools
    for s, eps in episodes.items():
        assert len(eps) == dict(SPLITS)[s]
        for e in eps:
            dates = set(e.lineage["capture_dates"])
            assert dates == ({"06-05"} if s == "ood" else {"05-15"})
            assert len(set(e.lineage["item_ids"])) == e.n_items and e.lineage["items_requested"] == 16
            assert e.lineage["ood_kind"] == ("proxy_within_dataset" if s == "ood" else None)
            assert e.lineage["data_version"] == "reconstructed_v2" and e.lineage["role"] == m.ROLE_OF_SPLIT[s]
            assert e.lineage["item_ids"] and set(e.lineage["item_ids"]) <= set(adapter._pools()[s])
    for s in ("val", "id", "ood"):                              # held-out splits: 16 images each, never reused
        assert [e.n_items for e in episodes[s]] == [16] * dict(SPLITS)[s]
        assert not any(e.lineage["reused_items"] for e in episodes[s])
    again = adapter.build_episodes("id", 4, seed=5)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["id"]]
    assert adapter.build_episodes("id", 4, seed=6)[0].lineage["item_ids"] != episodes["id"][0].lineage["item_ids"]
    assert [e.lineage["item_ids"] for e in adapter.build_episodes("id", 2, seed=5)] == \
        [e.lineage["item_ids"] for e in episodes["id"][:2]]      # prefix-stable


def test_src_episodes_are_single_capture_and_leave_other_captures_visible(adapter, episodes):
    groups = adapter._source_groups()
    assert {g: len(v) for g, v in groups.items()} == {"P0030692": 11, "P0030855": 8, "P0030947": 13}
    seen = set()
    for e in episodes["src"]:
        caps = e.lineage["capture_groups"]
        assert len(caps) == 1 and e.n_items == len(groups[caps[0]])
        seen.add(caps[0])
        assert not set(e.lineage["visible_train_groups"]) & set(caps)
        assert not set(e.lineage["dev_groups"]) & (set(caps) | set(e.lineage["visible_train_groups"]))
        assert e.lineage["n_visible_train"] > 0 and e.lineage["n_dev"] > 0
    assert seen == set(groups) and episodes["src"][0].lineage["reused_items"]
    rep = adapter.build_episodes("rep", 3, seed=5)                 # rep = frozen copy of the src composition
    assert [e.lineage["item_ids"] for e in rep] == [e.lineage["item_ids"] for e in episodes["src"][:3]]
    assert all(e.split == "rep" and e.lineage["role"] == "source" for e in rep)
    small = adapter.build_episodes("src", 3, seed=5, items_per_episode=4)
    assert [e.n_items for e in small] == [4, 4, 4] and all(len(e.lineage["capture_groups"]) == 1 for e in small)


def test_held_out_episodes_shrink_and_record_the_request(adapter):
    eps = adapter.build_episodes("id", 5, seed=1)                  # 5 x 16 > 64 -> 64 // 5 = 12 images each
    assert [e.n_items for e in eps] == [12] * 5 and {e.lineage["items_requested"] for e in eps} == {16}
    assert len({i for e in eps for i in e.lineage["item_ids"]}) == 60
    with pytest.raises(ValueError):
        adapter.build_episodes("val", 33, seed=1)
    with pytest.raises(ValueError):
        adapter.build_episodes("nope", 1, seed=1)


def test_visible_labelled_data_never_overlaps_the_episode(adapter, episodes):
    src_ids = set(adapter._pools()["src"])
    for s, eps in episodes.items():
        for e in eps:
            train, dev = adapter._visible(e.lineage["item_ids"])
            assert set(train) <= src_ids and set(dev) <= src_ids and not set(train) & set(dev)
            assert not (set(train) | set(dev)) & set(e.lineage["item_ids"])
            cap = lambda ids: {adapter._record(i).group for i in ids}     # noqa: E731
            assert not cap(train) & cap(e.lineage["item_ids"]) and not cap(dev) & cap(e.lineage["item_ids"])
            assert not cap(train) & cap(dev)
            assert e.lineage["n_visible_train"] == len(train) and e.lineage["n_dev"] == len(dev)
        if s != "src":                                       # held-out: all three source captures are free
            assert (len(train), len(dev)) == (24, 8)
    names = {t.name for t in episodes["id"][0].tools}
    expected = {"load_train", "load_eval_inputs", "load_dev_inputs", "score_dev"}
    # A GPU/remote-host run additionally exposes the one-step frozen Mask2Former route.  CPU-only
    # test environments intentionally keep the original tool surface.
    if phenoseg_m2f.available():
        expected.add("predict_pretrained")
        expected.add("score_pretrained_dev")
    if phenoseg_hapt.available():
        expected.add("predict_hapt")
    if phenobench_weyler.available():
        expected.add("predict_weyler")
    assert names == expected


def test_pretrained_tool_is_label_free_and_returns_submit_shape(adapter, episodes, monkeypatch):
    """The optional specialist route consumes only episode RGB and returns the required y dict."""
    ep = episodes["id"][0]
    assert "`y` output" in ep.objective and "submitted directly" in ep.objective
    tool = ep.tool("predict_pretrained")
    if tool is None:
        pytest.skip("frozen Mask2Former worker is not available in this environment")
    calls = {}

    def fake_predict(images, **kwargs):
        calls["shape"] = np.asarray(images).shape
        calls["kwargs"] = kwargs
        z = np.zeros((ep.n_items, ep.size, ep.size), dtype=np.int32)
        return {"semantics": z, "plant_instances": z, "leaf_instances": z}

    monkeypatch.setattr(phenoseg_m2f, "predict_panoptic", fake_predict)
    out = tool.fn({}, {})
    assert set(out) == {"y"} and set(out["y"]) == {"semantics", "plant_instances", "leaf_instances"}
    assert calls["shape"] == (ep.n_items, ep.size, ep.size, 3)
    assert calls["kwargs"] == {"plant_threshold": 0.8, "leaf_threshold": 0.8, "batch_size": 4}


def test_weyler_tool_is_label_free_and_returns_submit_shape(adapter, episodes, monkeypatch):
    ep = episodes["id"][0]
    tool = ep.tool("predict_weyler")
    if tool is None:
        pytest.skip("frozen Weyler checkpoint is not available in this environment")
    calls = {}

    def fake_predict(images, batch_size=1):
        calls["shape"] = np.asarray(images).shape
        z = np.zeros((ep.n_items, ep.size, ep.size), dtype=np.int32)
        return {"semantics": z, "plant_instances": z, "leaf_instances": z}

    monkeypatch.setattr(phenobench_weyler, "predict_panoptic", fake_predict)
    out = tool.fn({}, {})
    assert set(out) == {"y"} and set(out["y"]) == {"semantics", "plant_instances", "leaf_instances"}
    assert calls["shape"] == (ep.n_items, ep.size, ep.size, 3)


def test_hapt_tool_is_label_free_and_returns_submit_shape(adapter, episodes, monkeypatch):
    ep = episodes["id"][0]
    tool = ep.tool("predict_hapt")
    if tool is None:
        pytest.skip("frozen HAPT checkpoint is not available in this environment")
    calls = {}

    def fake_predict(images):
        calls["shape"] = np.asarray(images).shape
        z = np.zeros((ep.n_items, ep.size, ep.size), dtype=np.int32)
        return {"semantics": z, "plant_instances": z, "leaf_instances": z}

    monkeypatch.setattr(phenoseg_hapt, "predict_panoptic", fake_predict)
    out = tool.fn({}, {})
    assert set(out) == {"y"} and set(out["y"]) == {"semantics", "plant_instances", "leaf_instances"}
    assert calls["shape"] == (ep.n_items, ep.size, ep.size, 3)


def test_tools_expose_only_visible_data(adapter, episodes):
    e = episodes["id"][0]
    ev = e.tool("load_eval_inputs").fn({}, {})
    assert set(ev) == {"images", "names"} and ev["images"].shape == (16, 512, 512, 3) and ev["images"].dtype == np.uint8
    tr = e.tool("load_train").fn({}, {})
    assert tr["semantics"].shape == (24, 512, 512) and float(tr["plant_visibility"].max()) <= 1.0
    assert not set(tr["names"]) & set(ev["names"])
    dv = e.tool("load_dev_inputs").fn({}, {})
    assert set(dv) == {"images", "names"} and dv["images"].shape == (8, 512, 512, 3)
    for name in ev["names"] + tr["names"] + dv["names"]:            # opaque handles, no file names / capture ids
        assert re.fullmatch(r"ph-[0-9a-f]{8}", name)
    assert len(set(ev["names"] + tr["names"] + dv["names"])) == 16 + 24 + 8
    _, dev = adapter._visible(e.lineage["item_ids"])
    gts = [adapter._gt(i) for i in dev]
    oracle = {k: np.stack([g[k] for g in gts]) for k in ("semantics", "plant_instances", "leaf_instances")}
    s = e.tool("score_dev").fn({"prediction": oracle}, {})
    assert s["pq_plus"] == pytest.approx(100.0)
    with pytest.raises(ValueError, match="score_dev"):
        e.tool("score_dev").fn({"prediction": {"semantics": np.zeros((8, 4, 4), int)}}, {})
    assert "P00" not in e.objective and "05-15" not in e.objective


def test_objective_documents_the_domain_library(adapter, episodes):
    ep = episodes["val"][0]
    assert "scilib.phenoseg" in ep.objective and "fit_predict" in ep.objective and "pq_plus" in ep.objective
    if ep.tool("predict_hapt") is not None:
        assert "diagnostic-only" in ep.objective and "do not submit" in ep.objective


def test_rgb_only_mode(adapter):
    a = m.Adapter(expose_source_labels=False)
    eps = a.build_episodes("ood", 2, seed=3)
    assert [{t.name for t in e.tools} for e in eps] == [{"load_eval_inputs"}] * 2
    assert all(e.lineage["source_labels_exposed"] is False and e.lineage["n_visible_train"] == 0 for e in eps)
    src = a.build_episodes("src", 3, seed=3)
    assert [e.lineage["item_ids"] for e in src] == [e.lineage["item_ids"] for e in adapter.build_episodes("src", 3, seed=3)]
    assert "load_train" not in src[0].objective


def test_evaluator_reference_oracle_malformed(adapter, episodes):
    details = []
    for e in (episodes["id"][1], episodes["ood"][0]):
        items = e.lineage["item_ids"]
        r = e.evaluate(adapter.reference_prediction(items), None)
        assert r.hard_ok() and not r.accepted and r.z == 0
        assert r.primary == pytest.approx(r.details["reference"]) and 5.0 < r.primary < 80.0
        assert 0.0 < r.metrics["mean_image_pq_plus"] <= 100.0
        gts = [adapter._gt(i) for i in items]
        oracle = {k: np.stack([g[k] for g in gts]) for k in ("semantics", "plant_instances", "leaf_instances")}
        ro = e.evaluate(oracle, None)
        assert ro.primary == pytest.approx(100.0) and ro.accepted and ro.z == 1
        details.append(ro.details)
    e = episodes["id"][1]
    bad = e.evaluate({"semantics": np.zeros((16, 8, 8), int)}, None)
    assert not bad.h["output_structure"] and bad.z == 0 and bad.primary is None
    z = np.zeros((16, 512, 512), int)
    bad2 = e.evaluate({"semantics": z + 7, "plant_instances": z, "leaf_instances": z}, None)
    assert bad2.h["output_structure"] and not bad2.h["label_values"] and bad2.z == 0
    assert adapter.pooled_metric(details) == pytest.approx(100.0)


def test_src_reference_scores_through_evaluate(adapter, episodes):
    e = episodes["src"][1]                                       # the 8-image capture: cheapest src episode
    r = e.evaluate(adapter.reference_prediction(e.lineage["item_ids"]), None)
    assert r.hard_ok() and not r.accepted and r.primary == pytest.approx(r.details["reference"])
    assert e.lineage["items_requested"] == 16 and e.n_items == 8


def test_split_plan_integration(adapter):
    from scienceclaw.bench.splits import SplitPlan
    from scienceclaw.config import BenchConfig

    plan = SplitPlan.build(BenchConfig(disciplines=["FoR30"]), {"FoR30": adapter})   # no LineageOverlapError
    w = plan.manifest()["warnings"]
    assert not [x for x in w if "returned" in x]
    assert not [x for x in w if "reused" in x and not x.startswith("FoR30/src")]      # only src repeats images
    assert len(plan.episodes["src"]["FoR30"]) == 7 and len(plan.episodes["ood"]["FoR30"]) == 4
    assert [e.n_items for e in plan.episodes["ood"]["FoR30"]] == [16] * 4


def test_cache_is_keyed_by_the_row_hashes(adapter, tmp_path):
    a = m.Adapter(cache_dir=str(tmp_path))
    item = adapter._pools()["val"][0]
    x = a._load(item)
    cfile = tmp_path / "scale2" / (item.replace("/", "__") + ".npz")
    assert cfile.exists() and x["images"].shape == (512, 512, 3) and x["plant_visibility"].dtype == np.uint8
    with np.load(cfile) as z:
        assert str(z["fingerprint"]) == a._record(item).fingerprint
    cfile.write_bytes(b"not a zip")                               # unreadable entry is rebuilt, not trusted
    assert np.array_equal(a._load(item)["images"], x["images"])
    with np.load(cfile) as z:
        assert str(z["fingerprint"]) == a._record(item).fingerprint
    np.savez_compressed(cfile, src_bytes=np.int64(1), **x)        # v1-style entry (no fingerprint) is rebuilt
    assert np.array_equal(a._load(item)["semantics"], x["semantics"])
    with np.load(cfile) as z:
        assert "fingerprint" in z.files


def test_unavailable_reports_reason(tmp_path):
    ok, why = m.Adapter(data_root=str(tmp_path)).available()
    assert not ok and "missing" in why and why.startswith("FoR30")


def test_tampered_role_file_or_missing_png_is_rejected(adapter, tmp_path):
    dl = adapter.delivery()
    root = dl.delivery_root
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / dl.base.parent.name).symlink_to(dl.base.parent, target_is_directory=True)
    for src in [dl.catalog_path, *dl.role_files.values()]:
        dst = tmp_path / src.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    fake = m.Adapter(data_root=str(tmp_path / "datasets"), cache_dir=str(tmp_path / "cache"))
    ok, why = fake.available()
    assert ok, why                                                      # a faithful copy of the delivery is accepted
    assert fake.delivery().delivery_root == tmp_path
    with (tmp_path / dl.role_files["id"].relative_to(root)).open("a") as fh:
        fh.write("\n")
    ok, why = m.Adapter(data_root=str(tmp_path / "datasets")).available()
    assert not ok and "sha256" in why
    # a row that points at a PNG that is not there
    shutil.copy(dl.role_files["id"], tmp_path / dl.role_files["id"].relative_to(root))
    cat = tmp_path / dl.catalog_path.relative_to(root)
    text = cat.read_text().replace(dl.entry["partition_role_files"]["id"]["expected_sha256"], "")
    cat.write_text(text)                                                # empty expected sha256 = role file not hash-checked
    (tmp_path / dl.role_files["id"].relative_to(root)).write_text(
        "".join(l.replace("05-15_00053_P0030859.png", "05-15_99999_P0030859.png") if "00053_P0030859" in l else l
                for l in dl.role_files["id"].read_text().splitlines(keepends=True)))
    ok, why = m.Adapter(data_root=str(tmp_path / "datasets")).available()
    assert not ok and "download incomplete" in why and "99999" in why, why


@pytest.mark.skipif(os.environ.get("SCIENCECLAW_OFFICIAL_SCORERS") != "1" or not VISION_PY.exists(),
                    reason="set SCIENCECLAW_OFFICIAL_SCORERS=1 (needs environments/vision with the official scorer)")
def test_matches_official_scorer(adapter, tmp_path):
    from PIL import Image

    items = adapter._pools()["id"][:2] + adapter._pools()["ood"][:2]
    pred = adapter.reference_prediction(items)
    for j, i in enumerate(items):
        name, g = adapter._record(i).name, adapter._load(i)
        for k in m.FOLDERS:
            d = tmp_path / "gt" / "val" / k
            d.mkdir(parents=True, exist_ok=True)
            a = g[k]
            Image.fromarray(a if k in ("images", "plant_visibility", "leaf_visibility") else a.astype(np.uint16)).save(d / name)
        for k in pred:
            d = tmp_path / "pred" / k
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred[k][j].astype(np.uint16)).save(d / name)
    out = subprocess.run([str(VISION_PY), "-m", "phenobench.evaluation.phenobench_eval", "--task", "hierarchical",
                          "--phenobench_dir", str(tmp_path / "gt"), "--prediction_dir", str(tmp_path / "pred"),
                          "--split", "val"], capture_output=True, text=True, timeout=600,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}).stdout
    official = {k: float(re.search(rf"{re.escape(lbl)}\s*:\s*([0-9.]+)", out).group(1))
                for k, lbl in [("iou_soil", "IoU (soil)"), ("iou_weed", "IoU (weed)"), ("pq_crop", "PQ (crop)"),
                               ("pq_leaf", "PQ (leaf)"), ("pq_plus", "PQ+")]}
    ours = m.aggregate(adapter._stats(items, pred))
    for k, v in official.items():
        assert ours[k] == pytest.approx(v, abs=0.011), k
