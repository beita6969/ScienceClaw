"""FoR32 MSD Task04 Hippocampus adapter tests (skipped when the data are not available)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from scienceclaw.bench.tasks import for32_msd_hippocampus as m

SPLITS = [("src", 7), ("val", 2), ("id", 4), ("ood", 4)]


def test_dice_and_surface_dice_known_answers():
    a = np.zeros((10, 10, 10), bool)
    b = np.zeros((10, 10, 10), bool)
    a[2:6, 2:6, 2:6] = True
    b[2:6, 2:6, 4:8] = True
    assert m.dice(a, b) == pytest.approx(2 * 32 / 128)
    assert m.dice(a, a) == 1.0 and m.dice(np.zeros_like(a), np.zeros_like(a)) == 1.0
    assert m.dice(a, np.zeros_like(a)) == 0.0
    assert m.surface_dice(a, a, (1.0, 1.0, 1.0)) == 1.0
    assert 0.0 < m.surface_dice(a, b, (1.0, 1.0, 1.0)) < 1.0


def test_atlas_roundtrip():
    lab = np.zeros((20, 30, 20), np.uint8)
    lab[5:15, 5:15, 5:15] = 1
    lab[5:15, 15:25, 5:15] = 2
    atlas = m.build_atlas([lab, lab])
    pred = m.atlas_predict(atlas, lab.shape)
    assert pred.shape == lab.shape and m.dice(pred == 1, lab == 1) > 0.9


@pytest.fixture(scope="module")
def adapter():
    a = m.Adapter()
    ok, why = a.available()
    if not ok:
        pytest.skip(why)
    return a


@pytest.fixture(scope="module")
def episodes(adapter):
    return {s: adapter.build_episodes(s, n, seed=3) for s, n in SPLITS}


def test_splits_subject_disjoint_and_deterministic(adapter, episodes):
    idx, pools = adapter._index(), adapter._pools()
    items = {s: {i for e in eps for i in e.lineage["item_ids"]} for s, eps in episodes.items()}
    items["train"], items["dev"] = set(pools["train"]), set(pools["dev"])
    subj = {s: {idx[c]["subject"] for c in v} for s, v in items.items()}
    for a, b in combinations(items, 2):
        assert not items[a] & items[b], (a, b)
        assert not subj[a] & subj[b], (a, b)
    for s, eps in episodes.items():
        for e in eps:
            ids = e.lineage["item_ids"]
            assert len(ids) == len(set(ids)) == 16
            encs = {idx[c]["encoding"] for c in ids}
            assert (encs <= {"uint8_0_255", "float_1e5_scale"}) if s == "ood" else encs == {"float_1e3_scale"}
    assert episodes["src"][0].lineage["reused_items"]          # 7 x 16 > 40 source volumes
    assert not any(e.lineage["reused_items"] for e in episodes["id"] + episodes["ood"])
    again = adapter.build_episodes("ood", 4, seed=3)
    assert [e.lineage["item_ids"] for e in again] == [e.lineage["item_ids"] for e in episodes["ood"]]


def test_tools_expose_only_visible_data(adapter, episodes):
    e = episodes["id"][1]
    ev = e.tool("load_eval_inputs").fn({}, {})
    assert set(ev) == {"images", "case_ids", "spacing_mm"}
    assert ev["case_ids"] == e.lineage["item_ids"]
    for c, img in zip(ev["case_ids"], ev["images"]):
        vol, lab, _ = adapter._volume(c)
        assert img.dtype == np.float32 and img.shape == lab.shape and np.array_equal(img, vol)
        assert not np.array_equal(img, lab)
    tr = e.tool("load_train").fn({}, {})
    assert not set(tr["case_ids"]) & set(ev["case_ids"])
    assert len(tr["images"]) == len(tr["labels"]) == 28
    dv = e.tool("load_dev_inputs").fn({}, {})
    assert set(dv) == {"images", "case_ids", "spacing_mm"}
    s = e.tool("score_dev").fn({"predictions": [adapter._volume(c)[1] for c in dv["case_ids"]]}, {})
    assert s["mean_dsc"] == pytest.approx(1.0)


def test_evaluator_reference_oracle_malformed(adapter, episodes):
    details = []
    for e in (episodes["id"][0], episodes["ood"][0]):
        cases = e.lineage["item_ids"]
        shapes = [adapter._volume(c)[1].shape for c in cases]
        r = e.evaluate([m.atlas_predict(adapter._atlas(), s) for s in shapes], None)
        assert r.hard_ok() and not r.accepted and r.z == 0
        assert r.primary == pytest.approx(r.details["reference"]) and r.details["norm_score"] == pytest.approx(1.0)
        assert 0.3 < r.primary < 0.95
        ro = e.evaluate([adapter._volume(c)[1] for c in cases], None)
        assert ro.primary == pytest.approx(1.0) and ro.accepted and ro.z == 1
        details.append(ro.details)
        bad = e.evaluate([np.zeros((2, 2, 2), np.uint8)] * 16, None)
        assert not bad.h["output_structure"] and bad.z == 0 and bad.primary is None
        bad2 = e.evaluate([np.full(s, 3, np.uint8) for s in shapes], None)
        assert bad2.h["output_structure"] and not bad2.h["label_values"] and bad2.z == 0
    assert adapter.pooled_metric(details) == pytest.approx(1.0)


def test_objective_documents_the_domain_library(adapter, episodes):
    ep = episodes["val"][0]
    assert "scilib.hippo_unet.fit_predict" in ep.objective and "no-self-training policy" in ep.objective
    assert "manifest marks FoR32 blocked" in ep.objective
    assert "stop without a formal submit" in ep.objective
