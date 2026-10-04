"""scilib.phenoseg: metric parity with the adapter, interface text, sandbox import, fit / predict on synthetic scenes."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

import scilib
from scilib import phenoseg as ps
from scienceclaw.bench.tasks import for30_phenobench as m
from scienceclaw.runtime.integrity import scan_code

H = 96


def _scene(seed: int, n_crop: int = 3, n_weed: int = 4):
    """Brown soil, large dark-green discs (crop, each made of two overlapping 'leaf' discs) and small light-green weeds."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :H]
    img = np.zeros((H, H, 3), np.float32)
    img[:] = (0.45, 0.33, 0.24)
    sem = np.zeros((H, H), np.int16)
    plant = np.zeros((H, H), np.int32)
    leaf = np.zeros((H, H), np.int32)
    lid = 0
    for k in range(n_crop):
        cy, cx = 20 + 28 * (k % 3), 18 + 26 * ((k + seed) % 3)
        for j, (dy, dx) in enumerate([(-4, -4), (4, 4)]):
            d = (yy - cy - dy) ** 2 + (xx - cx - dx) ** 2 <= 8 ** 2
            lid += 1
            leaf[d & (leaf == 0)] = lid
        d = (leaf > 0) & (plant == 0) & (np.abs(yy - cy) < 14) & (np.abs(xx - cx) < 14)
        plant[d] = k + 1
        sem[d] = 1
    for k in range(n_weed):
        cy, cx = rng.integers(8, H - 8, 2)
        d = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 3 ** 2) & (sem == 0)
        sem[d] = 2
    img[sem == 1] = (0.10, 0.45, 0.10)
    img[sem == 2] = (0.35, 0.65, 0.15)
    img += rng.normal(0, 0.02, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), sem, plant, leaf


def _stack(seeds):
    sc = [_scene(s) for s in seeds]
    return tuple(np.stack(x) for x in zip(*sc))


# ---------------------------------------------------------------------------------------------------- interface
def test_describe_lists_every_public_name():
    text = scilib.describe("phenoseg")
    for name in ps.__all__:
        if name not in ("DEFAULT_PARAMS", "PixelModel"):
            assert name in text
    assert "fit_predict" in text and "panoptic_from_probs" in text


def test_code_node_may_import_scilib():
    assert scan_code("from scilib.phenoseg import fit_predict\n\ndef run(inputs, config):\n    return {}\n") == []


# ---------------------------------------------------------------------------------------------------- metric parity
def test_pq_plus_matches_the_adapter_metric():
    rng = np.random.default_rng(1)
    n = 3
    imgs, sem, plant, leaf = _stack([0, 1, 2])
    gt = dict(semantics=sem, plant_instances=plant, leaf_instances=leaf,
              plant_visibility=np.where(plant > 0, rng.choice([0.3, 1.0], plant.shape), 0.0),
              leaf_visibility=np.where(leaf > 0, rng.choice([0.4, 0.9], leaf.shape), 0.0))
    # a perturbed prediction: shifted semantics, merged / dropped instances
    pred = dict(semantics=np.roll(sem, 2, axis=1), plant_instances=np.roll(plant, 3, axis=2),
                leaf_instances=np.where(np.roll(leaf, 1, axis=1) % 3 == 0, 0, np.roll(leaf, 1, axis=1)))
    ours = ps.pq_plus(pred, gt)
    theirs = m.aggregate([m.hierarchical_image_stats({k: v[i] for k, v in pred.items()},
                                                     {k: v[i] for k, v in gt.items()}) for i in range(n)])
    for k in ("pq_plus", "iou_soil", "iou_weed", "pq_crop", "pq_leaf"):
        assert ours[k] == pytest.approx(theirs[k], abs=1e-6), k
    # identical prediction and ground truth
    perfect = ps.pq_plus(gt, gt)
    assert perfect["pq_plus"] == pytest.approx(100.0)


def test_pq_plus_visibility_keys_are_optional():
    _, sem, plant, leaf = _stack([3])
    gt = dict(semantics=sem, plant_instances=plant, leaf_instances=leaf)
    assert ps.pq_plus(gt, gt)["pq_plus"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------------------------------- instances
def test_split_instances_separates_touching_discs():
    yy, xx = np.mgrid[:60, :90]
    mask = ((yy - 30) ** 2 + (xx - 28) ** 2 <= 14 ** 2) | ((yy - 30) ** 2 + (xx - 46) ** 2 <= 11 ** 2)
    lab = ps.split_instances(mask, sigma=1.0, min_dist=6, min_area=20)
    assert lab.dtype == np.int32 and len(np.unique(lab[lab > 0])) == 2 and ((lab > 0) == mask).all()
    one = ps.split_instances(mask, sigma=1.0, min_dist=40, min_area=20)
    assert len(np.unique(one[one > 0])) == 1
    assert ps.split_instances(np.zeros((20, 20), bool), 1.0, 4, 5).max() == 0


def test_growth_scale_orders_by_plant_size_and_accepts_three_inputs():
    yy, xx = np.mgrid[:H, :H]
    small = (yy - 48) ** 2 + (xx - 48) ** 2 <= 8 ** 2
    large = (yy - 48) ** 2 + (xx - 48) ** 2 <= 30 ** 2
    s = ps.growth_scale(np.stack([small, large]), plant_size=10.0)
    assert s[0] < s[1] and 0.7 <= s.min() and s.max() <= 3.0
    sem = np.where(np.stack([small, large]), 3, 0)          # partial crop counts as crop
    assert ps.growth_scale(sem, 10.0) == pytest.approx(s)
    probs = np.zeros((2, H, H, 3), np.float32)
    probs[..., 0] = 1
    probs[0][small] = (0, 1, 0)
    probs[1][large] = (0, 1, 0)
    assert ps.growth_scale(probs, 10.0) == pytest.approx(s, abs=0.15)
    assert ps.growth_scale(np.zeros((1, H, H), bool))[0] == 1.0


def test_panoptic_from_probs_layout_and_scale_rule():
    _, sem, _, _ = _stack([0, 1])
    probs = np.eye(3, dtype=np.float32)[np.where(sem == 3, 1, sem)]
    out = ps.panoptic_from_probs(probs, dict(plant_size=10.0))
    assert set(out) == {"semantics", "plant_instances", "leaf_instances"}
    assert out["semantics"].dtype == np.int16 and out["plant_instances"].dtype == np.int32
    assert all(v.shape == (2, H, H) for v in out.values())
    assert (out["semantics"] == 1).sum() > 0 and (out["semantics"] == 2).sum() > 0
    assert (out["plant_instances"][out["semantics"] != 1] == 0).all()
    # with a very small reference size every image is 'grown': plants are plain connected components (3 discs per image)
    big = ps.panoptic_from_probs(probs, dict(plant_size=2.0))
    assert len(np.unique(big["plant_instances"][0][big["plant_instances"][0] > 0])) <= 3


# ---------------------------------------------------------------------------------------------------- fit / predict
def test_fit_predict_synthetic_deterministic_picklable_and_in_layout():
    tr_i, tr_s, tr_p, tr_l = _stack([0, 1, 2, 3])
    te_i, te_s, te_p, te_l = _stack([10, 11])
    kw = dict(pixels_per_image=1500, n_estimators=15, stride=2)
    a = ps.fit_predict(tr_i, tr_s, [te_i, te_i[:1]], **kw)
    b = ps.fit_predict(tr_i, tr_s, [te_i, te_i[:1]], **kw)
    assert isinstance(a, list) and len(a) == 2
    for k in a[0]:
        assert np.array_equal(a[0][k], b[0][k])
    assert a[0]["semantics"].shape == (2, H, H) and a[1]["semantics"].shape == (1, H, H)
    single = ps.fit_predict(tr_i, tr_s, te_i, **kw)
    assert isinstance(single, dict) and np.array_equal(single["semantics"], a[0]["semantics"])
    # quality on separable synthetic scenes and the metric runs on the result
    q = ps.pq_plus(a[0], dict(semantics=te_s, plant_instances=te_p, leaf_instances=te_l))
    assert q["iou_crop"] > 80 and q["iou_soil"] > 95
    # the adapter's output validation accepts the layout
    ad = m.Adapter()
    ad.size = H
    coerced, why = ad._coerce(a[0], 2)
    assert coerced is not None, why


def test_model_is_picklable_and_probs_are_float16():
    tr_i, tr_s, _, _ = _stack([0, 1])
    model = ps.fit_pixel_classifier(tr_i, tr_s, pixels_per_image=900, n_estimators=8)
    model2 = pickle.loads(pickle.dumps(model))
    p1, p2 = ps.predict_probs(model, tr_i[:1]), ps.predict_probs(model2, tr_i[:1])
    assert p1.dtype == np.float16 and p1.shape == (1, H, H, 3) and np.array_equal(p1, p2)
    assert np.allclose(p1.astype(np.float32).sum(-1), 1.0, atol=5e-3)
    assert model.n_trees == 8 and model.plant_size > 0
    f = ps.pixel_features(tr_i)
    assert f.shape == (2, H, H, 35) and f.dtype == np.float32 and ps.pixel_features(tr_i[0]).shape == (H, H, 35)


def test_budget_stops_boosting_and_missing_class_raises():
    tr_i, tr_s, _, _ = _stack([0, 1])
    model = ps.fit_pixel_classifier(tr_i, tr_s, pixels_per_image=900, n_estimators=40, budget_s=0.0)
    assert 1 <= model.n_trees < 40
    with pytest.raises(ValueError):
        ps.fit_pixel_classifier(tr_i, np.where(tr_s == 2, 0, tr_s), pixels_per_image=900, n_estimators=3)


def test_oof_probs_never_splits_groups():
    imgs, sem, _, _ = _stack([0, 1, 2, 3])
    P = ps.oof_probs(imgs, sem, n_folds=2, groups=[0, 0, 1, 1], pixels_per_image=900, n_estimators=8)
    assert P.shape == (4, H, H, 3) and P.dtype == np.float16
    # each image is predicted by a model that never saw its own group: check a leave-group-out model reproduces fold 0
    model = ps.fit_pixel_classifier(imgs[2:], sem[2:], pixels_per_image=900, n_estimators=8)
    assert np.array_equal(P[:2], ps.predict_probs(model, imgs[:2]))
    with pytest.raises(ValueError):
        ps.oof_probs(imgs, sem, groups=[0, 0, 0, 0])
