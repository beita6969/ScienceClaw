"""Tests for scilib.hippo on synthetic blob volumes (no dataset files are read)."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import hippo

SHAPE = (24, 32, 24)


def _make_case(seed: int, scale: float = 1.0, offset: float = 0.0):
    """Two touching ellipsoids (anterior in front of posterior) on a noisy background, with a random position jitter."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")).astype(float)
    jit = rng.integers(-1, 2, size=3)
    c1 = np.array([12, 11, 12]) + jit
    c2 = np.array([12, 20, 12]) + jit
    lab = np.zeros(SHAPE, np.uint8)
    for c, cl in ((c1, 1), (c2, 2)):
        d = ((g[0] - c[0]) / 4.0) ** 2 + ((g[1] - c[1]) / 5.0) ** 2 + ((g[2] - c[2]) / 4.0) ** 2
        lab[d <= 1] = cl
    img = 0.25 + 0.6 * (lab > 0) + 0.05 * (lab == 2) + 0.06 * rng.standard_normal(SHAPE)
    return (img * scale + offset).astype(np.float32), lab


@pytest.fixture(scope="module")
def cases():
    return [_make_case(i, scale=[1.0, 1000.0, 255.0][i % 3], offset=[0.0, 5.0, 0.0][i % 3]) for i in range(8)]


def test_dsc_and_metric():
    a = np.zeros((4, 4, 4), np.uint8)
    b = a.copy()
    assert hippo.dsc(a, b) == 1.0
    a[:2] = 1
    b[:1] = 1
    assert hippo.dsc(a, b) == pytest.approx(2 * 16 / (32 + 16))
    gt = np.zeros((4, 4, 4), np.uint8)
    gt[:2] = 1
    gt[2:] = 2
    pred = gt.copy()
    pred[3:] = 0
    d1, d2 = hippo.case_dsc(pred, gt)
    assert d1 == 1.0 and d2 == pytest.approx(2 * 16 / (16 + 32))
    assert hippo.mean_dsc([pred, gt], [gt, gt]) == pytest.approx(((d1 + d2) / 2 + 1.0) / 2)


def test_normalize_intensity_is_scale_free():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 1.0, size=(10, 12, 14))
    for method in ("robust", "rank"):
        a = hippo.normalize_intensity(x, method)
        b = hippo.normalize_intensity(1e5 * x + 300.0, method)
        assert a.dtype == np.float32 and a.shape == x.shape
        np.testing.assert_allclose(a, b, atol=1e-4)
    r = hippo.normalize_intensity(x, "rank")
    assert 0.0 <= r.min() and r.max() <= 1.0
    np.testing.assert_allclose(hippo.normalize_intensity(np.exp(x), "rank"), r, atol=1e-6)
    with pytest.raises(ValueError):
        hippo.normalize_intensity(x, "nope")
    with pytest.raises(ValueError):
        hippo.normalize_intensity(x[0])
    assert np.isfinite(hippo.normalize_intensity(np.full((4, 4, 4), 7.0))).all()


def test_subject_groups():
    assert hippo.subject_groups(["hippocampus_1", "hippocampus_2", "hippocampus_3", "hippocampus_260"]) == [1, 1, 2, 130]
    with pytest.raises(ValueError):
        hippo.subject_groups(["nodigits"])


def test_location_atlas(cases):
    labels = [c[1] for c in cases]
    at = hippo.LocationAtlas().fit(labels)
    p = at.prob(SHAPE)
    assert p.shape == (3, *SHAPE) and p.dtype == np.float32
    np.testing.assert_allclose(p.sum(0), 1.0, atol=1e-4)
    assert p[1][12, 11, 12] > 0.5 and p[2][12, 20, 12] > 0.5 and p[0][0, 0, 0] > 0.99
    assert at.support(SHAPE, 0).sum() > 0
    assert at.support(SHAPE, 3).sum() > at.support(SHAPE, 0).sum()
    assert at.prob(SHAPE) is at.prob(SHAPE)
    with pytest.raises(ValueError):
        hippo.LocationAtlas().fit([np.zeros(SHAPE, np.uint8)])


def test_postprocess_cleans_up_components():
    lab = np.zeros(SHAPE, int)
    lab[8:16, 8:16, 8:16] = 1
    lab[8:16, 16:20, 8:16] = 2
    lab[12, 12, 12] = 0            # hole
    lab[2, 2, 2] = 1               # stray voxel
    prob = np.eye(3)[lab].transpose(3, 0, 1, 2).astype(np.float32)
    out = hippo.postprocess(prob)
    assert out.dtype == np.uint8 and out[2, 2, 2] == 0 and out[12, 12, 12] > 0
    assert out.max() == 2 and (out == 1).sum() > 0 and (out == 2).sum() > 0
    empty = np.zeros((3, 4, 4, 4), np.float32)
    empty[0] = 1
    assert hippo.postprocess(empty).sum() == 0
    sm = hippo.postprocess(prob, smooth=1.0, background_weight=1.5)
    assert sm.shape == SHAPE


def test_label_fusion_and_features(cases):
    train, tgt = cases[:6], cases[6]
    fus = hippo.label_fusion(tgt[0], [c[0] for c in train], [c[1] for c in train], k=3)
    assert fus.shape == (3, *SHAPE)
    np.testing.assert_allclose(fus.sum(0), 1.0, atol=1e-3)
    assert hippo.dsc(fus.argmax(0) == 1, tgt[1] == 1) > 0.6
    at = hippo.LocationAtlas().fit([c[1] for c in train])
    X, names = hippo.voxel_features(tgt[0], at, fus)
    assert X.shape == (int(np.prod(SHAPE)), len(names)) and X.dtype == np.float32 and np.isfinite(X).all()
    X0, n0 = hippo.voxel_features(tgt[0])
    assert X0.shape[1] < X.shape[1] and set(n0) < set(names)


@pytest.fixture(scope="module")
def small_kw():
    return dict(n_estimators=30, fusion_k=3, margin=3, fusion_max_shift=2)


def test_fit_predict_synthetic_shapes_and_quality(cases, small_kw):
    train, ev = cases[:6], cases[6:]
    ids = [f"hippocampus_{i + 1}" for i in range(6)]
    preds = hippo.fit_predict([c[0] for c in train], [c[1] for c in train], [c[0] for c in ev], train_ids=ids, **small_kw)
    assert len(preds) == 2
    for p, c in zip(preds, ev):
        assert p.shape == c[0].shape and p.dtype == np.uint8 and set(np.unique(p)) <= {0, 1, 2}
    assert hippo.mean_dsc(preds, [c[1] for c in ev]) > 0.8
    again = hippo.fit_predict([c[0] for c in train], [c[1] for c in train], [c[0] for c in ev], train_ids=ids, **small_kw)
    assert all(np.array_equal(a, b) for a, b in zip(preds, again))


def test_input_validation(cases, small_kw):
    imgs, labs = [c[0] for c in cases[:4]], [c[1] for c in cases[:4]]
    with pytest.raises(ValueError):
        hippo.fit_predict(imgs, labs[:3], imgs, **small_kw)
    with pytest.raises(ValueError):
        hippo.fit_predict(imgs[:2], labs[:2], imgs, **small_kw)
    with pytest.raises(ValueError):
        hippo.fit_predict(imgs, [np.zeros((3, 3, 3), np.uint8)] * 4, imgs, **small_kw)
    with pytest.raises(RuntimeError):
        hippo.HippocampusSegmenter().predict(imgs[0])


def test_cross_validate_single_fold(cases, small_kw):
    ids = [f"hippocampus_{i + 1}" for i in range(8)]
    out = hippo.cross_validate([c[0] for c in cases], [c[1] for c in cases], ids, n_folds=2, folds=[0], **small_kw)
    assert 0.5 < out["mean_dsc"] <= 1.0
    per = np.array(out["per_case"])
    assert per.shape == (8, 2) and np.isnan(per[:, 0]).sum() == 4


def test_describe_lists_module():
    text = scilib.describe("hippo")
    assert "fit_predict" in text and "label_fusion" in text


def test_documented_defaults_match_signature():
    import inspect

    for name, prm in inspect.signature(hippo.HippocampusSegmenter.__init__).parameters.items():
        if name != "self":
            assert f"{name}={prm.default}" in hippo.__doc__, name
