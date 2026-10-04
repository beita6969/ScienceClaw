"""scilib.audiosep: metric parity with the FoR36 adapter, constant-gain fit, STFT round trip, soft-mask separator on
synthetic sources, grouped cross-validation, input validation, sandbox import, interface text."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import audiosep as A
from scienceclaw.bench.tasks import for36_musdb as m
from scienceclaw.runtime.integrity import scan_code

SR = A.SR


def _sources(seed: int, n: int = 2 * SR + 500) -> np.ndarray:
    """(4, n, 2) synthetic 'stems' with different spectra: band-limited noise (vocal-like, bass, hi-hat) and tones."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR

    def band(lo, hi, gain):
        sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
        return gain * sosfilt(sos, rng.standard_normal((n, 2)), axis=0)

    env = 0.5 + 0.5 * (np.sin(2 * np.pi * (1.5 + seed % 3) * t) > 0)               # on/off percussion
    vocals = band(300, 1800, 0.30) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))[:, None]
    drums = band(4000, 9000, 0.25) * env[:, None]
    bass = np.stack([0.30 * np.sin(2 * np.pi * (55 + 10 * seed) * t)] * 2, 1) + band(30, 120, 0.10)
    other = np.stack([0.15 * np.sin(2 * np.pi * (700 + 40 * seed) * t), 0.10 * np.sin(2 * np.pi * (900 + 40 * seed) * t)], 1)
    return np.stack([vocals, drums, bass, other]).astype(np.float32)


def _data(seeds):
    stems = np.stack([_sources(s) for s in seeds])
    return stems.sum(axis=1).astype(np.float32), stems


# ------------------------------------------------------------------------------------------------ metric
def test_sdr_scores_match_the_adapter_metric_including_nan_and_inf_rules():
    rng = np.random.default_rng(12345)
    sr = 1000
    ref = rng.normal(size=(5, 4, 3 * sr + 137, 2))
    ref[1, 2, sr:2 * sr] = 0                                                    # silent reference window (all targets skipped)
    ref[3, 1] = 0                                                               # a target without any valid window
    ref[4, 0, :, 1] = -ref[4, 0, :, 0]                                          # channel sum exactly 0 counts as silent
    est = 0.8 * ref + 0.3 * ref[:, [1, 2, 3, 0]] + 0.2 * rng.normal(size=ref.shape)
    est[0, 3, :sr] = 0                                                          # silent estimate window
    est[2] = ref[2]                                                             # perfect estimate -> +inf
    got = A.sdr_scores(ref, est, sr)
    exp = np.stack([m.item_scores(ref[i], est[i], sr=sr) for i in range(5)])
    assert got["item"].shape == (5, 4)
    assert np.array_equal(np.isnan(got["item"]), np.isnan(exp))
    assert np.allclose(got["item"][~np.isnan(exp)], exp[~np.isnan(exp)], atol=1e-9, equal_nan=False)
    assert np.isinf(got["item"][2]).all()
    assert got["sdr"] == pytest.approx(m.aggregate(exp))
    assert np.allclose(got["target_median"], [np.nanmedian(exp[:, j]) for j in range(4)])


def test_sdr_scores_is_nan_when_a_target_has_no_valid_item_and_depends_on_scale():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3, 4, 2000, 2))
    ref[:, 2] = 0
    assert np.isnan(A.sdr_scores(ref, ref + 0.1, 1000)["sdr"])
    ref = rng.normal(size=(3, 4, 2000, 2))
    est = ref + 0.1 * rng.normal(size=ref.shape)
    assert A.sdr_scores(ref, 0.5 * est, 1000)["sdr"] < A.sdr_scores(ref, est, 1000)["sdr"] - 3


# ------------------------------------------------------------------------------------------------ constant gain
def test_gain_only_fits_the_best_constant_gain_and_reports_the_level_effect():
    mix, stems = _data([0, 1, 2])
    out = A.gain_only(mix, stems, mix[:2])
    assert out["gains"].shape == (4,) and out["estimates"].shape == (2, 4) + mix.shape[1:]
    assert out["estimates"].dtype == np.float32
    # the reported SDRs are the metric of the scaled / unscaled mixture
    scaled = out["gains"][None, :, None, None] * mix[:, None]
    assert out["train_sdr"] == pytest.approx(A.sdr_scores(stems, scaled)["sdr"])
    assert out["mixture_sdr"] == pytest.approx(A.sdr_scores(stems, np.repeat(mix[:, None], 4, 1))["sdr"])
    assert out["train_sdr"] > out["mixture_sdr"] + 3                             # level alone, no separation
    assert abs(out["null_sdr"]) < 1e-3                                           # a no-signal estimate: 0 dB
    # no other gain on the log grid is better for any target (per-target medians)
    for j in range(4):
        base = A.sdr_scores(stems, scaled)["target_median"][j]
        for f in (0.8, 0.9, 1.1, 1.25):
            g = out["gains"].copy()
            g[j] *= f
            assert A.sdr_scores(stems, g[None, :, None, None] * mix[:, None])["target_median"][j] <= base + 1e-6
    assert "estimates" not in A.gain_only(mix, stems)


# ------------------------------------------------------------------------------------------------ transform
def test_stft_round_trip_and_shapes():
    x = _sources(3)[3, :SR + 777]
    Z = A.stft(x)
    assert Z.shape[:2] == (2, 1025) and np.iscomplexobj(Z)
    y = A.istft(Z, len(x))
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.max(np.abs(y - x)) < 1e-5


# ------------------------------------------------------------------------------------------------ separator
KW = dict(rows_per_excerpt=1500, n_estimators=15, num_leaves=15, min_samples_leaf=20)


def test_soft_mask_separator_separates_synthetic_sources_and_is_deterministic():
    tr_mix, tr_stems = _data([0, 1, 2, 3])
    te_mix, te_stems = _data([4, 5])
    est = A.separate(tr_mix, tr_stems, te_mix, **KW)
    assert est.shape == te_stems.shape and est.dtype == np.float32 and np.isfinite(est).all()
    again = A.separate(tr_mix, tr_stems, te_mix, **KW)
    assert np.array_equal(est, again)
    sdr = A.sdr_scores(te_stems, est)["sdr"]
    gain = A.gain_only(tr_mix, tr_stems, te_mix)
    sdr_gain = A.sdr_scores(te_stems, gain["estimates"])["sdr"]
    assert sdr > sdr_gain + 3.0 and sdr > 3.0
    # no exactly silent 1-s window even for a target that is silent in the mask
    fr = est[:, :, : (est.shape[2] // SR) * SR].reshape(est.shape[0], 4, -1, SR, 2)
    assert not np.any(np.all(fr.sum(axis=4) == 0, axis=3))
    # a list of arrays gives a list of estimate arrays, same numbers as the single call
    a, b = A.separate(tr_mix, tr_stems, [te_mix[:1], te_mix[1:]], **KW)
    assert np.array_equal(np.concatenate([a, b]), est)


def test_estimates_scale_with_the_input_level():
    tr_mix, tr_stems = _data([0, 1, 2])
    te_mix, _ = _data([4])
    sep = A.SoftMaskSeparator(**KW).fit(tr_mix, tr_stems)
    e1, e2 = sep.separate(te_mix), sep.separate(0.5 * te_mix)
    assert np.allclose(e2, 0.5 * e1, atol=1e-5)
    silent = sep.separate(np.zeros_like(te_mix))
    assert np.isfinite(silent).all() and np.abs(silent).max() < 1e-6


def test_oracle_mask_sdr_is_high_on_synthetic_sources():
    mix, stems = _data([0, 1])
    assert A.oracle_mask_sdr(mix, stems) > 10


# ------------------------------------------------------------------------------------------------ cross-validation
def test_folds_never_split_a_group():
    groups = ["a", "a", "b", "c", "c", "d", "e", "e"]
    folds = A._folds(groups, 8, 4)
    assert sorted(np.concatenate(folds).tolist()) == list(range(8))
    for g in set(groups):
        idx = {i for i, x in enumerate(groups) if x == g}
        assert sum(bool(idx & set(f.tolist())) for f in folds) == 1
    assert len(A._folds(None, 5, 10)) == 5
    with pytest.raises(ValueError, match="one label per training excerpt"):
        A._folds(["a", "b"], 3, 2)
    with pytest.raises(ValueError, match="at least 2 distinct groups"):
        A._folds(["a", "a", "a"], 3, 2)


def test_cross_validate_reports_separator_gain_only_and_mixture():
    mix, stems = _data([0, 1, 2, 3])
    cv = A.cross_validate(mix, stems, groups=list("aabb"), n_folds=2, **KW)
    assert cv["folds"] == 2 and cv["item"].shape == (4, 4) and cv["target_median"].shape == (4,)
    for k in ("sdr", "gain_only_sdr", "mixture_sdr", "null_sdr", "gain_only_target_median", "mixture_target_median",
              "null_target_median"):
        assert k in cv
    assert cv["sdr"] > cv["gain_only_sdr"] + 3.0 and cv["gain_only_sdr"] > cv["mixture_sdr"] + 3.0
    assert abs(cv["null_sdr"]) < 1e-3                                             # error = reference: 0 dB
    cv2 = A.cross_validate(mix, stems, groups=list("aabb"), n_folds=2, **KW)
    assert cv["sdr"] == cv2["sdr"] and cv["gain_only_sdr"] == cv2["gain_only_sdr"]


# ------------------------------------------------------------------------------------------------ validation
def test_shape_errors_are_informative():
    mix, stems = _data([0, 1])
    with pytest.raises(ValueError, match=r"looks like stems"):
        A.separate(stems, stems, mix)
    with pytest.raises(ValueError, match=r"train_stems must have shape \(k, 4, n, 2\)"):
        A.separate(mix, mix, mix)
    with pytest.raises(ValueError, match="does not match train_mixtures"):
        A.separate(mix, stems[:1], mix)
    with pytest.raises(ValueError, match=r"mixtures must have shape \(k, n, 2\)"):
        A.separate(mix, stems, mix[..., 0])                                     # rejected before any fitting
    with pytest.raises(ValueError, match=r"mixtures must have shape \(k, n, 2\)"):
        A.separate(mix, stems, [mix, mix[..., 0]])
    with pytest.raises(ValueError, match="learning_rate"):
        A.SoftMaskSeparator(learning_rate=0)
    bad = stems.copy()
    bad[0, 0, 5, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        A.separate(mix, bad, mix)
    with pytest.raises(ValueError, match="same shape"):
        A.sdr_scores(stems, stems[:1])
    with pytest.raises(RuntimeError, match="fit"):
        A.SoftMaskSeparator().separate(mix)


# ------------------------------------------------------------------------------------------------ integration
def test_describe_lists_every_public_name():
    text = scilib.describe("audiosep")
    for name in A.__all__:
        assert name in text


def test_code_node_may_import_scilib_audiosep():
    code = ("import numpy as np\nfrom scilib.audiosep import separate, cross_validate, gain_only\n\n"
            "def run(inputs, config):\n    return {}\n")
    assert scan_code(code) == []
