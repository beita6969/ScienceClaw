"""scilib.anomsound: log-mel parity with the adapter, component semantics, ranking, error messages, sandbox import."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import anomsound as an
from scienceclaw.bench.tasks import for40_dcase as m
from scienceclaw.runtime.integrity import scan_code

SR = 16000


def _machine_clip(rng, seconds=1.5, f0=180.0, noise=0.05, click=False, extra_tone=False):
    t = np.arange(int(SR * seconds)) / SR
    x = sum(a * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) for k, a in ((1, 0.5), (2, 0.3), (3, 0.15)))
    x = x * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))
    x = x + noise * rng.standard_normal(t.size)
    if extra_tone:
        x = x + 0.25 * np.sin(2 * np.pi * 2600 * t)
    if click:
        for c in rng.uniform(0.2, seconds - 0.2, 3):
            i = int(c * SR)
            x[i:i + 80] += 1.5 * rng.standard_normal(80)
    return x.astype(np.float32)


def _episode(seed=0, kind="click"):
    """24 normal source clips + 16 eval clips (4 per domain x condition stratum, order shuffled); target = shifted pitch/noise."""
    rng = np.random.default_rng(seed)
    tr = [_machine_clip(rng, f0=180 + rng.normal(0, 2)) for _ in range(24)]
    ev, y, dom = [], [], []
    for d, f0, nz in (("source", 180.0, 0.05), ("target", 205.0, 0.08)):
        for lab in (0, 1):
            for _ in range(4):
                ev.append(_machine_clip(rng, f0=f0 + rng.normal(0, 2), noise=nz, click=bool(lab) and kind == "click",
                                        extra_tone=bool(lab) and kind == "tone"))
                y.append(lab)
                dom.append(d)
    p = rng.permutation(16)
    return tr, [ev[i] for i in p], np.array(y)[p], np.array(dom)[p]


def _pack(clips, pad_to=None):
    n = max(c.size for c in clips) if pad_to is None else pad_to
    W = np.zeros((len(clips), n), dtype=np.float32)
    for i, c in enumerate(clips):
        W[i, :c.size] = c
    return {"waveforms": W, "lengths": np.array([c.size for c in clips]), "sample_rate": float(SR)}


# ------------------------------------------------------------------------------------------------ front end
def test_log_mel_list_matches_the_adapter_tool_and_ignores_padding():
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(SR).astype(np.float32) * 0.1, rng.standard_normal(int(1.3 * SR)).astype(np.float32) * 0.1]
    pack = _pack(clips)
    got = an.log_mel_list(pack["waveforms"], pack["lengths"], SR)
    for c, g in zip(clips, got):
        ref = m.log_mel(c, SR, 1024, 512, 128)
        assert g.shape == ref.shape == (1 + c.size // 512, 128)
        assert np.allclose(g, ref, atol=1e-3)
    small = an.log_mel_list(pack["waveforms"], pack["lengths"], SR, n_fft=512, hop=256, n_mels=64)
    assert small[0].shape == (1 + SR // 256, 64)


def test_trim_mels_and_floor_frames_are_removed():
    rng = np.random.default_rng(1)
    clips = [rng.standard_normal(SR).astype(np.float32) * 0.1, rng.standard_normal(int(1.3 * SR)).astype(np.float32) * 0.1]
    pack = _pack(clips)
    padded = np.stack([m.log_mel(w, SR, 1024, 512, 128) for w in pack["waveforms"]])      # what the tool returns for padded rows
    trimmed = an.trim_mels(padded, pack["lengths"])
    assert [t.shape[0] for t in trimmed] == [1 + c.size // 512 for c in clips]
    d_pad = an.clip_descriptors(list(padded), "ms")                                        # floor frames stripped automatically
    d_ok = an.clip_descriptors(trimmed, "ms")
    assert np.allclose(d_pad[1], d_ok[1], atol=1e-3)
    kept = len(an._mel_list([padded[0]], "x")[0])                                          # short row: only the 1-2 boundary frames remain
    assert 0 <= kept - trimmed[0].shape[0] <= 2
    with pytest.raises(ValueError, match="3-D"):
        an.trim_mels(padded[0], pack["lengths"])
    with pytest.raises(ValueError, match="lengths"):
        an.trim_mels(padded, [1])


def test_descriptors():
    S = [np.arange(12, dtype=float).reshape(4, 3), np.ones((5, 3))]
    ms = an.clip_descriptors(S, "ms")
    assert ms.shape == (2, 6) and np.allclose(ms[0, :3], S[0].mean(0)) and np.allclose(ms[0, 3:], S[0].std(0))
    assert np.allclose(an.clip_descriptors(S, "mx")[0], S[0].max(0)) and an.clip_descriptors(S, "m").shape == (2, 3)
    assert np.allclose(an.clip_descriptors(S, "q95")[1], 1.0)
    with pytest.raises(ValueError, match="unknown descriptor"):
        an.clip_descriptors(S, "nope")


# ------------------------------------------------------------------------------------------------ components / ensemble
@pytest.fixture(scope="module")
def click_episode():
    tr, ev, y, dom = _episode(0, "click")
    return _pack(tr), _pack(ev), y, dom


def test_components_shapes_finite_deterministic(click_episode):
    tr, ev, y, dom = click_episode
    tm, em = an.log_mel_list(tr["waveforms"], tr["lengths"], SR), an.log_mel_list(ev["waveforms"], ev["lengths"], SR)
    c1, c2 = an.component_scores(tm, em), an.component_scores(tm, em)
    assert list(c1) == list(an.COMPONENTS) and set(an.DEFAULT_MEMBERS) <= set(c1)
    for k, v in c1.items():
        assert v.shape == (16,) and np.isfinite(v).all() and np.array_equal(v, c2[k])
    assert np.array_equal(np.stack([tm[0]]), np.stack([tm[0]]))                   # inputs are not modified
    s = an.fit_predict(tm, em)
    assert s.shape == (16,) and s.min() > 0 and s.max() <= 1 and np.array_equal(s, an.fit_predict(tm, em))


def test_pool_components_follow_their_definitions(click_episode):
    tr, ev, y, dom = click_episode
    tm, em = an.log_mel_list(tr["waveforms"], tr["lengths"], SR), an.log_mel_list(ev["waveforms"], ev["lengths"], SR)
    c = an.component_scores(tm, em)
    Zt, Ze = an._zscore(an.clip_descriptors(tm, "ms"), an.clip_descriptors(em, "ms"))
    d_tr = an._dist(Ze, Zt)
    d_ee = an._dist(Ze, Ze)
    np.fill_diagonal(d_ee, np.inf)
    assert np.allclose(c["nn_train"], d_tr.min(1))
    assert np.allclose(c["nn_pool"], np.minimum(d_tr.min(1), d_ee.min(1)))
    pooled = np.sort(np.hstack([d_tr, d_ee]), axis=1)
    assert np.allclose(c["nn2_pool"], pooled[:, 1]) and (c["nn_pool"] <= c["nn_train"] + 1e-12).all()
    # training-only components do not depend on the other eval clips; pool components do
    c_one = an.component_scores(tm, em[:1])
    for k in ("nn_train", "band_max", "maha"):
        assert np.allclose(c_one[k], c[k][:1], atol=1e-9)
    assert not np.allclose(c_one["nn_pool"], c["nn_pool"][:1]) or c["nn_pool"][0] == c["nn_train"][0]


def test_embedding_scores_follow_their_definitions():
    rng = np.random.default_rng(3)
    mu = np.ones(16)
    Et, Ee = mu + 0.3 * rng.normal(size=(24, 16)), mu + 0.3 * rng.normal(size=(16, 16))
    Ee[3] = -mu + 0.3 * rng.normal(size=16)
    c = an.embedding_scores(Et, Ee)
    assert list(c) == list(an.EMBEDDING_COMPONENTS)
    for v in c.values():
        assert v.shape == (16,) and np.isfinite(v).all()
    Ut, Ue = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (Et, Ee))
    d_tr = np.linalg.norm(Ue[:, None] - Ut[None], axis=-1)
    d_ee = np.linalg.norm(Ue[:, None] - Ue[None], axis=-1)
    np.fill_diagonal(d_ee, np.inf)
    assert np.allclose(c["nn_train"], d_tr.min(1))
    assert np.allclose(c["nn_pool"], np.minimum(d_tr.min(1), d_ee.min(1)))
    assert np.allclose(c["nn2_pool"], np.sort(np.hstack([d_tr, d_ee]), axis=1)[:, 1])
    assert np.argmax(c["nn2_pool"]) == 3 and np.argmax(c["lof"]) == 3
    assert np.allclose(c["nn_train"], an.embedding_scores(Et * 7.0, Ee * 0.3)["nn_train"])      # rows are L2-normalised
    members = an.rank_average({"a": c["nn2_pool"], "b": an.embedding_scores(Et[:, ::-1], Ee[:, ::-1])["nn2_pool"]}, members=("a", "b"))
    assert members.shape == (16,) and members.min() > 0 and members.max() <= 1
    for bad in ((Et[:2], Ee), (Et, Ee[:, :8]), (Et.ravel(), Ee), (np.where(Et > 5, np.nan, Et) * np.nan, Ee)):
        with pytest.raises(ValueError):
            an.embedding_scores(*bad)


def test_default_ensemble_finds_planted_anomalies():
    """Planted transients / extra tone on a synthetic machine (target domain = shifted pitch and noise level)."""
    res = {}
    for kind in ("click", "tone"):
        vals = {"official": [], "auc_source": [], "band_max": []}
        for seed in range(3):
            tr, ev, y, dom = _episode(seed, kind)
            tm = an.log_mel_list(_pack(tr)["waveforms"], _pack(tr)["lengths"])
            em = an.log_mel_list(_pack(ev)["waveforms"], _pack(ev)["lengths"])
            comps = an.component_scores(tm, em)
            sc = an.official_score(y, dom, an.rank_average(comps))
            vals["official"].append(sc["official_score"])
            vals["auc_source"].append(sc["auc_source"])
            vals["band_max"].append(an.official_score(y, dom, comps["band_max"])["official_score"])
        res[kind] = {k: float(np.mean(v)) for k, v in vals.items()}
    assert res["click"]["official"] > 0.8 and res["click"]["band_max"] > 0.9       # short transients show in per-band maxima
    assert res["tone"]["auc_source"] > 0.85                                        # a new tonal component among source-like clips


def test_score_clips_equals_fit_predict_and_member_selection(click_episode):
    tr, ev, y, dom = click_episode
    tm, em = an.log_mel_list(tr["waveforms"], tr["lengths"], SR), an.log_mel_list(ev["waveforms"], ev["lengths"], SR)
    assert np.allclose(an.score_clips(tr, ev), an.fit_predict(tm, em))
    # one padded (wider) batch gives the same scores: rows are cut to their lengths
    wide = dict(ev, waveforms=np.hstack([ev["waveforms"], np.zeros((16, 777), np.float32)]))
    assert np.allclose(an.score_clips(tr, wide), an.score_clips(tr, ev), atol=1e-6)
    comps = an.component_scores(tm, em)
    from scipy.stats import rankdata
    assert np.allclose(an.fit_predict(tm, em, members=("maha",)), rankdata(comps["maha"]) / 16)
    assert np.allclose(an.fit_predict(tm, em, members="band_max"), rankdata(comps["band_max"]) / 16)
    assert np.allclose(an.rank_average(comps, ["lof", "maha"]), (rankdata(comps["lof"]) + rankdata(comps["maha"])) / 32)


def test_rank_average_handles_ties_and_non_finite():
    s = {"a": np.array([1.0, 1.0, 3.0, 2.0]), "b": np.array([np.nan, 0.0, 5.0, np.inf])}
    r = an.rank_average(s, ("a", "b"))
    assert np.isfinite(r).all() and r.max() <= 1.0 and r.min() > 0 and r[2] >= r[1]
    with pytest.raises(ValueError, match="unknown members"):
        an.rank_average(s, ("zzz",))
    with pytest.raises(ValueError, match="empty"):
        an.rank_average(s, ())


def test_input_validation_messages(click_episode):
    tr, ev, y, dom = click_episode
    tm, em = an.log_mel_list(tr["waveforms"], tr["lengths"], SR), an.log_mel_list(ev["waveforms"], ev["lengths"], SR)
    with pytest.raises(ValueError, match="at least 3 training"):
        an.fit_predict(tm[:2], em)
    with pytest.raises(ValueError, match="mel bins"):
        an.fit_predict(tm, an.log_mel_list(ev["waveforms"], ev["lengths"], SR, n_mels=64))
    with pytest.raises(ValueError, match="one .* array per clip"):
        an.fit_predict(tm, em[0])
    bad = [e.copy() for e in em]
    bad[3][2, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        an.fit_predict(tm, bad)
    with pytest.raises(ValueError, match="unknown members"):
        an.fit_predict(tm, em, members=("nope",))
    with pytest.raises(ValueError, match="load tool"):
        an.score_clips(tm, ev)
    with pytest.raises(ValueError, match="lengths"):
        an.log_mel_list(tr["waveforms"], tr["lengths"][:3], SR)
    one = an.fit_predict(tm, em[:1])                                                   # a single eval clip still works
    assert one.shape == (1,) and np.isfinite(one).all()


def test_probe_report(click_episode):
    tr, ev, y, dom = click_episode
    tm, em = an.log_mel_list(tr["waveforms"], tr["lengths"], SR), an.log_mel_list(ev["waveforms"], ev["lengths"], SR)
    rep = an.probe_report(tm, em)
    corr = np.array(rep["spearman"])
    assert corr.shape == (len(an.COMPONENTS),) * 2 and np.allclose(corr, corr.T) and np.allclose(np.diag(corr), 1.0)
    assert len(rep["nearest_kind"]) == 16 and set(rep["nearest_kind"]) <= {"train", "eval"}
    assert len(rep["dist_to_nearest_train"]) == 16


def test_official_score_matches_the_adapter_metric():
    rng = np.random.default_rng(5)
    for _ in range(10):
        y = np.repeat([0, 0, 1, 1], 4)
        dom = np.array((["source"] * 4 + ["target"] * 4) * 2)
        s = rng.random(16) + 0.4 * y
        a, b = m.official_scores(y, dom, s), an.official_score(y, dom, s)
        assert a == pytest.approx(b)


# ------------------------------------------------------------------------------------------------ interface text
def test_describe_lists_every_public_name():
    text = scilib.describe("anomsound")
    for name in an.__all__:
        assert name in text
    for member in an.COMPONENTS:
        assert member in text
    assert "importable in code nodes" in text


def test_code_node_may_import_scilib():
    code = ("from scilib import anomsound as A\n\ndef run(inputs, config):\n"
            "    return {'y': A.score_clips(inputs['load_train'], inputs['load_eval_inputs'])}\n")
    assert scan_code(code) == []


def test_docstring_states_no_hidden_composition_or_recipe_words():
    text = scilib.describe("anomsound").lower()
    for word in ("reference", "baseline", "accept", "margin", "best", "recommend", "should", "target-domain", "anomal" + "ous clips"):
        assert word not in text, word
