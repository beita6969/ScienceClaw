"""Pretrained-model options of the domain libraries (forecast, loadforecast, aquatics, anomsound): the default call is unchanged,
the option mixes / adds exactly what its documentation says (checked with stand-ins for scilib.tsfm / scilib.audioenc, so no GPU),
and bad arguments fail with a clear message."""
from __future__ import annotations

import numpy as np
import pytest

from scilib import aquatics as aq
from scilib import anomsound as an
from scilib import forecast as fc
from scilib import loadforecast as lf
from scilib import audioenc, tsfm
from scilib import udparse as up
from scilib import udparse_pretrained as upp

SR = 16000


class _FakeTsfm:
    """Stand-in for tsfm.forecast: constant ``value`` for every quantile; records the calls."""

    def __init__(self, value):
        self.value, self.calls = value, []

    def __call__(self, histories, horizon, quantiles=(0.1, 0.5, 0.9), model="chronos_2", context_length=None, lengths=None):
        self.calls.append({"n": len(histories), "horizon": horizon, "quantiles": tuple(quantiles), "model": model,
                           "context": context_length, "inputs": [np.asarray(h, dtype=float) for h in histories]})
        v = self.value(len(histories), len(quantiles)) if callable(self.value) else self.value
        return np.full((len(histories), horizon, len(quantiles)), v, dtype=np.float32) if np.isscalar(v) else np.asarray(v, np.float32)


# ------------------------------------------------------------------------------------------------ forecast (FoR35)
def _panel(n=3, T=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return [50 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, T) + 5 * i for i in range(n)]


def test_forecast_default_has_no_pretrained_call(monkeypatch):
    fake = _FakeTsfm(1.0)
    monkeypatch.setattr(tsfm, "forecast", fake)
    h = _panel()
    a = fc.fit_predict(h, 6, 12, methods=("repeat_season", "theta"), global_models=())
    b = fc.fit_predict(h, 6, 12, methods=("repeat_season", "theta"), global_models=(), pretrained=0.0)
    assert np.array_equal(a, b) and fake.calls == []


def test_forecast_pretrained_mix_is_the_documented_weighted_average(monkeypatch):
    fake = _FakeTsfm(np.log1p(100.0))                                   # model output in log1p space -> 100 after expm1
    monkeypatch.setattr(tsfm, "forecast", fake)
    h = _panel()
    base = fc.fit_predict(h, 6, 12, methods=("repeat_season", "theta"), global_models=())
    mix = fc.fit_predict(h, 6, 12, methods=("repeat_season", "theta"), global_models=(), pretrained=0.25)
    assert np.allclose(mix, 0.75 * base + 0.25 * 100.0, atol=1e-4)
    call = fake.calls[-1]
    assert call["n"] == 3 and call["horizon"] == 6 and call["quantiles"] == (0.5,) and call["model"] == "chronos_2" and call["context"] == 120
    assert np.allclose(call["inputs"][0], np.log1p(h[0]))                # all-positive series go through log1p


def test_forecast_pretrained_forecast_context_and_negative_series(monkeypatch):
    fake = _FakeTsfm(3.0)
    monkeypatch.setattr(tsfm, "forecast", fake)
    h = [np.linspace(-5, 5, 200), np.linspace(1, 9, 200)]
    out = fc.pretrained_forecast(h, 4, model="chronos_bolt", context=50)
    call = fake.calls[-1]
    assert call["model"] == "chronos_bolt" and call["context"] == 50 and all(x.size == 50 for x in call["inputs"])
    assert np.allclose(call["inputs"][0], h[0][-50:])                    # contains negatives -> unchanged
    assert np.allclose(call["inputs"][1], np.log1p(h[1][-50:]))
    assert np.allclose(out[0], 3.0) and np.allclose(out[1], np.expm1(3.0))


def test_forecast_pretrained_argument_checks():
    with pytest.raises(ValueError, match="pretrained must be"):
        fc.fit_predict(_panel(), 6, 12, methods=("repeat_season",), global_models=(), pretrained=1.5)
    with pytest.raises(ValueError, match="pretrained must be"):
        fc.fit_predict(_panel(), 6, 12, methods=("repeat_season",), global_models=(), pretrained=-0.1)


# ------------------------------------------------------------------------------------------------ loadforecast (FoR33)
def _model_and_windows():
    from test_scilib_loadforecast import _synthetic
    load, ids, cats, starts = _synthetic(nb=3, T=1500)
    m = lf.fit(load, starts, ids, cats)
    w = lf.history_windows(load, starts, ids, cats, stride=48, first_hour=1200)
    return m, w


def test_loadforecast_candidates_default_keys_and_pretrained_keys(monkeypatch):
    fake = _FakeTsfm(7.5)
    monkeypatch.setattr(tsfm, "forecast", fake)
    m, w = _model_and_windows()
    base = m.candidates(w["context"], w["target_start"], w["building_id"], w["category"])
    assert tuple(base) == ("yesterday", "mean7", "median7", "core", "ml", "ens") == lf.CANDIDATES and fake.calls == []
    ext = m.candidates(w["context"], w["target_start"], w["building_id"], w["category"], pretrained=True)
    assert tuple(ext) == lf.CANDIDATES + lf.PRETRAINED_CANDIDATES
    n = len(w["context"])
    assert np.allclose(ext["chronos2"], 7.5) and ext["chronos2"].shape == (n, 24)
    assert np.allclose(ext["ens_chronos2"], 0.5 * base["ens"] + 0.5 * 7.5)
    for k in lf.CANDIDATES:
        assert np.array_equal(ext[k], base[k])
    call = fake.calls[-1]
    assert call["n"] == n and call["horizon"] == 24 and call["quantiles"] == (0.5,) and call["model"] == "chronos_2" and call["context"] is None
    assert all(x.size == 168 for x in call["inputs"])


def test_loadforecast_negative_model_output_is_clipped_at_zero(monkeypatch):
    monkeypatch.setattr(tsfm, "forecast", _FakeTsfm(-3.0))
    m, w = _model_and_windows()
    ext = m.candidates(w["context"], w["target_start"], w["building_id"], w["category"], pretrained=True)
    assert (ext["chronos2"] == 0).all() and np.allclose(ext["ens_chronos2"], 0.5 * ext["ens"])


def test_loadforecast_backtest_scores_include_the_pretrained_names(monkeypatch):
    from test_scilib_loadforecast import _synthetic
    monkeypatch.setattr(tsfm, "forecast", _FakeTsfm(lambda n, q: 10.0))
    load, ids, cats, starts = _synthetic(nb=3, T=1500)
    bt = lf.backtest_history(load, starts, ids, cats, holdout_hours=480, stride=48, pretrained=True)
    assert set(lf.PRETRAINED_CANDIDATES) <= set(bt["scores"])
    plain = lf.backtest_history(load, starts, ids, cats, holdout_hours=480, stride=48)
    assert not set(lf.PRETRAINED_CANDIDATES) & set(plain["scores"])


# ------------------------------------------------------------------------------------------------ aquatics (FoR41)
def _aq_data():
    from test_scilib_aquatics import _series
    return _series()


def _quantile_fake():
    """q25 / q50 / q75 = 1.0 / 2.0 / 3.0 for oxygen items; 11 / 12 / 15 for temperature (second call)."""
    state = {"i": 0}

    def val(n, q):
        base = [(1.0, 2.0, 3.0), (11.0, 12.0, 15.0)][state["i"] % 2]
        state["i"] += 1
        return np.broadcast_to(np.array(base, dtype=np.float32), (n, aq.H, 3))
    return val


def test_aquatics_default_unchanged_and_pretrained_zero_makes_no_call(monkeypatch):
    fake = _FakeTsfm(1.0)
    monkeypatch.setattr(tsfm, "forecast", fake)
    hist, dates = _aq_data()
    a = aq.fit_predict(hist, dates)
    b = aq.fit_predict(hist, dates, pretrained=0.0)
    assert all(np.array_equal(a[k], b[k]) for k in a) and fake.calls == []


def test_aquatics_pretrained_mixes_mu_and_sigma_as_documented(monkeypatch):
    fake = _FakeTsfm(_quantile_fake())
    monkeypatch.setattr(tsfm, "forecast", fake)
    hist, dates = _aq_data()
    base = aq.fit_predict(hist, dates)
    mix = aq.fit_predict(hist, dates, pretrained=0.5)
    n = hist.shape[0]
    mu_o = np.clip(2.0, 0, 45)
    sd_o = np.clip((3.0 - 1.0) / 1.349, 0.05, 50)
    mu_t = np.clip(12.0, 0, 45)
    sd_t = np.clip((15.0 - 11.0) / 1.349, 0.05, 50)
    assert np.allclose(mix["oxygen_mu"], 0.5 * base["oxygen_mu"] + 0.5 * mu_o)
    assert np.allclose(mix["oxygen_sigma"], 0.5 * base["oxygen_sigma"] + 0.5 * sd_o)
    assert np.allclose(mix["temperature_mu"], 0.5 * base["temperature_mu"] + 0.5 * mu_t)
    assert np.allclose(mix["temperature_sigma"], 0.5 * base["temperature_sigma"] + 0.5 * sd_t)
    assert len(fake.calls) == 2 and all(c["quantiles"] == (0.25, 0.5, 0.75) and c["horizon"] == 30 and c["n"] == n for c in fake.calls)
    assert all(x.size == hist.shape[1] for x in fake.calls[0]["inputs"])      # the whole history is passed


def test_aquatics_items_with_fewer_than_three_observed_days_keep_the_library_value(monkeypatch):
    fake = _FakeTsfm(_quantile_fake())
    monkeypatch.setattr(tsfm, "forecast", fake)
    hist, dates = _aq_data()
    hist = hist.copy()
    obs = np.flatnonzero(np.isfinite(hist[0, :, 0]))
    hist[0, :, 0] = np.nan
    hist[0, obs[-2:], 0] = 5.0                                                    # 2 observed oxygen days
    hist[0, obs[-1] + 0, 0] = 5.0
    base = aq.fit_predict(hist, dates)
    mix = aq.fit_predict(hist, dates, pretrained=0.5)
    assert np.array_equal(mix["oxygen_mu"][0], base["oxygen_mu"][0]) and np.array_equal(mix["oxygen_sigma"][0], base["oxygen_sigma"][0])
    assert not np.allclose(mix["oxygen_mu"][1], base["oxygen_mu"][1])
    assert fake.calls[0]["n"] == hist.shape[0] - 1


def test_aquatics_pretrained_argument_check():
    hist, dates = _aq_data()
    with pytest.raises(ValueError, match="pretrained must be"):
        aq.fit_predict(hist, dates, pretrained=2.0)


# ------------------------------------------------------------------------------------------------ anomsound (FoR40)
def _clips():
    from test_scilib_anomsound import _episode, _pack
    tr, ev, y, dom = _episode(0, "click")
    return _pack(tr), _pack(ev)


class _FakeEmbed:
    def __init__(self):
        self.calls = []

    def __call__(self, waveforms, lengths=None, sample_rate=16000.0, model="ast_audioset", kind="pooled"):
        self.calls.append((np.asarray(waveforms).shape, model, kind, float(sample_rate)))
        rng = np.random.default_rng(abs(hash(model)) % 1000 + np.asarray(waveforms).shape[0])
        d = 16
        return rng.standard_normal((np.asarray(waveforms).shape[0], d)).astype(np.float32)


def test_anomsound_default_score_clips_unchanged(monkeypatch):
    fe = _FakeEmbed()
    monkeypatch.setattr(audioenc, "embed", fe)
    tr, ev = _clips()
    a = an.score_clips(tr, ev)
    b = an.score_clips(tr, ev, embed=None)
    assert np.array_equal(a, b) and fe.calls == []


def test_anomsound_embed_option_is_the_rank_average_of_the_named_components(monkeypatch):
    fe = _FakeEmbed()
    monkeypatch.setattr(audioenc, "embed", fe)
    tr, ev = _clips()
    s = an.score_clips(tr, ev, embed=["ast_audioset", "clap_htsat"])
    assert [(c[1], c[2]) for c in fe.calls] == [("ast_audioset", "pooled"), ("ast_audioset", "pooled"),
                                                  ("clap_htsat", "pooled"), ("clap_htsat", "pooled")]
    raw = {}
    for m in ("ast_audioset", "clap_htsat"):
        te, ee = fe(tr["waveforms"], tr["lengths"], SR, m), fe(ev["waveforms"], ev["lengths"], SR, m)
        raw[f"{m}:nn2_pool"] = an.embedding_scores(te, ee)["nn2_pool"]
    want = an.rank_average(raw, list(raw))
    assert s.shape == (16,) and np.allclose(s, want)


def test_anomsound_embed_with_members_adds_the_log_mel_members_and_checks_names(monkeypatch):
    monkeypatch.setattr(audioenc, "embed", _FakeEmbed())
    tr, ev = _clips()
    only_emb = an.score_clips(tr, ev, embed="ast_audioset")
    both = an.score_clips(tr, ev, members=("maha",), embed="ast_audioset")
    assert not np.allclose(only_emb, both)
    with pytest.raises(ValueError, match="unknown embed_members"):
        an.score_clips(tr, ev, embed="ast_audioset", embed_members=("maha",))
    with pytest.raises(ValueError, match="unknown members"):
        an.score_clips(tr, ev, members=("nope",), embed="ast_audioset")
    with pytest.raises(ValueError, match="at least one model"):
        an.score_clips(tr, ev, embed=[])


# ------------------------------------------------------------------------------------------------ interface text
@pytest.mark.parametrize("mod,names", [(fc, ("pretrained_forecast", "pretrained")), (lf, ("PRETRAINED_CANDIDATES", "chronos2")),
                                        (aq, ("pretrained",)), (an, ("embed", "embed_members")),
                                        (up, ("pretrained=True", "udparse_pretrained"))])
def test_documentation_mentions_the_option_and_stays_factual(mod, names):
    doc = mod.__doc__
    for n in names:
        assert n in doc
    low = doc.lower()
    for word in ("recommend", " should ", "state of the art", "worst", "naive", "persistence", "last value", "last-value"):
        assert word not in low


# ------------------------------------------------------------------------------------------------------------ udparse
def test_udparse_pretrained_option_routes_to_the_pretrained_parser(monkeypatch):
    calls = []

    def fake(sents, **kw):
        calls.append(len(sents))
        return [{"head": [0] + [1] * (len(s) - 1), "deprel": ["root"] + ["dep"] * (len(s) - 1), "upos": ["X"] * len(s)} for s in sents]

    monkeypatch.setattr(upp, "available", lambda: True)
    monkeypatch.setattr(upp, "parse_gold_tokens", fake)
    out = up.fit_predict([], [[["a", "b", "c"], ["d"]], [["e", "f"]]], pretrained=True)
    assert calls == [2, 1]
    assert out[0][0] == {"head": [0, 1, 1], "deprel": ["root", "dep", "dep"]}
    assert [len(x) for x in out] == [2, 1]


def test_udparse_pretrained_option_fails_clearly_when_unavailable(monkeypatch):
    monkeypatch.setattr(upp, "available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        up.fit_predict([], [[["a"]]], pretrained=True)
