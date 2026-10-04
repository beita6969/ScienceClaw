"""FoR36 MUSDB18 adapter: museval-v4 SDR, roles/pools of the data team's full_v1, decoding, episodes.

Data-dependent tests skip when ``<DATA_ROOT>/for36-musdb18/full_v1`` (or a runnable ffmpeg) is absent; excerpt
decoding is cached in ``cache/tasks/FoR36`` (a cold cache costs ~15 s here, a warm one ~1 s).
"""
from __future__ import annotations

import numpy as np
import pytest

from scienceclaw.bench.splits import SplitPlan
from scienceclaw.bench.tasks import for36_musdb as m
from scienceclaw.bench.tasks._adapter_utils_for36_46_49_52 import PoolExhausted
from scienceclaw.config import BenchConfig

AD = m.Adapter()
HAVE_ROLES = (AD.base / "sampling-manifest.json").is_file()
OK, WHY = AD.available()
needs_roles = pytest.mark.skipif(not HAVE_ROLES, reason=f"no FoR36 full_v1 under {AD.base}")
needs_audio = pytest.mark.skipif(not OK, reason=f"FoR36 unavailable: {WHY}")


# ------------------------------------------------------------------------------------------------ metric
def test_framewise_sdr_matches_museval_reference_values():
    """Values computed with museval 0.4.1 ``metrics.bss_eval(..., window=hop=1000, framewise_filters=False,
    bsseval_sources_version=False)`` on the same deterministic signals."""
    rng = np.random.default_rng(12345)
    sr = 1000
    ref = rng.normal(size=(4, 3 * sr, 2))
    ref[2, sr:2 * sr] = 0
    est = 0.8 * ref + 0.3 * ref[[1, 2, 3, 0]] + 0.2 * rng.normal(size=ref.shape)
    expected = np.array([[7.806709, np.nan, 7.842842], [7.843025, np.nan, 7.849838],
                         [7.759789, np.nan, 7.621154], [7.515533, np.nan, 7.648863]])
    got = m.framewise_sdr(ref, est, sr)
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    assert np.allclose(got[~np.isnan(got)], expected[~np.isnan(expected)], atol=1e-5)


def test_sdr_known_snr_perfect_and_aggregation():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(4, 4000, 2))
    noise = rng.normal(size=ref.shape)
    noise *= np.sqrt(np.sum(ref ** 2) / np.sum(noise ** 2)) / 10 ** (10 / 20)     # global 10 dB SNR
    s = m.item_scores(ref, ref + noise, sr=1000)
    assert np.all(np.abs(s - 10.0) < 0.6)
    assert np.all(np.isinf(m.item_scores(ref, ref.copy(), sr=1000)))
    per = np.array([[1.0, 2.0, np.nan, 4.0], [3.0, 4.0, 5.0, np.nan], [5.0, 0.0, 7.0, 8.0]])
    assert m.aggregate(per) == pytest.approx(np.mean([3.0, 2.0, 6.0, 6.0]))
    assert m.aggregate(np.full((2, 4), np.nan)) is None


def test_norm_score_db():
    from scienceclaw.bench.tasks._adapter_utils_for36_46_49_52 import norm_score_db
    assert norm_score_db(3.0, 0.0) == pytest.approx(10 ** 0.3)
    assert norm_score_db(-20.0, 0.0) == pytest.approx(0.01)
    assert norm_score_db(50.0, 0.0) == 10.0
    assert norm_score_db(None, 0.0) == 0.0


# ------------------------------------------------------------------------------------------------ availability
def test_unavailable_reason_names_the_missing_dataset(tmp_path):
    ok, why = m.Adapter(data_root=tmp_path).available()
    assert not ok and "sampling-manifest.json" in why and "full_v1" in why


@needs_roles
def test_unavailable_reason_when_ffmpeg_cannot_run(tmp_path):
    ok, why = m.Adapter(ffmpeg=tmp_path / "no-such-ffmpeg").available()
    assert not ok and "ffmpeg" in why


# ------------------------------------------------------------------------------------------------ roles / pools
@needs_roles
def test_pools_follow_the_data_team_roles():
    data = m._load(AD.root)
    assert data.track_counts == {"src": 64, "val": 14, "id": 50, "ood": 0, "train": 16, "dev": 6}
    assert data.pools["ood"] == []
    by_pool = {p: {t.track_id for t in data.tracks.values() if t.pool == p} for p in data.track_counts}
    assert sum(len(v) for v in by_pool.values()) == 150 == len(data.tracks)         # roles are track-disjoint
    assert all(t.startswith("musdb18/test/") for t in by_pool["id"])
    assert all(t.startswith("musdb18/train/") for p in ("src", "val", "train", "dev") for t in by_pool[p])
    # every excerpt lies inside its track, and belongs to exactly one pool
    seen: set[str] = set()
    for p in ("src", "val", "id", "train", "dev"):
        assert not seen & set(data.pools[p])
        seen |= set(data.pools[p])
        for i in data.pools[p]:
            ex = data.excerpts[i]
            assert ex.track.pool == p
            assert (ex.track.slots[ex.k] + 1) * m.EXCERPT_S <= ex.track.duration_s
    assert len(data.pools["src"]) == 128 and len(data.pools["val"]) == 70 and len(data.pools["id"]) == 150
    assert len(data.pools["dev"]) == 24 and 60 <= len(data.pools["train"]) <= 64
    again = m._load(AD.root)                                                          # deterministic
    assert again.pools == data.pools


@needs_audio
def test_available_reports_full_v1_sizes():
    assert OK and "full_v1" in WHY and "'ood': 0" in WHY


@needs_audio
def test_build_episodes_sizes_disjointness_determinism_and_empty_ood():
    a = AD.build_episodes("id", 4, seed=11)
    v = AD.build_episodes("val", 2, seed=12)
    s = AD.build_episodes("src", 8, seed=13)
    assert (len(a), len(v), len(s)) == (4, 2, 8) and all(e.n_items == 16 for e in a + v + s)
    ids = {sp: [i for e in eps for i in e.lineage["item_ids"]] for sp, eps in (("id", a), ("val", v), ("src", s))}
    assert all(len(x) == len(set(x)) for x in ids.values())          # no item twice, even across episodes
    assert not (set(ids["id"]) & set(ids["val"]) | set(ids["id"]) & set(ids["src"]) | set(ids["val"]) & set(ids["src"]))
    tracks = {sp: {t for e in eps for t in e.lineage["tracks"]} for sp, eps in (("id", a), ("val", v), ("src", s))}
    assert not (tracks["id"] & tracks["val"] or tracks["id"] & tracks["src"] or tracks["val"] & tracks["src"])
    vis = {AD._data.get().excerpts[i].track.track_id for e in a + v + s
           for i in e.lineage["train_item_ids"] + e.lineage["dev_item_ids"]}
    assert vis and not vis & set().union(*tracks.values())          # visible D_E comes from the reserve tracks
    assert all(t.startswith("musdb18/test/") for t in tracks["id"])
    assert [e.lineage["item_ids"] for e in AD.build_episodes("id", 4, seed=11)] == [e.lineage["item_ids"] for e in a]
    assert [e.lineage["item_ids"] for e in AD.build_episodes("id", 2, seed=11)] == [e.lineage["item_ids"] for e in a[:2]]
    assert [e.lineage["item_ids"] for e in AD.build_episodes("id", 2, seed=99)] != [e.lineage["item_ids"] for e in a[:2]]
    assert AD.build_episodes("ood", 3, seed=14) == []
    with pytest.raises(PoolExhausted):
        AD.build_episodes("val", 5, seed=12)
    with pytest.raises(ValueError):
        AD.build_episodes("rep", 1, seed=1)


@needs_audio
def test_split_plan_tolerates_the_empty_ood():
    cfg = BenchConfig(disciplines=["FoR36"], items_per_episode=16, rounds=3, n_val=2, n_id=4, n_ood=2, seed=5)
    plan = SplitPlan.build(cfg, {"FoR36": AD})
    assert [len(plan.episodes[sp]["FoR36"]) for sp in ("src", "val", "id", "ood")] == [3, 2, 4, 0]
    assert any("FoR36/ood: adapter returned 0 of 2" in w for w in plan.warnings)
    assert len(plan.rep_until(3)) == 3
    assert plan.manifest()["schema"]                                # lineage is JSON-serialisable


# ------------------------------------------------------------------------------------------------ decoding
@needs_audio
def test_decoded_excerpts_are_verified_finite_stereo_and_roughly_additive():
    data = AD._data.get()
    tr = data.tracks[sorted(t for t, x in data.tracks.items() if x.pool == "id")[3]]
    store = AD._store.get()
    mix, stems = store.get(tr)
    n = int(m.EXCERPT_S * m.SR)
    assert mix.shape == (len(tr.slots), n, 2) and stems.shape == (len(tr.slots), 4, n, 2)
    assert mix.dtype == stems.dtype == np.float32 and np.isfinite(mix).all() and np.isfinite(stems).all()
    assert np.abs(mix).max() > 0.01
    assert not np.array_equal(mix, stems[:, 0])
    res = stems.sum(axis=1) - mix                                   # AAC: the stems only sum approximately to the mix
    snr = 10 * np.log10(np.sum(mix ** 2) / np.sum(res ** 2))
    assert snr > 10.0
    again = store.get(tr)
    assert again[0] is mix                                          # memory LRU hit
    if store.verified:                                              # pinned binary: PCM sha256 checked in _decode
        mix2, stems2 = m._ExcerptStore(AD._ffmpeg, AD._pinned_sha256)._decode(tr)
        assert np.array_equal(mix, mix2) and np.array_equal(stems, stems2)


@needs_audio
def test_decode_stream_rejects_a_wrong_hash():
    tr = next(t for t in AD._data.get().tracks.values() if t.pool == "id")
    idx, sha = tr.streams["mixture"]
    assert m.decode_stream(AD._ffmpeg, tr.path, idx).shape[1] == 2
    with pytest.raises(RuntimeError, match="decoded_pcm_sha256"):
        m.decode_stream(AD._ffmpeg, tr.path, idx, "0" * 64)


# ------------------------------------------------------------------------------------------------ episodes
@needs_audio
@pytest.mark.parametrize("split", ["id", "val"])
def test_episode_reference_oracle_constraints(split):
    data = AD._data.get()
    e = AD.build_episodes(split, 1, seed=5, items_per_episode=2)[0]
    ht = e.tool("separate_htdemucs")
    assert ht is not None and set(ht.inputs) == {"mixtures"} and set(ht.outputs) == {"estimates"}
    assert "mixture-only" in ht.inputs["mixtures"].description and "staged" in ht.description
    n_len = int(m.EXCERPT_S * m.SR)
    mixes = e.tool("load_eval_inputs").fn({}, {})["mixtures"]
    assert mixes.shape == (2, n_len, 2) and mixes.dtype == np.float32
    assert e.lineage["pool"] == "iid" and e.lineage["ood_kind"] is None and e.split == split
    assert "scilib.audiosep" in e.objective and "separate(" in e.objective          # the domain library is documented
    assert "`estimates` output port" in e.objective and "directly as y" in e.objective
    if split == "id":
        tr = e.tool("load_train").fn({}, {})
        assert tr["stems"].shape == (m.N_TRAIN, 4, n_len, 2) and tr["sample_rate"] == m.SR
        assert len(tr["track_ids"]) == m.N_TRAIN and all(isinstance(t, str) for t in tr["track_ids"])
        assert e.tool("load_train").fn({}, {})["track_ids"] == tr["track_ids"]      # deterministic, opaque labels
        assert not any("musdb" in t or "/" in t for t in tr["track_ids"])
        dv = e.tool("load_dev_inputs").fn({}, {})
        assert dv["dev_mixtures"].shape == (m.N_DEV, n_len, 2)
        sd = e.tool("score_dev").fn({"dev_estimates": np.repeat(dv["dev_mixtures"][:, None], 4, axis=1)}, {})
        assert sd["dev_sdr"] == pytest.approx(sd["dev_reference_sdr"])
        with pytest.raises(ValueError):
            e.tool("score_dev").fn({"dev_estimates": np.zeros((1, 4, n_len, 2))}, {})
    ref_y = np.repeat(mixes[:, None], 4, axis=1)
    rv = e.evaluate(ref_y, None)
    assert rv.hard_ok() and rv.primary == pytest.approx(rv.details["reference"]) and not rv.accepted
    assert rv.details["norm_score"] == pytest.approx(1.0)
    oracle = np.stack([AD._clip(data.excerpts[i])[1] for i in e.lineage["item_ids"]])
    ov = e.evaluate(oracle, None)
    assert ov.accepted and ov.primary > rv.primary + 50 and ov.details["norm_score"] == 10.0
    noisy = oracle + 0.05 * np.random.default_rng(1).normal(size=oracle.shape).astype(np.float32)
    nv = e.evaluate(noisy, None)
    assert rv.primary < nv.primary < ov.primary
    h, _ = e.check_constraints(oracle[:1], None)
    assert not h["output_shape"]
    bad = oracle.copy()
    bad[0, 1, : m.SR] = 0.0
    h, msgs = e.check_constraints(bad, None)
    assert h["output_shape"] and not h["no_silent_window"]
    bad2 = oracle.copy()
    bad2[1, 2, 5, 0] = np.nan
    h, _ = e.check_constraints(bad2, None)
    assert not h["finite"]
    assert e.evaluate(bad2, None).primary is None
    pooled = AD.pooled_metric([ov.details["pooled_payload"], rv.details["pooled_payload"]])
    assert pooled is not None and not np.isnan(pooled)
    assert AD.pooled_metric([{"item_sdr": None, "ref_item_sdr": rv.details["pooled_payload"]["ref_item_sdr"]}]) \
        == pytest.approx(rv.details["reference"])
