"""Pretrained-model wrappers of scilib (htdemucs, Stanza French, DINOv2 pixel classifier): weight lookup, switches, interface
text, and the array plumbing with faked networks (no torch / weights needed)."""
from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scilib  # noqa: E402
from scilib import _pretrained as pre
from test_adapter_visible_text import scan_acceptance  # noqa: E402
from scilib import audiosep_pretrained as ap
from scilib import phenoseg as ps
from scilib import phenoseg_deep as pd
from scilib import phenoseg_sam as psam
from scilib import udparse as up
from scilib import udparse_pretrained as upp
from scienceclaw.runtime.integrity import scan_code

MODULES = ("audiosep_pretrained", "udparse_pretrained", "phenoseg_deep", "phenoseg_sam")


# ------------------------------------------------------------------------------------------------ weight lookup / switches
def test_model_path_prefers_env_root(tmp_path, monkeypatch):
    (tmp_path / "demucs").mkdir()
    monkeypatch.setenv("SCIENCECLAW_MODELS", str(tmp_path))
    assert pre.model_path("demucs") == tmp_path / "demucs"
    assert pre.model_path("no-such-dir-xyz") is None


def test_switched_off(monkeypatch):
    for k in ("SCIENCECLAW_NO_PRETRAINED", "SCIENCECLAW_NO_SCILIB"):
        monkeypatch.delenv(k, raising=False)
    assert not pre.switched_off()
    monkeypatch.setenv("SCIENCECLAW_NO_PRETRAINED", "1")
    assert pre.switched_off()
    monkeypatch.delenv("SCIENCECLAW_NO_PRETRAINED")
    monkeypatch.setenv("SCIENCECLAW_NO_SCILIB", "1")
    assert pre.switched_off()


@pytest.mark.parametrize("name", MODULES)
def test_unavailable_without_weights(name, tmp_path, monkeypatch):
    monkeypatch.setattr(pre, "MODEL_ROOTS", ())
    monkeypatch.setenv("SCIENCECLAW_MODELS", str(tmp_path))
    mod = importlib.import_module(f"scilib.{name}")
    assert mod.available() is False
    assert scilib.describe_extra(name) == ""


@pytest.mark.parametrize("name", MODULES)
def test_describe_extra_shown_only_when_available(name, monkeypatch):
    mod = importlib.import_module(f"scilib.{name}")
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    monkeypatch.setattr(mod, "available", lambda: True)
    text = scilib.describe_extra(name)
    assert text.startswith(f"\n\nLibrary `scilib.{name}`, importable in code nodes:")
    monkeypatch.setenv("SCIENCECLAW_NO_SCILIB", "1")
    assert scilib.describe_extra(name) == ""


@pytest.mark.parametrize("name", MODULES)
def test_available_false_when_switched_off(name, monkeypatch):
    mod = importlib.import_module(f"scilib.{name}")
    monkeypatch.setenv("SCIENCECLAW_NO_PRETRAINED", "1")
    assert mod.available() is False


# ------------------------------------------------------------------------------------------------ visible text (F5)
BANNED = [r"success criterion", r"accepted iff", r"acceptance", r"margin", r"reference (recipe|method|score|value)",
          r"\bthreshold of\b", r"\bexpected (score|LAS|PQ|SDR)"]


@pytest.mark.parametrize("name", MODULES)
def test_docs_state_facts_only(name):
    mod = importlib.import_module(f"scilib.{name}")
    doc = mod.__doc__
    for pat in BANNED:
        assert not re.search(pat, doc, re.I), (name, pat)
    assert not scan_acceptance(doc)
    assert "Model" in doc and re.search(r"trained|self-supervised", doc), "the description must say what the network was trained on"


# ------------------------------------------------------------------------------------------------ sandbox import
@pytest.mark.parametrize("name,fn", [("audiosep_pretrained", "separate_pretrained"), ("udparse_pretrained", "parse_gold_tokens"),
                                     ("phenoseg_deep", "fit_predict_deep"),
                                     ("phenoseg_sam", "fit_sam_selector")])
def test_import_passes_static_scan(name, fn):
    assert scan_code(f"from scilib.{name} import {fn}\n\ndef run(inputs, config):\n    return {{}}\n") == []


def test_sandbox_forwards_model_root(monkeypatch, tmp_path):
    from scienceclaw.runtime import sandbox
    (tmp_path / "hf").mkdir()
    monkeypatch.setenv("SCIENCECLAW_MODELS", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SOME_API_KEY", "secret")
    env = sandbox.build_env(tmp_path / "work")
    assert env["SCIENCECLAW_MODELS"] == str(tmp_path) and env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["HF_HOME"] == str(tmp_path / "hf") and env["HF_HUB_OFFLINE"] == "1"
    assert "SOME_API_KEY" not in env


# ------------------------------------------------------------------------------------------------ htdemucs plumbing
class _FakeDemucs:
    samplerate = 44100
    sources = ["drums", "bass", "other", "vocals"]


def _fake_run(model, wav, shifts, overlap, device):
    assert wav.dtype == np.float32 and wav.shape[0] == 2
    scale = np.array([1.0, 2.0, 3.0, 4.0], np.float32)[:, None, None]      # in Demucs order: drums, bass, other, vocals
    return wav[None] * scale


@pytest.fixture
def fake_demucs(monkeypatch):
    monkeypatch.setattr(ap, "_local_ok", lambda: True)
    monkeypatch.setattr(ap, "_load", lambda name, dev: _FakeDemucs())
    monkeypatch.setattr(ap, "_run", _fake_run)
    monkeypatch.setattr(ap, "torch_device", lambda d="auto": "cpu")
    monkeypatch.setattr(ap, "set_cpu_threads", lambda n=4: None)


def test_separate_pretrained_order_scale_and_length(fake_demucs):
    rng = np.random.default_rng(0)
    n = 22050
    x = (0.1 * rng.normal(size=(3, n, 2)) + 0.02).astype(np.float32)
    est = ap.separate_pretrained(x)
    assert est.shape == (3, 4, n, 2) and est.dtype == np.float32
    # output order = vocals, drums, bass, other; the fake network multiplies the normalised mixture by 4, 1, 2, 3 (Demucs
    # order drums, bass, other, vocals = 1, 2, 3, 4), and the result is de-normalised back to the mixture scale
    for i in range(3):
        wav = ap._resample(x[i].T, ap.SR, 44100)
        mu = wav.mean(0).mean()
        for j, k in enumerate((4.0, 1.0, 2.0, 3.0)):
            expect = ap._fit_length(ap._resample(k * (wav - mu) + mu, 44100, ap.SR), n).T
            np.testing.assert_allclose(est[i, j], expect, atol=1e-4)


def test_separate_pretrained_silence_and_shape_errors(fake_demucs):
    est = ap.separate_pretrained(np.zeros((1, 4000, 2), np.float32))
    assert est.shape == (1, 4, 4000, 2) and not est.any()
    with pytest.raises(ValueError):
        ap.separate_pretrained(np.zeros((1, 4000)))


def test_separate_pretrained_rejects_bad_runtime_parameters_and_empty_is_noop(fake_demucs, monkeypatch):
    with pytest.raises(ValueError, match="finite"):
        ap.separate_pretrained(np.array([[[np.nan, 0.0]]], np.float32))
    with pytest.raises(ValueError, match="sample_rate"):
        ap.separate_pretrained(np.zeros((1, 4000, 2), np.float32), sample_rate=0)
    with pytest.raises(ValueError, match="shifts"):
        ap.separate_pretrained(np.zeros((1, 4000, 2), np.float32), shifts=-1)
    with pytest.raises(ValueError, match="overlap"):
        ap.separate_pretrained(np.zeros((1, 4000, 2), np.float32), overlap=1.0)
    monkeypatch.setattr(ap, "_load", lambda *a: (_ for _ in ()).throw(AssertionError("empty batch loaded model")))
    out = ap.separate_pretrained(np.zeros((0, 4000, 2), np.float32))
    assert out.shape == (0, 4, 4000, 2) and out.dtype == np.float32


def test_separate_pretrained_needs_weights(monkeypatch):
    monkeypatch.setattr(ap, "_local_ok", lambda: False)
    monkeypatch.delenv("SCIENCECLAW_REMOTE_SPOOL", raising=False)
    with pytest.raises(RuntimeError):
        ap.separate_pretrained(np.zeros((1, 100, 2)))


# ------------------------------------------------------------------------------------------------ Stanza plumbing
def _fake_pipeline(calls, break_kind=None):
    def nlp(batch):
        calls.append([len(b) for b in batch])
        sents = []
        for forms in batch:
            n = len(forms)
            heads = [i + 2 if i + 1 < n else 0 for i in range(n)]                # right-branching chain
            rels = ["nsubj:pass" if i == 0 else "dep" for i in range(n)]
            if break_kind == "cycle" and n > 2:
                heads = [2, 1] + heads[2:]
            words = [types.SimpleNamespace(head=h, deprel=r, upos="NOUN") for h, r in zip(heads, rels)]
            if break_kind == "count":
                words = words[:-1]
            sents.append(types.SimpleNamespace(words=words))
        return types.SimpleNamespace(sentences=sents)
    return nlp


@pytest.fixture
def fake_stanza(monkeypatch):
    calls: list = []
    monkeypatch.setattr(upp, "available", lambda: True)
    monkeypatch.setattr(upp, "_local_ok", lambda: True)
    monkeypatch.setattr(upp, "torch_device", lambda d="auto": "cpu")
    monkeypatch.setattr(upp, "set_cpu_threads", lambda n=4: None)
    monkeypatch.setattr(upp, "_pipeline", lambda dev, model=None: _fake_pipeline(calls))
    return calls


def test_parse_gold_tokens_order_and_format(fake_stanza):
    sents = [["a", "b", "c", "d"], {"words": [{"form": "x"}]}, [], ["p", "q"], ["r", "s", "t"]]
    out = upp.parse_gold_tokens(sents, batch_size=2)
    assert [len(o["head"]) for o in out] == [4, 1, 0, 2, 3]
    assert out[2] == {"head": [], "deprel": [], "upos": []}
    for o in out:
        assert set(o) == {"head", "deprel", "upos"}
        if o["head"]:
            assert not up.check_parse(o, len(o["head"]))
    assert out[0]["deprel"][0] == "nsubj:pass"
    assert fake_stanza == [[1, 2], [3, 4]]                                        # sorted by length, batched, empties skipped


def test_parse_gold_tokens_invalid_tree_falls_back(monkeypatch):
    calls: list = []
    monkeypatch.setattr(upp, "available", lambda: True)
    monkeypatch.setattr(upp, "_local_ok", lambda: True)
    monkeypatch.setattr(upp, "torch_device", lambda d="auto": "cpu")
    monkeypatch.setattr(upp, "set_cpu_threads", lambda n=4: None)
    monkeypatch.setattr(upp, "_pipeline", lambda dev, model=None: _fake_pipeline(calls, "cycle"))
    out = upp.parse_gold_tokens([["a", "b", "c", "d"]])
    assert not up.check_parse(out[0], 4)


def test_parse_gold_tokens_word_count_change_raises(monkeypatch):
    monkeypatch.setattr(upp, "available", lambda: True)
    monkeypatch.setattr(upp, "_local_ok", lambda: True)
    monkeypatch.setattr(upp, "torch_device", lambda d="auto": "cpu")
    monkeypatch.setattr(upp, "set_cpu_threads", lambda n=4: None)
    monkeypatch.setattr(upp, "_pipeline", lambda dev, model=None: _fake_pipeline([], "count"))
    with pytest.raises(RuntimeError):
        upp.parse_gold_tokens([["a", "b", "c"]])


def test_parse_gold_tokens_needs_weights(monkeypatch):
    monkeypatch.setattr(upp, "available", lambda: False)
    monkeypatch.setattr(upp, "_local_ok", lambda: False)
    with pytest.raises(RuntimeError):
        upp.parse_gold_tokens([["a"]])
    assert upp.parse_gold_tokens([[]]) == [{"head": [], "deprel": [], "upos": []}]


def test_parse_gold_tokens_passes_model_path(monkeypatch):
    seen: list = []
    monkeypatch.setattr(upp, "available", lambda: True)
    monkeypatch.setattr(upp, "_local_ok", lambda: True)
    monkeypatch.setattr(upp, "torch_device", lambda d="auto": "cpu")
    monkeypatch.setattr(upp, "set_cpu_threads", lambda n=4: None)
    monkeypatch.setattr(upp, "_pipeline", lambda dev, model=None: (seen.append(model), _fake_pipeline([]))[1])
    upp.parse_gold_tokens([["a", "b"]])
    upp.parse_gold_tokens([["a", "b"]], model="/tmp/adapted.pt")
    assert seen == [None, "/tmp/adapted.pt"]


def test_finetune_label_mapping_and_conllu():
    known = {"nsubj", "nsubj:pass", "expl:pv", "expl:subj", "expl:comp", "nmod", "obl:arg", "dep", "root"}
    assert upp._closest_label("nsubj:pass", "x", known) == "nsubj:pass"
    assert upp._closest_label("expl", "se", known) == "expl:pv"
    assert upp._closest_label("expl", "Il", known) == "expl:subj"
    assert upp._closest_label("expl", "y", known) == "expl:comp"
    assert upp._closest_label("nmod:range", "x", known) == "nmod"                  # subtype unknown: base label
    assert upp._closest_label("obl", "x", known) == "obl:arg"                       # base unknown: a label with the same base
    assert upp._closest_label("weird", "x", known) == "dep"
    sents = [{"words": [{"form": "Il", "lemma": "il", "upos": "PRON", "feats": "", "head": 2, "deprel": "expl"},
                        {"form": "pleut", "lemma": "pleuvoir", "upos": "VERB", "head": 0, "deprel": "root"}]}]
    rows = [r.split("\t") for r in upp._conllu(sents, known).splitlines() if r and not r.startswith("#")]
    assert [r[7] for r in rows] == ["expl:subj", "root"] and [r[6] for r in rows] == ["2", "0"]
    assert rows[0][5] == "_" and all(len(r) == 10 for r in rows)
    with pytest.raises(ValueError):
        upp._conllu([{"words": [{"form": "a", "head": 5, "deprel": "dep"}]}], known)


def test_finetune_is_disabled_before_any_backend_or_data_access(monkeypatch):
    events = []

    # These hooks fail the test if the implementation inspects weights,
    # checks local availability, or attempts to use the remote worker before
    # returning the policy error.
    monkeypatch.setattr(upp, "model_path", lambda *args, **kwargs: events.append("model_path"))
    monkeypatch.setattr(upp, "_local_ok", lambda: events.append("local") or True)
    monkeypatch.setattr(upp._remote, "enabled", lambda: events.append("remote") or True)
    monkeypatch.setattr(upp._remote, "call", lambda *args, **kwargs: events.append("remote_call"))

    with pytest.raises(RuntimeError, match="finetune.*disabled.*frozen-inference"):
        # An empty input also proves the policy check precedes argument/data
        # validation; no cache, worker, or trainer should be reached.
        upp.finetune([])
    assert events == []


def test_las_of_pretrained_output_scores(fake_stanza):
    gold = [{"words": [{"form": "a", "head": 2, "deprel": "nsubj:pass"}, {"form": "b", "head": 0, "deprel": "root"}]}]
    out = upp.parse_gold_tokens(gold)
    assert up.las_uas(gold, out)["las"] == 0.5                                     # right head, wrong relation for word 2 ("dep" vs "root")


# ------------------------------------------------------------------------------------------------ DINOv2 pixel classifier
def _scene(seed, H=64):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :H]
    img = np.zeros((H, H, 3), np.float32)
    img[:] = (0.45, 0.33, 0.24)
    sem = np.zeros((H, H), np.int16)
    for cy, cx in ((18, 18), (44, 40)):
        sem[(yy - cy) ** 2 + (xx - cx) ** 2 <= 9 ** 2] = 1
    for cy, cx in ((50, 12), (10, 50)):
        sem[(yy - cy) ** 2 + (xx - cx) ** 2 <= 4 ** 2] = 2
    img[sem == 1] = (0.10, 0.45, 0.10)
    img[sem == 2] = (0.35, 0.65, 0.15)
    img += rng.normal(0, 0.02, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), sem


def _fake_patch_features(images, backbone="dinov2_large", size=728, device="auto"):
    """Grid of 4x4-pixel block means of the RGB image, tiled to 12 channels (stands in for the encoder's patch grid)."""
    X = np.asarray(images).astype(np.float32) / 255.0
    n, H, W, _ = X.shape
    g = X.reshape(n, H // 8, 8, W // 8, 8, 3).mean((2, 4))
    return np.concatenate([g, g ** 2, g * 0.5, np.roll(g, 1, axis=-1)], axis=-1)


@pytest.fixture
def fake_dino(monkeypatch):
    monkeypatch.setattr(pd, "patch_features", _fake_patch_features)


def test_grid_size():
    assert pd._grid_size(512, 512, 728) == (52, 52)
    assert pd._grid_size(512, 1024, 728) == (26, 52)
    assert pd._grid_size(3, 3, 14) == (1, 1)


def test_interp_matches_grid_values_and_border():
    g = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    H = W = 4                                                                     # cell centres at pixel-centre coordinates 1 and 3
    r = pd._interp(g, H, W, np.array([0, 3]), np.array([0, 3]))                   # outside the outermost centres: border value
    np.testing.assert_allclose(r[0], g[0, 0])
    np.testing.assert_allclose(r[1], g[1, 1])
    m = pd._interp(g, H, W, np.array([1.5]), np.array([1.5]))                     # pixel centre 2.0: halfway between the cells
    np.testing.assert_allclose(m[0], g.reshape(4, 3).mean(0), rtol=1e-5)


def test_fit_predict_deep_shapes_and_semantics(fake_dino):
    tr = [_scene(s) for s in range(4)]
    imgs, sems = np.stack([t[0] for t in tr]), np.stack([t[1] for t in tr])
    te = np.stack([_scene(s)[0] for s in (10, 11)])
    model = pd.fit_deep_classifier(imgs, sems, pca_dim=6, n_estimators=30, pixels_per_image=3000)
    assert model.components.shape == (6, 12) and model.n_trees >= 5 and model.plant_size > 1
    probs = pd.predict_deep_probs(model, te, stride=2)
    assert probs.shape == (2, 64, 64, 3) and probs.dtype == np.float16
    np.testing.assert_allclose(probs.astype(np.float32).sum(-1), 1.0, atol=5e-3)
    truth = np.stack([_scene(s)[1] for s in (10, 11)])
    assert (probs.argmax(-1) == truth).mean() > 0.9
    pred_a = pd.fit_predict_deep(imgs, sems, [te, imgs[:1]], seed=0, pca_dim=6, n_estimators=30, pixels_per_image=3000)
    pred_b = pd.fit_predict_deep(imgs, sems, te, seed=0, pca_dim=6, n_estimators=30, pixels_per_image=3000)
    assert len(pred_a) == 2 and set(pred_b) == {"semantics", "plant_instances", "leaf_instances"}
    assert pred_b["semantics"].shape == (2, 64, 64)
    np.testing.assert_array_equal(pred_a[0]["semantics"], pred_b["semantics"])   # deterministic given the seed


def test_fit_deep_without_hand_features(fake_dino):
    tr = [_scene(s) for s in range(3)]
    imgs, sems = np.stack([t[0] for t in tr]), np.stack([t[1] for t in tr])
    model = pd.fit_deep_classifier(imgs, sems, pca_dim=5, use_hand=False, n_estimators=20, pixels_per_image=1500)
    assert model.booster.num_feature() == 5
    assert pd.predict_deep_probs(model, imgs[:1]).shape == (1, 64, 64, 3)


def test_fit_deep_needs_all_classes(fake_dino):
    imgs = np.stack([_scene(0)[0]])
    with pytest.raises(ValueError):
        pd.fit_deep_classifier(imgs, np.zeros((1, 64, 64), np.int16), pca_dim=4)


def test_score_against_pq_plus(fake_dino):
    tr = [_scene(s) for s in range(4)]
    imgs, sems = np.stack([t[0] for t in tr]), np.stack([t[1] for t in tr])
    pred = pd.fit_predict_deep(imgs, sems, imgs, pca_dim=6, n_estimators=30, pixels_per_image=3000)
    gt = dict(semantics=sems, plant_instances=(sems == 1).astype(np.int32), leaf_instances=(sems == 1).astype(np.int32))
    r = ps.pq_plus(pred, gt)
    assert r["iou_crop"] > 80 and r["iou_soil"] > 90


# ------------------------------------------------------------------------------------------------ SAM 2.1 proposal selector
def _pool_of(shapes, H=32):
    M = np.zeros((len(shapes), H, H), bool)
    for k, (y0, y1, x0, x1) in enumerate(shapes):
        M[k, y0:y1, x0:x1] = True
    return M


def test_nms_removes_near_duplicates():
    pytest.importorskip("torch")
    M = _pool_of([(0, 10, 0, 10), (0, 10, 0, 10), (0, 10, 1, 11), (20, 30, 20, 30)])
    keep = psam._nms(M, np.array([0.5, 0.9, 0.4, 0.7]), 0.8)
    assert set(keep.tolist()) == {1, 3}


def test_select_orders_by_score_and_limits_overlap():
    M = _pool_of([(0, 10, 0, 10), (0, 10, 5, 15), (20, 30, 20, 30), (0, 4, 0, 4)])
    score = np.array([0.9, 0.8, 0.6, 0.2])
    assert psam._select(M, score, 0.3, 0.25) == [0, 2]                            # mask 1 overlaps mask 0 by half; mask 3 is below the threshold
    assert psam._select(M, score, 0.3, 0.6) == [0, 1, 2]
    assert psam._select(M, score, 0.3, 0.25, allowed=np.array([False, True, True, True])) == [1, 2]


def test_compose_classes_by_crop_share_and_clips_leaves_to_crop():
    H = 32
    M = _pool_of([(2, 12, 2, 12), (18, 24, 18, 24), (2, 7, 2, 7)])
    probs = np.zeros((H, H, 3), np.float32)
    probs[..., 0] = 1
    probs[2:12, 2:12] = (0, 0.9, 0.1)
    probs[18:24, 18:24] = (0, 0.1, 0.9)
    F = np.zeros((3, len(psam.FEATS)), np.float32)
    F[:, psam.FEATS.index("share")] = [0.9, 0.1, 0.9]
    pool = dict(masks=M)
    prm = dict(psam.DEFAULT_PARAMS)
    out = psam._compose(pool, F, np.array([0.9, 0.8, 0.1]), np.array([0.2, 0.2, 0.9]), probs, prm)
    assert out["semantics"][5, 5] == 1 and out["semantics"][20, 20] == 2 and out["semantics"][30, 30] == 0
    assert out["plant_instances"][5, 5] == 1 and out["plant_instances"][20, 20] == 0        # weed regions get no plant id
    assert out["leaf_instances"][4, 4] == 1 and out["leaf_instances"][9, 9] == 0            # the leaf proposal is the small crop mask
    assert out["plant_instances"].dtype == np.int32


def test_visibility_normalisation():
    assert psam._norm_vis(None, 3) == [None, None, None]
    v = psam._norm_vis(np.full((1, 2, 2), 255, np.uint8), 1)
    assert float(v.max()) == 1.0 and float(psam._norm_vis(np.full((1, 2, 2), 0.7), 1).max()) == pytest.approx(0.7)


def test_selector_pickle_round_trip(tmp_path):
    sel = psam.SamSelector("pix", "mp", "ml", {"plant_thr": 0.3}, 4, 1.5)
    psam.save_selector(sel, str(tmp_path / "s.pkl"))
    back = psam.load_selector(str(tmp_path / "s.pkl"))
    assert (back.pixel_model, back.plant_scorer, back.params, back.n_train) == ("pix", "mp", {"plant_thr": 0.3}, 4)


def test_mask_features_and_targets_on_toy_pool():
    torch = pytest.importorskip("torch")
    H = 32
    M = _pool_of([(2, 12, 2, 12), (18, 24, 18, 24)])
    pool = dict(masks=M, sam=np.array([0.9, 0.8], np.float32), level=np.array([0, 1]), ptype=np.array([0, 2]))
    probs = np.zeros((H, H, 3), np.float32)
    probs[..., 0] = 1
    probs[2:12, 2:12] = (0, 0.9, 0.1)
    probs[18:24, 18:24] = (0, 0.1, 0.9)
    F = psam._mask_feats(pool, probs, "cpu")
    assert F.shape == (2, len(psam.FEATS))
    assert F[0, psam.FEATS.index("share")] == pytest.approx(0.9, abs=1e-4) and F[1, psam.FEATS.index("share")] == pytest.approx(0.1, abs=1e-4)
    assert F[0, psam.FEATS.index("area")] == 100 and F[0, psam.FEATS.index("fill")] == pytest.approx(1.0)
    sem = np.zeros((H, H), np.int16)
    sem[2:12, 2:12] = 1
    inst = np.zeros((H, H), np.int32)
    inst[2:12, 2:12] = 1
    tp, tl = psam._targets(pool, inst, inst, sem, np.ones((H, H)), np.ones((H, H)), "cpu")
    np.testing.assert_allclose(tp, [1.0, 0.0], atol=1e-6)
    tp2, _ = psam._targets(pool, inst, inst, sem, np.full((H, H), 0.4), np.full((H, H), 0.4), "cpu")     # half-hidden instance is ignored
    np.testing.assert_allclose(tp2, [0.0, 0.0])
