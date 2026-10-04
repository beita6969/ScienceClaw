"""Content-addressed array transfer of the remote bridge: chunking, dedupe, integrity, worker and broker round trip (no network)."""
from __future__ import annotations

import base64
import gzip
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "remote"))

import blobs  # noqa: E402
import broker as brk  # noqa: E402
import worker as wk  # noqa: E402
from scilib import _remote  # noqa: E402
from scilib import audiosep_pretrained as ap  # noqa: E402
from scilib import hippo_unet as hu  # noqa: E402
from scilib import phenoseg_sam as psam  # noqa: E402
from test_scilib_remote import broker  # noqa: E402,F401  (fixture)


def _arr(k=3, seed=0):
    return np.random.default_rng(seed).normal(size=(k, 300, 300)).astype(np.float32)      # 360 KB per row -> 2 rows per chunk


def _args(a):
    return _remote.encode({"mixtures": a, "n": 3, "small": np.arange(4)})


def test_pack_replaces_big_arrays_only_and_unpack_restores_them(tmp_path):
    a = _arr(5)
    new: dict = {}
    packed = blobs.pack(_args(a), set(), new)
    assert set(packed["mixtures"]) == {"__rows__"} and len(packed["mixtures"]["__rows__"]) == 3        # 2+2+1 rows
    assert "__nd__" in packed["small"] and packed["n"] == 3
    blobs.store(tmp_path, {h: base64.b64encode(r).decode() for h, r in new.items()})
    back = _remote.decode(blobs.unpack(packed, tmp_path))
    assert np.array_equal(back["mixtures"], a) and back["mixtures"].dtype == np.float32 and np.array_equal(back["small"], np.arange(4))


def test_known_chunks_are_not_sent_again_and_slices_reuse_them():
    a = _arr(4)
    new1: dict = {}
    p1 = blobs.pack(_args(a), set(), new1)
    new2: dict = {}
    blobs.pack(_args(a), set(new1), new2)
    assert new1 and not new2
    new3: dict = {}
    p3 = blobs.pack(_args(a[2:]), set(new1), new3)                       # same chunk boundaries -> the same chunk
    assert not new3 and p3["mixtures"]["__rows__"] == p1["mixtures"]["__rows__"][1:]
    new4: dict = {}
    blobs.pack(_args(a[:3]), set(new1), new4)                            # a different split of the rows: one new chunk
    assert len(new4) == 1


def test_missing_or_corrupt_chunks_are_reported(tmp_path):
    a = _arr(2)
    new: dict = {}
    packed = blobs.pack(_args(a), set(), new)
    with pytest.raises(blobs.MissingBlobs) as e:
        blobs.unpack(packed, tmp_path)
    assert set(e.value.missing) == set(new)
    blobs.store(tmp_path, {h: base64.b64encode(r).decode() for h, r in new.items()})
    h = next(iter(new))
    (tmp_path / f"{h}.npy").write_bytes(b"garbage")
    with pytest.raises(blobs.MissingBlobs):
        blobs.unpack(packed, tmp_path)
    with pytest.raises(ValueError):
        blobs.store(tmp_path, {h: base64.b64encode(b"x").decode()})


class _IO:
    def __init__(self, data=b""):
        self.buffer = io.BytesIO(data)


def _run_worker(monkeypatch, tmp_path, body: bytes) -> dict:
    out = _IO()
    monkeypatch.setattr(sys, "stdin", _IO(body))
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(wk, "BLOB_ROOT", tmp_path / "blobs")
    wk.main()
    return json.loads(gzip.decompress(out.buffer.getvalue()))


def test_broker_and_worker_round_trip_with_dedupe_and_recovery(tmp_path, monkeypatch):
    seen = []

    def fake(mixtures, **kw):
        seen.append(np.asarray(mixtures).copy())
        return np.asarray(mixtures) * 2.0
    monkeypatch.setattr(ap, "separate_pretrained", fake)
    sent = []

    def fake_ssh(a, body):
        sent.append(json.loads(gzip.decompress(body)))
        return _run_worker(monkeypatch, tmp_path, body)
    monkeypatch.setattr(brk, "_run_ssh", fake_ssh)

    a = _arr(3)
    raw = json.dumps({"module": "audiosep_pretrained", "fn": "separate_pretrained", "args": _args(a)}).encode()
    known: set = set()
    out = brk.run_remote(SimpleNamespace(), raw, known)
    assert out["ok"] and np.array_equal(_remote.decode(out["result"]), a * 2) and np.array_equal(seen[-1], a)
    assert sent[-1]["blobs"] and len(known) == 2
    out = brk.run_remote(SimpleNamespace(), raw, known)                                   # host has the chunks: none are sent
    assert out["ok"] and "blobs" not in sent[-1] and np.array_equal(seen[-1], a)
    for f in (tmp_path / "blobs").glob("*.npy"):                                          # host lost them: one retry with full data
        f.unlink()
    n = len(sent)
    out = brk.run_remote(SimpleNamespace(), raw, known)
    assert out["ok"] and len(sent) == n + 2 and "blobs" in sent[-1] and np.array_equal(seen[-1], a)


def test_worker_serves_audio_but_not_arbitrary_functions(monkeypatch, tmp_path):
    body = gzip.compress(json.dumps({"module": "os", "fn": "system", "args": {}}).encode())
    out = _run_worker(monkeypatch, tmp_path, body)
    assert not out["ok"] and "not served" in out["error"]
    assert ("audiosep_pretrained", "separate_pretrained") in wk.ALLOWED


def test_separate_pretrained_routes_to_the_gpu_host(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(ap, "_local_ok", lambda *args: False)
    assert ap.available()
    box["handler"] = lambda fn, a: np.zeros((len(a["mixtures"]), 4, a["mixtures"].shape[1], 2), np.float32) + 1
    x = np.zeros((2, 50, 2), np.float32)
    out = ap.separate_pretrained(x, shifts=1)
    assert out.shape == (2, 4, 50, 2) and out.dtype == np.float32 and np.all(out == 1)
    assert seen[0]["module"] == "audiosep_pretrained" and seen[0]["fn"] == "separate_pretrained"
    args = _remote.decode(seen[0]["args"])
    assert args["shifts"] == 1 and args["model"] == "htdemucs" and args["mixtures"].shape == (2, 50, 2)
    assert ap.separate_pretrained(np.zeros((0, 50, 2), np.float32)).shape == (0, 4, 50, 2)


def test_separate_pretrained_unavailable_without_stack_or_spool(monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    monkeypatch.setattr(ap, "_local_ok", lambda *args: False)
    assert not ap.available()
    with pytest.raises(RuntimeError, match="not available"):
        ap.separate_pretrained(np.zeros((1, 10, 2), np.float32))


def test_sam_selector_is_fitted_and_used_on_the_gpu_host(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(psam, "_local_ok", lambda: False)
    assert psam.available()

    def handler(fn, a):
        if fn == "fit_sam_selector":
            assert a["images"].shape == (2, 8, 8, 3) and a["plant_visibility"] is None and a["grid"] == 16
            return {"path": "/models/abc/selector.pkl", "params": {"plant_thr": 0.3}, "n_train": 2, "fit_s": 1.5}
        assert a["model"] == "/models/abc/selector.pkl"
        n = len(a["images"])
        return {k: np.zeros((n, 8, 8), np.int64) for k in ("semantics", "plant_instances", "leaf_instances")}

    box["handler"] = handler
    imgs = np.zeros((2, 8, 8, 3), np.uint8)
    lab = np.zeros((2, 8, 8), np.int64)
    m = psam.fit_sam_selector(imgs, lab, lab, lab, grid=16)
    assert isinstance(m, psam.RemoteSelector) and m.path.endswith("selector.pkl") and m.n_train == 2
    out = psam.predict_sam_panoptic(m, imgs[:1])
    assert out["semantics"].shape == (1, 8, 8)
    assert [p["fn"] for p in seen] == ["fit_sam_selector", "predict_sam_panoptic"]


def test_sam_selector_path_must_come_from_the_host(monkeypatch, tmp_path):
    monkeypatch.setattr(wk, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(wk, "BLOB_ROOT", tmp_path / "blobs")
    outside = tmp_path / "x.pkl"
    outside.write_bytes(b"x")
    req = {"module": "phenoseg_sam", "fn": "predict_sam_panoptic",
           "args": _remote.encode({"model": str(outside), "images": np.zeros((1, 4, 4, 3), np.uint8), "device": "cpu"})}
    out = _run_worker(monkeypatch, tmp_path, gzip.compress(json.dumps(req).encode()))
    assert not out["ok"] and "fit_sam_selector" in out["error"]
    assert ("phenoseg_sam", "fit_sam_selector") in wk.ALLOWED and ("phenoseg_sam", "predict_sam_panoptic") in wk.ALLOWED


def test_hippo_unet_is_fitted_and_used_on_the_gpu_host(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(hu, "_local_ok", lambda: False)
    assert hu.available()

    def handler(fn, a):
        if fn == "fit_unet":
            assert len(a["train_images"]) == 2 and a["train_images"][0].shape == (5, 6, 7) and a["n_models"] == 2
            return {"path": "/models/abc/unet.pt", "params": {"base": 16}, "n_train": 2, "fit_s": 2.5}
        if fn == "predict_unet":
            assert a["model"] == "/models/abc/unet.pt" and a["tta"] is True
            return [np.zeros(x.shape, np.uint8) for x in a["images"]]
        assert fn == "fit_predict" and len(a["eval_images"]) == 1 and a["seed"] == 3
        return [np.ones(x.shape, np.uint8) for x in a["eval_images"]]

    box["handler"] = handler
    imgs = [np.zeros((5, 6, 7), np.float32)] * 2
    labs = [np.zeros((5, 6, 7), np.uint8)] * 2
    m = hu.fit_unet(imgs, labs, n_models=2)
    assert isinstance(m, hu.RemoteUNet) and m.path.endswith("unet.pt") and m.n_train == 2
    out = hu.predict_unet(m, imgs[:1])
    assert out[0].shape == (5, 6, 7) and out[0].dtype == np.uint8
    assert hu.fit_predict(imgs, labs, imgs[:1], seed=3)[0].all()
    assert [p["fn"] for p in seen] == ["fit_unet", "predict_unet", "fit_predict"]


def test_hippo_unet_model_path_must_come_from_the_host(monkeypatch, tmp_path):
    monkeypatch.setattr(wk, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(wk, "BLOB_ROOT", tmp_path / "blobs")
    outside = tmp_path / "x.pt"
    outside.write_bytes(b"x")
    for fn in ("predict_unet", "predict_proba"):
        req = {"module": "hippo_unet", "fn": fn, "args": _remote.encode({"model": str(outside), "images": [np.zeros((4, 4, 4), np.float32)]})}
        out = _run_worker(monkeypatch, tmp_path, gzip.compress(json.dumps(req).encode()))
        assert not out["ok"] and "fit_unet" in out["error"]
    assert {("hippo_unet", f) for f in ("fit_unet", "predict_unet", "predict_proba", "fit_predict")} <= wk.ALLOWED


@pytest.mark.skipif(not hu._local_ok(), reason="torch is not installed here")
def test_hippo_unet_trains_and_predicts_on_cpu():
    rng = np.random.default_rng(0)
    imgs, labs = [], []
    for _ in range(3):
        a = rng.normal(size=(16, 16, 16)).astype(np.float32)
        y = np.zeros((16, 16, 16), np.uint8)
        y[4:12, 4:12, 4:8] = 1
        y[4:12, 4:12, 8:12] = 2
        imgs.append(a + 3 * y); labs.append(y)
    m = hu.fit_unet(imgs, labs, n_models=1, iters=8, base=4, batch=2, device="cpu")
    p = hu.predict_unet(m, imgs[:2], device="cpu")
    assert [x.shape for x in p] == [(16, 16, 16)] * 2 and p[0].dtype == np.uint8
    p2 = hu.predict_unet(m, imgs[:2], device="cpu")
    assert all((a == b).all() for a, b in zip(p, p2))
