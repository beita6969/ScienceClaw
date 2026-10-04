"""scilib.phenoseg_m2f: availability, input checks, the assembly of the three output arrays, remote routing, whitelist and the factual
interface text (no torch, no checkpoints)."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import _remote, phenoseg_m2f as m2f
from test_scilib_remote import broker  # noqa: F401  (fixture)

IMG = np.zeros((2, 8, 8, 3), dtype=np.uint8)


def test_unavailable_without_stack_or_spool(monkeypatch):
    monkeypatch.delenv(_remote.SPOOL_ENV, raising=False)
    monkeypatch.setattr(m2f, "_local_ok", lambda: False)
    assert not m2f.available()
    assert scilib.describe_extra("phenoseg_m2f") == ""
    with pytest.raises(RuntimeError, match="neither local checkpoints"):
        m2f.predict_panoptic(IMG)


def test_describe_extra_shown_only_when_available(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    monkeypatch.setattr(m2f, "available", lambda: True)
    assert scilib.describe_extra("phenoseg_m2f").startswith("\n\nLibrary `scilib.phenoseg_m2f`, importable in code nodes:")


def test_bad_inputs():
    with pytest.raises(ValueError, match="uint8"):
        m2f.predict_panoptic(IMG.astype(float))
    with pytest.raises(ValueError, match="uint8"):
        m2f.predict_panoptic(IMG[0])
    with pytest.raises(ValueError, match="uint8"):
        m2f.predict_panoptic(IMG[:0])
    with pytest.raises(ValueError, match="plant_threshold"):
        m2f.predict_panoptic(IMG, plant_threshold=0.0)
    with pytest.raises(ValueError, match="leaf_threshold"):
        m2f.predict_panoptic(IMG, leaf_threshold=1.5)
    with pytest.raises(ValueError, match="batch_size"):
        m2f.predict_panoptic(IMG, batch_size=0)


def test_assemble_layout_and_leaf_clipping():
    plant = np.full((6, 6), -1)
    plant[:, :2] = 7                    # soil stuff segment
    plant[:3, 2:] = 3                   # crop plant
    plant[3:, 2:4] = 5                  # weed plant
    plant[3:, 4:] = 9                   # crop plant
    pinfo = [{"id": 7, "label_id": 0}, {"id": 3, "label_id": 1}, {"id": 5, "label_id": 2}, {"id": 9, "label_id": 1}]
    leaf = np.full((6, 6), -1)
    leaf[:, :] = 1                      # one leaf segment over the whole image, crop
    leaf[:, 5] = 2                      # a second crop leaf
    leaf[4:, :] = 4                     # a weed segment of the leaf model (never a leaf)
    linfo = [{"id": 1, "label_id": 1}, {"id": 2, "label_id": 1}, {"id": 4, "label_id": 2}]
    r = m2f.assemble(plant, pinfo, leaf, linfo)
    assert set(r) == {"semantics", "plant_instances", "leaf_instances"} and all(v.shape == (6, 6) for v in r.values())
    assert (r["semantics"][:, :2] == 0).all() and (r["semantics"][:3, 2:] == 1).all() and (r["semantics"][3:, 2:4] == 2).all()
    assert sorted(np.unique(r["plant_instances"]).tolist()) == [0, 1, 2, 3] and (r["plant_instances"][:, :2] == 0).all()
    assert (r["leaf_instances"][r["semantics"] != 1] == 0).all()               # leaves only inside predicted crop pixels
    assert set(np.unique(r["leaf_instances"]).tolist()) <= {0, 1, 2} and (r["leaf_instances"][:3, 2] == 1).all()
    assert (r["leaf_instances"][:4, 5] == 2).all() or (r["leaf_instances"][:3, 5] == 2).all()


def test_assemble_empty_prediction_is_all_soil():
    z = np.full((4, 5), -1)
    r = m2f.assemble(z, [], z, [])
    assert all((v == 0).all() and v.shape == (4, 5) for v in r.values())


def test_remote_routing(broker, monkeypatch):  # noqa: F811
    box, seen, _ = broker
    monkeypatch.setattr(m2f, "_local_ok", lambda: False)
    assert m2f.available()
    got = []

    def handler(fn, a):
        got.append(a)
        n, H, W = a["images"].shape[:3]
        return {k: np.zeros((n, H, W), dtype=np.int32) for k in ("semantics", "plant_instances", "leaf_instances")}

    box["handler"] = handler
    out = m2f.predict_panoptic(IMG, plant_threshold=0.7, leaf_threshold=0.6, batch_size=2)
    assert set(out) == {"semantics", "plant_instances", "leaf_instances"} and out["semantics"].shape == (2, 8, 8)
    assert seen[-1]["module"] == "phenoseg_m2f" and seen[-1]["fn"] == "predict_panoptic"
    a = got[-1]
    assert a["plant_threshold"] == 0.7 and a["leaf_threshold"] == 0.6 and a["batch_size"] == 2 and a["images"].shape == (2, 8, 8, 3)


def test_worker_whitelist_contains_the_call():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("sc_worker", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "remote" / "worker.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    assert ("phenoseg_m2f", "predict_panoptic") in w.ALLOWED


def test_remote_argument_names_match_the_function():
    import inspect
    assert {"images", "plant_threshold", "leaf_threshold", "batch_size"} <= set(inspect.signature(m2f.predict_panoptic).parameters)


def test_interface_text_is_factual():
    txt = m2f.__doc__.lower()
    for word in ("recommend", "should", "best", "baseline", "margin", "accept", "state of the art", "worst", "naive", "persistence"):
        assert word not in txt
    for name in m2f.__all__:
        assert name in m2f.__doc__
