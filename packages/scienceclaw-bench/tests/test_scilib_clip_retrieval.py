"""Focused safety and visible-data tests for the optional FoR45 CLIP reference."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scilib import clip_retrieval as cr


def test_provenance_is_fail_closed_without_a_staged_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(cr, "_weights", lambda model: None)
    monkeypatch.setattr(cr, "have_module", lambda name: False)
    p = cr.provenance()
    assert p["available"] is False
    assert p["network_allowed"] is False and p["labels_used"] is False and p["frozen"] is True
    assert p["weights_path"] is None and p["weights_sha256"] is None
    with pytest.raises(RuntimeError, match="frozen encoder unavailable"):
        cr.encode_images(np.zeros((1, 4, 4, 3), dtype=np.uint8))


def test_provenance_records_digest_and_local_only_metadata(monkeypatch, tmp_path):
    weight = tmp_path / "open_clip_vit_b32.pt"
    weight.write_bytes(b"approved-checkpoint-fixture")
    monkeypatch.setattr(cr, "_weights", lambda model: weight)
    monkeypatch.setattr(cr, "have_module", lambda name: True)
    monkeypatch.delenv("SCIENCECLAW_NO_PRETRAINED", raising=False)
    monkeypatch.delenv("SCIENCECLAW_NO_SCILIB", raising=False)
    p = cr.provenance()
    assert p["available"] is True
    assert p["weights_sha256"] == hashlib.sha256(weight.read_bytes()).hexdigest()
    assert p["weights_bytes"] == weight.stat().st_size
    assert p["network_allowed"] is False and p["labels_used"] is False


def test_remote_worker_provenance_and_embedding_route(monkeypatch):
    calls = []

    def fake_call(module, fn, args):
        calls.append((module, fn, args))
        if fn == "provenance":
            return {"available": True, "weights_sha256": "official", "network_allowed": False,
                    "labels_used": False, "frozen": True, "reason": None}
        return np.ones((1, 512), dtype=np.float32)

    monkeypatch.setattr(cr, "have_module", lambda name: False)
    monkeypatch.setattr(cr._remote, "enabled", lambda: True)
    monkeypatch.setattr(cr._remote, "call", fake_call)
    z = cr.encode_images(np.zeros((1, 4, 4, 3), dtype=np.uint8))
    assert z.shape == (1, 512)
    assert [fn for _, fn, _ in calls] == ["provenance", "encode_images"]
    assert calls[-1][0] == "clip_retrieval"


def test_retrieve_uses_only_visible_same_language_rows_and_explicit_fallback(monkeypatch):
    train_images = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    train_images[0, 0, 0, 0] = 1
    train_images[1, 0, 0, 0] = 2
    train_images[2, 0, 0, 0] = 3
    train = [
        {"iso_lang": "x", "caption": "x near", "has_image": True},
        {"iso_lang": "y", "caption": "y near", "has_image": True},
        {"iso_lang": "x", "caption": "x hidden candidate", "has_image": False},
    ]
    items = [{"iso_lang": "x"}, {"iso_lang": "y"}, {"iso_lang": "z"}]
    eval_images = np.stack([train_images[0], train_images[1], train_images[2]])

    def fake_encode(images, **kwargs):
        arr = np.asarray(images)
        code = arr[:, 0, 0, 0].astype(float)
        return np.column_stack([code, 4.0 - code]).astype(np.float32)

    monkeypatch.setattr(cr, "encode_images", fake_encode)
    got = cr.retrieve_captions(train, train_images, items, eval_images, model="open_clip_vit_b32",
                                fallback={"z": "z visible medoid"})
    assert got == ["x near", "y near", "z visible medoid"]


def test_retrieve_empty_visible_images_does_not_load_encoder(monkeypatch):
    train = [{"iso_lang": "x", "caption": "visible caption", "has_image": False}]
    train_images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    items = [{"iso_lang": "x"}]
    eval_images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(cr, "encode_images", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not load")))
    assert cr.retrieve_captions(train, train_images, items, eval_images, fallback={"x": "visible caption"}) == ["visible caption"]


def test_retrieve_caption_medoid_honors_top_k_without_target_access(monkeypatch):
    train_images = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    train = [
        {"iso_lang": "x", "caption": "unrelated token", "has_image": True},
        {"iso_lang": "x", "caption": "red cat", "has_image": True},
        {"iso_lang": "x", "caption": "red dog", "has_image": True},
    ]
    items = [{"iso_lang": "x"}]
    eval_images = np.zeros((1, 4, 4, 3), dtype=np.uint8)

    def fake_encode(images, **kwargs):
        # Rank the unrelated caption first; the two red captions form the
        # text-side medoid among the top three visible neighbours.
        arr = np.asarray(images)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(arr), 1))

    monkeypatch.setattr(cr, "encode_images", fake_encode)
    nearest = cr.retrieve_captions(train, train_images, items, eval_images, k=3)
    medoid = cr.retrieve_captions(train, train_images, items, eval_images, k=3, selection="caption_medoid")
    assert nearest == ["unrelated token"]
    assert medoid == ["red cat"]
    with pytest.raises(ValueError, match="selection"):
        cr.retrieve_captions(train, train_images, items, eval_images, selection="bad")
