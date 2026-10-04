"""Contract tests for the read-only InnerEye-HS binary smoke wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from scilib import hippo_innereeye as ie


def test_invalid_volume_is_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(ie, "_load", lambda *_: (_ for _ in ()).throw(AssertionError("must not load")))
    with pytest.raises(ValueError, match="finite"):
        ie.predict_binary(np.array([[[np.nan]]], dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        ie.predict_binary(np.zeros((2, 3), dtype=np.float32))


def test_checkpoint_path_uses_explicit_file(monkeypatch, tmp_path):
    p = tmp_path / "last.ckpt"
    p.write_bytes(b"placeholder")
    monkeypatch.setenv("SCIENCECLAW_INNEREYE_CHECKPOINT", str(p))
    assert ie.checkpoint_path() == p


@pytest.mark.skipif(ie.have_module("torch") is False, reason="torch is optional in the local test environment")
def test_random_binary_smoke_roundtrip(monkeypatch, tmp_path):
    import torch

    net = ie._network()()
    p = tmp_path / "last.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in net.state_dict().items()}}, p)
    monkeypatch.setenv("SCIENCECLAW_INNEREYE_CHECKPOINT", str(p))
    monkeypatch.setattr(ie, "_MODEL", None)
    out = ie.predict_binary(np.zeros((1, 16, 16, 16), dtype=np.float32), device="cpu")
    assert out.shape == (1, 16, 16, 16)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 1})
