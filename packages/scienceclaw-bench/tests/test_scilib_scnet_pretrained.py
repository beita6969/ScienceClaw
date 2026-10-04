"""Lightweight contract tests for the frozen FoR36 SCNet wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from scilib import scnet_pretrained as sp


def test_empty_batch_does_not_load_model(monkeypatch):
    monkeypatch.setattr(sp, "_local_ok", lambda: True)
    monkeypatch.setattr(sp, "_load", lambda *_: (_ for _ in ()).throw(AssertionError("loaded")))
    out = sp.separate_pretrained(np.zeros((0, 120, 2), np.float32))
    assert out.shape == (0, 4, 120, 2) and out.dtype == np.float32


def test_contract_rejects_invalid_inputs(monkeypatch):
    monkeypatch.setattr(sp, "_local_ok", lambda: True)
    with pytest.raises(ValueError, match="shape"):
        sp.separate_pretrained(np.zeros((1, 120), np.float32))
    with pytest.raises(ValueError, match="finite"):
        sp.separate_pretrained(np.array([[[np.nan, 0.0]]], np.float32))
    with pytest.raises(ValueError, match="iterations"):
        sp.separate_pretrained(np.zeros((1, 120, 2), np.float32), iterations=3)


def test_missing_route_fails_closed(monkeypatch):
    monkeypatch.setattr(sp, "_local_ok", lambda: False)
    monkeypatch.setattr(sp._remote, "enabled", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        sp.separate_pretrained(np.zeros((1, 120, 2), np.float32))
