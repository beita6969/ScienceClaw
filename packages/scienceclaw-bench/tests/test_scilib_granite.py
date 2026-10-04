"""Offline checks for the optional Granite component wrapper."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scilib import granite


def _checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["TinyTimeMixerForPrediction"]}))
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    return tmp_path


def test_unavailable_without_explicit_local_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv(granite.MODEL_ENV, raising=False)
    assert granite.available() is False
    with pytest.raises(RuntimeError, match="checkpoint is unavailable"):
        granite.forecast(np.zeros((1, granite.CONTEXT_LENGTH)), model_dir=tmp_path / "missing")


def test_context_shape_and_finite_checks(tmp_path):
    ckpt = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="shape"):
        granite.forecast(np.zeros((1, granite.CONTEXT_LENGTH - 1)), model_dir=ckpt)
    with pytest.raises(ValueError, match="finite"):
        x = np.zeros((1, granite.CONTEXT_LENGTH))
        x[0, 0] = np.nan
        granite.forecast(x, model_dir=ckpt)


def test_frozen_forward_contract_with_fake_stack(tmp_path, monkeypatch):
    ckpt = _checkpoint(tmp_path)

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class FakeTorch:
        float32 = object()

        @staticmethod
        def as_tensor(x, dtype=None, device=None):
            assert device == "cpu"
            return FakeTensor(x)

        @staticmethod
        def inference_mode():
            class Ctx:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Ctx()

    class FakeModel:
        def __init__(self):
            self.eval_called = False
            self.to_device = None
            self.kwargs = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.to_device = device
            return self

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return type("Output", (), {"prediction_outputs": FakeTensor(np.ones((2, 96, 1), dtype=np.float32))})()

    model = FakeModel()
    monkeypatch.setattr(granite, "_load", lambda path, device: (model.eval().to(device), FakeTorch))
    out = granite.forecast(np.zeros((2, granite.CONTEXT_LENGTH)), model_dir=ckpt, device="cpu")
    assert out.shape == (2, granite.PREDICTION_LENGTH, 1)
    assert out.dtype == np.float32 and np.all(out == 1.0)
    assert model.eval_called and model.to_device == "cpu"
    assert model.kwargs["return_loss"] is False
