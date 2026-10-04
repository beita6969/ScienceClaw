"""Contract tests for the local-only BEATs component."""
from __future__ import annotations

import numpy as np
import pytest

from scilib import beats


def test_requires_explicit_local_assets(tmp_path, monkeypatch):
    monkeypatch.delenv(beats.MODEL_ENV, raising=False)
    monkeypatch.delenv(beats.SOURCE_ENV, raising=False)
    assert beats.available() is False
    with pytest.raises(RuntimeError, match="checkpoint is unavailable"):
        beats.encode(np.zeros(160), model_path=tmp_path / "missing", source_path=tmp_path)


def test_native_rate_and_finite_checks(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"checkpoint")
    (tmp_path / "BEATs.py").write_text("# placeholder")
    with pytest.raises(ValueError, match="sample_rate"):
        beats.encode(np.zeros(160), sample_rate=8000, model_path=ckpt, source_path=tmp_path)
    with pytest.raises(ValueError, match="finite"):
        x = np.zeros(160)
        x[0] = np.nan
        beats.encode(x, model_path=ckpt, source_path=tmp_path)


def test_frozen_embedding_contract(tmp_path, monkeypatch):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"checkpoint")
    source = tmp_path / "source"
    source.mkdir()
    (source / "BEATs.py").write_text("# placeholder")

    class FakeTensor:
        shape = (2, 4, 768)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones(self.shape, dtype=np.float32)

    class FakeTorch:
        float32 = object()

        @staticmethod
        def as_tensor(x, dtype=None, device=None):
            assert tuple(x.shape) == (2, 160) and device == "cpu"
            return x

        @staticmethod
        def inference_mode():
            class Context:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Context()

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

        def extract_features(self, x):
            return FakeTensor(), None

    monkeypatch.setattr(beats, "_load", lambda *args: (FakeModel(), FakeTorch))
    out = beats.encode(np.zeros((2, 160)), model_path=ckpt, source_path=source, device="cpu")
    assert out.shape == (2, 4, 768)
    assert out.dtype == np.float32 and np.all(out == 1.0)
