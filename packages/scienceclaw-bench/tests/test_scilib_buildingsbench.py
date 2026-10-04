"""Contract tests for the local-only BuildingsBench component."""
from __future__ import annotations

import numpy as np
import pytest

from scilib import buildingsbench


def _features(n=2):
    return {name: np.zeros((n, buildingsbench.TOTAL_LENGTH, 1), dtype=np.float32)
            for name in buildingsbench.FEATURES}


def test_requires_explicit_local_assets(tmp_path, monkeypatch):
    monkeypatch.delenv(buildingsbench.MODEL_ENV, raising=False)
    monkeypatch.delenv(buildingsbench.SOURCE_ENV, raising=False)
    assert buildingsbench.available() is False
    with pytest.raises(RuntimeError, match="checkpoint/source"):
        buildingsbench.forecast(_features(1), model_path=tmp_path / "missing", source_path=tmp_path)


def test_feature_contract(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"checkpoint")
    source = tmp_path / "source" / "buildings_bench" / "models"
    source.mkdir(parents=True)
    (source / "transformers.py").write_text("# placeholder")
    (source.parent / "configs").mkdir()
    (source.parent / "configs" / "TransformerWithGaussian-L.toml").write_text("[model]\n")
    with pytest.raises(ValueError, match="exactly"):
        buildingsbench.forecast({"load": np.zeros((1, buildingsbench.TOTAL_LENGTH, 1))}, model_path=ckpt, source_path=source.parent.parent)
    with pytest.raises(ValueError, match="shape"):
        bad = _features(1)
        bad["load"] = np.zeros((1, buildingsbench.CONTEXT_LENGTH, 1))
        buildingsbench.forecast(bad, model_path=ckpt, source_path=source.parent.parent)


def test_frozen_forecast_contract(tmp_path, monkeypatch):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"checkpoint")
    source = tmp_path / "source" / "buildings_bench" / "models"
    source.mkdir(parents=True)
    (source / "transformers.py").write_text("# placeholder")
    (source.parent / "configs").mkdir()
    (source.parent / "configs" / "TransformerWithGaussian-L.toml").write_text("[model]\n")

    class FakeTensor:
        shape = (2, buildingsbench.PREDICTION_LENGTH, 2)

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
        long = object()

        @staticmethod
        def as_tensor(x, dtype=None, device=None):
            assert device == "cpu"
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

        def __call__(self, x):
            assert x["load"].shape == (2, buildingsbench.TOTAL_LENGTH, 1)
            return FakeTensor()

    monkeypatch.setattr(buildingsbench, "_load", lambda *args: (FakeModel(), FakeTorch))
    out = buildingsbench.forecast(_features(), model_path=ckpt, source_path=source.parent.parent, device="cpu")
    assert out.shape == (2, buildingsbench.PREDICTION_LENGTH, 2)
    assert out.dtype == np.float32 and np.all(out == 1.0)
