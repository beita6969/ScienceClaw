"""Contract tests for the frozen public HAPT FoR30 route."""
from __future__ import annotations

import numpy as np
import pytest

import scilib
from scilib import phenoseg_hapt as hapt


def test_unavailable_without_checkpoint(monkeypatch):
    monkeypatch.delenv("SCIENCECLAW_HAPT_CHECKPOINT", raising=False)
    monkeypatch.setattr(hapt, "model_path", lambda *parts: None)
    assert hapt.checkpoint_path() is None and not hapt.available()
    assert scilib.describe_extra("phenoseg_hapt") == ""


def test_input_contract_and_center_postprocessing():
    with pytest.raises(ValueError, match="uint8"):
        hapt.predict_panoptic(np.zeros((1, 8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty"):
        hapt.predict_panoptic(np.zeros((0, 8, 8, 3), dtype=np.uint8))
    sem = np.zeros((12, 12), dtype=np.int16)
    sem[2:10, 2:10] = 1
    labels = hapt._instances(np.zeros((12, 12), np.float32), np.zeros((2, 12, 12), np.float32), sem,
                              nms_kernel=3, grouping_dist=2.0, threshold_fraction=0.8)
    assert labels.shape == sem.shape and labels.dtype == np.int32 and np.all(labels == 0)


def test_worker_whitelist_contains_hapt_route():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "remote" / "worker.py"
    spec = importlib.util.spec_from_file_location("sc_worker_hapt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert ("phenoseg_hapt", "predict_panoptic") in mod.ALLOWED
