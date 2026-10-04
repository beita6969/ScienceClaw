"""transfer_metrics: all-cell vs off-diagonal FWT, GEM FWT/BWT, Chaudhry forgetting, negative transfer."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scienceclaw.bench.metrics import transfer_metrics

R = [[1.0, 2.0, -1.0],
     [3.0, 4.0, 5.0],
     [0.0, 6.0, 7.0]]


def test_allcell_vs_offdiag_and_nt():
    t = transfer_metrics(R)
    assert t["FWT_allcell"] == pytest.approx(27 / 9)          # all 9 cells
    assert t["FWT_offdiag"] == pytest.approx(15 / 6)          # 2 - 1 + 3 + 5 + 0 + 6 over 6 off-diagonal cells
    assert t["FWT_allcell"] != t["FWT_offdiag"]
    assert t["NT"] == pytest.approx(1 / 6)                    # only -1 is negative (0 is neither)
    assert (t["n_offdiag"], t["n_offdiag_positive"], t["n_offdiag_negative"]) == (6, 4, 1)


def test_gem_and_chaudhry_definitions():
    t = transfer_metrics(R)
    assert t["FWT_GEM"] == pytest.approx((2.0 + 5.0) / 2)     # R[0,1], R[1,2]
    assert t["FWT_upper"] == pytest.approx((2.0 - 1.0 + 5.0) / 3)
    assert t["BWT"] == pytest.approx(((0 - 1) + (6 - 4)) / 2)
    # forgetting: j=0: max(1, 3) - 0 = 3; j=1: max(2, 4) - 6 = -2
    assert t["forgetting"] == pytest.approx((3 - 2) / 2)
    assert t["T"] == 3


def test_baseline_shifts_forward_but_not_backward_transfer():
    t = transfer_metrics(R, baseline=[1.0, 1.0, 1.0])
    assert t["FWT_allcell"] == pytest.approx(2.0)
    assert t["FWT_offdiag"] == pytest.approx(1.5)
    assert t["FWT_GEM"] == pytest.approx(2.5)
    assert t["NT"] == pytest.approx(2 / 6)
    assert t["BWT"] == pytest.approx(0.5) and t["forgetting"] == pytest.approx(0.5)


def test_nan_cells_and_shape_checks():
    t = transfer_metrics(np.array([[1.0, np.nan], [0.5, 1.0]]))
    assert t["BWT"] == pytest.approx(-0.5) and t["forgetting"] == pytest.approx(0.5)
    assert math.isnan(t["FWT_GEM"]) and t["n_offdiag"] == 1
    one = transfer_metrics([[2.0]])
    assert one["FWT_allcell"] == 2.0 and math.isnan(one["FWT_offdiag"]) and math.isnan(one["BWT"])
    with pytest.raises(ValueError):
        transfer_metrics([[1.0, 2.0]])
    with pytest.raises(ValueError):
        transfer_metrics(R, baseline=[0.0])
