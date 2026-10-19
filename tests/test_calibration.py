import matplotlib
import numpy as np
import pytest

from src.calibration import LogitRecalibrator, plot_reliability

matplotlib.use("Agg")


def _logit(p):
    return np.log(p / (1 - p))


def _expit(z):
    return 1 / (1 + np.exp(-z))


def test_recalibrated_mean_matches_observed(synthetic_credit):
    y = synthetic_credit["default"]
    too_high = _expit(_logit(synthetic_credit["pd_true"]) + 1)

    recal = LogitRecalibrator().fit(too_high, y)

    assert recal.predict_proba(too_high).mean() == pytest.approx(y.mean(), rel=1e-3)
    assert recal.a_ == pytest.approx(-1, abs=0.3)
    assert recal.b_ == pytest.approx(1, abs=0.25)


def test_calibrated_pds_left_alone(synthetic_credit):
    p, y = synthetic_credit["pd_true"], synthetic_credit["default"]
    recal = LogitRecalibrator().fit(p, y)

    assert recal.a_ == pytest.approx(0, abs=0.3)
    assert recal.b_ == pytest.approx(1, abs=0.25)


def test_extreme_pds_are_clipped(synthetic_credit):
    recal = LogitRecalibrator().fit(synthetic_credit["pd_true"], synthetic_credit["default"])
    out = recal.predict_proba([0.0, 1.0])
    assert np.isfinite(out).all()
    assert 0 < out[0] < out[1] < 1


def test_plot_reliability_saves(synthetic_credit, tmp_path):
    y, p = synthetic_credit["default"], synthetic_credit["pd_true"]
    path = tmp_path / "figures" / "rel.png"

    g = plot_reliability({"Scorecard": p, "LightGBM": p}, y, "test", path=path)

    assert path.exists()
    assert len(g.axes_dict) == 2
