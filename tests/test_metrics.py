import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from src.metrics import (
    _firm_rows,
    bootstrap_metric,
    evaluate,
    ks_statistic,
    paired_bootstrap,
    reliability,
)


def test_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    scores = evaluate(y, p)

    assert scores["roc_auc"] == 1
    assert scores["ks"] == 1
    assert scores["n_pos"] == 2


def test_base_rate_pd_has_no_skill(synthetic_credit):
    y = synthetic_credit["default"].to_numpy()
    scores = evaluate(y, np.full(len(y), y.mean()))

    assert scores["roc_auc"] == 0.5
    assert scores["ks"] == 0
    assert scores["brier_skill"] == pytest.approx(0)


def test_ks_matches_scipy(synthetic_credit):
    y = synthetic_credit["default"].to_numpy()
    p = synthetic_credit["pd_true"].to_numpy()
    expected = ks_2samp(p[y == 1], p[y == 0]).statistic
    assert ks_statistic(y, p) == pytest.approx(expected)


def test_firm_rows_partition_rows():
    groups = np.array(["b", "a", "b", "c", "a"])
    rows = _firm_rows(groups)

    assert sorted(np.concatenate(rows).tolist()) == [0, 1, 2, 3, 4]
    for r in rows:
        assert len(set(groups[r])) == 1


def test_bootstrap_brackets_point_estimate(synthetic_credit):
    y, p = synthetic_credit["default"], synthetic_credit["pd_true"]
    est, lo, hi = bootstrap_metric(y, p, roc_auc_score, n_boot=200)

    assert est == roc_auc_score(y, p)
    assert lo < est < hi
    assert bootstrap_metric(y, p, roc_auc_score, n_boot=200) == (est, lo, hi)


def test_firm_bootstrap_is_wider_for_repeated_rows(synthetic_credit):
    # each firm's row copied 5 times: rows are not independent, firms are
    y = np.repeat(synthetic_credit["default"].to_numpy()[:1000], 5)
    p = np.repeat(synthetic_credit["pd_true"].to_numpy()[:1000], 5)
    groups = np.repeat(np.arange(1000), 5)

    _, lo_row, hi_row = bootstrap_metric(y, p, roc_auc_score, n_boot=200)
    _, lo_firm, hi_firm = bootstrap_metric(y, p, roc_auc_score, n_boot=200, groups=groups)

    assert hi_firm - lo_firm > 1.5 * (hi_row - lo_row)


def test_paired_bootstrap(synthetic_credit):
    y, p = synthetic_credit["default"], synthetic_credit["pd_true"]
    noise = np.random.default_rng(1).random(len(y))

    assert paired_bootstrap(y, p, p, roc_auc_score, n_boot=100) == (0, 0, 0)

    diff, lo, hi = paired_bootstrap(y, p, noise, roc_auc_score, n_boot=200)
    assert 0 < lo < diff < hi


def test_reliability_rows_strategy(synthetic_credit):
    y, p = synthetic_credit["default"], synthetic_credit["pd_true"]
    r = reliability(y, p, n_bins=10)

    assert r["n"].tolist() == [500] * 10
    assert r["mean_pred"].is_monotonic_increasing
    assert (r["obs_low"] >= 0).all()
    assert (r["obs_low"] <= r["observed"]).all()
    assert (r["observed"] <= r["obs_high"]).all()


def test_reliability_defaults_strategy(synthetic_credit):
    y, p = synthetic_credit["default"], synthetic_credit["pd_true"]
    r = reliability(y, p, n_bins=10, strategy="defaults")

    assert r["n"].sum() == len(y)
    assert r["defaults"].sum() == y.sum()
    assert r["defaults"].max() - r["defaults"].min() <= 1


def test_reliability_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        reliability([0, 1], [0.1, 0.9], strategy="width")
