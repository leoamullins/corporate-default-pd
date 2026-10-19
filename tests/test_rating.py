import numpy as np
import pytest

from src.config import MASTER_SCALE
from src.rating import assign_grade, grade_table

EDGES = [0.01, 0.05, 0.2]


def test_grades_follow_bands():
    p = np.array([0.0, 0.005, 0.03, 0.1, 0.5, 1.0])
    assert assign_grade(p, EDGES).tolist() == [1, 1, 2, 3, 4, 4]


def test_band_edges_are_right_closed():
    # a PD sitting exactly on an edge belongs to the lower (better) grade
    assert assign_grade(np.array(EDGES), EDGES).tolist() == [1, 2, 3]


def test_master_scale_spans_all_grades():
    mids = np.sqrt(np.array([1e-5] + MASTER_SCALE) * np.array(MASTER_SCALE + [0.5]))
    assert assign_grade(mids).tolist() == list(range(1, len(MASTER_SCALE) + 2))


def test_master_scale_is_increasing():
    assert np.all(np.diff(MASTER_SCALE) > 0)


def test_grade_table_counts_and_rates():
    p = np.array([0.005, 0.005, 0.03, 0.03, 0.03, 0.1])
    y = np.array([0, 1, 0, 0, 1, 1])
    t = grade_table(p, y, EDGES)

    assert t.index.tolist() == [1, 2, 3, 4]
    assert t["n"].tolist() == [2, 3, 1, 0]
    assert t["defaults"].tolist() == [1, 1, 1, 0]
    assert t["pct_book"].sum() == pytest.approx(1)
    assert t.loc[2, "observed"] == pytest.approx(1 / 3)
    assert t.loc[2, "expected"] == pytest.approx(0.09)
    assert t.loc[1, "mean_pd"] == pytest.approx(0.005)


def test_grade_table_band_labels():
    t = grade_table([0.1], [0], EDGES)
    assert t["band"].tolist() == ["0.00%–1.00%", "1.00%–5.00%", "5.00%–20.00%", "20.00%–100.00%"]


def test_empty_grades_have_no_test_results():
    t = grade_table([0.005, 0.03], [0, 1], EDGES)
    empty = t.loc[[3, 4]]

    assert (empty["n"] == 0).all()
    assert empty[["observed", "obs_low", "obs_high", "binom_p"]].isna().all().all()


def test_wilson_interval_brackets_observed_rate():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.001, 0.3, size=2000)
    y = rng.binomial(1, p)
    t = grade_table(p, y, EDGES)

    assert (t["obs_low"] <= t["observed"]).all()
    assert (t["observed"] <= t["obs_high"]).all()


def test_binomial_test_flags_miscalibration():
    rng = np.random.default_rng(1)
    p = np.full(2000, 0.03)

    good = grade_table(p, rng.binomial(1, 0.03, size=2000), EDGES)
    bad = grade_table(p, rng.binomial(1, 0.12, size=2000), EDGES)

    assert good.loc[2, "binom_p"] > 0.05
    assert bad.loc[2, "binom_p"] < 1e-6


def test_grade_table_accepts_series(synthetic_credit):
    t = grade_table(synthetic_credit["pd_true"], synthetic_credit["default"])

    assert t["n"].sum() == len(synthetic_credit)
    assert t["defaults"].sum() == synthetic_credit["default"].sum()
