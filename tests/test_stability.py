import numpy as np
import pandas as pd
import pytest

from src.stability import psi, psi_edges, psi_flag, psi_table


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_identical_samples_give_zero(rng):
    x = rng.normal(size=2000)
    assert psi(x, x) == pytest.approx(0)


def test_shift_increases_psi(rng):
    x = rng.normal(size=5000)
    small = psi(x, rng.normal(0.1, size=5000))
    large = psi(x, rng.normal(1.0, size=5000))
    assert 0 < small < large
    assert large > 0.25


def test_invariant_to_monotone_transform(rng):
    e, a = rng.normal(size=3000), rng.normal(0.3, size=3000)
    assert psi(np.exp(e), np.exp(a)) == pytest.approx(psi(e, a))


def test_empty_bin_stays_finite(rng):
    # one reference bin has no actual observations at all
    x = rng.normal(size=5000)
    value = psi(x, x[x < 1])
    assert np.isfinite(value)
    assert value < 1


def test_out_of_range_values_are_counted():
    e = np.arange(100.0)
    assert psi(e, e + 1000) > 1


def test_nans_are_ignored(rng):
    x = rng.normal(size=1000)
    assert psi(np.r_[x, np.nan], np.r_[x, np.nan, np.nan]) == pytest.approx(0)


def test_empty_input_returns_nan():
    assert np.isnan(psi([np.nan, np.nan], [1.0, 2.0]))
    assert np.isnan(psi([1.0, 2.0], []))


def test_ties_collapse_edges():
    x = np.r_[np.zeros(80), np.arange(1.0, 21.0)]
    assert len(psi_edges(x, 10)) - 1 < 10


def test_flag_bands():
    assert [psi_flag(v) for v in [0.05, 0.1, 0.2, 0.3]] == ["stable", "monitor", "monitor", "shift"]
    assert psi_flag(np.nan) is None


def test_psi_table_columns(rng):
    ref = pd.DataFrame({"a": rng.normal(size=500), "b": np.r_[np.zeros(400), rng.normal(size=100)]})
    other = ref.copy()
    other.loc[:49, "a"] = np.nan
    t = psi_table(ref, {"val": other}, ["a", "b"])

    assert t.columns.tolist() == ["n_bins", "missing_ref", "psi_val", "flag_val", "missing_val"]
    assert t.loc["a", "n_bins"] == 10
    assert t.loc["b", "n_bins"] < 10
    assert t.loc["a", "missing_val"] == pytest.approx(0.1)
    assert t.loc["b", "flag_val"] == "stable"
