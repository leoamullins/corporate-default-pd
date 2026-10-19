import numpy as np
import pandas as pd
import pytest

from src.scorecard import (
    WOEBinner,
    fine_edges,
    monotone_edges,
    trend_violations,
    woe_table,
)


def test_fine_edges_are_contiguous_with_ties():
    x = np.repeat([1.0, 2.0, 3.0], 100)
    edges = fine_edges(x, n_bins=20)

    assert edges[0][0] == -np.inf
    assert edges[-1][1] == np.inf
    for (_, hi), (lo, _) in zip(edges[:-1], edges[1:]):
        assert hi == lo
    assert len(edges) == len({e[0] for e in edges})


def test_woe_table_counts_and_missing(synthetic_credit):
    x = synthetic_credit["tl_ta"].copy()
    x.iloc[:50] = np.nan
    y = synthetic_credit["default"]

    table = woe_table(x, y, fine_edges(x.dropna(), 5), missing=True)

    assert table["bins"].iloc[-1] == "missing"
    assert table["n"].iloc[-1] == 50
    assert table["n"].sum() == len(x)
    assert table["bad"].sum() == y.sum()
    assert (table["iv"] >= 0).all()
    # riskier bins have lower WOE
    assert table["woe"].iloc[:-1].is_monotonic_decreasing


def test_trend_violations():
    rates = pd.Series([0.1, 0.2, 0.15, 0.3])
    assert trend_violations(rates, 1).tolist() == [False, False, True, False]
    assert trend_violations(rates, -1).tolist() == [False, True, False, True]

    with pytest.raises(ValueError):
        trend_violations(rates, 0)


def test_monotone_edges(synthetic_credit):
    x, y = synthetic_credit["tl_ta"], synthetic_credit["default"]
    edges = monotone_edges(x, y, trend=1, n_bins=20, min_bad=20)
    table = woe_table(x, y, edges)

    assert len(edges) > 1
    assert (table["rate"].diff().dropna() > 0).all()
    assert (table["bad"] >= 20).all()


def test_binner_scores_rare_missing_as_neutral(synthetic_credit):
    X = synthetic_credit[["tl_ta", "mve_tl"]].copy()
    y = synthetic_credit["default"]
    # missing only on non-defaults, so too few defaults to trust its WOE
    goods = y[y == 0].index[:30]
    X.loc[goods, "tl_ta"] = np.nan

    binner = WOEBinner({"tl_ta": 1, "mve_tl": -1}).fit(X, y)
    woe = binner.transform(X)

    assert woe.shape == X.shape
    assert (woe.loc[goods, "tl_ta"] == 0).all()

    ordered = woe.loc[X["tl_ta"].sort_values().dropna().index, "tl_ta"]
    assert ordered.is_monotonic_decreasing


def test_binner_iv_ranks_signal_above_noise(synthetic_credit):
    X = synthetic_credit[["tl_ta", "mve_tl"]]
    binner = WOEBinner({"tl_ta": 1, "mve_tl": -1}).fit(X, synthetic_credit["default"])
    assert binner.iv().index.tolist() == ["tl_ta", "mve_tl"]
