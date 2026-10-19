import numpy as np
import pandas as pd

from src.features import add_ratios, auc_by_lag


def _accounts(**overrides):
    row = {f"x{i}": 100.0 for i in range(1, 19)}
    row.update(overrides)
    return row


def test_ratios_on_known_row():
    df = pd.DataFrame([_accounts(x1=60, x5=20, x10=200, x14=30, x17=80)])
    r = add_ratios(df).iloc[0]

    assert r["tl_ta"] == 0.4
    assert r["wc_ta"] == 0.15
    assert r["quick"] == (60 - 20) / 30
    assert r["log_ta"] == np.log(200)
    assert r["neg_eq"] == 0


def test_nonpositive_denominators_give_nan():
    df = pd.DataFrame([_accounts(x10=0, x14=0), _accounts(x10=-50, x14=-1)])
    r = add_ratios(df)

    for c in ["tl_ta", "ni_ta", "wc_ta", "re_ta", "log_ta", "quick"]:
        assert r[c].isna().all(), c


def test_negative_equity_flag():
    df = pd.DataFrame([_accounts(x10=100, x17=150), _accounts(x10=100, x17=50)])
    assert add_ratios(df)["neg_eq"].tolist() == [1, 0]


def test_input_not_modified():
    df = pd.DataFrame([_accounts()])
    before = df.copy()
    add_ratios(df)
    pd.testing.assert_frame_equal(df, before)


def test_auc_by_lag_stops_when_too_few_failures():
    years = np.arange(2000, 2010)
    survivors = pd.DataFrame(
        {"year": np.repeat(years, 10), "is_fail": False, "lag": np.nan, "f": 1.0}
    )
    # 30 failures at t-0 and t-1, only 5 at t-2; failing firms have lower f
    pos = pd.DataFrame(
        {
            "year": 2009,
            "is_fail": True,
            "lag": [0] * 30 + [1] * 30 + [2] * 5,
            "f": 0.0,
        }
    )
    tr = pd.concat([survivors, pos], ignore_index=True)

    out = auc_by_lag(tr, survivors, ["f"], min_pos=25)

    assert out.index.tolist() == ["t-0", "t-1"]
    assert (out["f"] == 1.0).all()  # folded, so a feature falling with risk still scores 1
    assert out["n_neg"].tolist() == [100, 90]  # survivors cut to year <= last - k
