import numpy as np
import pandas as pd
import pytest

from src.benchmark import ZScoreToPD, altman_zpp, apply_caps, fit_caps, zpp_zone


def test_zpp_on_known_row():
    df = pd.DataFrame({"x1": [50], "x14": [30], "x10": [200], "x15": [40], "x12": [20], "x17": [80]})
    # wc_ta 0.1, re_ta 0.2, ebit_ta 0.1, bve_tl 1.5
    assert altman_zpp(df).iloc[0] == pytest.approx(3.555)


def test_zones_use_altman_cutoffs():
    z = pd.Series([-5, 1.1, 1.5, 2.6, 10])
    assert zpp_zone(z).tolist() == ["distress", "distress", "grey", "grey", "safe"]


def test_caps_clip_outliers():
    df = pd.DataFrame({"a": np.r_[np.arange(100.0), 1e6], "b": np.r_[-1e6, np.arange(100.0)]})
    caps = fit_caps(df)
    capped = apply_caps(df, caps)

    assert capped["a"].max() == caps.loc[0.99, "a"]
    assert capped["b"].min() == caps.loc[0.01, "b"]
    assert capped.shape == df.shape


def test_z_to_pd_falls_as_z_rises(synthetic_credit):
    z = -synthetic_credit["tl_ta"]
    model = ZScoreToPD().fit(z, synthetic_credit["default"])

    pd_grid = model.predict_proba(np.linspace(-3, 3, 20))
    assert (np.diff(pd_grid) < 0).all()
