import numpy as np
import pytest
import pandas as pd


@pytest.fixture
def toy_panel_csv(tmp_path):
    rows = [
        ("C", "failed", 2015),
        ("A", "alive", 2011),
        ("B", " Failed", 2010),
        ("A", "Alive ", 2009),
        ("C", "FAILED", 2013),
        ("B", "failed", 2011),
        ("A", "alive", 2013),
        ("C", "failed", 2016),
        ("B", "failed ", 2009),
        ("A", "alive", 2010),
        ("C", "failed", 2014),
        ("A", "alive", 2012),
    ]
    df = pd.DataFrame(rows, columns=["company_name", "status_label", "Year "])

    path = tmp_path / "to y.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_credit():
    """5000 firm-years where risk rises with tl_ta and mve_tl is pure noise"""
    rng = np.random.default_rng(0)
    n = 5000

    tl_ta = rng.normal(size=n)
    mve_tl = rng.normal(size=n)
    pd_true = 1 / (1 + np.exp(-(-3 + 1.2 * tl_ta)))

    return pd.DataFrame(
        {
            "company_name": np.repeat([f"F{i}" for i in range(n // 5)], 5),
            "year": np.tile(np.arange(2000, 2010), n // 10),
            "tl_ta": tl_ta,
            "mve_tl": mve_tl,
            "pd_true": pd_true,
            "default": (rng.random(n) < pd_true).astype(int),
        }
    )
