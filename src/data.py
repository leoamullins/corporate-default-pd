import pandas as pd
from src.config import FEATURES
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# split boundaries from the dataset paper
TRAIN_YEARS = (1999, 2011)
VAL_YEARS = (2012, 2014)
TEST_YEARS = (2015, 2018)


def load_data(path="data/american_bankruptcy.csv"):
    df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]
    df["year"] = df["year"].astype(int)
    df["firm_failed"] = (df["status_label"].str.strip().str.lower() == "failed").astype(
        int
    )

    last_year = df.groupby("company_name")["year"].transform("max")
    df["default"] = ((df["firm_failed"] == 1) & (df["year"] == last_year)).astype(int)

    df = df.sort_values(["company_name", "year"]).reset_index(drop=True)
    return df


def _slice_years(df, bounds):
    lo, hi = bounds
    return df[df["year"].between(lo, hi)].copy()


def split_by_year(df):
    return (
        _slice_years(df, TRAIN_YEARS),
        _slice_years(df, VAL_YEARS),
        _slice_years(df, TEST_YEARS),
    )


def get_xy(df, features=FEATURES):
    """expects ratios already added, see features.add_ratios"""
    return df[features], df["default"]


def check_splits(train, val, test):
    names = ["train", "val", "test"]
    frames = [train, val, test]

    summary = pd.DataFrame(
        {
            "rows": [len(f) for f in frames],
            "year_min": [f["year"].min() for f in frames],
            "year_max": [f["year"].max() for f in frames],
            "defaults": [int(f["default"].sum()) for f in frames],
            "base_rate": [f["default"].mean() for f in frames],
        },
        index=names,
    )

    firms = [set(f["company_name"]) for f in frames]
    overlaps = {
        "train_val": len(firms[0] & firms[1]),
        "train_test": len(firms[0] & firms[2]),
        "val_test": len(firms[1] & firms[2]),
    }

    return summary, overlaps


def year_folds(df, n_splits=3, test_years=2):
    """expanding-window folds over years"""
    years = np.sort(df["year"].unique())

    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_years)
    for fit_idx, check_idx in tss.split(years):
        fit_part = df[df["year"].isin(years[fit_idx])]
        check_part = df[df["year"].isin(years[check_idx])]
        yield fit_part, check_part
