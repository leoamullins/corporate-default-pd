import pandas as pd
import numpy as np
from scipy.stats import binomtest

from src.config import MASTER_SCALE


def assign_grade(pd_, edges=MASTER_SCALE):
    bins = [0] + edges + [1]
    return pd.cut(pd_, bins=bins, labels=False, include_lowest=True) + 1


def grade_table(pd_, y, edges=MASTER_SCALE):
    bins = [0] + edges + [1]
    grades = range(1, len(bins))
    labels = [f"{lo:.2%}–{hi:.2%}" for lo, hi in zip(bins[:-1], bins[1:])]

    df = pd.DataFrame(
        {
            "grade": assign_grade(np.asarray(pd_), edges),
            "pd": np.asarray(pd_),
            "y": np.asarray(y),
        }
    )

    t = (
        df.groupby("grade")
        .agg(
            n=("y", "size"),
            defaults=("y", "sum"),
            mean_pd=("pd", "mean"),
            expected=("pd", "sum"),
        )
        .reindex(grades)
    )
    t.insert(0, "band", labels)
    t["n"] = t["n"].fillna(0).astype(int)
    t["defaults"] = t["defaults"].fillna(0).astype(int)

    t["pct_book"] = t["n"] / t["n"].sum()
    t["observed"] = t["defaults"] / t["n"].replace(0, np.nan)

    t[["obs_low", "obs_high", "binom_p"]] = np.nan

    for g, row in t.iterrows():
        if row.n == 0:
            continue

        r = binomtest(int(row.defaults), int(row.n), row.mean_pd)
        ci = r.proportion_ci(method="wilson")
        t.loc[g, ["obs_low", "obs_high", "binom_p"]] = [ci.low, ci.high, r.pvalue]

    return t
