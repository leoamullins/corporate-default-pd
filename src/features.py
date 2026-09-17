import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _safe_div(num, den):
    """Nan if den <= 0 undefined or sign-flipped ratio"""
    return num / den.where(den > 0)


def add_ratios(df):
    """credit rations on the confirmed X1-X18 mapping"""
    r = df.copy()

    r["tl_ta"] = _safe_div(df.x17, df.x10)  # leverage
    r["ltd_ta"] = _safe_div(df.x11, df.x10)
    r["mve_tl"] = _safe_div(df.x8, df.x17)
    r["neg_eq"] = (df.x17 > df.x10).astype(int)
    r["ni_ta"] = _safe_div(df.x6, df.x10)  # profitability
    r["ebit_ta"] = _safe_div(df.x12, df.x10)
    r["gp_sales"] = _safe_div(df.x13, df.x9)
    r["ca_cl"] = _safe_div(df.x1, df.x14)  # liquidity
    r["wc_ta"] = _safe_div(df.x1 - df.x14, df.x10)
    r["quick"] = _safe_div(df.x1 - df.x5, df.x14)
    r["sales_ta"] = _safe_div(df.x9, df.x10)  # activity
    r["re_ta"] = _safe_div(df.x15, df.x10)  # structure
    r["log_ta"] = np.log(df.x10.where(df.x10 > 0))
    return r


def auc_by_lag(tr, survivors, features, max_lag=5, min_pos=25):
    last_year = tr["year"].max()
    out = {}
    for k in range(max_lag + 1):
        pos = tr[tr["is_fail"] & (tr["lag"] == k)]
        if len(pos) < min_pos:
            break
        # a failure k years out can only be seen on rows up to last_year - k, so survivors
        # are cut to the same years; it also keeps rows of firms failing after the window out
        neg = survivors[survivors["year"] <= last_year - k]
        row = {}
        for c in features:
            s = pd.concat([pos[[c]].assign(y=1), neg[[c]].assign(y=0)])
            v = s[c].replace([np.inf, -np.inf], np.nan)
            a = roc_auc_score(s["y"], v.fillna(v.median()))
            row[c] = max(a, 1 - a)
        out[f"t-{k}"] = {**row, "n_pos": len(pos), "n_neg": len(neg)}

    return pd.DataFrame(out).T
