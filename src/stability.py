import numpy as np
import pandas as pd

# conventional PSI thresholds: < 0.1 stable, 0.1-0.25 monitor, > 0.25 shift
PSI_BANDS = [0.1, 0.25]
PSI_LABELS = ["stable", "monitor", "shift"]


def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def psi_edges(expected, n_bins=10):
    e = _clean(expected)
    # quantile edges from reference sample, open-ended so out-of-range values still count
    inner = np.quantile(e, np.linspace(0, 1, n_bins + 1))[1:-1]
    return np.r_[-np.inf, np.unique(inner), np.inf]


def psi(expected, actual, n_bins=10, smoothing=0.5):
    e, a = _clean(expected), _clean(actual)
    if len(e) == 0 or len(a) == 0:
        return np.nan

    edges = psi_edges(e, n_bins)

    # add a pseudo-count to every bin so an empty bin doesn't blow up the log, then renormalise
    e_cnt = np.histogram(e, edges)[0] + smoothing
    a_cnt = np.histogram(a, edges)[0] + smoothing
    e_share = e_cnt / e_cnt.sum()
    a_share = a_cnt / a_cnt.sum()

    # psi sum
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def psi_flag(value):
    if np.isnan(value):
        return None
    return PSI_LABELS[np.searchsorted(PSI_BANDS, value, side="right")]


def psi_table(ref, others, columns, n_bins=10):
    rows = {}
    for col in columns:
        ref_clean = _clean(ref[col])
        # ties can collapse quantile edges, so record how many bins were actually used
        n_used = len(psi_edges(ref_clean, n_bins)) - 1 if len(ref_clean) else 0
        row = {"n_bins": n_used, "missing_ref": ref[col].isna().mean()}
        for name, df in others.items():
            value = psi(ref[col], df[col], n_bins)
            row[f"psi_{name}"] = value
            row[f"flag_{name}"] = psi_flag(value)
            row[f"missing_{name}"] = df[col].isna().mean()
        rows[col] = row
    return pd.DataFrame.from_dict(rows, orient="index")
