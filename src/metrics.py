from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import numpy as np
import pandas as pd
from src.config import SEED


def _firm_rows(groups):
    """row pos for each firm"""
    codes, _ = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes))[:-1])


def _resample(rng, n, firm_rows=None):
    """row indices for one bootstrap sample: rows, or whole firms if given"""
    if firm_rows is None:
        return rng.integers(0, n, size=n)
    pick = rng.integers(0, len(firm_rows), size=len(firm_rows))
    return np.concatenate([firm_rows[i] for i in pick])


def evaluate(y_true, y_prob, ref_rate=None):
    """ref_rate: constant PD that Brier skill is measured against (defaults to this split's own base rate)"""
    a = average_precision_score(y_true, y_prob)
    if ref_rate is None:
        ref_rate = y_true.mean()
    ref_brier = brier_score_loss(y_true, np.full(len(y_true), ref_rate))
    return {
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
        "base_rate": y_true.mean(),
        "pr_auc": a,
        "pr_auc_lift": a / y_true.mean(),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "ref_rate": ref_rate,
        "brier_skill": 1 - brier_score_loss(y_true, y_prob) / ref_brier,
        "mean_pred": y_prob.mean(),
        "ks": ks_statistic(y_true, y_prob),
    }


def bootstrap_metric(
    y_true, y_prob, metric_fn, n_boot=1000, seed=SEED, alpha=0.05, groups=None
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    scores = []

    firm_rows = None if groups is None else _firm_rows(groups=groups)

    for _ in range(n_boot):
        idx = _resample(rng, n, firm_rows)
        yt, yp = y_true[idx], y_prob[idx]
        # firm draws vary in size, so compare against this sample's length
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(metric_fn(yt, yp))

    scores = np.array(scores)
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return metric_fn(y_true, y_prob), lo, hi


def paired_bootstrap(
    y_true,
    prob_a,
    prob_b,
    metric_fn,
    n_boot=1000,
    seed=SEED,
    alpha=0.05,
    groups=None,
):
    """metric(a) - metric(b), both scored on the same resample (rows, or firms if groups given)"""
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    diffs = []

    firm_rows = None if groups is None else _firm_rows(groups=groups)

    for _ in range(n_boot):
        idx = _resample(rng, n, firm_rows)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        diffs.append(metric_fn(yt, prob_a[idx]) - metric_fn(yt, prob_b[idx]))

    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return metric_fn(y_true, prob_a) - metric_fn(y_true, prob_b), lo, hi


def ks_statistic(y_true, y_prob):
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    g = d.groupby("p")["y"].agg(bad="sum", n="count")
    g["good"] = g["n"] - g["bad"]
    cum_bad = g["bad"].cumsum() / g["bad"].sum()
    cum_good = g["good"].cumsum() / g["good"].sum()
    return float((cum_good - cum_bad).abs().max())


def reliability(y_true, y_prob, n_bins=10, strategy="rows", z=1.96):
    """mean predicted vs observed default rate per bin of predicted PD, with a Wilson interval on observed

    strategy="rows": bins hold equal numbers of rows
    strategy="defaults": bins hold equal numbers of defaults, so each point is about equally precise
    """
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    if strategy == "rows":
        # rank first to avoid the duplicate edge issue
        d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    elif strategy == "defaults":
        d = d.sort_values("p", kind="stable")
        # a bin closes once it holds its share of the defaults
        before = d["y"].cumsum() - d["y"]
        d["bin"] = np.minimum(before * n_bins // d["y"].sum(), n_bins - 1)
    else:
        raise ValueError("strategy must be 'rows' or 'defaults'")
    r = d.groupby("bin").agg(
        n=("y", "size"),
        defaults=("y", "sum"),
        mean_pred=("p", "mean"),
        observed=("y", "mean"),
    )
    # Wilson rather than normal: stays inside [0, 1] and is sensible for bins with 0 defaults
    n, p = r["n"], r["observed"]
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    r["obs_low"] = (centre - half).clip(lower=0)
    r["obs_high"] = centre + half
    return r
