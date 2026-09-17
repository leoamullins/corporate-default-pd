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


def evaluate(y_true, y_prob):
    a = average_precision_score(y_true, y_prob)
    return {
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
        "base_rate": y_true.mean(),
        "pr_auc": a,
        "pr_auc_lift": a / y_true.mean(),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "brier_skill": 1
        - brier_score_loss(y_true, y_prob) / (y_true.mean() * (1 - y_true.mean())),
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
