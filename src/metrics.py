from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import numpy as np
import pandas as pd

SEED = 42


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
    }


def bootstrap_metric(y_true, y_prob, metric_fn, n_boot=1000, seed=SEED, alpha=0.05):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        scores.append(metric_fn(yt, yp))

    scores = np.array(scores)
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return metric_fn(y_true, y_prob), lo, hi


def ks_statistic(y_true, y_prob):
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)}).sort_values(
        "p"
    )
    cum_bad = (d.y == 1).cumsum() / (d.y == 1).sum()
    cum_good = (d.y == 0).cumsum() / (d.y == 0).sum()
    return float((cum_good - cum_bad).abs().max())
