import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from src.metrics import reliability


class LogitRecalibrator:
    """corrected logit, fitted on a sample later than models training data"""

    def __init__(self, eps=1e-6):
        self.eps = eps

    def _logit(self, p):
        p = np.clip(np.asarray(p, dtype=float), self.eps, 1 - self.eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        self.model = LogisticRegression(C=np.inf).fit(self._logit(p), y)
        self.a_ = self.model.intercept_[0]
        self.b_ = self.model.coef_[0, 0]
        return self

    def predict_proba(self, p):
        return self.model.predict_proba(self._logit(p))[:, 1]


# plotting: one panel per model, colours fixed per model so before and after figures match
COLORS = {"Z''": "#2a78d6", "Scorecard": "#eb6834", "LightGBM": "#1baf7a"}
LIM = (3e-4, 0.6)


def _draw_panel(data, color, **kwargs):
    """diagonal, 95% Wilson intervals, points"""
    ax = plt.gca()
    ax.plot(LIM, LIM, color="#9a9a94", lw=1, ls="--", zorder=1)
    ax.errorbar(
        data["mean_pred"],
        data["observed"],
        yerr=[data["observed"] - data["obs_low"], data["obs_high"] - data["observed"]],
        fmt="none",
        ecolor=color,
        alpha=0.5,
        lw=1.5,
    )
    ax.scatter(
        data["mean_pred"],
        data["observed"],
        s=60,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )


def plot_reliability(preds, y_true, title, path=None, n_bins=10):
    """reliability per model on log axes, 10 groups with equal numbers of defaults

    preds: {model name: PDs}, names as in COLORS, all scored on the same rows as y_true
    """
    y_true = np.asarray(y_true)
    names = list(preds)
    rel = {
        n: reliability(y_true, preds[n], n_bins=n_bins, strategy="defaults")
        for n in names
    }
    long = pd.concat(rel, names=["model", "bin"]).reset_index()

    with sns.axes_style("whitegrid", {"grid.color": "#e6e6e1"}):
        g = sns.FacetGrid(
            long,
            col="model",
            col_order=names,
            hue="model",
            hue_order=names,
            palette=COLORS,
            height=4.6,
            despine=True,
        )
        g.map_dataframe(_draw_panel)

    g.set(xscale="log", yscale="log", xlim=LIM, ylim=LIM)
    g.set_axis_labels("Mean predicted PD", "Observed default rate")
    g.set_titles(col_template="")  # replaced by the left-aligned titles below
    for name, ax in g.axes_dict.items():
        ratio = np.mean(preds[name]) / y_true.mean()
        ax.set_title(
            f"{name}  (mean PD / observed = {ratio:.2f})", loc="left", fontsize=11
        )
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mtick.FixedLocator([0.001, 0.01, 0.1]))
            axis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
            axis.set_minor_locator(mtick.NullLocator())
    g.figure.suptitle(title, x=0.01, ha="left", fontsize=11)
    g.tight_layout()

    if path is not None:
        path.parent.mkdir(exist_ok=True)
        g.savefig(path, dpi=150)
    return g
