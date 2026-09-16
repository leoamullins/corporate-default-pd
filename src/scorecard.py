import numpy as np
import pandas as pd


def fine_edges(x, n_bins=20):
    """Quantile cut points"""
    edges = [-np.inf]
    for n in range(1, n_bins):
        q = np.quantile(x, n / n_bins)
        if q > edges[-1]:
            edges.append(q)

    edges.append(np.inf)

    return list(zip(edges[:-1], edges[1:]))


def woe_table(x, y, edges):
    # sort the data into bins
    bins = [edge[0] for edge in edges] + [edges[-1][1]]
    binned_data = pd.Series(pd.cut(x, bins), name="bins")

    data = pd.concat([binned_data, y], axis=1)

    grouped = data.groupby("bins", observed=False)

    table = grouped[y.name].agg(n="count", bad="sum")
    table["good"] = table["n"] - table["bad"]
    table["rate"] = table["bad"] / table["n"]

    # +0.5 in every bin stops log(0) when a bin has no defaults;
    # adding 0.5 * len(table) to the total keeps each share column summing to 1
    pct_good = (table["good"] + 0.5) / (table["good"].sum() + 0.5 * len(table))
    pct_bad = (table["bad"] + 0.5) / (table["bad"].sum() + 0.5 * len(table))

    table["woe"] = np.log(pct_good / pct_bad)
    table["iv"] = (pct_good - pct_bad) * table["woe"]

    table = table.reset_index()
    return table


def trend_violations(default_rates, trend):
    """True where the default rate moves against trend (+1 rising, -1 falling)"""
    if trend not in (1, -1):
        raise ValueError("trend must be 1 (increasing) or -1 (decreasing)")

    return default_rates.diff() * trend <= 0


def monotone_edges(x, y, trend, n_bins=20, min_bad=20):

    edges = fine_edges(x, n_bins)

    while len(edges) > 1:

        table = woe_table(x, y, edges)
        default_rates = table["rate"]

        violations = trend_violations(default_rates, trend)

        # Check minimum number of defaults
        bad_violations = table["bad"] < min_bad

        if not violations.any() and not bad_violations.any():
            # break if we have no violations or we have reached the min bad criterion
            break

        # prioritising violations with too few defaults
        if bad_violations.any():
            i = bad_violations[bad_violations].index[0]

            if i == 0:
                merge_left = i
                merge_right = i + 1

            elif i == len(edges) - 1:
                merge_left = i - 1
                merge_right = i

            elif table["bad"].iloc[i - 1] <= table["bad"].iloc[i + 1]:
                merge_left = i - 1
                merge_right = i
            else:
                merge_left = i
                merge_right = i + 1

        else:
            i = violations[violations].index[0]  # index of first violation

            merge_left, merge_right = i - 1, i

        new_bin = (edges[merge_left][0], edges[merge_right][1])

        edges[merge_left : merge_right + 1] = [new_bin]

    return edges


class WOEBinner:
    def __init__(self, trends, n_bins=20, min_bad=20):
        self.trends = trend
        self.n_bins = n_bins
        self.min_bad = min_bad

    def fit(self, X, y):
        """learn bins and woe for each feature (training set)"""
        self.edges_ = {}
        self.tables_ = {}
        for feature, trend in self.trends.items():
            edges = monotone_edges(X[feature], y, trend, self.n_bins, self.min_bad)
            self.edges_[feature] = edges
            self.tables_[feature] = woe_table(X[feature], y, edges)
        return self

    def transform(self, X):
        """replace each value with its bin's WOE, derived from fit"""
        out = {}
        for feature, edges in self.edges_.items():
            bins = [e[0] for e in edges] + [edges[-1][1]]
            bin_number = pd.cut(X[feature], bins, labels=False)
            out[feature] = self.tables_[feature]["woe"].to_numpy()[bin_number]
        return pd.DataFrame(out, index=X.index)

    def iv(self):
        """total IV per feature, descending"""
        return pd.Series(
            {f: t["iv"].sum() for f, t in self.tables_.items()}
        ).sort_values(ascending=False)
