import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.config import ALTMAN_ZPP, ZPP_ZONES


def zpp_components(df):
    """the compnenets for Z'' scoring mathcing ALTMAN_ZPP"""
    return pd.DataFrame(
        {
            "wc_ta": (df.x1 - df.x14) / df.x10,
            "re_ta": df.x15 / df.x10,
            "ebit_ta": df.x12 / df.x10,
            "bve_tl": (df.x10 - df.x17) / df.x17,  # BOOK equity, per the 1995 spec
        },
        index=df.index,
    )


def fit_caps(df, p=0.01):
    return df.quantile([p, 1 - p])


def apply_caps(df, caps):
    return df.clip(caps.iloc[0], caps.iloc[1], axis=1)


def altman_zpp(df, caps=None):
    comp = zpp_components(df)
    if caps is not None:
        comp = apply_caps(comp, caps)
    return sum(ALTMAN_ZPP[c] * comp[c] for c in ALTMAN_ZPP)


def zpp_zone(z):
    edges = [lo for lo, _, _ in ZPP_ZONES] + [ZPP_ZONES[-1][1]]
    labels = [name for _, _, name in ZPP_ZONES]
    return pd.cut(z, bins=edges, labels=labels)


class ZScoreToPD:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, z, y):
        self.model.fit(np.asarray(z).reshape(-1, 1), y)
        return self

    def predict_proba(self, z):
        return self.model.predict_proba(np.asarray(z).reshape(-1, 1))[:, 1]
