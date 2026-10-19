import numpy as np
import pandas as pd

from src.gbm import cv_lgb, feature_gain, fit_lgb, monotone_constraints

FEATURES = ["tl_ta", "mve_tl"]
FAST = {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 50}


def test_monotone_constraints():
    assert monotone_constraints(["mve_tl", "log_ta", "tl_ta", "unknown"]) == [-1, 0, 1, 0]


def test_fit_respects_risk_trend(synthetic_credit):
    model = fit_lgb(synthetic_credit, None, FEATURES, params=FAST)

    grid = pd.DataFrame({"tl_ta": np.linspace(-3, 3, 50), "mve_tl": 0.0})
    assert (np.diff(model.predict_proba(grid)[:, 1]) >= 0).all()

    grid = pd.DataFrame({"tl_ta": 0.0, "mve_tl": np.linspace(-3, 3, 50)})
    assert (np.diff(model.predict_proba(grid)[:, 1]) <= 0).all()


def test_feature_gain(synthetic_credit):
    gain = feature_gain(fit_lgb(synthetic_credit, None, FEATURES, params=FAST))

    assert gain.sum() == 1
    assert gain.index[0] == "tl_ta"


def test_cv_lgb_folds_move_forward(synthetic_credit):
    cv = cv_lgb(synthetic_credit, FEATURES, params=FAST, patience=20)

    assert cv["check_years"].tolist() == ["2004-2005", "2006-2007", "2008-2009"]
    assert (cv["roc_auc"] > 0.6).all()
