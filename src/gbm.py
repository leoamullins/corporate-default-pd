import lightgbm as lgb
import pandas as pd

from src.config import GBM_PARAMS, GBM_UNCONSTRAINED, RISK_TREND, SEED
from src.data import year_folds
from src.metrics import evaluate


def monotone_constraints(features):
    return [0 if f in GBM_UNCONSTRAINED else RISK_TREND.get(f, 0) for f in features]


def fit_lgb(tr, val, features, params=GBM_PARAMS, patience=200):
    model = lgb.LGBMClassifier(
        **params,
        monotone_constraints=monotone_constraints(features),
        random_state=SEED,
        verbose=-1,
    )
    if val is None:
        model.fit(tr[features], tr["default"])
    else:
        model.fit(
            tr[features],
            tr["default"],
            eval_X=(val[features],),
            eval_y=(val["default"],),
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(patience, verbose=False)],
        )
    return model


def cv_lgb(tr, features, params=GBM_PARAMS, patience=200):
    rows = []
    for fit_part, check_part in year_folds(tr):
        model = fit_lgb(fit_part, check_part, features, params, patience)
        scores = evaluate(
            check_part["default"], model.predict_proba(check_part[features])[:, 1]
        )
        rows.append(
            {
                "check_years": f"{check_part['year'].min()}-{check_part['year'].max()}",
                "trees": model.best_iteration_,
                "n_pos": scores["n_pos"],
                "roc_auc": scores["roc_auc"],
                "ks": scores["ks"],
            }
        )
    return pd.DataFrame(rows)


def feature_gain(model):
    gain = pd.Series(
        model.booster_.feature_importance("gain"), index=model.feature_name_
    )
    return (gain / gain.sum()).sort_values(ascending=False)
