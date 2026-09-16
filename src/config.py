from pathlib import Path

import numpy as np

# repo root, so paths work whichever folder a notebook runs from
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "american_bankruptcy.csv"
OPTUNA_STORAGE = f"sqlite:///{ROOT / 'optuna.db'}"

SEED = 42

RATIOS = [
    "tl_ta",
    "ltd_ta",
    "mve_tl",
    "neg_eq",
    "ni_ta",
    "ebit_ta",
    "gp_sales",
    "ca_cl",
    "wc_ta",
    "quick",
    "sales_ta",
    "re_ta",
    "log_ta",
]

FEATURES = ["mve_tl", "tl_ta", "ni_ta", "quick", "re_ta", "log_ta"]

# which way default risk moves as each feature rises: +1 riskier, -1 safer
RISK_TREND = {
    "mve_tl": -1,
    "tl_ta": 1,
    "ni_ta": -1,
    "quick": -1,
    "re_ta": -1,
    "log_ta": -1,
}

ALTMAN_ZPP = {"wc_ta": 6.56, "re_ta": 3.26, "ebit_ta": 6.72, "bve_tl": 1.05}
ZPP_ZONES = [(-np.inf, 1.1, "distress"), (1.1, 2.6, "grey"), (2.6, np.inf, "safe")]

GBM_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "num_leaves": 7,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
}

GBM_UNCONSTRAINED = ["log_ta"]
