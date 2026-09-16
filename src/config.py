import numpy as np

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
RISK_TREND = {"mve_tl": -1, "tl_ta": 1, "ni_ta": -1, "quick": -1, "re_ta": -1, "log_ta": -1}

ALTMAN_ZPP = {"wc_ta": 6.56, "re_ta": 3.26, "ebit_ta": 6.72, "bve_tl": 1.05}
ZPP_ZONES = [(-np.inf, 1.1, "distress"), (1.1, 2.6, "grey"), (2.6, np.inf, "safe")]
