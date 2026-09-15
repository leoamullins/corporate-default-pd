# One-Year Corporate Default Prediction

A probability of default model for US-listed companies, built on accounting ratios. Pairs logistic regression with gradient boosting and evaluates both on calibration as well as discrimination.

**Target**: does the firm file for Chapter 7 or Chapter 11 in the next fiscal year?

## Data

[American Companies Bankruptcy Dataset](https://github.com/sowide/bankruptcy_dataset)

- 78682 firm-year observations
- 8262 companies listed on the NYSE and NASDAQ
- 1999 - 2018
- 18 accounting features per firm year

Splits are time-ordered, as defined by dataset authors.

| Split   | Years       |
| --------| -------     |
| Train   | 1999-2011   |
| Val     | 2012-2014   |
| Test    | 2015-2018   |

## Label Correction

**The shipped label does not match the documented one.** This is corrected before any modelling and it changes the problem substantially. The dataset documentation states that the fiscal year prior to a Chapter 7 or 11 filing is labelled as 1 and all other firm-years are labelled as 0. What ships is `status_label`, which is **firm level**: a company that failed in 2011 carries the "failed" mark on every row it has back to 1999.

| Check                            | As Shipped |
| --------                         | -------    |
| Firms with >= 1 positive         | 609        |
| Positive Rows                    | 5220       |
| Mean positives per failed firm   | 8.57       |

**Why it matters.** The as-shipped target is "does this firm eventually fail within the observation window," not a one-year PD. The label is not a function of that year's financials i.e. a firm's healthy 1999 accounts are labelled by a 2011 event. Any model trained on it learns persistent firm characteristics
that correlate with eventual failure, and reports an optimistic score for a question nobody asks.
The base rates gave it away. US public-company bankruptcy runs near 1% a year; the shipped labels give 7.9% in train, falling to 2.3% in test. That decline is the artefact, not an economic trend: failed firms' back-catalogues are concentrated in the early years and have delisted out of the panel by 2015. The correction. For each failed firm, label 1 on its final observed year only; 0 everywhere earlier. This restores the documented design rather than imposing a new one.

After the correction, the base rates became approx 1% in the train, val and test datasets matching the average US year-on-year default rate.

See `notebooks/01_data_checks.ipynb` for more details.

## Split Integrity

Firms appear in more than one split (3,290 across train-val etc.) this is expected behaviour and is not leakage.

The time split partitions firm-years not firms individually, so any company which survives past the boundary will appear in both splits.

Predicting a firms outcome in 2016, from data in 1999-2011 train split is what the model does. Each default even appears exactly once - the target marks a single year where the firm stops filing afterward. Verified above.

The residual risk is hindsight inside a feature, not firm identity. Checked two ways.

Univariately on train, no ratio behaves like a leaked outcome. The strongest, `mve_tl`, reaches 0.80 AUC — high, but consistent with a market-based solvency measure rather than a label in disguise.

The direct test re-indexes each failed firm's history by years-to-failure and scores the ratios at each lag against surviving firm-years.

| Years before filing | `mve_tl` | `tl_ta` | `ni_ta` | `quick` |
| --------| ------- | ------- | ------- | ------- |
| t-0   | 0.804    | 0.773   | 0.731   | 0.746   |
| t-1   | 0.706    | 0.700   | 0.683   | 0.665   |
| t-3   | 0.626    | 0.616   | 0.614   | 0.601   |
| t-5   | 0.602    | 0.585   | 0.611   | 0.575   |

A leaked feature would score near 1.0 at t-0 and collapse to chance a year earlier. This decays gradually and stays above chance five years out, which is early warning rather than hindsight.

**The one caveat is `mve_tl`.** Market value at the fiscal year end before a filing already reflects investors pricing in the collapse, and the jump from 0.706 at t-1 to 0.804 at t-0 measures roughly how much of its strength comes from that. Not leakage — the price is observable at the time — but the model is partly inheriting the market's forecast rather than deriving distress from fundamentals. An accounting-only ablation is reported alongside the main model.

See `notebooks/02_EDA.ipynb` for more details.

## Features

The dataset ships 18 raw accounting line items. The documented X1-X18 mapping was confirmed arithmetically rather than taken on trust:

| Identity | Holds in |
| --------| ------- |
| `x16` Total Revenue == `x9` Net Sales | 100.00% |
| `x13` Gross Profit == `x16` − `x2` COGS | 100.00% |
| `x4` EBITDA == `x12` EBIT + `x3` D&A | 100.00% |

Two consequences. `x9` and `x16` are the same column (Spearman 1.000000), so one has to go before any selection step or it will be chosen twice. `x13` and `x4` are deterministic functions of other fields. The 18 columns carry 15 independent quantities.

**Raw levels are not the model input.** $500m of liabilities means nothing without the asset base behind it. Scored univariately on train, the strongest raw field reaches 0.74 AUC and does so largely by proxying firm size; the same information expressed as ratios reaches 0.80.

Thirteen candidate ratios were built across the standard credit dimensions — leverage, profitability, liquidity, activity, structure. Denominators are safe: `x10` and `x17` are strictly positive throughout, so no ratio produces an infinity and no imputation layer is needed.

## Feature Selection

Thirteen ratios is more than the signal supports. Redundancy was located by clustering the Spearman correlation matrix — distance `1 − |rho|`, average linkage, cut at `|rho| > 0.7` — which collapsed 13 ratios into 9 groups.

Clustering is unsupervised: it finds where the redundancy is but cannot say which member to keep. Univariate AUC and information value decide that within each group.

| Group | Kept | Dropped, and why |
| --------| ------- | ------- |
| Profitability | `ni_ta` (0.727 AUC, 0.729 IV) | `ebit_ta` — EBIT is before interest, and interest is what breaks a levered firm |
| Liquidity | `quick` (0.742, 0.835) | `ca_cl`, `wc_ta` — the quick ratio strips inventory, which cannot be sold at book in distress |
| Leverage | `tl_ta` (0.769, 1.144), `mve_tl` (0.800, 1.240) | `neg_eq`, `ltd_ta` — on a one-year horizon, long-term debt is the debt that is not due |
| Activity | — | `sales_ta` (0.562) — asset turnover measures industry, not distress |
| Margin | — | `gp_sales` (0.600) — gross margin measures industry, not distress |

Final six: `mve_tl`, `tl_ta`, `ni_ta`, `quick`, `re_ta`, `log_ta`.

**Why it matters.** Pruning costs 0.010 CV AUC, which sits inside one fold standard deviation. What it buys is coefficient stability.

| Feature set | CV AUC | Fold sd | Wrong-signed coefficients |
| --------| ------- | ------- | ------- |
| All 13   | 0.7963  | 0.0137  | 6 |
| Selected 6 | 0.7792 | 0.0119  | 1 |

With all 13 in a logistic regression, six coefficients take the wrong sign — including *both* leverage ratios, so the model asserts that more debt reduces default risk. That is collinearity splitting weight across redundant inputs, not a finding, and a scorecard whose coefficients contradict credit logic cannot be defended at validation regardless of its AUC. The justification for pruning is interpretability, not information retention.

One trap worth recording. `neg_eq` is exactly `1[tl_ta > 1]`, yet the clustering placed it in a group of its own because binarising a continuous variable caps its rank correlation with the parent — here at 0.498. Correlation clustering detects linear redundancy, not functional dependence. It is dropped because WOE binning of `tl_ta` will place a cut at 1.0 and recover the same information. As a standalone flag it is still striking: 9.9% of firm-years have liabilities exceeding assets, and they default at 3.26% against 0.50% for the rest.

See `notebooks/02_EDA.ipynb` for more details.

## Benchmark: Altman Z''

Scored with the **published 1995 coefficients and no refitting** — the point of a benchmark is that it is fixed.

Z'' rather than the original 1968 Z-score because this panel is mixed-sector and Z'' was built for non-manufacturers. It also drops Sales/Total Assets, a choice the feature selection above independently arrived at: `sales_ta` scored 0.562 AUC, barely above chance.

| Split | ROC AUC | PR-AUC lift | KS |
| --------| ------- | ------- | ------- |
| Train | 0.7634 | 2.49 | 0.456 |
| Val   | 0.7878 | 2.39 | 0.582 |
| Test  | 0.7731 | 2.37 | 0.526 |

Discrimination rises out of time, which is further evidence the splits are sound.

The published distress bands give a ready-made rating scale. On the test window:

| Zone | Firm-years | Defaults | Rate | Lift |
| --------| ------- | ------- | ------- | ------- |
| Distress < 1.1 | 5,058 | 105 | 2.08% | 2.14 |
| Grey 1.1 - 2.6 | 1,942 | 5   | 0.26% | 0.27 |
| Safe > 2.6     | 5,282 | 9   | 0.17% | 0.18 |

**88.2% of test defaults fall in the distress zone, which is 41.2% of the book.**

**Why it matters.** This is the bar. Four ratios with coefficients fixed in 1995 reach 0.773 out of time, and anything built here has to clear that to justify its own complexity.

A note on Brier skill, which `evaluate()` reports and which sits at approximately zero here — and will for every model in this project. At a 1% base rate, squared error is dominated by the mass of correctly-predicted non-defaults, leaving almost no room for sharpness to register. It is not evidence that the model adds nothing: the same predictions score 0.773 AUC and 0.526 KS. Calibration is assessed on reliability curves and calibration-in-the-large instead.

See `notebooks/03_benchmarks.ipynb` for more details.