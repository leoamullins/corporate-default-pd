# One-Year Corporate Default Prediction

A probability of default model for US-listed companies, built on accounting ratios. Pairs logistic regression with gradient boosting and evaluates both on calibration as well as discrimination.

**Target**: does the firm file for Chapter 7 or Chapter 11 in the next fiscal year?

## Data

[US Company Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset) on Kaggle, file `american_bankruptcy.csv`.

- 78,682 firm-year observations, 1999 - 2018
- 8,971 anonymised company IDs (`C_1`, `C_2`, ...), listed on the NYSE and NASDAQ. The dataset's own description says 8,262 companies; the file has 8,971.
- 18 accounting features per firm-year
- Columns: `company_name`, `status_label`, `year`, `X1` to `X18`
- SHA-256: `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`

**Use the Kaggle file.** The same panel is also published [on GitHub](https://github.com/sowide/bankruptcy_dataset) as `american_bankruptcy_dataset.csv`, with `fyear`, `Division` and `MajorGroup` columns. That file uses a different column mapping for `X1` to `X18`, partly on a different scale: `C_1`'s 1999 total assets are `X10` = 740.998 in the Kaggle file and `X2` = 740998 on GitHub. The identity checks under [Features](#features) hold on every row of the Kaggle file and on almost none of the GitHub file (at most 0.01%), so the feature definitions here do not apply to it.

Splits are time-ordered, as defined by dataset authors.

| Split   | Years       |
| --------| -------     |
| Train   | 1999-2011   |
| Val     | 2012-2014   |
| Test    | 2015-2018   |

## Reproduce

1. Download `american_bankruptcy.csv` from the [Kaggle page](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset) and save it as `data/american_bankruptcy.csv` (`data/` is not in git). To check it is the same file, run `shasum -a 256 data/american_bankruptcy.csv` and compare with the hash above.
2. Install [uv](https://docs.astral.sh/uv/) and run `uv sync` in the repo root. This creates `.venv` with Python 3.14 and installs `src/` as a package, so the notebooks can import it from any folder.
3. Run the notebooks in order, `01_data_checks` to `05_lightgbm`, with the `.venv` kernel. From the command line:

   ```bash
   for nb in notebooks/0*.ipynb; do uv run jupyter execute --inplace "$nb"; done
   ```

   The first run of notebook 05 also runs both Optuna searches (130 trials, a few minutes) and saves them to `optuna.db`, which later runs reuse.

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

The direct test re-indexes each failed firm's history by years-to-failure and scores the ratios at each lag against surviving firm-years. At lag k the failed rows can only come from years up to 2011 − k, so the survivors are cut to the same years; otherwise anything that drifts over time, such as firm size, reads as early warning.

| Years before filing | `mve_tl` | `tl_ta` | `ni_ta` | `quick` |
| --------| ------- | ------- | ------- | ------- |
| t-0   | 0.804    | 0.773   | 0.731   | 0.746   |
| t-1   | 0.707    | 0.701   | 0.681   | 0.665   |
| t-3   | 0.625    | 0.617   | 0.609   | 0.599   |
| t-5   | 0.606    | 0.589   | 0.603   | 0.572   |

A leaked feature would score near 1.0 at t-0 and collapse to chance a year earlier. This decays gradually and stays above chance five years out, which is early warning rather than hindsight.

**The one caveat is `mve_tl`.** Market value at the fiscal year end before a filing already reflects investors pricing in the collapse, and the jump from 0.707 at t-1 to 0.804 at t-0 measures roughly how much of its strength comes from that. Not leakage — the price is observable at the time — but the model is partly inheriting the market's forecast rather than deriving distress from fundamentals. An accounting-only ablation is reported alongside the main model.

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

One trap worth recording. `neg_eq` is exactly `1[tl_ta > 1]`, yet the clustering placed it in a group of its own because binarising a continuous variable caps its rank correlation with the parent — here at 0.498. Correlation clustering detects linear redundancy, not functional dependence. It is dropped because WOE binning of `tl_ta` should recover the same information. As a standalone flag it is still striking: 9.9% of firm-years have liabilities exceeding assets, and they default at 3.26% against 0.50% for the rest.

See `notebooks/02_EDA.ipynb` for more details.

## Benchmark: Altman Z''

Scored with the **published 1995 coefficients and no refitting** — the point of a benchmark is that it is fixed.

Z'' rather than the original 1968 Z-score because this panel is mixed-sector and Z'' was built for non-manufacturers. It also drops Sales/Total Assets, a choice the feature selection above independently arrived at: `sales_ta` scored 0.562 AUC, barely above chance.

| Split | ROC AUC | PR-AUC lift | KS |
| --------| ------- | ------- | ------- |
| Train | 0.7634 | 2.49 | 0.456 |
| Val   | 0.7878 | 2.39 | 0.581 |
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

A note on Brier skill, which `evaluate()` reports and which sits at approximately zero here and stays small for every model in this project (0.045 for the scorecard and 0.084 for LightGBM on test). At a 1% base rate, squared error is dominated by the mass of correctly-predicted non-defaults, leaving almost no room for sharpness to register. It is not evidence that the model adds nothing: the same predictions score 0.773 AUC and 0.526 KS. Calibration is assessed on reliability curves and calibration-in-the-large instead.

Calibration-in-the-large already shows drift. Z'' is mapped to a PD by a one-variable logistic regression fitted on train (0.72% base rate). It predicts a mean PD of 0.76% on both val and test, against observed rates of 0.83% and 0.97%, so test PDs come out about 22% too low. The default rate is simply higher in 2015-2018 than in the fitting window. This is left uncorrected, since the benchmark is meant to stay fixed, but every model fitted on train will face the same shift.

See `notebooks/03_benchmarks.ipynb` for more details.

## Scorecard: WOE Logistic Regression

Each ratio is cut into bins, and every value is replaced by its bin's **weight of evidence**: WOE = ln(share of non-defaulters in the bin ÷ share of defaulters in the bin). Positive means safer than average, negative riskier, zero uninformative. A logistic regression is then fitted on the six WOE columns.

**Binning** is fitted on train only. Each ratio starts as 20 quantile bins, and neighbours are merged until every bin holds at least 20 defaults (5% of train defaults) and the default rate moves in the direction credit logic predicts: up with `tl_ta`, down with the rest. Values beyond the train range fall into the end bins.

| Feature | IV |
| --------| ------- |
| `mve_tl` | 1.327 |
| `tl_ta`  | 1.047 |
| `quick`  | 0.836 |
| `ni_ta`  | 0.769 |
| `re_ta`  | 0.539 |
| `log_ta` | 0.087 |

Because positive WOE always means safer, every coefficient should be negative. Five are. `log_ta` comes out at +0.003, effectively zero.

Against the benchmark (Z'' in brackets):

| Split | ROC AUC | PR-AUC lift | KS |
| --------| ------- | ------- | ------- |
| Train | 0.8503 (0.7634) | 6.28 (2.49)  | 0.576 (0.456) |
| Val   | 0.9093 (0.7878) | 9.20 (2.39)  | 0.684 (0.581) |
| Test  | 0.9062 (0.7731) | 10.65 (2.37) | 0.679 (0.526) |

**Why it matters.** The scorecard clears the benchmark by 0.13 AUC out of time, and its PR-AUC lift is more than four times the benchmark's. `mve_tl` carries the largest coefficient (−0.687), so the accounting-only ablation matters here.

Calibration drifts the same way as the benchmark. Mean predicted PD on test is 0.69% against 0.97% observed, about 29% too low.

See `notebooks/04_scorecard.ipynb` for more details.

## LightGBM

Gradient boosting on the same six ratios, fed in raw. Tree splits depend only on the order of values, so the extreme values that break a logistic regression need no WOE or capping. With 403 training defaults the risk is overfitting, so the trees are kept small: 7 leaves, at least 200 firm-years per leaf, a 0.02 learning rate and 80% of rows sampled per tree. There are no class weights, which would distort the PDs.

**Monotone constraints** fix the direction each ratio can move the PD while the others are held fixed, using the same directions as the scorecard binning, so no part of the model can say that more debt lowers risk. `log_ta` is left free: the scorecard gave it no weight, so it has no reliable direction.

**The number of trees** is chosen by cross-validation over whole years within train. Each fold fits on every earlier year and stops early on the next two, and no fold sees val or test.

| Fit | Check | Check defaults | Trees | ROC AUC |
| --------| ------- | ------- | ------- | ------- |
| 1999–2005 | 2006–2007 | 110 | 212 | 0.851 |
| 1999–2007 | 2008–2009 | 81  | 373 | 0.921 |
| 1999–2009 | 2010–2011 | 60  | 279 | 0.907 |

Mean CV AUC is 0.893 ± 0.037. That spread is why settings are compared across folds rather than on one val set of 87 defaults.

**Tuning.** Optuna searched for better settings, scored by the same mean CV AUC. Neither search beat the current settings by more than noise, so they are kept.

| Settings | Trials | Best CV AUC | Gain |
| --------| ------- | ------- | ------- |
| Current | – | 0.8931 | – |
| 3 tuned: tree size, leaf size, shrinkage | 30 | 0.8936 | < 0.001 |
| 11 tuned: adds row and feature sampling, smoothing, binning and more | 100 | 0.8960 | +0.003 |

The 30 trials of the first search already varied by about 0.003, so gains this small are noise. The best 11-setting trial also needed 1,256 trees against the current 279. The model's performance does not depend much on these settings.

The final model is refitted on all of train with the current settings, the median 279 trees and no early stopping, so val is an honest out-of-time check alongside test.

| Split | ROC AUC | PR-AUC lift | KS |
| --------| ------- | ------- | ------- |
| Train | 0.8850 | 13.55 | 0.623 |
| Val   | 0.9126 | 21.68 | 0.720 |
| Test  | 0.9248 | 18.36 | 0.733 |

All three models on the same test rows:

| | Z'' | Scorecard | LightGBM |
| --------| ------- | ------- | ------- |
| ROC AUC | 0.7731 | 0.9062 | 0.9248 |
| PR-AUC lift | 2.37 | 10.65 | 18.36 |
| KS | 0.526 | 0.679 | 0.733 |
| Mean PD (observed 0.97%) | 0.76% | 0.69% | 0.70% |

**Is the gap real?** With 119 test defaults, each model's AUC is uncertain on its own, so the models are compared with a paired bootstrap: both are scored on the same resampled firms and the difference is recorded.

| Comparison | Difference | 95% interval |
| --------| ------- | ------- |
| LightGBM − scorecard, ROC AUC | +0.019 | +0.008 to +0.031 |
| LightGBM − scorecard, PR-AUC  | +0.075 | +0.035 to +0.127 |
| Scorecard − Z'', ROC AUC      | +0.133 | +0.102 to +0.162 |
| Scorecard − Z'', PR-AUC       | +0.080 | +0.053 to +0.115 |

**Why it matters.** LightGBM's lead over the scorecard is small but real, and it is concentrated at the risky end of the book, where its PR-AUC lift is 18.4 against 10.6. The cost is interpretability: there is no points table, so explaining a decision needs per-firm contributions rather than a lookup.

Where the gain comes from:

| Feature | Share of gain |
| --------| ------- |
| `mve_tl` | 44% |
| `ni_ta`  | 18% |
| `log_ta` | 13% |
| `quick`  | 10% |
| `re_ta`  | 8%  |
| `tl_ta`  | 6%  |

`mve_tl` dominates here too, so the accounting-only ablation applies to both models. `log_ta`, worthless in the scorecard, earns 13%: left unconstrained, the trees use size in combination with the other ratios. `tl_ta` ranks last because it shares a correlation cluster with `mve_tl`.

PDs are too low out of time, as with the other models: 0.57% on val and 0.70% on test, against 0.83% and 0.97% observed.

See `notebooks/05_lightgbm.ipynb` for more details.
