# corporate-default-pd: BofA-ready fixes

**Milestone:** BofA-ready, due **4 October 2026** (application deadline 11 October)
**Repo:** `leoamullins/corporate-default-pd`

| Evening | Tasks |
|---|---|
| 1 | M1, M6 |
| 2 | M5, M4 |
| 3 | M2, M3 |
| 4 | S1, S2 |
| 5 | S3, S4, P1 to P4, then the final rerun |

Rules:
- One fix per commit, referencing the issue (`... (fixes #N)`).
- For anything that changes a number, record the old and new values in the issue before closing it, then update the README table and any text that depends on it.

---

## Must fix

### M1: The data link doesn't match the code
- [ ] Work out which file you actually use (Kaggle copy or another release): record its URL, file name and row count.
- [ ] Replace the README data link with that exact source.
- [ ] Add a note that the GitHub release (`american_bankruptcy_dataset.csv`, with `fyear`, `division` and `majorgroup` columns) uses a **different column mapping**, and that your identity checks fail on it.
- [ ] Add a **Reproduce** section: download link → where to put the file → `uv sync` → notebook order 01 to 05.

**Done when:** a fresh clone plus those instructions runs notebook 01 without edits.

### M2: Accounting-only ablation
- [ ] Refit the scorecard and LightGBM without `mve_tl`, using the same splits and settings.
- [ ] Add to the README: test ROC AUC, PR-AUC lift and KS, with and without `mve_tl`, for both models.
- [ ] Add one sentence on how much of the performance is the market's forecast.

**Done when:** both README mentions of the ablation point to a real table.

### M3: Calibration
- [ ] Add `reliability(y_true, y_prob, n_bins=10)` to `metrics.py` (quantile bins, predicted vs observed rate per bin).
- [ ] Plot reliability curves for all three models on test, in one figure.
- [ ] Add a calibration-in-the-large table: mean PD vs observed, per split, per model.
- [ ] Add one line on the fix you would apply (e.g. intercept recalibration to a long-run default rate).

**Done when:** the README's calibration claims are backed by a figure and a table.

### M4: Firm-level cluster bootstrap
- [ ] Add a shared helper that returns resampled row indices by drawing **firms** with replacement.
- [ ] Use it in both `bootstrap_metric` and `paired_bootstrap` (add a `groups` argument).
- [ ] Keep the check that skips samples with no defaults.
- [ ] Rerun the paired comparisons, and record the row-level vs firm-level intervals side by side.
- [ ] If the LightGBM − scorecard interval now crosses 0, soften "small but real".

**Done when:** the code matches the word "firms" in the README.

### M5: KS with tied scores
- [ ] In `ks_statistic`, group by unique score (summing goods and bads) before the cumulative sums.
- [ ] Rerun KS for all models, and record old vs new (the scorecard is the one most likely to change).
- [ ] Update every KS table in the README.

**Done when:** the order of rows within a tie can't change the result.

### M6: Denominators and infinities
- [ ] Check `x14` (current liabilities) and `x17` (total liabilities): zero or negative counts in your file.
- [ ] Decide: drop the bad rows, or set the affected ratios to NaN and handle them. Document the choice.
- [ ] Fix or remove the README line "no ratio produces an infinity".
- [ ] Check whether any NaN could reach `WOEBinner.transform` (a NaN bin index will break it).

**Done when:** the README states the denominator checks for every ratio that is used.

---

## Should fix

### S1: Make the scorecard a real scorecard
- [ ] Scale scores to points using base points and PDO (e.g. 600 points at 50:1 odds, 20 PDO).
- [ ] Output a points table: feature → bin → WOE → points.
- [ ] Show the bin tables (count, defaults, rate, WOE) for each feature.
- [ ] Add markdown to notebook 04 (it currently has none).

### S2: Refit on train + val
- [ ] Refit the scorecard and LightGBM on 1999 to 2014, and score test.
- [ ] Add as an extra row next to the train-only results.
- [ ] Note whether the PD underprediction shrinks.

### S3: Document where numbers come from
- [ ] State that CV AUC is for comparing settings (the check folds are used for early stopping and tuning), not an unbiased estimate.
- [ ] Record how the "current" LightGBM settings were chosen.
- [ ] State that Z'' components are winsorised at train's 1st and 99th percentiles, and that the zones use the capped scores.

### S4: Small code fixes
- [ ] Set the logistic regression penalty explicitly (scorecard and `ZScoreToPD`).
- [ ] Brier skill: state the reference rate, or use the train base rate.
- [ ] Lower `requires-python` unless 3.14 is needed.

---

## Polish

### P1: Top of the README
- [ ] 3-line summary: problem, data, result.
- [ ] Three-model test table (Z'', scorecard, LightGBM).
- [ ] One figure (the reliability curves from M3 or PR curves).
- [ ] Capture table: share of the 119 test defaults caught in the riskiest 5% / 10% / 20% of firm-years, for each model.
- [ ] Fill in the GitHub description and topics.

### P2: Label Correction section
- [ ] Add a "Corrected" column (609 / 609 / 1.00).
- [ ] Add a before/after base-rate row (7.9% → 0.72%, etc.).
- [ ] Reformat "The correction. For each failed firm…".

### P3: Remove or move
- [ ] Delete "Discrimination rises out of time, which is further evidence the splits are sound."
- [ ] Move the scorecard and LightGBM Brier numbers out of the benchmark section.
- [ ] Replace "Verified above" with a notebook link.

### P4: Typos
- [ ] "Each default even appears" → **event**
- [ ] "a firms outcome" → **firm's**
- [ ] Comma splice: "…(3,290 across train-val etc.) this is expected"
- [ ] Docstrings: "compnenets", "mathcing", "rations"

### P5: Interview prep (one line each in the README)
- [ ] Firms that leave the panel without failing (acquisitions) are labelled 0: label noise.
- [ ] Firms still alive in 2018 are right-censored.
- [ ] How the PD underprediction would be fixed.

---

## Final check
- [ ] Clear all notebook outputs, then re-execute 01 to 05 in order with `nbclient`.
- [ ] Every README number matches the notebook outputs.
- [ ] Fresh clone + Reproduce section works end to end.
- [ ] All `must` issues closed.
- [ ] Update the CV bullets with any numbers that changed.