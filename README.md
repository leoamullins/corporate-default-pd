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

| Split   | Years |
| --------| ------- |
| Train  | 1999-2011    |
| Val    | 2012-2014     |
| Test   | 2015-2018    |

## Label Correction

**The shipped label does not match the documented one.** This is corrected before any modelling and it changes the problem substantially. The dataset documentation states that the fiscal year prior to a Chapter 7 or 11 filing is labelled as 1 and all other firm-years are labelled as 0. What ships is `status_label`, which is **firm level**: a company that failed in 2011 carries the "failed" mark on every row it has back to 1999.

 Check  | As Shipped |
| --------| ------- |
| Firms with >= 1 positive  | 609    |
| Positive Rows    | 5220    |
| Mean positives per failed firm   | 8.57    |

**Why it matters.** The as-shipped target is "does this firm eventually fail within the observation window," not a one-year PD. The label is not a function of that year's financials i.e. a firm's healthy 1999 accounts are labelled by a 2011 event. Any model trained on it learns persistent firm characteristics
that correlate with eventual failure, and reports an optimistic score for a question nobody asks.
The base rates gave it away. US public-company bankruptcy runs near 1% a year; the shipped labels give 7.9% in train, falling to 2.3% in test. That decline is the artefact, not an economic trend: failed firms' back-catalogues are concentrated in the early years and have delisted out of the panel by 2015. The correction. For each failed firm, label 1 on its final observed year only; 0 everywhere earlier. This restores the documented design rather than imposing a new one.

After the correction, the base rates became approx 1% in the train, val and test datasets matching the average US year-on-year default rate. 

See `notebooks/01_data_checks.ipynb` for more details.

## Split Integrity 
Firms appear in more than one split (3,290 across train-val etc.) this is expected behaviour and is not leakage. 

The time split partitions firm-years not firms individually, so any company which survives past the boundary will appear in both splits.

Predicting a firms outcome in 2016, from data in 1999-2011 train split is what the model does. Each default even appears exactly once - the target marks a single year where the firm stops filing afterward. Verified above.

The residual risk is hindsight inside a feature not firm identity. Checked via univariate AUC per feature. [Result - Tbd]