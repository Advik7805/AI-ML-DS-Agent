# Churn Model Card — Northwind Connect

Owner: Data Science, Northwind Connect · Model: `churn-gbm` · Tenant: Northwind Global

## Task
Binary classification: will this customer churn within the next billing cycle?

## Data
3,333 telecom customers. Positive rate 14.5%. Features: usage minutes and call counts across
day/evening/night/international windows, plan flags, voicemail message count, account length
and customer-service call count.

## Preprocessing
The four `*_charge` columns are dropped: charge is `minutes * rate`, so they are deterministic
functions of the minute columns and would introduce perfect collinearity.

## Training
Gradient boosting (220 trees, depth 3, lr 0.06) against a logistic-regression baseline,
selected by **PR-AUC** — at a 14.5% positive rate ROC-AUC flatters almost any classifier.
SMOTE is applied to the training fold only, after the split.

## Folds
Four disjoint folds: fit / calibrate / threshold / test. Probabilities are calibrated with a
sigmoid map over a `FrozenEstimator`, so the shipped score is a monotone function of the exact
margin SHAP decomposes. The operating threshold is chosen by F1 on the reserved fold, not at
the default 0.50.

## Explanation
TreeSHAP, exact in raw log-odds. Attributions sum to the uncalibrated margin and do **not**
sum to the calibrated probability.

## Known limitations
Service calls and international plan dominate. The model has not been audited for differential
performance across `state` or `area_code`.
