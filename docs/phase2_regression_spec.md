# Phase 2 regression specification

**Written 8 October 2026, before any regression model was fitted or scored.** The classification test had already been scored, and its exploratory diagnostics showed that median purchase-to-delivery days fell in June–August 2018. No regression test score had been computed. Changes to this specification after validation results are seen must be recorded below with a reason.

## Question and target

How many days will a delivered order take from purchase to customer delivery, predicted just after checkout? The target is `delivery_days` in `data/phase2_order_table.csv` (fractional days, delivered orders only). It is conditional on eventual delivery, like the classification target.

The regression has two uses: a delivery-time estimate for fulfilment planning, and an input to classification version 2 (below).

## Data, features and splits

Unchanged from [`phase2_feature_spec.json`](../data/phase2_feature_spec.json) and the classification work: the same 20 checkout inputs, the same eligible training orders (75,099), the same five chronological validation windows with outcome-aware fitting sets, and the same later test (19,363 orders from 26 May 2018). Preprocessing is fitted inside each training fold: median imputation, one-hot categories, and for linear models only, scaling, cyclic hour and `log1p` of the skewed inputs listed in `linear_log_candidates`. The target is not transformed.

## Candidates (fixed before fitting)

| Name | Model | Settings |
|---|---|---|
| `mean_baseline` | Training-fold mean | none |
| `route_baseline` | Predict `route_typical_days` (fold median where missing) | no fitting beyond the median |
| `linear_baseline` | LinearRegression | none |
| `ridge_1`, `ridge_10`, `ridge_100` | Ridge | alpha 1, 10, 100 |
| `tree_baseline` | DecisionTreeRegressor | unrestricted |
| `tree_depth6` | DecisionTreeRegressor | max depth 6, min leaf 50 |
| `tree_depth12` | DecisionTreeRegressor | max depth 12, min leaf 20 |
| `forest_baseline` | RandomForestRegressor, 200 trees | unrestricted, min leaf 1 |
| `forest_depth12` | RandomForestRegressor, 200 trees | max depth 12, min leaf 20 |
| `forest_leaf20` | RandomForestRegressor, 200 trees | unrestricted depth, min leaf 20 |

Seed 42 wherever randomness is involved; `max_features="sqrt"` for forests, matching classification.

## Selection and reporting

- **Selection metric:** mean validation MAE across the five windows. RMSE and R² are reported alongside. The lowest mean MAE wins; if two candidates differ by less than 0.05 days, choose the simpler one (linear before tree before forest, fewer parameters first).
- Report per-window MAE for the selected model and the best of each family, and how many windows each beats the other in.
- Interpretation: linear coefficients for a linear winner; permutation importance on validation windows for a tree winner.
- **Test:** after selection is recorded, refit the selected model on all eligible training orders and score the test once. Report MAE, RMSE, R² overall and by purchase month, plus the mean signed error (bias) by month.
- **Monthly refit (pre-specified deployment cadence):** also score the test with the selected model refitted at 26 May, 1 July and 1 August, each time on all orders purchased and delivered before that date. Same model and settings; no reselection.

## Classification version 2 (designed after the version 1 test)

This design was motivated by the version 1 test result, so its test score is informative but not an independent holdout. Version 1 remains the primary classification result.

- **Input:** `gap = predicted_delivery_days − promised_days`, from the selected regression model.
- **Model:** logistic regression on `gap` alone (`C=1`, no class weighting). Its training rows are out-of-sample regression predictions: within each validation window, regression and then logistic are fitted on that window's fitting set using an inner time split (the earlier 80% of fitting orders, by purchase time, train the regression; the later 20%, whose outcomes are known, train the logistic step).
- **Validation comparison** with version 1 uses the same five windows: mean PR-AUC, mean ROC-AUC and top-10% precision. Version 2 is reported whatever the result.
- **Test:** single fit and monthly refit, as above. Capacity-based reporting at 1, 5, 10 and 20%. No probability cutoff is selected.

## Change log

None yet.
