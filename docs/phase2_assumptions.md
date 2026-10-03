# Phase 2 modelling assumptions and preprocessing decisions

Use this as the source for the assumptions and preprocessing discussion in the Phase 2 report. It records the current decisions; an open item is not a verified property of the Olist data. The implementation is in [`notebooks/phase2/data_prep.ipynb`](../notebooks/phase2/data_prep.ipynb).

## Business question and prediction point

| Topic | Current decision | Report implication or check |
|---|---|---|
| Prediction time | Immediately after checkout and submission of payment details, before payment approval or dispatch. | Use only fields plausibly available then. Exclude approval, carrier, delivery, status and review fields. The source-timing audit is below. |
| Late-delivery target | A delivered order is late only when its actual delivery **calendar date** is after the estimated delivery date; delivery on that date is on time. | This target is defined **among orders eventually delivered**. It must not be reported as the unconditional probability of lateness for every new purchase. |
| Cancelled and unavailable orders | Do not label these as late: cancellation and stock unavailability have different causes and no delivery date. | They are outside the current lateness and delivery-time targets. If the business later needs a risk score for every purchase, define and evaluate a separate non-delivery outcome or broader promise-failure target. |
| Open or unfinished orders | Do not label an unfinished order late solely because no delivery timestamp is recorded. | Its outcome may be unknown at extraction; assess censoring before changing the target population. |
| Delivery-time regression | Days from purchase to actual delivery, including fractional days, for delivered orders. | This is conditional on delivery. No finite delivery-time label exists for an order that never arrives. |

## Features and preprocessing

| Topic | Current decision | Report implication or check |
|---|---|---|
| Location data | Customer and seller tables provide ZIP-code **prefixes** and states, not order-specific latitude/longitude. The geolocation table provides many coordinate observations per prefix. | The notebook uses the median latitude and longitude for each prefix as an approximate location, then calculates Haversine distance between the customer and seller approximations. This is not an exact door-to-door distance. |
| Geolocation reference timing | ZIP centroids currently use the complete geolocation file, which has no observation timestamps. | Treat this as a static reference-map assumption, not as proven historically available at each cutoff. It uses no delivery labels, but its as-of availability cannot be checked from these files. |
| Multiple items and sellers | Aggregate to one row per order. Use the furthest seller's estimated distance, `same_state` only when every seller is in the customer's state, and the largest item's product category by volume. | These are simplifying assumptions about which parcel drives the order-level outcome; check their impact for multi-seller orders. |
| Missing and extreme values | Keep orders with missing predictors; mark unknown categories explicitly and leave numeric gaps for fold-fitted imputation. Remove invalid geolocation coordinates and exact duplicate points; keep plausible long-delivery and high-value orders. | Document the 42 removed geolocation points, ZIP-prefix median aggregation, and model-specific imputation. Outlier thresholds in the notebook are diagnostic, not exclusion rules. |
| Purchase-time feature availability | Item, seller, product, promise and submitted payment details are candidate inputs at checkout. | Their checkout availability is a modelling assumption. The audit below explains the reasoning and uncertainty for each group. |

### Checkout-time feature audit (20 model inputs)

We assume the intended prediction moment is immediately after checkout and payment submission. The raw CSVs do not provide historical snapshots or recording timestamps for the estimated delivery promise, items, payments, product catalog, seller profile, or geolocation reference. A third-party copy of the dataset description is useful context but does not establish when these values were available for each order. The table states assumptions and checks; it is not proof that every feature was captured at checkout.

| Inputs | Source and checkout-time assessment |
|---|---|
| `purchase_hour`, `purchase_dayofweek` | Derived only from `order_purchase_timestamp`; available when the purchase event is recorded. (`purchase_month` is no longer an input; see the feature changes below.) |
| `route_typical_days`, `promise_slack` | The typical purchase-to-delivery days on the order's seller-state to customer-state route, averaged over earlier orders **delivered before this purchase**, shrunk toward the overall average known at that moment (20 pseudo-orders). `promise_slack` is `promised_days` minus that value. Uses no information from the order itself or from later orders; relies on the same assumption as `promised_days` that the stored estimate is the checkout promise. Route uses the furthest seller's state, the same simplification as `seller_state`. |
| `seller_ship_days` | The furthest seller's average purchase-to-carrier-handover days, from earlier orders **handed to the carrier before this purchase** (handovers within 0 to 60 days only), shrunk toward the overall average known at that moment (10 pseudo-orders). The seller's past behaviour is public history at checkout; the carrier timestamp is used only for earlier orders. For multi-seller orders only the furthest seller is used. |
| `promised_days` | Purchase timestamp and `order_estimated_delivery_date`. We assume the stored estimate is the promise shown at checkout; the CSV cannot verify that it was not later revised. |
| `customer_state` | Customer table; plausibly the checkout delivery address, but the file has no address-change history. |
| `n_items`, `n_products`, `n_sellers`, `total_price`, `total_freight` | Order-item rows; the purchased basket and quoted amounts are plausibly known when payment details are submitted. Item rows have no creation timestamp, so this remains an assumption. |
| `seller_state`, `same_state`, `distance_km` | Seller identity comes from order items; seller location and customer ZIP prefix come from profile tables. ZIP-prefix centroids are a static reference-map assumption, and the distance is approximate. Seller/profile history is unavailable. |
| `product_category`, `total_weight_g`, `total_volume_cm3` | Product catalog joined to purchased items. These are plausible listing attributes at checkout, but the file cannot show whether catalog values changed later. The largest-volume category is an order-level simplification. |
| `payment_type`, `max_installments` | We assume payment method and installments have been submitted by the prediction point. The notebook uses the lowest `payment_sequential` row and maximum installments across **all** payment rows. There are 2,875 delivered orders with multiple payment rows (about 3.0%); absent row timestamps, later payment edits cannot be ruled out. This aggregation has greater timing uncertainty; run a sensitivity model without these two inputs. |

No model input uses approval, carrier, actual-delivery, order-status or review fields. The delivered-status filter is a **target-population decision**, not a checkout-time input: the current models estimate outcomes conditional on eventual delivery. A model fitted on this table alone cannot be claimed to estimate the unconditional late-delivery risk of every new order.

### Feature changes after validation (schema version 2)

These changes were decided on primary **training and validation rows only**; no test-period score was produced. The list of ideas was written down before testing, each idea was tried once against the same five time-based validation windows, and the results are in `results/phase2/classification/ladder_summary.csv`.

| Change | Reason | Evidence to cite |
|---|---|---|
| Removed `purchase_month` (kept in the table as a reference column) | With orders from late 2016 to mid-2018, each calendar month has at most one earlier year of examples. The models learn that a specific past month was bad and apply it to the next period. | Ladder step 9: mean validation scores improved for both models when it was dropped (better in 5 of 5 windows for the forest, 3 of 5 for the logistic model). |
| Added `route_typical_days` and `promise_slack` | Lateness compares delivery with the promise, so how generous the promise is for that route matters; a model cannot easily work this out itself. | Ladder step 10. |
| Added `seller_ship_days` | A diagnostic that deliberately used post-purchase information showed the time until the seller hands the parcel to the carrier is the strongest late-delivery signal. A seller's past handover speed is the closest checkout-time proxy. | Ladder step 11. |
| `log1p` of skewed inputs for the linear model only | Heavy right skew in price, freight, distance, weight and volume. Adopted for the linear model because it did not lower mean validation PR-AUC (decision recorded in `selection.json`). | Ladder step 12 (better in 4 of 5 windows). |
| Tried and **not** adopted | A peak-season flag (10 Nov to 20 Dec) as a cross-year replacement for month made validation scores worse. Recent order volume (last 7 days), the seller's past late rate and the platform's recent late rate gave no consistent gain. | Quick checks on training and validation rows; the peak-season flag result is in the review notes, the others were not added to the saved ladder. |

All three history features are computed in `data_prep.ipynb` through `phase2_features.py`, which only uses records dated strictly before each purchase. `tests/test_phase2_features.py` checks this on synthetic data, and `tests/test_phase2_feature_contract.py` recomputes both `route_typical_days` and `seller_ship_days` by brute force for random samples of real orders (the seller check uses the stored `furthest_seller_id` reference column and the raw carrier timestamps).

Because hyperparameters were first tuned on the original feature set and then held fixed while features changed, the ladder isolates the effect of each feature change. All 13 declared candidates were then re-run on the final feature set, and the selection comes from that final run.

### Chunk 1 feature contract

The versioned [`phase2_feature_spec.json`](../data/phase2_feature_spec.json) freezes the **candidate** feature lists, target rules, prediction point, missing-value policies, and split definitions shared by both modelling tasks. The saved table contains one row per eligible delivered order; identifiers, timestamps, targets, and split/fold columns are metadata, never predictors. The 20 candidate inputs are the starting set, not a claim that every feature will improve validation performance. Any later addition or removal must be justified using training-period evidence and recorded before final test evaluation.

Seven numeric inputs can be missing: `distance_km`, `total_weight_g`, `total_volume_cm3`, `max_installments`, and the three history features (for the few hundred earliest orders, before enough history exists). Fit median imputation separately within each training fold, inside the model Pipeline; use the fitted values on that fold's validation rows. Categories already use the literal `unknown` without learned statistics. Fit encoders, scalers, feature selection, and any probability calibration within training folds; choose class weights and alert thresholds without later-period test outcomes. The two payment inputs are the specified sensitivity comparison, not an automatic exclusion.

## Evaluation and use of exploratory analysis

| Topic | Current decision | Report implication or check |
|---|---|---|
| Primary test | Fixed purchase-date cutoff of 2018-05-26; use earlier purchases for training only when their delivery label existed before that date. Five expanding-time validation windows end at least 60 days before the test cutoff. More recent labelled purchases remain available for the final fit but are not CV validation rows. | Orders spanning each training cutoff are unavailable to that snapshot. The 60-day validation maturity gap sharply reduces selection of only quick deliveries; 15 delivered purchases before the gap still had outcomes after the test cutoff, so residual censoring remains. The later test period is for final evaluation, not tuning. |
| Secondary test | A seeded, stratified random 80/20 split is prepared as a separate same-history benchmark. | Decide whether to run and report this optional comparison before final evaluation; it cannot establish future-period performance. |
| Phase 1 EDA | The [project brief](../ref/IT5006%20Project%20Description%20-%20AY%202026_27%20Semester%201.pdf) explicitly asks Phase 1 to explore relationships and candidate problems to inform Phase 2. The full-dataset EDA is therefore part of the intended workflow and legitimately motivates the business question and initial features. | Retain and cite those exploratory findings. For Phase 2 model comparison and tuning, use the training period and its validation folds; do not repeatedly use later-test scores to revise features or hyperparameters. Describe the final holdout as an internal retrospective test, not wholly independent external validation. |
| End of delivered-order coverage | The latest purchase with a usable delivered outcome is 2018-08-29. The raw file has 25 later purchases: 24 cancelled and one shipped. | August is where the current delivered-only test naturally ends. The CSVs do not establish the extraction date or why the later tail is sparse; check possible end-of-data censoring before interpreting later-period performance. |
| Seasonal coverage of the primary test | The later-purchase test runs from late May through August 2018 and contains no November Black Friday period. Its late rate is 3.48%, versus 7.10% in primary training. | Report the period shift and avoid claiming the test measures peak-season performance. The lower test late rate alone does not establish that any model score will be lower or higher. |

### Time order and class imbalance

Late delivery is uncommon, but our main question is performance on later purchases. We therefore prioritise a chronological split. A stratified **random** split can preserve the late-order percentage, but it would mix later orders into training; when the rate changes over time, forcing the primary periods to have equal percentages would hide that change. This is a deliberate trade-off, consistent with scikit-learn's guidance to use time-aware evaluation for time-dependent data and its description of stratification as a safeguard against folds missing a class ([cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)).

For the primary experiment, check the late-order count in every training and validation fold, retain each period's natural class distribution, compare unweighted and class-weighted models using training/validation data only, and report PR-AUC together with precision, recall, MCC and the late-order rate. Choose any alert threshold on validation data, not the later test. The five current validation folds contain 310, 1,051, 363, 754 and 1,576 late orders respectively; the modelling code also checks that both classes appear in every fit and validation fold. These safeguards address imbalance without claiming that the two periods have the same late-order rate. The prepared stratified random split remains available as a separate comparison; its inclusion in the final report is still to be decided.

## Open before reporting model performance

1. In model notebooks, put fitted preprocessing, feature selection, calibration and threshold choice inside the appropriate training folds. The current data-preparation notebook does not fit these components.
2. Keep the agreed Phase 2 business claim explicitly conditional on delivery. A separate non-delivery outcome would be needed for an all-purchase operational score.
3. Check how excluding undelivered and recently purchased orders changes the apparent late rate and delivery-time distribution.
4. Compare models with and without `payment_type` and `max_installments` because payment-row timing cannot be established from the CSVs. The classification comparison is recorded in the [Chunk 2 review](phase2_classification_review.md).

## Classification implementation decisions

The linear model applies `log1p` to the skewed inputs listed in the feature contract (if adopted, see `selection.json`), then scales them; trees use the raw values. Use one-hot encoding for both linear and tree models, fitted within each training fold, because the nominal categories have no natural ordering. Scale numeric inputs and use a sine/cosine hour pair for logistic regression; trees retain raw numeric inputs and hour. The prior and unweighted models retain the natural class distribution; compare specified weighted variants without resampling validation rows.

Select candidates by mean per-fold average precision (`average_precision_score`), reported as PR-AUC (average precision). Show per-fold prevalence because the metric depends on the class mix. Choose a provisional alert threshold by maximising MCC over pooled validation predictions, with exact ties favouring fewer alerts. The same validation predictions select and describe this threshold, so its reported metrics are optimistic selection evidence; final performance requires the reserved holdout. No operational cost or alert-capacity constraint has been supplied.
