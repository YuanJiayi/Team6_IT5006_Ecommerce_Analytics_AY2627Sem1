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

### Checkout-time feature audit (18 model inputs)

We assume the intended prediction moment is immediately after checkout and payment submission. The raw CSVs do not provide historical snapshots or recording timestamps for the estimated delivery promise, items, payments, product catalog, seller profile, or geolocation reference. A third-party copy of the dataset description is useful context but does not establish when these values were available for each order. The table states assumptions and checks; it is not proof that every feature was captured at checkout.

| Inputs | Source and checkout-time assessment |
|---|---|
| `purchase_hour`, `purchase_dayofweek`, `purchase_month` | Derived only from `order_purchase_timestamp`; available when the purchase event is recorded. |
| `promised_days` | Purchase timestamp and `order_estimated_delivery_date`. We assume the stored estimate is the promise shown at checkout; the CSV cannot verify that it was not later revised. |
| `customer_state` | Customer table; plausibly the checkout delivery address, but the file has no address-change history. |
| `n_items`, `n_products`, `n_sellers`, `total_price`, `total_freight` | Order-item rows; the purchased basket and quoted amounts are plausibly known when payment details are submitted. Item rows have no creation timestamp, so this remains an assumption. |
| `seller_state`, `same_state`, `distance_km` | Seller identity comes from order items; seller location and customer ZIP prefix come from profile tables. ZIP-prefix centroids are a static reference-map assumption, and the distance is approximate. Seller/profile history is unavailable. |
| `product_category`, `total_weight_g`, `total_volume_cm3` | Product catalog joined to purchased items. These are plausible listing attributes at checkout, but the file cannot show whether catalog values changed later. The largest-volume category is an order-level simplification. |
| `payment_type`, `max_installments` | We assume payment method and installments have been submitted by the prediction point. The notebook uses the lowest `payment_sequential` row and maximum installments across **all** payment rows. There are 2,875 delivered orders with multiple payment rows (about 3.0%); absent row timestamps, later payment edits cannot be ruled out. This aggregation has greater timing uncertainty; run a sensitivity model without these two inputs. |

No model input uses approval, carrier, actual-delivery, order-status or review fields. The delivered-status filter is a **target-population decision**, not a checkout-time input: the current models estimate outcomes conditional on eventual delivery. A model fitted on this table alone cannot be claimed to estimate the unconditional late-delivery risk of every new order.

### Chunk 1 feature contract

The versioned [`phase2_feature_spec.json`](../data/phase2_feature_spec.json) freezes the **candidate** feature lists, target rules, prediction point, missing-value policies, and split definitions shared by both modelling tasks. The saved table contains one row per eligible delivered order; identifiers, timestamps, targets, and split/fold columns are metadata, never predictors. The 18 candidate inputs are the starting set, not a claim that every feature will improve validation performance. Any later addition or removal must be justified using training-period evidence and recorded before final test evaluation.

Four numeric inputs can be missing: `distance_km`, `total_weight_g`, `total_volume_cm3`, and `max_installments`. Fit median imputation separately within each training fold, inside the model Pipeline; use the fitted values on that fold's validation rows. Categories already use the literal `unknown` without learned statistics. Fit encoders, scalers, feature selection, and any probability calibration within training folds; choose class weights and alert thresholds without later-period test outcomes. The two payment inputs are the specified sensitivity comparison, not an automatic exclusion.

## Evaluation and use of exploratory analysis

| Topic | Current decision | Report implication or check |
|---|---|---|
| Primary test | Fixed purchase-date cutoff of 2018-05-26; use earlier purchases for training only when their delivery label existed before that date. Five expanding-time validation windows end at least 60 days before the test cutoff. More recent labelled purchases remain available for the final fit but are not CV validation rows. | Orders spanning each training cutoff are unavailable to that snapshot. The 60-day validation maturity gap sharply reduces selection of only quick deliveries; 15 delivered purchases before the gap still had outcomes after the test cutoff, so residual censoring remains. The later test period is for final evaluation, not tuning. |
| Secondary test | Keep the seeded, stratified random 80/20 split as a separate same-history benchmark. | Report it separately; its score does not establish future-period performance. |
| Phase 1 EDA | The [project brief](../ref/IT5006%20Project%20Description%20-%20AY%202026_27%20Semester%201.pdf) explicitly asks Phase 1 to explore relationships and candidate problems to inform Phase 2. The full-dataset EDA is therefore part of the intended workflow and legitimately motivates the business question and initial features. | Retain and cite those exploratory findings. For Phase 2 model comparison and tuning, use the training period and its validation folds; do not repeatedly use later-test scores to revise features or hyperparameters. Describe the final holdout as an internal retrospective test, not wholly independent external validation. |
| End of delivered-order coverage | The latest purchase with a usable delivered outcome is 2018-08-29. The raw file has 25 later purchases: 24 cancelled and one shipped. | August is where the current delivered-only test naturally ends. The CSVs do not establish the extraction date or why the later tail is sparse; check possible end-of-data censoring before interpreting later-period performance. |
| Seasonal coverage of the primary test | The later-purchase test runs from late May through August 2018 and contains no November Black Friday period. Its late rate is 3.48%, versus 7.10% in primary training. | Report the period shift and avoid claiming the test measures peak-season performance. The lower test late rate alone does not establish that any model score will be lower or higher. |

## Open before reporting model performance

1. In model notebooks, put fitted preprocessing, feature selection, calibration and threshold choice inside the appropriate training folds. The current data-preparation notebook does not fit these components.
2. Keep the agreed Phase 2 business claim explicitly conditional on delivery. A separate non-delivery outcome would be needed for an all-purchase operational score.
3. Check how excluding undelivered and recently purchased orders changes the apparent late rate and delivery-time distribution.
4. Compare models with and without `payment_type` and `max_installments` because payment-row timing cannot be established from the CSVs. The classification comparison is recorded in the [Chunk 2 review](phase2_classification_review.md).

## Classification implementation decisions

Use one-hot encoding for both linear and tree models, fitted within each training fold, because the nominal categories have no natural ordering. Scale numeric inputs and use a sine/cosine hour pair for logistic regression; trees retain raw numeric inputs and hour. The prior and unweighted models retain the natural class distribution; compare specified weighted variants without resampling validation rows.

Select candidates by mean per-fold average precision (`average_precision_score`), reported as PR-AUC (average precision). Show per-fold prevalence because the metric depends on the class mix. Choose a provisional alert threshold by maximising MCC over pooled validation predictions, with exact ties favouring fewer alerts. The same validation predictions select and describe this threshold, so its reported metrics are optimistic selection evidence; final performance requires the reserved holdout. No operational cost or alert-capacity constraint has been supplied.
