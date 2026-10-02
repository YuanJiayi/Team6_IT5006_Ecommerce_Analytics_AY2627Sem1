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
| Purchase-time feature availability | Item, seller, product, promise and submitted payment details are candidate inputs at checkout. | The source-timing audit below identifies which fields have direct timing evidence and which still rely on a checkout-snapshot assumption. |

### Checkout-time feature audit (18 model inputs)

The [Olist data dictionary reproduced here](https://github.com/aduverger/olist/blob/master/data/README.md) says that the estimated delivery date was given to the customer at purchase and describes payment type and installments as the customer's chosen method and count. The raw CSVs have no item, payment, product-catalog, seller-profile, or geolocation observation timestamps, so their historical values cannot be proved from these files. This is a source-timing audit, not a claim that every input was recorded in an immutable checkout snapshot.

| Inputs | Source and checkout-time assessment |
|---|---|
| `purchase_hour`, `purchase_dayofweek`, `purchase_month` | Derived only from `order_purchase_timestamp`; directly available at checkout. |
| `promised_days` | Purchase timestamp and `order_estimated_delivery_date`; the data dictionary explicitly says the estimate was shown at purchase. |
| `customer_state` | Customer table; plausibly the checkout delivery address, but the file has no address-change history. |
| `n_items`, `n_products`, `n_sellers`, `total_price`, `total_freight` | Order-item rows; the purchased basket and quoted amounts are plausibly known when payment details are submitted. Item rows have no creation timestamp, so this remains an assumption. |
| `seller_state`, `same_state`, `distance_km` | Seller identity comes from order items; seller location and customer ZIP prefix come from profile tables. ZIP-prefix centroids are a static reference-map assumption, and the distance is approximate. Seller/profile history is unavailable. |
| `product_category`, `total_weight_g`, `total_volume_cm3` | Product catalog joined to purchased items. These are plausible listing attributes at checkout, but the file cannot show whether catalog values changed later. The largest-volume category is an order-level simplification. |
| `payment_type`, `max_installments` | Payment method and installments are chosen at checkout, consistent with the agreed prediction point. However, the notebook uses the lowest `payment_sequential` row and maximum installments across **all** payment rows. There are 2,875 delivered orders with multiple payment rows (about 3.0%); absent row timestamps, later payment edits cannot be ruled out. Treat these two inputs as provisional until a checkout payment snapshot or timestamp evidence is available; run a sensitivity model without them. |

No model input uses approval, carrier, actual-delivery, order-status or review fields. The delivered-status filter is a **target-population decision**, not a checkout-time input: the current models estimate outcomes conditional on eventual delivery. A model fitted on this table alone cannot be claimed to estimate the unconditional late-delivery risk of every new order.

## Evaluation and use of exploratory analysis

| Topic | Current decision | Report implication or check |
|---|---|---|
| Primary test | Fixed purchase-date cutoff of 2018-05-26; use earlier purchases for training only when their delivery label existed before that date. Five expanding-time validation windows end at least 60 days before the test cutoff. More recent labelled purchases remain available for the final fit but are not CV validation rows. | Orders spanning each training cutoff are unavailable to that snapshot. The 60-day validation maturity gap sharply reduces selection of only quick deliveries; 15 delivered purchases before the gap still had outcomes after the test cutoff, so residual censoring remains. The later test period is for final evaluation, not tuning. |
| Secondary test | Keep the seeded, stratified random 80/20 split as a separate same-history benchmark. | Report it separately; its score does not establish future-period performance. |
| Phase 1 EDA | The [project brief](../ref/IT5006%20Project%20Description%20-%20AY%202026_27%20Semester%201.pdf) explicitly asks Phase 1 to explore relationships and candidate problems to inform Phase 2. The full-dataset EDA is therefore part of the intended workflow and legitimately motivates the business question and initial features. | Retain and cite those exploratory findings. For Phase 2 model comparison and tuning, use the training period and its validation folds; do not repeatedly use later-test scores to revise features or hyperparameters. Describe the final holdout as an internal retrospective test, not wholly independent external validation. |
| End of delivered-order coverage | The latest purchase with a usable delivered outcome is 2018-08-29. The raw file has 25 later purchases: 24 cancelled and one shipped. | August is where the current delivered-only test naturally ends. The CSVs do not establish the extraction date or why the later tail is sparse; check possible end-of-data censoring before interpreting later-period performance. |

## Open before reporting model performance

1. In model notebooks, put fitted preprocessing, feature selection, calibration and threshold choice inside the appropriate training folds. The current data-preparation notebook does not fit these components.
2. Decide whether the Phase 2 business claim remains explicitly conditional on delivery, or whether a separate non-delivery outcome is needed for an all-purchase operational score.
3. Check how excluding undelivered and recently purchased orders changes the apparent late rate and delivery-time distribution.
4. Compare models with and without `payment_type` and `max_installments` because payment-row timing cannot be established from the CSVs.
