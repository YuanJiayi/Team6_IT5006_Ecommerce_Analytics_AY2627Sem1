# Phase 2 modelling assumptions and preprocessing decisions

Use this as the source for the assumptions and preprocessing discussion in the Phase 2 report. It records the current decisions; an open item is not a verified property of the Olist data. The implementation is in [`notebooks/phase2/data_prep.ipynb`](../notebooks/phase2/data_prep.ipynb).

## Business question and prediction point

| Topic | Current decision | Report implication or check |
|---|---|---|
| Prediction time | Immediately after checkout and submission of payment details, before payment approval or dispatch. | Use only fields plausibly available then. Exclude approval, carrier, delivery, status and review fields. Audit source timing for every retained feature before modelling. |
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
| Purchase-time feature availability | Item, seller, product, promise and submitted payment details are candidate inputs at checkout. | **Open audit:** verify each source field's recording time, especially payment aggregates and item records. A column-name screen alone does not prove availability. |

## Evaluation and use of exploratory analysis

| Topic | Current decision | Report implication or check |
|---|---|---|
| Primary test | Fixed purchase-date cutoff of 2018-05-26; use earlier purchases for training only when their delivery label existed before that date. Five expanding-time validation windows are inside the training period. | Orders spanning each cutoff are unavailable to that training snapshot. The later test period is for final evaluation, not tuning. |
| Secondary test | Keep the seeded, stratified random 80/20 split as a separate same-history benchmark. | Report it separately; its score does not establish future-period performance. |
| Phase 1 EDA | Phase 1 already examined delivery outcomes across the full dataset and informed some candidate features. | This retrospective exposure limits any claim that the later test was completely untouched. From now on, use training-period data for target-based EDA and feature/model selection; reserve the later test for final evaluation and disclose the Phase 1 context. |
| End of delivered-order coverage | The latest purchase with a usable delivered outcome is 2018-08-29. The raw file has 25 later purchases: 24 cancelled and one shipped. | August is where the current delivered-only test naturally ends. The CSVs do not establish the extraction date or why the later tail is sparse; check possible end-of-data censoring before interpreting later-period performance. |

## Open before reporting model performance

1. Complete the as-of-checkout feature audit and verify that fitted preprocessing and feature selection stay within each training fold.
2. Decide whether the Phase 2 business claim remains explicitly conditional on delivery, or whether a separate non-delivery outcome is needed for an all-purchase operational score.
3. Check how excluding undelivered and recently purchased orders changes the apparent late rate and delivery-time distribution.
