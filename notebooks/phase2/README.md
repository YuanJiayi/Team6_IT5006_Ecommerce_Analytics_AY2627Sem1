# Phase 2 modelling notebooks

The shared `data_prep.ipynb` notebook builds the Phase 2 order table, defines both evaluation splits, and records the rationale and limits. Add classification and regression notebooks here as the team defines the Phase 2 problems. Build modelling tables from the raw Olist CSVs in `data/`, with one row per order and features available at the prediction point. Reusable preprocessing code can be added after the modelling design is agreed.

Use the primary chronological split and expanding validation indices in [data_prep.ipynb](data_prep.ipynb) for model selection and future-order claims. The random split is a separate secondary benchmark. Complete the notebook's remaining leakage audit before fitting models.

## Changes from the original data-preparation notebook

These are the three substantive changes to discuss with the teammate who prepared the first version:

1. **Lateness rule (`1fcd790`):** compare actual and estimated **calendar dates**. Delivery on the promised date is on time. The earlier full-timestamp comparison called 1,292 same-date deliveries late; the corrected delivered-order count is 6,534 late out of 96,470.
2. **Evaluation split (`d2f0e05`):** the primary experiment now tests later purchases from 2018-05-26 and uses expanding-time validation. Earlier purchases delivered after each training cutoff cannot provide labels at that cutoff. The original seed-42 stratified random 80/20 split and folds remain in `random_split` and `random_cv_fold` as a separate same-history benchmark.
3. **Notebook location and paths (`52026f2`):** moved `data_prep.ipynb` from `milestone2/` into `notebooks/phase2/`, removed the duplicate protocol document, and made its setup locate the root `data/` folder from either the repository root or this folder.

The [Phase 2 assumptions log](../../docs/phase2_assumptions.md) records the agreed checkout-time prediction point, the delivered-only scope of the current targets, and checks still needed before reporting model performance. The commit messages preserve the detailed history; this summary is the handoff for review.

The checkout-time audit now lists all 18 features and their source-timing assumptions in that log. The CSVs do not prove historical checkout availability; payment aggregation has particular uncertainty because payment rows have no timestamps, so compare models with and without those inputs. The primary validation folds now end 60 days before the test cutoff. The former latest fold excluded 1,802 orders delivered after the cutoff and understated its late rate (1.38% versus 5.75% for all delivered purchases in that window). Recent labelled training orders still enter the final fit, but not validation.
