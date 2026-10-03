# Phase 2 modelling notebooks

The shared `data_prep.ipynb` notebook builds the Phase 2 order table, defines both evaluation splits, and records the rationale and limits. The versioned `data/phase2_feature_spec.json` is the shared candidate-feature and target contract. Classification and regression notebooks should load the saved order table and contract; rerun data preparation only when that contract changes. Fitted preprocessing belongs inside each model Pipeline and its training folds.

Use the primary chronological split and expanding validation indices in [data_prep.ipynb](data_prep.ipynb) for model selection and future-order claims. The random split is a separate secondary benchmark. The checkout-timing assumptions are documented; model-specific pipeline and test-use checks remain for later chunks.

## Review checkpoints

1. Shared feature engineering: complete; versioned contract and raw-data checks are committed.
2. [Classification](classification.ipynb): validation comparison, payment sensitivity, and alert threshold ready for review. See the [results interpretation](../../docs/phase2_classification_review.md).
3. Regression: after the classification review.
4. Voting ensembles: combine the selected linear and tree approaches after both task reviews.
5. Final evaluation and report: primary holdout once, separate random benchmark, and report evidence.

To reproduce classification from the repository root, run `python phase2_classification.py` in an environment with `requirements.txt` installed. The notebook reads the completed results and exports figures; set `RERUN_TRAINING=True` to repeat training there. It checks input and implementation hashes before displaying saved results. The helper at the repository root keeps fitting and metric code testable. Outputs are in `results/phase2/classification/`; `selection.json` records settings, versions, seed, and input hashes. Classification uses only primary training/validation rows. The later test and secondary random benchmark are reserved for the final evaluation chunk.

## Changes from the original data-preparation notebook

These are the three substantive changes to discuss with the teammate who prepared the first version:

1. **Lateness rule (`1fcd790`):** compare actual and estimated **calendar dates**. Delivery on the promised date is on time. The earlier full-timestamp comparison called 1,292 same-date deliveries late; the corrected delivered-order count is 6,534 late out of 96,470.
2. **Evaluation split (`d2f0e05`):** the primary experiment now tests later purchases from 2018-05-26 and uses expanding-time validation. Earlier purchases delivered after each training cutoff cannot provide labels at that cutoff. The original seed-42 stratified random 80/20 split and folds remain in `random_split` and `random_cv_fold` as a separate same-history benchmark.
3. **Notebook location and paths (`52026f2`):** moved `data_prep.ipynb` from `milestone2/` into `notebooks/phase2/`, removed the duplicate protocol document, and made its setup locate the root `data/` folder from either the repository root or this folder.

The [Phase 2 assumptions log](../../docs/phase2_assumptions.md) records the agreed checkout-time prediction point, the delivered-only scope of the current targets, and checks still needed before reporting model performance. The commit messages preserve the detailed history; this summary is the handoff for review.

The checkout-time audit now lists all 18 features and their source-timing assumptions in that log. The CSVs do not prove historical checkout availability; payment aggregation has particular uncertainty because payment rows have no timestamps, so compare models with and without those inputs. The primary validation folds now end 60 days before the test cutoff. The former latest fold excluded 1,802 orders delivered after the cutoff and understated its late rate (1.38% versus 5.75% for all delivered purchases in that window). Recent labelled training orders still enter the final fit, but not validation.

## Reconciliation with teammate's 3 October update

The update merged into `origin/main` from `data-preprocessing` (`43d01f1`) independently makes the same calendar-date lateness correction. We compared its saved table against this notebook's table: the same 96,470 order IDs, all 18 candidate features, and both targets agree. Its notebook still lived under `milestone2/`; the merged version remains here under `notebooks/phase2/`.

The evaluation assignments differ. The teammate's 80th-percentile timestamp cutoff splits the calendar day of 26 May 2018 and places earlier purchases in training even when their delivery happened after that cutoff. Its expanding folds likewise use older purchase blocks without checking whether those orders' outcomes were known when the validation block starts. This version keeps the whole-day cutoff, the `unavailable_at_cutoff` group, and outcome-aware fit indices. The teammate's useful reminder that the May–August test has no Black Friday season is now in the assumptions log. Do not combine the teammate's `time_block` or saved split labels with this version's `cv_fold` and `split` columns.
