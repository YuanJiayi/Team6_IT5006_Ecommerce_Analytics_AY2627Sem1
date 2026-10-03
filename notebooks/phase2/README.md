# Phase 2 modelling notebooks

Start with [data_prep.ipynb](data_prep.ipynb). It builds the one-row-per-delivered-order table from the raw Olist tables, explains the targets and the chronological validation design, and saves the shared [feature contract](../../data/phase2_feature_spec.json). The current contract has 20 candidate inputs. It excludes `purchase_month` as a model feature and adds historical route delivery time, promise slack and seller handover speed; all three use events completed before the current purchase. The prepared order table remains shared with the planned regression task.

[classification.ipynb](classification.ipynb) is the executed, plain-language review of the saved classification results. It explains the problem, validation choices, feature signal, model and feature ladder, interpretation, and risk ranking. The code that produced the saved model results is [`phase2_classification.py`](../../phase2_classification.py); point-in-time historical features are implemented in [`phase2_features.py`](../../phase2_features.py). The notebook **only displays** saved results and plots training-period distributions. It does not train a model or read test-period outcomes. Its figures are generated in the notebook, not taken from older PNG files in the results directory.

To rerun the display notebook from a fresh kernel, open it from the repository root or this directory and choose the `it5006-proj` environment. To rerun the model experiment separately from the repository root, use `it5006-proj/bin/python phase2_classification.py`. The notebook checks the input and implementation hashes in [`selection.json`](../../results/phase2/classification/selection.json) before showing the saved outputs. Current results and supporting tables are under [`results/phase2/classification/`](../../results/phase2/classification/). The [assumptions log](../../docs/phase2_assumptions.md) and [classification review](../../docs/phase2_classification_review.md) explain the methodological limits.

## Review sequence

1. Shared preparation and feature contract: complete.
2. Classification validation review: complete for discussion; the later-period test is reserved.
3. Delivery-time regression: next.
4. Voting comparison of the selected linear and tree-based models: after both task reviews.
5. Final evaluation and report: one later-period test after choices are fixed.

## Changes from the original data-preparation notebook

1. **Lateness rule (`1fcd790`):** compare actual and estimated calendar dates. Delivery on the promised date counts as on time.
2. **Evaluation split (`d2f0e05`):** test later purchases from 2018-05-26; use earlier purchases for a training snapshot only when their outcomes were known. The teammate's seeded random split is retained as a separate same-history benchmark.
3. **Location and paths (`52026f2`):** keep data preparation in `notebooks/phase2/` and locate the root `data/` directory from either supported launch location.
4. **Feature contract:** the later feature revision removed `purchase_month` as a model input and added point-in-time route and seller summaries. These changes were assessed on primary training and validation windows only. They did not change the delivered-order target or use the later-period test.

The teammate's later update made the same calendar-date lateness correction. Its saved order targets and original candidate features matched the earlier local table, but its split used a timestamp within the cutoff day and did not require labels to be available at each fit date. The current notebook retains the whole-day cutoff and outcome-aware training indices. See the [assumptions log](../../docs/phase2_assumptions.md) for remaining checkout-time uncertainties, especially payment rows.
