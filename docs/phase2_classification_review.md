# Phase 2 classification validation review

The readable, executed [classification notebook](../notebooks/phase2/classification.ipynb) is the main review. It plots the saved evidence in [`results/phase2/classification/`](../results/phase2/classification/) without fitting models or consulting the reserved later-period test. The data contract now has 20 candidate inputs, including three features built only from earlier completed orders. `purchase_month` is no longer a model input.

## What the comparison shows

The prior-only baseline has mean per-window average precision (PR-AUC) of 0.1067. On the final feature set, regularised logistic regression (`C=0.1`) reaches 0.2003 and the best random forest reaches 0.2000 across the same five chronological validation windows. The gap is too small to establish a meaningful winner. Logistic regression is provisionally selected by the declared highest-mean rule; both family winners remain candidates for the voting comparison.

The [saved ladder](../results/phase2/classification/ladder_summary.csv) shows the contribution of each change. Class weighting lowered the original logistic model's mean PR-AUC relative to its unweighted default. Dropping month improved the forest in every window and the logistic model in three of five. Route promise and seller history improved the forest enough to close its early gap. A log transform of skewed numeric inputs further improved the linear model. Paired removal of the two payment inputs changes mean PR-AUC by less than 0.001 for each selected configuration; this does not prove the fields were available at checkout.

A shuffled five-fold check within the *primary training period* produces higher scores than the chronological windows. It is a diagnostic of how much easier mixed-period evaluation is, not an evaluation on the reserved later period. The selected model's validation ranking is useful for prioritising work: the top 10% of orders within each window contains 23.5% of late orders, 2.34 times random selection. Its fixed MCC alert threshold is less compelling and was chosen and described on the same pooled validation predictions, so those alert metrics are optimistic.

## Boundaries

These are validation results for orders that eventually have a recorded delivery. They do not estimate unconditional risk for every new purchase. Historical checkout availability of some source fields remains an assumption. The late rate varies strongly by period, and residual end-of-data censoring remains possible. There is no final fitted model or primary test-period score yet. See the [assumptions log](phase2_assumptions.md) for the feature-timing audit and the [feature contract](../data/phase2_feature_spec.json) for the exact inputs.
