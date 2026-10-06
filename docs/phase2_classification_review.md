# Phase 2 classification review

The readable, executed [classification notebook](../notebooks/phase2/classification.ipynb) is the main review. It visibly fits the declared models and computes the validation evidence, then compares its live results with [`results/phase2/classification/`](../results/phase2/classification/). After model selection, its final section fits on eligible primary training orders and scores the later test period. The data contract has 20 candidate inputs, including three features built only from earlier completed orders. `purchase_month` is no longer a model input.

## What the comparison shows

The prior-only baseline has mean per-window average precision (PR-AUC) of 0.1067. On the final feature set, regularised logistic regression (`C=0.1`) reaches 0.2003 and the best random forest reaches 0.2000 across the same five chronological validation windows. Equal-weight voting reaches 0.2050, a small and uneven gain. The gap is too small to establish a meaningful winner. Logistic regression was selected before the held-out test because it is simpler and its mean validation score is effectively tied with the alternatives.

The [saved ladder](../results/phase2/classification/ladder_summary.csv) shows the contribution of each change. Class weighting lowered the original logistic model's mean PR-AUC relative to its unweighted default. Dropping month improved the forest in every window and the logistic model in three of five. Route promise and seller history improved the forest enough to close its early gap. A log transform of skewed numeric inputs further improved the linear model. Paired removal of the two payment inputs changes mean PR-AUC by less than 0.001 for each selected configuration; this does not prove the fields were available at checkout.

A shuffled five-fold check within the *primary training period* produces higher scores than the chronological windows. It is a diagnostic of how much easier mixed-period evaluation is, not an evaluation on the reserved later period. The selected model's validation ranking is useful for prioritising work: the top 10% of orders within each window contains 23.5% of late orders, 2.34 times random selection. Its fixed MCC alert threshold is less compelling and was chosen and described on the same pooled validation predictions, so those alert metrics are optimistic.

## Boundaries

The final test has 19,363 delivered orders, 674 late. Logistic regression reaches PR-AUC 0.0651 and ROC-AUC 0.668. At the validation-selected 0.0788 cutoff, it flags 8,316 orders: 434 true late and 7,882 false alerts, with 240 late orders missed. Precision is 5.2%, recall 64.4%, F1 0.097 and MCC 0.082. This supports modest risk ranking but not the chosen yes/no alert rule. The result is final evaluation evidence, not an invitation to retune on test labels.

These results apply to orders that eventually have a recorded delivery. They do not estimate unconditional risk for every new purchase. Historical checkout availability of some source fields remains an assumption. The late rate varies strongly by period, and residual end-of-data censoring remains possible. See the [assumptions log](phase2_assumptions.md) for the feature-timing audit and the [feature contract](../data/phase2_feature_spec.json) for the exact inputs.

## Reader revision completed

The notebook was reviewed cell by cell for a clear purpose, a computation or plot that can address it, and a nearby explanation of the actual output. The broad original-input grid now uses three focused examples. The month/payment snapshot chart was removed because it could not establish repeatable seasonality. One price example and a skewness table now explain the log transform; the validation ladder still decides whether it helps. The heatmap now connects overlapping inputs to later coefficient interpretation. The history-feature box plots describe their observed associations; their extra single-feature ROC-AUC chart was removed after reader review because it added an abstract second measure without a needed decision. The feature ladder uses named stages, and model-interpretation and alert figures are separated by short explanations. An unused single-feature validation diagnostic and repeated captions were removed.

The modelling calculations, saved-result checks and final held-out test remain in visible cells. The notebook was executed from a fresh kernel; the live validation comparisons matched the saved reference, and the later-period test was then scored once with the frozen model and cutoff.

## Carry into the Phase 2 report

Explain that `promise_slack = promised_days - route_typical_days` wherever all three are present. Logistic regression can use these overlapping inputs with regularisation, but individual coefficients and odds ratios are **not isolated effects**. If the report needs an isolated coefficient claim, first compare non-redundant input pairs using training and validation only; do not use the held-out test to choose among them. Do not discard an input automatically if it helps the forest.
