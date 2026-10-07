# Classification report revision log

Run `it5006-proj/bin/python reports/phase2_report_evidence.py` from the repository root to regenerate the new CSV tables, 1,000-resample intervals, standalone PDF figures, and inline LaTeX figures. The original model grid, feature ladder, and fixed threshold come from `phase2_classification.py` and its saved outputs in `results/phase2/classification/`. The evidence script refits only the already selected logistic model for the previously reported held-out evaluation; it makes no new model or threshold choice.

## Numbers added or given new context

- Test PR-AUC 0.0651 now has a 95% order-bootstrap interval of 0.0564–0.0788; ROC-AUC 0.668 has 0.651–0.685. The test late-rate baseline 0.0348 and ratio 1.87 are shown together. The validation mean ROC-AUC 0.677 is explicitly distinguished from the pooled final test score.
- Each validation window and the test now show mean predicted risk alongside observed late rate, PR-AUC, PR-AUC/late-rate ratio, and ROC-AUC. Validation mean predicted risks are 3.2%, 3.2%, 2.7%, 4.5%, and 6.1%, all below their observed rates; the test mean is 9.82%, above its 3.48% observed rate. Window ROC-AUCs are 0.680, 0.653, 0.623, 0.719, and 0.711.
- Top 1%, 5%, 10%, and 20% validation precision/recall means and exploratory test precision/recall are added in a top-k table and a cumulative-gains figure. Test precision is 12.4%, 8.8%, 6.8%, and 5.3%, with bootstrap intervals 7.7–17.0%, 7.0–10.4%, 5.8–8.0%, and 4.6–6.0%. Corresponding recalls are 3.6%, 12.6%, 19.6%, and 30.4%. The 10% test list remains 1,937 orders and 132 late orders.
- The original 0.0788 cutoff and all its alert counts, precision, recall, F1, and MCC are unchanged. They moved to a secondary table after the capacity-based ranking view. The 9.5% versus 42.9% flagged shares are now discussed with a calibration figure and period means; no cause is asserted.
- Exploratory June, July, and August test PR-AUCs are 0.060, 0.063, and 0.074; ROC-AUCs are 0.761, 0.634, and 0.522; monthly PR-AUC/late-rate ratios are 5.14, 1.88, and 1.20, and top-10% precision is 3.8%, 7.8%, and 7.4%. Each has a 95% order-bootstrap interval in the report and exact values in `phase2_evidence/monthly_test.csv`.
- The 193 in-progress purchases without delivery outcomes are shown by purchase month: May 3, June 46, July 74, August 70. Their unknown final outcomes are not imputed or used to explain the test result.
- The break-even expression `review cost < 0.068 × intervention effectiveness × late-delivery cost` is illustrative. Intervention effectiveness and both costs remain unmeasured.

## Wording and interpretation changed

- Route history now says explicitly that an earlier delivery must be known before the current purchase; seller handover history uses the corresponding handover time. `phase2_features.py` now asserts the strict time ordering at the point where history records are selected.
- Both promised-date and payment availability at checkout are named as assumptions. The existing payment-removal result stays; no promise-removal result is invented.
- The modelling sequence is explicit: original-feature grid chose logistic `C=0.1`, the feature ladder used that setting, then all 13 candidates were rerun on final features.
- The 0.2003 versus 0.1999 logistic/forest difference, the logistic model winning three of five windows against the forest, the 0.2050 versus 0.2003 voting difference, and 0.001–0.005 feature steps are described as uncertain across five windows, rather than established gains. Class weighting is described as a change to fitting and alert decisions, not a cure for imbalance.
- Forest permutation importance is identified as forest-specific and based on shuffling raw columns, including all one-hot indicators of a category together. Logistic coefficients are described separately for the selected model.
- The ranking claim is separated from the alert claim. The pooled test ROC-AUC is no longer presented as evidence of stable monthly performance: August's exploratory ROC-AUC is close to 0.5. Public Olist results are qualified by prediction timing, late definitions, and split design, with our own shuffled 0.303 versus chronological 0.200 comparison.
- The business conclusion keeps the original caution against costly intervention or customer messaging, and makes any low-cost watchlist a pilot requiring an action, capacity, measured costs, and prospective validation.
