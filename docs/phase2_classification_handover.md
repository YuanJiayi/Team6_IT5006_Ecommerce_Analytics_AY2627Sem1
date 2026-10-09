# Phase 2 classification handover

**Historical handover.** The dated next-step lists below record the state on 8 October. The [current joint report](../reports/phase2_joint_report.tex) and [evidence index](phase2_evidence_index.md) supersede them. Check Git status for live checkout state.

## Latest handover: 8 October 2026, after the recent-history audit (resume here)

The user paused report work to diagnose why checkout delivery-time regression overpredicted the later period. The frozen classification and regression models, their feature contract, and their original test scores were not changed. Current checkout: `josh`, four local commits ahead of the last fetched `origin/josh`, with only the new audit script, its outputs/notes, this handover edit, and its focused test uncommitted. Check `git status` again before continuing; no push was requested.

**Diagnosis.** The saved regression predicts 3.659 days too long on average on 19,363 later orders (MAE 5.001). On rows with valid stage timestamps, mean carrier-to-customer time fell from 9.947 days in training to 6.109 days in the test, while seller-to-carrier time fell from 2.955 to 2.368 days. Reweighting older seller-state/customer-state routes to the later route mix still gives about 12.64 days versus 8.60 actual days on those routes: most of the change is within routes. The existing `route_typical_days` is an all-earlier-history purchase-to-delivery mean, so it can lag this shift. Fitting ridge on only the last 90 or 180 days *without rebuilding that history feature* did not improve later-period MAE. These observations locate the measured change; the CSVs do not establish its operational cause. Phase 1's `notebooks/phase1/operational_capacity.ipynb` had already plotted weekly actual versus promised delivery time, but had not tested a recent-history forecast.

**New, reproducible exploratory results.** Start with [the recent-history audit](../experiments/phase2_recent_history_audit/README.md), its [script](../experiments/phase2_recent_history_audit.py), `checkout.csv`, and `handover.csv`. It compares only histories known strictly before each purchase or handover, uses the same five chronological validation windows, and leaves the frozen model intact. Adding a 90-day seller-state-to-customer-state **carrier-to-customer** history to checkout ridge lowered mean validation MAE from 5.810 to 5.679 days (better in all five windows); the already-open test changed from 5.001 to 4.490. A 30-day total purchase-to-delivery route history also helped (validation 5.763, opened test 4.501), so the experiment does not prove that separating the carrier leg is the only useful approach. Global recent delivery history alone worsened validation; recent sample weighting helped validation but barely changed the opened test.

At carrier handover, a simple update—**observed purchase-to-handover time plus the last 90 days' route-specific carrier-to-customer average**—lowered mean validation MAE from 5.809 to 5.304 days on the same eligible cohorts (better in all five windows). On the same 19,230 later orders it lowered MAE from 4.991 to 3.762. This is a later prediction, not a better checkout prediction. A 30-day handover average scored 2.916 on the opened test but was worse on validation (5.503), so do not choose it from that test result. The staged *classification* audit is separate evidence and does not prove regression performance.

**Resume decision.** Treat every new test comparison above as post hoc because these ideas followed an opened test. Review the per-window and monthly CSVs, then decide whether to formalize the 90-day carrier-history input and the simple handover update as exploratory candidates. Keep the frozen published model/result identifiable and seek a new future period for independent confirmation if available. The 90th-percentile promise model and joint report work below are deferred while this delivery-time diagnosis is the focus. The audit's two point-in-time tests pass with `it5006-proj/bin/python tests/test_phase2_recent_history_audit.py -q`; the script also checks that its all-history route calculation reproduces the saved feature.

## Earlier progress at 8 October 2026 (report due Sunday 11 October 23:59)

**Done**
- Report review: the numbers in the draft match the evidence. Main gaps: no story about the promise shift; pooled test ROC-AUC 0.668 hides within-month ROC-AUC ≈ 0.58; no success criteria, citations, feature-rationale table, train/validation/test tables, AI declaration or regression section; page budget probably tight.
- Promise shift: from about 23 May 2018 (truckers' strike) promises lengthened, then fell to a median of 13.4 days by August, below anything in training. This likely explains the cutoff failure, the over-prediction and the August ROC-AUC of 0.52. Tables: `experiments/phase2_promise_regime_audit/`. Post hoc regime features lose on validation (PR-AUC 0.1960 vs 0.2003) but help on the test (ROC-AUC 0.705; 0.723 with monthly refit). Report this as hindsight only.
- Regression ([spec](phase2_regression_spec.md), `phase2_regression.py`, `results/phase2/regression/`):
  - Selected ridge alpha 100 (validation MAE 5.81 vs forest 5.76, inside the 0.05-day margin).
  - Test scored once: MAE 5.00 (route-history baseline 5.21, mean baseline 6.27); monthly refit 4.90.
  - Bias +3.6 days because test deliveries were faster (8.8 vs 13.3-day mean), so test R² is −0.18.
- Regression diagnostics (`experiments/phase2_regression_diagnostics/`, training and validation only):
  - OLS fit: R² 0.264, adjusted R² 0.263, F 180.6.
  - Assumptions: heteroscedastic, heavy right tail, mild autocorrelation; VIF 16.7, 12.2, 5.3 for promise, slack and route.
  - A log target is worse (MAE 6.06).
  - Conclusion: these affect inference, not MAE-based selection.
- Public Olist R² ≈ 0.48 (shef4793 repo) comes from target outlier capping before the split, `review_score` as a feature, and a random split. Not comparable.

**Earlier proposed next work (deferred during the recent-history diagnosis)**
1. Spec, then agent build:
   - 90% quantile regression for promise dates, linear plus a tree-based model. Exclude promised days and slack. Evaluate on coverage, pinball loss, and comparison with Olist's promise (median 23.5 days, 92.9% on time).
   - A WLS validation check.
   - Clip predictions at 0.
   - Drop `promise_slack` for the coefficient table.
   - This replaces classification v2 in the spec; record the change in its change log.
2. Training-set scores for train/validation/test tables, regression error analysis by route, distance and state, and final-fit odds ratios and coefficients.
3. Report rewrite (classification plus regression in 6–8 pages). Literature citations from the Phase 1 report in `ref/`. Mechanical formatting is to go to Codex (CLI not yet installed). No LaTeX toolchain is installed locally.

**Historical status: 7 October 2026.** The snapshot below was a starting map for another agent. Its uncommitted-file description predates the latest handover above; check current Git status before making changes.

## Read in this order

1. [Project brief](../ref/IT5006%20Project%20Description%20-%20AY%202026_27%20Semester%201.pdf), [submitted Phase 1 report](../ref/Team6_Phase1_IT5006_AY2627Sem1-2.pdf), and [Phase 1 feedback](../ref/Phase%201%20Report%20feedback.txt) for the assignment and its context. The [Phase 1 validation audit](phase1_validation.md) records corrections made after submission.
2. [Phase 2 assumptions](phase2_assumptions.md) for the business question, prediction timing, target, feature availability, and split decisions. The [data preparation notebook](../notebooks/phase2/data_prep.ipynb) builds the prepared data; the exact modelling inputs and split settings are in [phase2_feature_spec.json](../data/phase2_feature_spec.json).
3. [Classification notebook](../notebooks/phase2/classification.ipynb) for the reader-facing, executable modelling story. [phase2_classification.py](../phase2_classification.py) is its independent validation reference; [phase2_features.py](../phase2_features.py) builds point-in-time features. The original validation files are in [results/phase2/classification/](../results/phase2/classification/).
4. [Classification review](phase2_classification_review.md), then the [current joint report](../reports/phase2_joint_report.tex) and [report changelog](../reports/phase2_report_changelog.md). The report evidence generator is [phase2_report_evidence.py](../reports/phase2_report_evidence.py).

## What is fixed, and what has already been evaluated

- The business question is late delivery at checkout, conditional on eventual delivery. An order delivered on its promised **calendar date** is on time. Approval, handover, delivery, review, and status after checkout are unavailable to the checkout model. Availability of the stored promise and payment fields at checkout is an assumption.
- The primary test boundary is **26 May 2018**. The five earlier validation windows use expanding, outcome-aware training sets. An earlier order may enter delivery history only when its outcome was known before the purchase being predicted. Missing-value fitting, encoding, and scaling stay within each training fold.
- The selected checkout model is logistic regression with `C=0.1`. Model selection used mean per-window average precision (called PR-AUC in the project). The validation-selected alert cutoff is **0.07878485347563163**. The selection preceded the held-out test. Do not retune either choice using test labels.
- The later test **has been opened and scored**. It contains 19,363 delivered orders, 674 calendar-date late. The selected model scored ROC-AUC **0.668** and average precision **0.0651** against a late rate of **0.0348**. Its fixed cutoff flags 42.9% of orders, with 5.2% precision and 64.4% recall. The exploratory top-10% list has 6.8% precision and about 1.96× lift. These results support modest ranking, not dependable individual alerts. See the report and generated evidence for intervals and monthly results.
- The `results/phase2/classification/selection.json` file is a **validation-only selection artifact** and still says the test is pending. Use the classification notebook's final section and `reports/phase2_evidence/` for the later test. Do not mistake that older status field for the current project status.

## Post hoc diagnostics: keep separate from model selection

These scripts were run **after** the test opened. They are reproducible diagnostics, not a fresh holdout or grounds to replace the fixed checkout model. Each uses seed 42 and saves plain tables.

| Question | Script and main output | Finding to check |
| --- | --- | --- |
| What becomes knowable at approval or carrier handover? | [phase2_staged_audit.py](../experiments/phase2_staged_audit.py), [staged tables](../experiments/phase2_staged_audit/TABLES.md) | Handover has stronger ranking on its eligible cohort, but is a later prediction time; cohorts and interventions differ. |
| Does the public project's timestamp label explain its higher ROC-AUC? | [phase2_label_definition_audit.py](../experiments/phase2_label_definition_audit.py), [frozen scores](../experiments/phase2_label_definition_audit/frozen_scores.csv) | Timestamp comparison flips 347 same-date deliveries, reproducing 1,021 test late orders; the frozen checkout ROC-AUC falls from 0.668 to 0.660. It explains the late-count gap, not the ROC gap. |
| Do point-in-time marketplace-climate inputs help? | [phase2_marketplace_climate_audit.py](../experiments/phase2_marketplace_climate_audit.py), [scores](../experiments/phase2_marketplace_climate_audit/scores.csv), [monthly and within-month diagnostics](../experiments/phase2_marketplace_climate_audit/test_months.csv) | Adding the bundle lowers mean validation scores for logistic and shallow XGBoost. Logistic test pooled ROC-AUC rises slightly, almost entirely from cross-month comparisons; within-month ranking barely moves. |
| What if the 2,008 boundary-spanning orders were added to training? | [eligibility sensitivity](../experiments/phase2_marketplace_climate_audit/eligibility_sensitivity.csv) | Test ROC-AUC rises from 0.668 to 0.676, but their outcomes were unknown at the boundary. This is an information sensitivity, not a valid real-time fit. |

The external project's [technical documentation](https://github.com/jedu28/olist-atraso/blob/main/docs/en/technical.md) and [feature code](https://github.com/jedu28/olist-atraso/blob/main/src/features.py) are useful comparisons. Its timestamp label, feature set, model, and test cohort differ from ours. [External comparison notes](phase2_external_olist_comparison.md) predate some of the later diagnostics above; verify claims against the scripts before reusing them.

## Reproduce or continue

From the repository root, use the local environment:

```sh
it5006-proj/bin/python reports/phase2_report_evidence.py
it5006-proj/bin/python experiments/phase2_staged_audit.py
it5006-proj/bin/python experiments/phase2_label_definition_audit.py
it5006-proj/bin/python experiments/phase2_marketplace_climate_audit.py
```

The scripts can refit models and regenerate output files; inspect the diff before retaining regenerated artifacts. The last two were run successfully for this handover. The climate script reproduced the saved frozen checkout test probabilities exactly (`max absolute difference = 0`). No report or frozen-model edit was made for the two most recent diagnostic requests.

Before editing, run `git status --short --branch`. At the **earlier** handover, the checkout was `josh` with local uncommitted edits to `phase2_features.py` and the LaTeX report, plus untracked report evidence and experiment scripts/outputs; the latest status is described at the top of this file. Preserve any current uncommitted work. Do not treat the public Olist repository's test metrics as directly comparable, and do not turn post hoc findings into prespecified results. The user asked for those earlier audits as **scripts and plain tables** and kept the frozen model, cutoff, boundary, and report unchanged.
