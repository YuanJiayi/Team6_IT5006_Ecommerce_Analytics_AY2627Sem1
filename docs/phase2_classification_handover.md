# Phase 2 classification handover

**Status: 7 October 2026.** This is a starting map for another agent. Check Git status and the current files before making changes; several recent analyses and the report draft are local, uncommitted work.

## Read in this order

1. [Project brief](../ref/IT5006%20Project%20Description%20-%20AY%202026_27%20Semester%201.pdf), [submitted Phase 1 report](../ref/Team6_Phase1_IT5006_AY2627Sem1-2.pdf), and [Phase 1 feedback](../ref/Phase%201%20Report%20feedback.txt) for the assignment and its context. The [Phase 1 validation audit](phase1_validation.md) records corrections made after submission.
2. [Phase 2 assumptions](phase2_assumptions.md) for the business question, prediction timing, target, feature availability, and split decisions. The [data preparation notebook](../notebooks/phase2/data_prep.ipynb) builds the prepared data; the exact modelling inputs and split settings are in [phase2_feature_spec.json](../data/phase2_feature_spec.json).
3. [Classification notebook](../notebooks/phase2/classification.ipynb) for the reader-facing, executable modelling story. [phase2_classification.py](../phase2_classification.py) is its independent validation reference; [phase2_features.py](../phase2_features.py) builds point-in-time features. The original validation files are in [results/phase2/classification/](../results/phase2/classification/).
4. [Classification review](phase2_classification_review.md), then the [current classification report draft](../reports/phase2_classification_draft.tex) and [report changelog](../reports/phase2_report_changelog.md). The report evidence generator is [phase2_report_evidence.py](../reports/phase2_report_evidence.py). Regression is being handled by teammates and is not completed in this classification draft.

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

Before editing, run `git status --short --branch`. At this handover the checkout is `josh` with local uncommitted edits to `phase2_features.py` and the LaTeX report, plus untracked report evidence and experiment scripts/outputs. Preserve these; they are not disposable scratch files. Do not treat the public Olist repository's test metrics as directly comparable, and do not turn post hoc findings into prespecified results. The user asked for the recent audits as **scripts and plain tables** and explicitly kept the frozen model, cutoff, boundary, and report unchanged.
