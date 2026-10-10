# Phase 2 handoff (10 October 2026)

## RESUME HERE (10 October 2026, evening)

**Read this section first.** Everything under "History" below describes the *previous* selection (tree_depth6 / forest_leaf20) and its numbers are stale. Keep it only as history.

**Deadline:** Phase 2, 11 October 2026, 23:59. Brief: `ref/IT5006 Project Description - AY 2026_27 Semester 1.pdf` (Phase 2 pp. 9–12, marking p. 16).

### State
- The tuned two-stage selection has run end to end. Smoke runs and full runs finished cleanly; 68 tests pass. New outputs are in `results/phase2/handover/` and `results/phase2/promise/` (new files: `adaptive_matched.csv`, `adaptive_validation_grid.csv`, `forecast_metrics_by_window.csv`).
- On an 8-core Mac the classifier took 23 min and the promise engine about 62 min; they can run in parallel.
- **Not done:** `reports/phase2_final_evidence.py` is not updated for the new outputs, so `reports/generated/numbers.tex` and every number in `reports/phase2_report.tex` are still from the previous selection. The report text has not been rewritten.

### Design (agreed with the user; do not change without asking)
- Families: Linear and Tree-based (single tree, random forest, gradient boosting), the same for both models.
- Stage 1: within each group (linear, tree, forest, boost) keep the configuration with the best mean over the five temporal validation windows.
- Stage 2: line-up = plain linear, tuned linear, plain tree, tuned tree, tuned forest, tuned boosting; paired one-standard-error rule (simplest entry within 1 SE of the best). Simplicity order: linear < tree < forest/boosting.
- Promise engine score: mean promise at exactly 95% mean validation on-time, each configuration tuned over step size γ (0 = fixed level) × on-time target. Classifier score: mean validation PR-AUC.
- Fixed: features, target, cohort, splits, 95% target, 5:1 benefit ratio.

### Results
**Late warning (classifier): tuned logistic regression (C = 0.01).**
- Validation PR-AUC: forest_l50_sqrt 0.369 (best), logistic_c0.01 0.367 (gap 0.002, SE 0.0095, so about 0.2 SE: chosen as simplest), boosting 0.354, tuned tree 0.308, plain logistic 0.343, plain tree 0.137. Checkout variant: logistic, PR-AUC 0.21 (vs 0.37 at handover, both validation).
- Test (late rate 3.5%, action share 14%): PR-AUC 0.23, ROC-AUC 0.82, precision 11.8%, recall 47%, net benefit negative. Random PR-AUC = 0.035.
- Validation → test PR-AUC drop (0.37 → 0.23) is mainly the late rate: per-window PR-AUC tracks lateness (W1 4.1% late → 0.33; W3 4.8% → 0.26; W5 20.7% → 0.47), ROC-AUC rose 0.77 → 0.82, and lift over random went 3.4× → 6.6×.
- Top permutation importance is now `handover_days` (0.130), then `remaining_slack` (0.085).
- Slack rule on test: PR-AUC 0.52 (beats the model). On validation the model beats it by +0.039 (CI 0.029–0.048).
- For reference only: the previous forest scored test PR-AUC 0.43. Other configurations were **not** scored on test.

**Promise engine: ridge (α = 1000) with a fixed coverage level (γ = 0).**
- γ = 0 was best for **all 31 configurations**: at matched 95% validation on-time, adaptive buffers give longer promises (ridge: γ 0 → 30.9 days, 0.002 → 32.5, 0.01 → 37.3, 0.05 → 35.7).
- Validation at 95%: ridge_a1000 31.18 days (best), forest +0.22 (SE 0.15), plain linear +0.27 (SE 0.21), boosting +0.28 (SE 0.36, within 1 SE but more complex), tuned tree +1.06, plain tree 46.9. Route average + same buffer 31.79 (reference, not a candidate). Selected level ≈ 0.971.
- Test: engine 24.5 days, 98.5% on time; Olist 22.05 days, 96.5%; route + buffer 27.4 days, 99.0%. Monthly mean promise at the same level: May 32, Jun 27, Jul 25, Aug 21 days. Olist fell to 93.8% on time in August (15.8-day promises); engine 98.2% (20.6 days).
- Matched to Olist's test on-time (post hoc, uses test outcomes to set the level): engine 18.9 days, 3.1 shorter than Olist; route average 18.8. So on test the gain comes from the calibrated buffer, not the ML forecast.

### Why the earlier "adaptive buffer" story fell apart
- The old headline (17.3 vs 22.1 days on test) came from the calm test period, where an adaptive buffer trims promises. On validation the old adaptive engine averaged 40 days vs Olist's 25 (it overreacted after Black Friday). The new matched comparison on validation exposes this.
- Lag has two sources in `phase2_promise.py`: (1) calibration rows are purchased 45–90 days before the day and already delivered (a newer window would be biased toward fast deliveries); (2) the adaptive feedback for a promise arrives only on its due date, about 3–4 weeks after purchase.
- "Fixed" is only the coverage level. The buffer in days is recalculated daily from the rolling calibration window, so it still tracks conditions, just slowly.
- Adaptive conformal inference (Gibbs & Candès 2021) is a research method; there is no evidence here that it is an industry standard. Quantile-of-forecast-error promising (Salari et al. 2022, JD.com) is published retail practice, and that is what the fixed-level buffer does. The Phase 1 literature review does not mention ACI or the one-SE rule.

### Decisions agreed with the user today
1. Keep the one-SE rule and the chosen models; do not change the selection rule after seeing test (that would be test-set selection).
2. Defend the classifier's validation → test drop with the late-rate argument and ROC-AUC.
3. The remaining-slack rule is not a baseline. Keep it as a short limitation (the user leaned this way; confirm).
4. Cite the one-SE rule as ISLR (James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, Sec. 6.1.3, p. 214, verified from the PDF). The current ESL Sec. 7.10 citation is **unverified**. Say "standard in statistical learning" (glmnet's default `lambda.1se`, verified), not "industry standard". State two caveats: we apply it across model types (our simplicity order is a judgment), and the SE comes from 5 non-independent windows. Add: "the gap is about a fifth of its standard error, so the choice is not sensitive to the tie-breaking rule."

### OPEN DECISION (the user is asking a teammate to review this)
How to present the promise engine. Options:
- **A (recommended).** Fixed coverage level with a daily rolling-recalibrated buffer is the chosen method; adaptive is a tested alternative that lost on validation because of feedback lag. Deployment: daily forecast + buffer; monthly review of on-time against 95% and of the level; analyst raises the level ahead of known peaks (data lags about 6 weeks); late-warning list at handover as the second layer. Pitch: more reliable than Olist on test (98.5% vs 96.5%, and 98.2% vs 93.8% in August) at 2.5 days longer; the level is a reliability-vs-length dial; tie to reviews (62% of late orders get 1–2 stars vs 9%). Lag reduction goes under future work.
- **B.** Try to reduce the lag tonight (newer calibration window with a censoring correction; the late warning as a leading signal; multi-γ ACI variants). Costs: redesign plus a ~1 h rerun, and it would be designed after seeing test results.

### Report: every section needs checking, not just the model sections
- Executive summary: rewrite (old models, old 17.3 vs 22.05 headline).
- Business problem: add the review-led framing; honest about repeat purchase (2.5% vs 3.0%) and that conversion cannot be measured (cite Salari et al.). **Fix an existing error:** it compares checkout validation PR-AUC (`\CheckoutPR`) with handover *test* PR-AUC (`\ClsTestPR`). Compare validation with validation (0.21 vs 0.37).
- Data section: mostly unchanged; check the Figure 1 caption (buffer line).
- Model 1 and Model 2: full rewrite; tuning grids, baseline-vs-tuned per group, the one-SE line-up, appendix table of every configuration. The old line "the model is essentially reading how much of the promise has been used" no longer matches (top input is `handover_days`).
- Limitations: update the buffer and peak claims; add the one-SE caveats; the classifier is trained against Olist's promise, not the engine's (linking them is a Phase 3 simulation).
- References: add ISLR; drop unused ones.
- AI declaration: the placeholder must be completed by the team.
- Length: 6–8 pages main body; the full configuration table goes in the appendix.

### Unchanged data facts (verified against `reports/generated/numbers.tex` and `results/phase2/final/supporting_facts.csv`)
96,470 delivered orders; handover cohort 96,272 (198 excluded); test 19,363 orders; late share about 4% normal, 14% Black Friday, 21% Feb–Mar 2018, 3.5% test; bad review 62% late vs 9% on time (95,824 rated orders); carrier leg 85% of delivery-time variance; repeat purchase 2.5% after late vs 3.0% after on time (3.0% of 93,350 customers ever reorder).

### Next steps
1. Settle the open decision with the user.
2. Update `reports/phase2_final_evidence.py` for the new outputs (`adaptive_selection.json`: `tuned`, `lineup`, `stage2`, `matched`; promise MAE/RMSE/R² from `forecast_metrics_by_window.csv`; classifier `selection.json`: `tuned`, `stage2`), rerun it.
3. Rewrite the report as above, then reread it end to end against the regenerated numbers.
4. Commit and push; the report is compiled online (no local LaTeX).

**User preferences:** settle the design before running; everything defensible and aligned with the brief; no report disclosure about when selection rules changed (user's request).

## History (previous selection; numbers below are stale)

Status of branch `phase2-final`, for the next agent. The design is in `docs/phase2_final_spec.md`. Course requirements are in `ref/IT5006 Project Description - AY 2026_27 Semester 1.pdf`: Phase 2 on pp. 9–12, marking on p. 16.

## Story

Olist makes two delivery decisions per order. One problem, two model types (brief Example 4):

1. **Checkout: the promise engine (regression).**
   - `tree_depth6` (depth-6 decision tree) forecasts delivery days. Chosen with the adaptive buffer and the one-standard-error rule; see the selection-rule change below.
   - Inputs: the spec features minus `promised_days` and `promise_slack`, plus `route_transit_90d`.
   - A self-correcting buffer is added to the forecast to give the promised date. The buffer level adapts daily to whether promises that just fell due were kept.
   - Code: `phase2_promise.py`, results in `results/phase2/promise/`.
2. **Carrier handover: the late warning (classification).**
   - `forest_leaf20` scores each order's risk of missing Olist's promise.
   - Operations works down the ranked daily list: expedite with the carrier, or message the customer.
   - Code: `phase2_handover_classifier.py`, results in `results/phase2/handover/`.

**Theme:** Olist's delivery conditions swing.

| Period | Orders late |
|---|---|
| Normal | ~4% |
| Black Friday 2017 | ~14% |
| Feb–Mar 2018 | ~21% |
| Test (Jun–Aug 2018) | 3.5% |

Both models beat their baselines. The hard part of deployment is keeping the buffer right through those swings.

## Baselines used in the report

| Model | Status quo | Simple heuristic | Simplest in family |
|---|---|---|---|
| Regression | Olist's promise | — | LinearRegression / DecisionTree variants |
| Classifier | No prioritisation (random list of the same size) | — | Logistic / decision tree variants |

The remaining-slack rule is **not** a report baseline. Prepare for the question "could a hand-built slack formula do as well?":

- Mostly, yes. Slack is the model's top input.
- On validation the model beat the rule: PR-AUC +0.038, 95% CI [0.031, 0.045].
- On test the rule won: 0.52 vs 0.43.

## Key numbers (all from the results folders)

**Promise engine, test**

| | Mean promise | On time |
|---|---|---|
| Tree + adaptive buffer (γ = 0.05) | 17.28 days | 95.41% |
| Olist | 22.05 days | 96.52% |
| Fixed buffer level (L = 0.975) | 26.8 days | 98.9% |

**Promise engine, validation and buffer behaviour**

- On validation the engine promised longer than Olist in all five windows (mean 40.1 vs 25.0 days; on time 95.9% vs 89.3%).
- Faster γ (0.1, 0.2) oscillates and loses coverage.
- Even at γ = 0.05, the adaptive buffer overreacted after Black Friday: about 57-day promises in Jan–Mar 2018. Treat this as a limitation that needs monitoring.

**Classifier**

| | PR-AUC | Late orders caught in top 10% |
|---|---|---|
| Validation | 0.366 | 34% |
| Validation, random | 0.107 | 10% |
| Test | 0.43 | 56% |

- At a 12% action share, net benefit is negative on test because only 3.5% of orders were late. Recommend sizing the list to current lateness.
- The checkout-stage variant has PR-AUC 0.20, so handover information doubles it.

## Report wording rule

Present the fixed and adaptive buffers as a method comparison: "a fixed level over-covers in calm periods; we recommend the adaptive level". Do **not** claim the regression's test set was scored only once, or that all settings were fixed before testing.

## Next steps

1. Optional appendix: rank-quality of the classifier against the engine's promise instead of Olist's, labelled as a simulation.
2. Write `reports/phase2_final_evidence.py`, producing:
   - `results/phase2/final/*.csv`;
   - `reports/generated/numbers.tex`, `\newcommand` macros for **every** number in the report;
   - TikZ/pgfplots figures, so the report compiles in the online editor.

   The key figure is the monthly late share, buffer and on-time rate, with Black Friday and Feb–Mar 2018 marked. Also compute the supporting facts in code:
   - carrier-leg share of delivery-time variance (~84%);
   - bad-review rate, late vs on time;
   - repeat customers by `customer_unique_id`;
   - optionally, on-time by customer region.
3. Write a new `reports/phase2_report.tex`.
   - 6–8 pages for the main body, 12pt; the appendix is unlimited.
   - Output PDF: `Team6_Phase2_IT5006_AY2627Sem1.pdf`.
   - Required:
     - MAE/RMSE/R²;
     - precision, recall, F1, ROC-AUC, PR-AUC and MCC;
     - a discussion of imbalance;
     - feature importance;
     - an AI declaration;
     - why the handover timestamp is legitimate at the handover prediction point;
     - why the split is temporal, not stratified;
     - why we select on promise length, not MAE (ridge has the best MAE, 5.72 vs 5.80).
   - Exec summary in plain words: what each model does and how operations uses it.
4. There is no local LaTeX. Install tectonic (ask the user first), or the user compiles online.

## Environment

- Python env `it5006-proj/` is not in git. Create it with `python -m venv it5006-proj && it5006-proj/bin/pip install -r requirements.txt`; scikit-learn brings `threadpoolctl`.
- Data CSVs are tracked in `data/`.
- Tests: `it5006-proj/bin/python -m unittest discover tests` (59 pass).
- Rerun: `it5006-proj/bin/python phase2_promise.py`, about 7 minutes, and `it5006-proj/bin/python phase2_handover_classifier.py`, about 2 minutes.

## Parked: decide later whether to include in the report

**Route average + same buffer** (simple-heuristic baseline for the promise engine). It separates the gain from the buffer approach from the gain from the forest.

| | Mean promise | On time |
|---|---|---|
| Route + adaptive buffer (test) | 18.86 days | 96.01% |

- Decomposition of the mean test promise: Olist 22.05 → route + buffer 18.86 → forest + buffer 17.79 days.
- The forest beats the route average in both validation and test.
- The course brief asks for a simple heuristic as well as the status quo. Check whether the report needs it before dropping it for good.
- Still in `docs/phase2_final_spec.md` and the results folders.

## Selection-rule change (10 October 2026)

- The spec's tie margins (0.1 day, 0.005 PR-AUC) had no justification. Both models now use the paired one-standard-error rule over the five validation windows: the simplest candidate whose mean gap to the best is within one standard error of that gap.
- The promise engine is now chosen with the adaptive buffer, the deployed system, not the fixed one. Per candidate, γ is the shortest-promise step reaching 95% mean validation on-time; candidates with none are ineligible.
- Result: `tree_depth6` (40.1 days, 95.9%), 1.17 ± 1.33 days behind the forest (38.9 days). Linear and ridge reach only 93.9–94.9% on time, so they are ineligible. The route average with the same buffer (39.6 days, 95.8%) is also within one SE of the forest.
- The classifier is unchanged: the forest beats logistic in 5/5 windows by 0.023 ± 0.006 PR-AUC.
- Disclosed in the report: the change was made after test results for the earlier forest engine and a fixed-buffer ridge engine had been seen.
- Evidence and report: `reports/phase2_final_evidence.py` → `reports/generated/`, `results/phase2/final/`; report `reports/phase2_report.tex` (not yet compiled).

