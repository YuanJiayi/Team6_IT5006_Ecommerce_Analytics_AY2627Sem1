# Phase 2 handoff (10 October 2026)

## RESUME HERE (work in progress, 10 October 2026)

**State:** the code for the tuned two-stage selection is written and unit-tested (68 tests pass), but the **full runs have not been done**. Everything under `results/phase2/` and the numbers in `reports/phase2_report.tex` are still from the previous selection (promise engine `tree_depth6`; classifier `forest_leaf20`) and will change.

**Agreed design (approved by the user; do not change without asking):**
- Families: Linear, and Tree-based (single tree, random forest, gradient boosting = `HistGradientBoosting*`), the same for both models (brief: 2–3 families, reused across tasks).
- Stage 1 (tuning): within each group, the configuration with the best mean score over the five temporal validation windows.
- Stage 2 (selection): line-up = plain linear, tuned linear, plain tree, tuned tree, tuned forest, tuned boosting; paired one-standard-error rule (simplest entry within 1 SE of the best). Shared code: `phase2_selection.py`.
- Promise engine score: mean promise at exactly 95% mean validation on-time, each configuration tuned with the adaptive buffer (γ grid × on-time target 0.85–0.98, `matched_scores`). MAE/RMSE/R² for every configuration in `forecast_metrics_by_window.csv`. Route average + buffer is a reference, not a candidate.
- Classifier score: mean validation PR-AUC. Operating share, slack-rule bootstrap, checkout variant unchanged.
- Grids are in `phase2_promise.py` and `phase2_handover_classifier.py` (`LINEAR/LOGISTIC`, `TREE`, `FOREST`, `BOOST`, `GROUPS`).
- Fixed, not reopened: features, target, cohort, splits, 95% target, 5:1 benefit ratio.

**Next steps, in order:**
1. Smoke runs (one config per group, a few minutes each), to a scratch folder:
   `it5006-proj/bin/python phase2_promise.py --smoke --out /tmp/smoke_promise` and
   `it5006-proj/bin/python phase2_handover_classifier.py --smoke --out /tmp/smoke_handover`. Check they finish and the JSON outputs look right.
2. Full runs, one after the other (2-core machine: ~1.5 h promise, ~40 min classifier):
   `it5006-proj/bin/python phase2_promise.py` then `it5006-proj/bin/python phase2_handover_classifier.py`.
3. Update `reports/phase2_final_evidence.py` for the new outputs (`adaptive_selection.json` now has `tuned`, `lineup`, `stage2`, `matched`; promise `validation_summary.csv` now holds only the chosen model and baselines, so MAE/RMSE/R² for all configs come from `forecast_metrics_by_window.csv`; classifier `selection.json` has `tuned`/`stage2`). Rerun it.
4. Rewrite the model sections of `reports/phase2_report.tex`: tuning grids, baseline-vs-tuned table per group, the one-SE line-up, an appendix table of every configuration, and the new numbers. Frame the regression criterion as coverage (on-time %) and interval width (promise length). Keep the explicit "why not MAE" paragraph.
5. Business framing (from a reviewer, agreed): tie results to bad reviews (62% of late orders get 1–2 stars vs 9% on time) and be honest that the repeat-purchase link is weak (2.5% vs 3.0%) and that conversion cannot be measured in this data (cite Salari et al. 2022). Do not reuse the old "4 days sooner at same reliability" headline unless the new results support it.
6. Commit and push. The report must still be compiled online (no LaTeX locally).

**User preferences for this work:** avoid repeated work, so settle the design before running; everything must be defensible and aligned with the course brief (`ref/IT5006 Project Description - AY 2026_27 Semester 1.pdf`, Phase 2 on pp. 9–12, marking on p. 16). Phase 2 deadline: 11 October 2026, 23:59. The user asked that the report not contain a disclosure about when selection rules changed.

**Environment:** `python -m venv it5006-proj && it5006-proj/bin/pip install -r requirements.txt`; tests: `it5006-proj/bin/python -m unittest discover tests`.


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

