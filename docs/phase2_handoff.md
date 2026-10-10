# Phase 2 handoff (10 October 2026)

## RESUME HERE (10 October 2026, night): the NEW DIRECTION plan is done

**Deadline:** Phase 2, 11 October 2026, 23:59. Brief: `ref/IT5006 Project Description - AY 2026_27 Semester 1.pdf` (Phase 2 pp. 9–12, marking p. 16).

All 15 steps of the "NEW DIRECTION" plan (see History below for its full text) are complete, tested and committed on `phase2-final`. The report compiles cleanly (`reports/phase2_report.tex`, 9 pages: 6-page main body + 3-page appendix) and `reports/Team6_Phase2_IT5006_AY2627Sem1.pdf` is the fresh compile. **Not yet done:** the AI declaration placeholder needs a team member to confirm it, and nobody has proofread the final PDF against the actual numbers end to end — do that before submitting.

### What changed in this pass
- `phase2_eta.py`: `select_candidate` now uses `phase2_selection.one_se_choice` (removed `TIE_MARGIN_DAYS`); `run()` now compares all candidates and selects separately for **both** the checkout and handover stages (previously checkout was fixed at `ridge_100` with no comparison). `selection.json` keeps `selected_candidate` = the handover model (the classifier still reads this key) and adds `checkout_selected_candidate`, per-stage `reason`/`stage2`/`lowest_mae_candidate`. The elapsed-plus-route rule baseline is handover-only (it needs handover-stage columns that don't exist at checkout).
- `phase2_selection.py`: docstring citation changed from ESL Sec. 7.10 to ISLR (James, Witten, Hastie & Tibshirani, Sec. 6.1.3, p. 214), with the "applied across model families" and "5 non-independent windows" caveats stated.
- `tests/test_phase2_eta.py`: `SelectionTests` rewritten for the per-window-scores one-SE signature.
- **Result:** at both stages, ridge alpha 100 is selected (checkout: within one SE of `forest_leaf20`, the best; handover: ridge 100 is itself the best, no simpler candidate is within one SE of it). Mean validation MAE: checkout 5.81 days (R² 0.167), handover 5.10 days (R² 0.338) — the handover update beats the checkout estimate in every validation window. Full numbers and the appendix tables are in the report.
- `phase2_handover_classifier.py` was rerun (its own selection does not depend on the regression, but the handover-selected model changed from `linear_baseline` to `ridge_100`, which its internal "estimate − promise" baseline reads — that comparison is not shown in the new report, but the saved results are now consistent). Selection is unchanged: tuned logistic regression (C = 0.01), validation PR-AUC 0.367 (forest_l50_sqrt is best at 0.369, within one SE and simpler, so logistic wins).
- `reports/phase2_final_evidence.py` was rewritten for the new story: regression numbers now come from `results/phase2/eta/` for **both** stages (ladder, one-SE rows, test metrics, coefficients/permutation importance, random-split benchmark); classifier numbers from `results/phase2/handover/` unchanged in substance but with the validation-vs-validation PR-AUC bug fixed (`\CheckoutPR` is now checkout validation, compared against `\ValSelPR` handover validation, not a test-period number); the promise engine and the remaining-slack rule are dropped entirely (no macros, no figures, no table rows). New appendix tables (`fig_appendix_regression.tex`, `fig_appendix_classifier.tex`) list every candidate. Found and fixed one macro-name collision (`CheckoutSelName` was being written by both the regression and classifier sections; the classifier's is now `ClsCheckoutSelName`).
- `reports/phase2_report.tex` is a full rewrite to the two-stage-estimate-plus-classifier story: executive summary, business problem, data/validation design, Model 1 (two-stage regression: ladder table, candidate selection at both stages, test results, random-split benchmark, feature importance), Model 2 (classifier: method, imbalance, operating point, test table, ranking quality, feature importance, monthly table — no slack-rule section), how Olist applies the models, limitations (including a short note that the promise buffer is a stretch goal, not a headline result), appendix (full configuration tables), references (ISLR added, promise-engine-only citations dropped), AI declaration placeholder.
- `README.md` and `notebooks/phase2/README.md` updated to point at the new report and the final scripts (`phase2_eta.py`, `phase2_handover_classifier.py`) instead of the old joint report / evidence index, and to note the notebooks are development history, not the final models.
- Stale `results/phase2/final/*` files left over from the old promise-engine evidence script (`monthly_late_share_and_buffer.csv`, `promise_adaptive_selection.csv`, `promise_test_per_month.csv`, `validation_windows_comparison.csv`) were removed since nothing regenerates or reads them any more.
- `it5006-proj/bin/python -m unittest discover tests`: 66/68 pass; the 2 failures (`test_dashboard_charts`, `test_dashboard_data`) are pre-existing missing-package errors (`pyarrow`, `streamlit` not installed in this container) unrelated to this work.

### Still true from before (unchanged)
- `results/phase2/handover/` is the classifier's own results; its candidate grid, selection rule and value-test machinery were not touched, only rerun.
- `phase2_promise.py`, `results/phase2/promise/`, `tests/test_phase2_promise.py` are the stretch goal; not touched, not referenced by the report. If there's time before the deadline, see "Promise engine: now the stretch goal" in History below for what's already known about it (Option A framing: fixed coverage level with daily rolling calibration beats the adaptive buffer on validation).
- The report is compiled locally in this session with `pdflatex` (available in this container, contrary to the earlier assumption that there was no local LaTeX) — `reports/Team6_Phase2_IT5006_AY2627Sem1.pdf` is the result of `pdflatex` run twice from `reports/phase2_report.tex`. It can still be compiled online if preferred; `reports/generated/*.tex` and `reports/phase2_report.tex` are everything needed.
- Unchanged data facts: 96,470 delivered orders; handover cohort 96,272 (198 excluded); test 19,230 orders (not 19,363 — that number in an earlier draft of this doc was wrong); late share about 4% normal, ~12% Black Friday, ~17% Feb–Mar 2018 (weighted), 3.5% test; bad review 62% late vs 9% on time (95,627 rated orders); carrier leg 85% of delivery-time variance; repeat purchase 2.5% after late vs 3.0% after on time (93,159 unique customers).

### Before submitting
1. A team member should read `reports/Team6_Phase2_IT5006_AY2627Sem1.pdf` end to end and fill in the AI declaration's bracketed confirmation.
2. Confirm the GitHub link in the report header is correct.
3. If there's spare time, the promise-engine stretch goal (see History) could be added as an extra appendix subsection, but it is not required — the report is complete and self-contained without it.

**User preferences (carried over):** everything defensible and aligned with the brief; no report disclosure about when the selection rules or the regression story changed.

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

