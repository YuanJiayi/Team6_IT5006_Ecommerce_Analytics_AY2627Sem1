# Phase 2 handoff (10 October 2026)

## RESUME HERE (10 October 2026, late evening)

**Read this section first.** Everything under "History" below describes the *previous* selection (tree_depth6 / forest_leaf20) and its numbers are stale. Keep it only as history.

**Deadline:** Phase 2, 11 October 2026, 23:59. Brief: `ref/IT5006 Project Description - AY 2026_27 Semester 1.pdf` (Phase 2 pp. 9–12, marking p. 16).

### NEW DIRECTION (agreed with the user, late 10 October): read this first
The regression is now **Pratik's two-stage delivery-time estimate** (checkout estimate + handover update), not the promise engine. The handover late-warning classifier stays. The promise engine becomes a **stretch goal**, done only if time remains after the steps below. The remaining-slack rule is not a baseline or comparison; do not reintroduce it.

**Story**
- Business problem: late deliveries lead to bad reviews, and delivery conditions are unstable over time, so a fixed estimate often misses. Stakeholder: Olist's delivery operations team, who need to know when orders will arrive and which orders need attention before the promise is broken.
- Regression, two stages. Checkout estimate: delivery days from order, route and seller information known at purchase; sets the customer's expected date. Handover update: re-predicts once the seller has handed the parcel to the carrier (adds time already used and recent route transit); gives a sharper updated arrival date.
- Classifier at handover: probability of missing the promised date; ranks the day's handed-over orders by risk.
- Both use the same two families (Linear, Tree-based), the simplest variant in each family as baseline, and the same tuning and one-SE selection.
- How Olist applies them: (1) checkout estimate sets the expected date; (2) at handover, a daily ranked risk list, worked from the top; (3) for each flagged order the updated estimate shows how late it is likely to be: expedite with the carrier if recoverable, otherwise message the customer with the new date; (4) refit and monitor regularly because conditions shift. This is the brief's Example 4 (dual framing).
- Stretch goal: a calibrated buffer on the checkout estimate turns it into a promise with a chosen on-time rate. Use a fixed coverage level with the daily rolling calibration, not the adaptive buffer (it lost on validation).

**Facts checked in this session**
- Pratik's code is already on `phase2-final` (commit `8e9a143`, identical to `origin/pratik` `c66c7be`): `phase2_eta.py`, `docs/phase2_two_stage_eta.md`, `results/phase2/eta/`, `tests/test_phase2_eta.py`. Nothing needs merging.
- The classifier already uses Pratik's selected handover model for its "estimate − promise" ranking: `phase2_handover_classifier.py:450–451` reads `results/phase2/eta/selection.json` (`selected_candidate`). So the regression must be re-selected **before** the classifier is rerun.
- Pratik's current selection uses a 0.05-day MAE tie margin (`TIE_MARGIN_DAYS`) and a fixed `SIMPLICITY` list, not our one-SE rule, and a smaller candidate set.
- Pratik's 0.41 R² is a handover model on a **random** split (his doc labels it secondary). Chronological: checkout about 0.17, handover about 0.34 (validation). Do not quote 0.41 as the result.

**Plan (steps before the stretch goal)**

*Context*
1. Read this section, `docs/phase2_two_stage_eta.md`, `phase2_eta.py`, `phase2_selection.py`, and the brief (Phase 2 pp. 9–12, marking p. 16).
2. `it5006-proj/bin/python -m unittest discover tests` must pass (68 tests).

*Decisions to confirm with the user before coding*
3. Regression selection metric: recommend mean validation MAE over the five windows; RMSE and R² reported alongside.
4. Checkout-stage inputs: recommend the promise engine's inputs (spec features minus `promised_days` and `promise_slack`, plus `route_transit_90d`) so the stretch goal can reuse the checkout model. Alternative: Pratik's frozen inputs, which include Olist's promise.
5. Handover-stage inputs: keep Pratik's.

*Align the regression with the agreed selection*
6. In `phase2_eta.py`, replace `TIE_MARGIN_DAYS` and `SIMPLICITY` with the two-stage method: Stage 1, best mean validation MAE within each group (linear, tree, forest, boost); Stage 2, line-up of plain linear, tuned linear, plain tree, tuned tree, tuned forest, tuned boosting with the paired one-SE rule from `phase2_selection.py`. Reuse the grids in `phase2_promise.py` (`LINEAR`, `TREE`, `FOREST`, `BOOST`, `GROUPS`). Run the selection for the checkout and handover stages separately.
7. Keep the baselines as reference rows, not candidates: training mean; for handover, Pratik's elapsed days + route transit rule.
8. Keep the information ladder.
9. Score test only after selection is recorded, once per stage, selected model only: MAE, RMSE, R², bias, overall and by month.
10. `results/phase2/eta/selection.json` gets `tuned`, `lineup`, `stage2` (same structure as the classifier) and must keep `selected_candidate` naming the handover-stage model, because the classifier reads it.
11. Update `tests/test_phase2_eta.py`; add a `--smoke` option like the other scripts; all tests pass.

*Runs*
12. Smoke run: `phase2_eta.py --smoke --out <scratch folder>`; check the JSON.
13. Full run: `phase2_eta.py`.
14. Rerun `phase2_handover_classifier.py` (about 25 min on 8 cores). Its selection should not change; its estimate − promise ranking will use the new handover model.
15. Commit code and results.

*Evidence*
16. Update `reports/phase2_final_evidence.py`: regression from `results/phase2/eta/` (both stages' selection, ladder, test metrics); classifier from the new `results/phase2/handover/`; remove or set aside the promise-engine parts for the stretch goal.
17. Rerun it to regenerate `reports/generated/numbers.tex`, figures and `results/phase2/final/`. Every report number must come from these files.

*Report (`reports/phase2_report.tex`)*
18. Rewrite to the story above: executive summary; business problem; data and validation design (mostly unchanged); regression section (checkout and handover, ladder, grids, baseline-vs-tuned per group, one-SE line-up, MAE/RMSE/R²); classifier section (logistic regression via one-SE, robustness sentence, imbalance metrics); how Olist applies the models; limitations; appendix table of every configuration.
19. Apply the fixes in "Report: every section needs checking" below: ISLR p. 214 citation (ESL section unverified); validation-vs-validation checkout/handover PR-AUC comparison; feature-importance wording (top input `handover_days`); classifier validation→test drop explained by the late rate plus ROC-AUC; any negative checkout test R² explained as drift; leave the AI declaration placeholder for the team.
20. Check length (6–8 pages main body) and reread the whole report against the regenerated numbers.

*Wrap-up*
21. Update this handoff, commit, push after the user confirms. The report is compiled online (no local LaTeX).

Estimate: 2–3 h code and tests, about 1.5 h runs, 2–3 h evidence and report.

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
3. The remaining-slack rule is not a baseline or a comparison in the report (user decision, late 10 October). The current report draft still has a slack-rule section and table row: remove them.
4. Cite the one-SE rule as ISLR (James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, Sec. 6.1.3, p. 214, verified from the PDF). The current ESL Sec. 7.10 citation is **unverified**. Say "standard in statistical learning" (glmnet's default `lambda.1se`, verified), not "industry standard". State two caveats: we apply it across model types (our simplicity order is a judgment), and the SE comes from 5 non-independent windows. Add: "the gap is about a fifth of its standard error, so the choice is not sensitive to the tie-breaking rule."

### Promise engine: now the stretch goal (findings to keep)
- Option A framing (fixed coverage level with daily rolling calibration; adaptive as a tested alternative that lost on validation) is the basis for the stretch goal.
- Validation trade-off (ridge_a1000, γ = 0): at Olist's validation on-time (89.3%, 25.0 days) the engine gives about 24.8 days and the route average about 24.9, so it does not beat Olist at equal reliability; it does hit a chosen target (95.0% on validation).
- A perfectly timed level would still not reach 95% in the Black Friday window (91% at the highest level tested, 0.98).
- Lead-indicator check (training data only, script not in the repo): 14-day handover delay, 14-day transit, 7-day volume, orders in transit and awaiting handover do not lead the shocks once trend is removed (weekly-change correlations −0.09 to 0.18). Orders in transit peaked 2–5 weeks after Black Friday and stayed high after Feb–Mar 2018. So lag fixes based on these signals are not supported.
- Untested ideas for shorter promises: buffers calibrated by segment (Mondrian conformal), or conformalized quantile regression. Test on validation only before building.

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
Follow the plan in "NEW DIRECTION" above.

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

