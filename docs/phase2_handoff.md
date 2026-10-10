# Phase 2 handoff (10 October 2026)

Status of branch `phase2-final`, for the next agent. The design is in `docs/phase2_final_spec.md`. Course requirements are in `ref/IT5006 Project Description - AY 2026_27 Semester 1.pdf`: Phase 2 on pp. 9–12, marking on p. 16.

## Story

Olist makes two delivery decisions per order. One problem, two model types (brief Example 4):

1. **Checkout: the promise engine (regression).**
   - `forest_leaf20` forecasts delivery days.
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
| Regression | Olist's promise | Route average + same buffer | LinearRegression / DecisionTree variants |
| Classifier | No prioritisation (random list of the same size) | — | Logistic / decision tree variants |

The remaining-slack rule is **not** a report baseline. Prepare for the question "could a hand-built slack formula do as well?":

- Mostly, yes. Slack is the model's top input.
- On validation the model beat the rule: PR-AUC +0.038, 95% CI [0.031, 0.045].
- On test the rule won: 0.52 vs 0.43.

## Key numbers (all from the results folders)

**Promise engine, test**

| | Mean promise | On time |
|---|---|---|
| Forest + adaptive buffer (γ = 0.05) | 17.79 days | 96.29% |
| Route + adaptive buffer | 18.86 days | 96.01% |
| Olist | 22.05 days | 96.52% |
| Fixed buffer level (L = 0.9775) | 27.3 days | 99.1% |

**Promise engine, validation and buffer behaviour**

- At matched reliability on validation, the engine and Olist are about equal: 24.5 vs 24.4 days. The forest beats the route average in both periods.
- Faster γ (0.1, 0.2) oscillates and loses coverage.
- Even at γ = 0.05, the adaptive buffer overreacted after Black Friday: about 52-day promises in Jan–Mar 2018. Treat this as a limitation that needs monitoring.

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
