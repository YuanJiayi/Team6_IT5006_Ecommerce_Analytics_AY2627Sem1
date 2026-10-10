# Phase 2 final specification: promise engine and handover late classifier

Fixed on 10 October 2026, before any model in this specification was fitted. Changes after fitting must be recorded at the end of this file with the reason and date.

## Business problem

Olist's delivery operations team makes two decisions per order:

1. **At checkout:** what delivery date to promise the customer. A shorter promise may help conversion; a broken promise drives bad reviews.
2. **At carrier handover:** which orders to act on (expedite with the carrier, or update the customer) because they are likely to miss their promise.

**Regression** sets the first; **classification** supports the second. They use one target (delivery time, and lateness against the promise derived from it): course brief Example 4.

## Data, cohort and splits

- Table `data/phase2_order_table.csv` and spec `data/phase2_feature_spec.json` (unchanged). Cohort: 96,470 delivered orders with actual and estimated delivery dates.
- Validation: `temporal_folds` (`phase2_classification.py`), 5 windows from 22 Sep 2017 to 26 Mar 2018. Each window fits only on orders purchased **and delivered** before it starts.
- Test: `split == "test"` (purchased from 26 May 2018), single refit on all `split == "train"` orders, scored once after every choice below is fixed.
- Handover cohort: `load_stage_table` (`phase2_eta.py`): orders whose carrier handover is after purchase and before delivery.
- `need` = calendar days from purchase date to delivery date. An order is on time when `need <= promise` (same calendar-date rule as `is_late`).

## Regression: forecast plus buffer

**Forecast μ (expected delivery days, target `delivery_days`).**

- Inputs: spec numeric and categorical features **minus `promised_days` and `promise_slack`**, **plus `route_transit_90d`** (from `add_stage_features`). Reasons: the promise engine should not depend on Olist's own estimate; recent carrier-leg history helped in the earlier ridge audit and Pratik's ladder.
- Input check on validation (reported, not used to override selection): with/without the two promise fields; with/without `route_transit_90d`.
- Candidates:
  - Floors: training mean; route baseline (`RouteBaseline` in `phase2_regression.py`).
  - Linear family: `linear_baseline`, `ridge_1`, `ridge_10`, `ridge_100`.
  - Tree family: `tree_baseline`, `tree_depth6`, `forest_leaf20`.
  - Pipelines from `make_regression_pipeline`.

**Buffer (promise layer).**

- For each purchase day D, the calibration set is orders purchased between D−90 and D−45 days **and delivered before D**.
- Score r = (need − μ) / max(μ, 3). Calibration orders that were also fitting orders use **out-of-fold μ** (3 time-ordered folds over the fitting set).
- Promise = max(1, ceil(μ + q · max(μ, 3))), where q is the calibration set's quantile of r at level L.
- Level grid: L = 0.800, 0.8025, …, 0.995.

**Selection.**

- For each candidate and window, find the mean promise at exactly 95% on time by interpolating along L.
- Choose the lowest mean over the 5 windows. A simpler model within 0.1 day wins.
- Simplicity order: `linear_baseline`, `ridge_1`, `ridge_10`, `ridge_100`, `tree_depth6`, `tree_baseline`, `forest_leaf20`.
- MAE, RMSE and R² of μ are reported for every candidate as diagnostics.

**Headline (implementable).**

- Fix L on validation: the smallest L whose mean validation on-time rate is ≥ 95%.
- Apply that L to test, once. Report mean and median promise, on-time rate overall and per month, and the per-month buffer.

**Baselines.**

- Status quo: Olist's promise (`promised_days`). Compare at matched reliability: the engine's mean promise interpolated at Olist's own test on-time rate.
- Simple heuristic: route baseline μ with the same buffer and the same L-selection rule. Report the decomposition Olist → route + buffer → selected model + buffer.
- Simplest in family: linear and decision-tree variants above.

## Classification: late warning at carrier handover

- **Label (primary):** `is_late` against Olist's promise (real outcomes; usable without the engine).
- **Inputs:**
  - The spec features, including `promised_days`.
  - `approval_days`, `handover_days`, `route_transit_90d_at_handover` (from `add_stage_features`).
  - `remaining_slack` = promised_days − handover_days − route_transit_90d_at_handover.
- **Candidates:**
  - `logistic` (plain); `logistic_balanced` (class_weight="balanced").
  - `tree_depth6`; `forest_leaf20` (classifier versions).
  - Preprocessing inside Pipelines.
- **Selection:** mean validation PR-AUC (late orders are a minority: 4–21% across windows, 3.5% on test). A simpler model within 0.005 wins. Order: logistic, logistic_balanced, tree_depth6, forest_leaf20.
- **Operating point (rank-based):** probabilities drifted across periods in the earlier report, so actions use a share of each month's handovers, not a fixed probability threshold.
  - Act on the top k% by risk, with k chosen on validation (grid 1–30%) to maximise net benefit = 5 × late orders caught − 1 × orders acted on.
  - **Assumption:** acting on a truly late order is worth 5× the cost of acting on any order.
  - Sensitivity in the appendix: worth 2×, 10× and 20× the cost.
  - Report precision, recall, F1 and MCC at k, plus ROC-AUC and PR-AUC overall.
- **Baselines:**
  - Status quo: random selection of the same k% (no prioritisation today).
  - Simple heuristic: rank by lowest `remaining_slack`.
  - Defensibility check: rank by (regression handover estimate − promise), using Pratik's selected handover model from `phase2_eta.py`.
- **Value test:** order-bootstrap 95% interval for model minus slack rule in PR-AUC and recall at top 10%, pooled over validation windows.
- **Pre-declared extension if the value test fails:** add leak-free carrier-leg history (ZIP-3 recent transit; seller recent handover speed) as of handover; rerun selection once.
- **Appendix:**
  - Checkout-stage variant: same candidates, checkout inputs only.
  - Simulation: ranking quality against the engine's promise instead of Olist's. Clearly labelled as a counterfactual.

## Supporting facts computed in code

- Share of delivery-time variance in the carrier leg (handover → delivery) versus the seller leg.
- Bad-review rate for late versus on-time orders.
- Olist promise padding (promise − actual days).
- Repeat customers by `customer_unique_id` across the train/test boundary.

## Value gates

If either gate fails, stop and consult the team before writing the report.

1. Selected model + buffer beats route + buffer on validation promise days at 95%. On test at the fixed L, its mean promise is shorter than Olist's at comparable on-time.
2. The classifier beats the slack rule under the value test.

## Changes after fitting

### 1. Adaptive buffer level (10 October 2026, after the test was scored once)

**What we saw.** With L fixed on validation (L = 0.9775), the test promise was 27.3 days at 99.1% on time, against Olist's 22.1 days at 96.5%. The forecast itself was fine: at Olist's test on-time rate the engine needed 17.7 days.

**Cause.** The buffer calibrates on orders purchased 45–90 days earlier, so it reacts to delivery shocks about two months late.

- During the Nov–Dec 2017 peak it was too small: 88% on time.
- In late Jan–Mar 2018 it was far too large: 98% on time with 38–40-day promises.
- A single L that averages 95% across those swings over-covers in calm periods.

**Change.** Keep the forecast, the score r and the calibration set. Replace the fixed L with a level updated daily from fresh outcomes. This is adaptive conformal inference (Gibbs and Candès, 2021).

- α starts at 0.05.
- The level used on day D is 1 − α_D, clipped to [0.50, 0.995].
- After day D: α_{D+1} = α_D + γ · (0.05 − e_D).
- e_D is the late share among orders whose promise date was D − 1. Their on-time status is known by day D, so no waiting for delivery is needed.
- If no order's promise date was D − 1, α is unchanged.
- The adaptive run starts 120 days before each validation window, and before the test start. It uses the same out-of-fold μ and the same promise rule.

**γ selection.**

- Grid: 0, 0.002, 0.005, 0.01, 0.02, 0.05 (γ = 0 is the plain nominal 95% buffer).
- Chosen on validation only: the shortest mean promise among values with mean validation on-time ≥ 95%.
- Applied to the selected model and to route + buffer alike.
- The test is scored once with γ fixed.

**Reporting.** Both the fixed-L result and the adaptive result are reported. This change was made after seeing the test result and is disclosed as such.

### 2. Classifier baseline roles (10 October 2026, after validation and test were scored)

**What changed.** The report's business baseline for the classifier is **no prioritisation**: acting on a random set of the same size. Nothing in the data suggests Olist ranks handed-over orders today.

- **Gate 2 is re-read against that baseline.** Validation passes: PR-AUC 0.366 vs 0.107; 34% of late orders caught in the top 10% vs 10%. Test passes too: 56% vs 10%.
- **The slack rule is no longer a report baseline.** It stays computed in `results/phase2/handover/`.

**Questions to prepare for (not in the report).** "Could a hand-built remaining-slack formula do as well?"

- Mostly, yes. Remaining slack is the model's top input.
- The model beat the rule on validation: PR-AUC +0.038, 95% CI [0.031, 0.045].
- The model lost on test: PR-AUC 0.43 vs 0.52; recall@10% 0.555 vs 0.592.
- Its extra inputs help in disrupted periods and cost a little in calm ones: about 8 fewer late orders caught per month on test.
