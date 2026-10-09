# Two-stage delivery-time estimate

**Written 9 October 2026, after the checkout regression test had been scored.** This extends the checkout regression in [phase2_regression_spec.md](phase2_regression_spec.md); it does not replace or edit it. Because the later test period was already open, every later-period score here is post hoc. The five chronological validation windows are the primary evidence.

Code: [`phase2_eta.py`](../phase2_eta.py). Tests: [`tests/test_phase2_eta.py`](../tests/test_phase2_eta.py). Results: [`results/phase2/eta/`](../results/phase2/eta/).

## Why change the problem statement

The checkout regression explains little of the variation in `delivery_days` (mean validation R² 0.17, later test −0.18). Three properties of the data set that ceiling, and no model choice removes them:

- Orders taking more than 30 days are 4.7% of orders and 58% of the target's variance.
- Carrier-to-customer transit holds about 84% of lead-time variance (76.5 of 91.2 days²), and no checkout input observes it.
- Delivery speed shifts over time (monthly means from 7.7 to 17.0 days). The later test period was 3.7 days faster than the model expected, which alone makes its R² negative.

Public Olist projects that report R² of 0.25 to 0.48 use a random split, later-stage inputs or target outlier filtering (see the benchmark below and [phase2_external_olist_comparison.md](phase2_external_olist_comparison.md)).

## Revised problem

**Stakeholder:** Olist's logistics and fulfilment operations team, as in Phase 1.

**Question:** how many days will this order take from purchase to delivery? The estimate is issued twice:

1. **At checkout**, to set expectations. This is the frozen Phase 2 regression: same 20 inputs, same Ridge model.
2. **At carrier handover**, when the carrier records receiving the parcel from the seller. The estimate is updated with the time already elapsed, so operations can re-notify customers and prioritise parcels that are now likely to run long.

**Target (both stages):** `delivery_days`, purchase to customer delivery. Keeping one target means the two stages are directly comparable.

**Population:** delivered orders with a carrier handover recorded after purchase and before delivery. This is 96,272 of 96,470 prepared orders; 198 are excluded.

**Success criterion:** the handover update must beat the checkout estimate on the same orders in every validation window, on MAE and R².

## Inputs added at handover

| Input | Meaning | Why it is available |
|---|---|---|
| `handover_days` | Days from purchase to carrier handover | Elapsed by definition at the prediction point |
| `approval_days` | Days from purchase to payment approval | Used only when approval is recorded at or before handover; otherwise missing |
| `route_transit_90d_at_handover` | Mean carrier-to-customer days on the seller-state to customer-state route over the previous 90 days | Built only from deliveries recorded strictly before the handover; shrunk toward the all-route mean with 20 pseudo-orders |

`route_transit_90d` is the same history built as of purchase. It is used only in the ladder below, to show what recent transit history adds at checkout.

## Validation design

Unchanged from the checkout regression: five chronological windows with outcome-aware fitting sets, preprocessing fitted inside each fold through a Pipeline, seed 42. Selection uses the same rule: lowest mean validation MAE, and within 0.05 days the simplest candidate wins.

## Results

### Information ladder (Ridge alpha 100 throughout, same orders at every rung)

| Rung | Mean validation R² | Lowest window R² | Mean validation MAE |
|---|---:|---:|---:|
| Checkout (frozen 20 inputs) | 0.167 | 0.060 | 5.81 |
| + recent route transit history | 0.214 | 0.078 | 5.68 |
| + elapsed days to handover and approval | 0.335 | 0.222 | 5.11 |
| Handover (history as of handover) | 0.338 | 0.227 | 5.10 |

The handover rung beats checkout in all five windows.

### Model comparison at handover (mean over five windows)

| Candidate | Family | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Training mean | baseline | 7.21 | 10.67 | −0.078 |
| Elapsed days + route transit average | baseline | 5.30 | 8.40 | 0.337 |
| LinearRegression | linear | 5.15 | 8.47 | 0.327 |
| Ridge alpha 1 | linear | 5.14 | 8.47 | 0.327 |
| Ridge alpha 10 | linear | 5.14 | 8.46 | 0.329 |
| Ridge alpha 100 | linear | 5.10 | 8.41 | 0.338 |
| Decision tree, unrestricted | tree-based | 6.84 | 11.62 | −0.330 |
| Decision tree, depth 6 | tree-based | 5.31 | 8.63 | 0.299 |
| Random forest, min leaf 20 | tree-based | 5.35 | 8.81 | 0.270 |
| Gradient boosting, defaults | tree-based | 5.21 | 8.65 | 0.297 |
| Gradient boosting, slower | tree-based | 5.22 | 8.67 | 0.294 |
| Voting (Ridge 100 + boosting) | ensemble | 5.11 | 8.47 | 0.327 |

**Selected:** LinearRegression. Ridge alpha 100 has the lowest MAE and the highest R², but LinearRegression is within the 0.05-day margin and simpler. The ensemble does not beat its linear member.

### Later period (post hoc, 19,230 orders from 26 May 2018)

| Stage | Model | MAE | RMSE | R² | Bias (days) |
|---|---|---:|---:|---:|---:|
| Checkout | Ridge alpha 100 | 5.00 | 6.41 | −0.175 | +3.65 |
| Checkout | Gradient boosting | 4.71 | 6.22 | −0.104 | +3.20 |
| Handover | LinearRegression (selected) | 3.89 | 5.37 | 0.176 | +2.76 |
| Handover | Gradient boosting | 3.71 | 5.25 | 0.212 | +2.50 |
| Handover | Voting | 3.79 | 5.26 | 0.211 | +2.69 |

### Same-period benchmark (random split; secondary)

Random 5-fold validation and a random 20% test from the saved `random_split`. Rows from every month appear on both sides, so this measures interpolation within the observed period, not forecasting.

| Stage | Model | Validation R² | Test R² | Test MAE |
|---|---|---:|---:|---:|
| Checkout | Ridge alpha 100 | 0.259 | 0.276 | 5.04 |
| Checkout | Gradient boosting | 0.309 | 0.332 | 4.78 |
| Handover | LinearRegression | 0.387 | 0.411 | 4.34 |
| Handover | Gradient boosting | 0.421 | 0.449 | 4.09 |
| Handover | Voting | 0.418 | 0.445 | 4.15 |

The checkout rows line up with public Olist results that use a held-out 20% (Ridge 0.25, tuned XGBoost 0.33), which suggests the gap to published R² is the evaluation design.

## What the evidence does and does not show

- Moving the prediction point to handover roughly doubles chronological R² (0.17 to 0.33) and turns the later-period R² positive.
- Most of that gain is the elapsed time itself. The rule baseline (elapsed days plus the route's recent transit average) matches the models on R² (0.337); the fitted models improve MAE by about 0.2 days.
- Linear models beat tree-based models under chronological validation. Tree-based models win under the random split. Trees fit the observed periods closely and transfer less well to a period with a different delivery speed.
- The handover estimate still over-predicts the later period by 2.5 to 2.8 days. The regime shift is reduced, not solved.
- The handover estimate is not available at checkout. It is a second, later forecast, and should not be reported as a better checkout model.

## Limitations

- Conditional on eventual delivery and on a recorded handover, like the earlier targets.
- Later-period scores are post hoc. A fresh future period would be needed for an independent claim.
- Results were produced with scikit-learn 1.9.1 and pandas 3.0.5; rerun `python phase2_eta.py` in the project environment to regenerate them.
