# Framing scan (10 October 2026)

Exploratory search for a regression and a classification model that beat both today's practice and the strongest simple rule. Written by Claude and Sonnet subagents in a session scratchpad, then copied here. **Exploratory, not yet reviewed or consolidated.** Numbers come from separate runs with slightly different baselines; compare within a file, not across files.

Common protocol: chronological; settings chosen on 5 monthly validation windows (Jan to 25 May 2018) with monthly refit on outcomes known before each window; test = purchases from 26 May 2018, scored once. All history features are as-of the prediction time.

## Outcome

Brief: `delivery-promise-plan.html` (published privately at https://claude.ai/artifact/4RWMHKDdX7F8w6DFCMTirq).

- **Regression: promise engine** (`geo_feat.py`, `geo_run.py`, `final_promise.py`; results `results_geo.md`, `results_final.md`). Two-stage GBM (seller handover days plus carrier-leg days) with a daily conformal margin. At a 95% target set on validation: test promise 17.1 days at 95.9% on time, against Olist 22.1 days at 96.5% and the best rule 17.6 days at 94.8%. Month-of-year leaked the regime and was removed (`results_geo_with_month.md`, `ablate.txt`).
- **Same model at handover vs Pratik** (`ho_*.py`, `results_handover.md`): test MAE 2.65 vs 3.89 on the same 19,230 orders; validation edge only 0.17 days.
- **Classification: late warning at handover** (`carrier.py`, `results_fail.md` Task 2, `results_carrier.md`). The relative-feature GBM catches 62% of late orders in a 10% list, the same as the slack rule (62%) and the promise engine's own flag (63%). Kept as the dual-framing classifier (course brief Example 4).

## Ruled out

| Framing | Files | Finding |
|---|---|---|
| Seller deadline-miss classifier | `miss_model.py`, `miss_fixed.py`, `nudge.py`, `results_nudge.md` | Beats rules at approval (31% vs 25%), but a reminder rule ('unshipped 12–24h before the deadline') catches every miss |
| Bad review at delivery | `rv_*.py`, `final_reviews.py`, `results_reviews.md`, `results_final.md` | First win was inflated: 91% of late-order bad reviews are written before delivery. On on-time orders the model ≈ 'n_items ≥ 2' rule |
| Fast-delivery badge | `badge_s1.py`, `badge_s2.py`, `results_badge.md` | Redundant with the promise engine's quantile |
| Fair model vs rule rematch | `fair.py`, `results.md` | Without live operational features, learned models tie the route rule |
| Failed orders, seller dormancy, approval delay, opportunity scan | `fail.py`, `seller_snap.py`, `approval.py`, `inv.py`, `size*.py`, `results_oppscan.md` | Small problems or small lift |
| Regression scan | `reg/`, `results_regscan.md` | Seller handover time real (MAE −15%); freight, seller GMV and weekly demand weak |
| First-pass hypotheses and decomposition | `hypotheses.py`, `decompose.py`, `scan.py` | Carrier leg = 84% of delivery-time variance; 57% of 1–2 star reviews are on on-time orders |

## Running the scripts

Most scripts hard-code the old scratchpad path (`/private/tmp/claude-501/.../scratchpad/`) for inputs and cached features (`feat.pkl`, `rv_base.pkl`, not copied because they are large and regenerable). Point those paths at this folder and regenerate the caches (`geo_feat.py` writes `feat.pkl`; `rv_base.py` writes `rv_base.pkl`) before running. `ho_pratik.py` expects Pratik's branch exported to `./pratik` (`git archive origin/pratik | tar -x -C pratik` plus a copy of `data/`). Run with `it5006-proj/bin/python -I <script> data`-style arguments as each script's header shows.

## Next steps

1. Consolidate the promise engine and the handover classifier into one reviewed pipeline with settings fixed in advance.
2. Decide with the team how to present the classifier (it ties the rule; its value is the decision it supports).
3. Add a monitoring/peak buffer plan (on-time fell to 91.7% in the Feb–Mar 2018 slowdown at the 95% target).
