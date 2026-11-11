# Checks used by the Phase 2 report

Run these from the repository root after the main model scripts. Outputs are included under `data/outputs/` so the report can be inspected without refitting.

| Command | Result used in report |
| --- | --- |
| `python src/experiments/slack_shape_ablation.py` | Slack-shape validation and test diagnostics in `data/outputs/phase2/handover/`. |
| `python src/experiments/three_feature_validation_ablation.py` | Combined three-feature validation comparison. |
| `python src/experiments/backlog_feature/run.py` | Buyer-state backlog validation ablation, saved in `data/outputs/experiments/backlog_feature/`. |
| `python src/experiments/unregularized_logistic_audit.py` | Plain logistic family baseline. |
| `python src/experiments/monthly_retraining/run.py` | Monthly classifier refit versus frozen fit and remaining-slack rule, saved in `data/outputs/experiments/monthly_retraining/`. `test_outputs.py` checks the saved outputs. |
| `python src/experiments/ranking_strategy/NN_name/run.py` | Phase 3 Task 4: late-order ranking strategy (model versus remaining-slack rule, mixes, promise-relative inputs, monthly refits, daily re-ranking), with gates declared in `ranking_strategy/EVALUATION_CRITERIA.md`. Saved in `data/outputs/experiments/ranking_strategy/`, including the refitted final Phase 2 models in `models/`. See `ranking_strategy/README.md` for the run order. |
| `python src/experiments/checkout_promise/run.py` | Phase 3: checkout ETA (ridge alpha=1000, no promise inputs) plus a monthly recalibrated buffer per customer region as the promised date, with the regional buffer diagnosis, retraining versus recalibration, the checkout gap signal, the drift gauge, threshold rules, grouping factors, the seasonal check and no-model benchmarks. Saved in `data/outputs/experiments/checkout_promise/` with two slide figures; `test_outputs.py` checks the headline numbers. Takes a few minutes. |
| `python src/experiments/parcel_inputs_ablation/run.py` | Phase 3: validation ablation of price, freight, weight and volume in the selected checkout ridge (alpha=1000), paired by window. Saved in `data/outputs/experiments/parcel_inputs_ablation/`. |
| `python src/experiments/handover_regression/run.py` | Phase 3: handover regression follow-up (plain linear versus ridge with paired one-SE comparisons, extended ridge grid, remaining-time shortfall against the promise, equal-capacity review queues under hypothetical carrier speed-ups). Saved in `data/outputs/experiments/handover_regression/`; the executed analysis is `notebooks/phase3_investigations/handover_regression.ipynb`. See `handover_regression/README.md`. |
| `python src/experiments/late_risk_calibration/run.py` | Phase 3: calibration of the selected late-risk classifier (logistic C=0.01). Mean predicted late probability against the actual late rate per validation window, test month and score decile. The probabilities follow the fitting history's late rate: about half the actual rate in validation (0.50) and about twice it in the test period (6.9% against 3.5%, every decile overpredicted), so the model is used as a ranking, not a probability. Saved in `data/outputs/experiments/late_risk_calibration/`; `test_outputs.py` checks the headline. |

The point-in-time `asof_mean` helper used by `src/olist/eta.py` lives in `src/olist/features.py` and is tested by `tests/test_phase2_recent_history_audit.py`. `recent_history_audit.py` keeps the standalone historical comparison, which also reads the included `data/outputs/phase2/regression/` prediction files; those older checkout outputs are inputs to that diagnostic, not the final regression metrics. The final regression comparison is in `data/outputs/phase2/eta/`.

The monthly test-period refresh and subgroup checks are descriptive analyses made after that period was available. Only chronological validation is used for the report's model-selection comparisons.
