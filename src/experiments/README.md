# Checks used by the Phase 2 report

Run these from the repository root after the main model scripts. Outputs are included under `data/outputs/` so the report can be inspected without refitting.

| Command | Result used in report |
| --- | --- |
| `python src/experiments/slack_shape_ablation.py` | Slack-shape validation and test diagnostics in `data/outputs/phase2/handover/`. |
| `python src/experiments/three_feature_validation_ablation.py` | Combined three-feature validation comparison. |
| `python src/experiments/backlog_feature/run.py` | Buyer-state backlog validation ablation, saved in `data/outputs/experiments/backlog_feature/`. |
| `python src/experiments/unregularized_logistic_audit.py` | Plain logistic family baseline. |
| `python src/experiments/monthly_retraining/run.py` | Monthly classifier refit versus frozen fit and remaining-slack rule, saved in `data/outputs/experiments/monthly_retraining/`. `test_outputs.py` checks the saved outputs. |

The point-in-time `asof_mean` helper used by `src/olist/eta.py` lives in `src/olist/features.py` and is tested by `tests/test_phase2_recent_history_audit.py`. `recent_history_audit.py` keeps the standalone historical comparison, which also reads the included `data/outputs/phase2/regression/` prediction files; those older checkout outputs are inputs to that diagnostic, not the final regression metrics. The final regression comparison is in `data/outputs/phase2/eta/`.

The monthly test-period refresh and subgroup checks are descriptive analyses made after that period was available. Only chronological validation is used for the report's model-selection comparisons.
