# Checks used by the Phase 2 report

Run these from `Phase2/` after the main model scripts. Outputs are included so the report can be inspected without refitting.

| Command | Result used in report |
| --- | --- |
| `python experiments/phase2_slack_shape_ablation.py` | Slack-shape validation and test diagnostics in `results/phase2/handover/`. |
| `python experiments/phase2_three_feature_validation_ablation.py` | Combined three-feature validation comparison. |
| `python experiments/backlog_feature/run.py` | Buyer-state backlog validation ablation. |
| `python experiments/phase2_unregularized_logistic_audit.py` | Plain logistic family baseline. |
| `python experiments/monthly_retraining/run.py` | Monthly classifier refit versus frozen fit and remaining-slack rule. |

`phase2_recent_history_audit.py` supplies the point-in-time `asof_mean` helper imported by `phase2_eta.py` and tested by `tests/test_phase2_recent_history_audit.py`. Its standalone historical comparison also reads the included `results/phase2/regression/` prediction files; those older checkout outputs are inputs to that diagnostic, not the final regression metrics. The final regression comparison is in `results/phase2/eta/`.

The monthly test-period refresh and subgroup checks are descriptive analyses made after that period was available. Only chronological validation is used for the report's model-selection comparisons.
