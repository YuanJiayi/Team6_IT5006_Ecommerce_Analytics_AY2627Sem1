"""Checks the saved calibration outputs: python src/experiments/late_risk_calibration/test_outputs.py"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "late_risk_calibration"

summary = json.loads((OUT / "summary.json").read_text())
deciles = pd.read_csv(OUT / "deciles.csv")
periods = pd.read_csv(OUT / "summary.csv")

assert summary["refit_max_abs_difference_from_saved_scores"] < 1e-9, "test scores are not the report's saved scores"
assert 1.9 < summary["test_predicted_to_actual"] < 2.05, "test probabilities are not about twice the late rate"
assert summary["test_deciles_overpredicted"] == 10, "some test decile is not overpredicted"
assert summary["validation_pooled_predicted_to_actual"] < 1, "validation probabilities are not below the late rate"
assert abs(summary["test_actual_late_rate"] - 672 / 19230) < 1e-12
test_months = periods[(periods.scope == "test") & (periods.period != "overall")]
assert (test_months.predicted_to_actual > 1).all(), "some test month is not overpredicted"
print("late-risk calibration output checks passed")
