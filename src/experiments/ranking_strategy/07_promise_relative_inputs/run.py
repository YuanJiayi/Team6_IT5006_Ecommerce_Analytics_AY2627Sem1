"""Replace the late-warning model's absolute timing inputs with inputs relative to the promise.

Run from the repository root:
    python src/experiments/ranking_strategy/07_promise_relative_inputs/run.py

The three new inputs are an approved exception to the rule against computing features by hand; they are calculated only
from load_cohort columns. A version that adds them to all existing inputs is reported on validation only.
Writes the candidate model and CSVs to data/outputs/experiments/ranking_strategy/07_promise_relative_inputs/.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "ranking_strategy" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent))

from task4_common import LABEL, evaluate, feature_columns, load_data, make_pipeline_for, validation_rows  # noqa: E402

REPLACED = ["promised_days", "promise_slack", "handover_days", "route_transit_90d_at_handover", "remaining_slack", "slack_shortfall"]
RELATIVE = ["promise_used_share", "slack_relative", "route_recent_vs_typical"]
CLIP = 5.0
CANDIDATE, ADDED = "relative_replaced", "relative_added"


def add_relative_inputs(table):
    out = table.copy()
    ratios = {"promise_used_share": out["handover_days"] / out["promised_days"],
              "slack_relative": out["remaining_slack"] / out["promised_days"],
              "route_recent_vs_typical": out["route_transit_90d_at_handover"] / out["route_typical_days"]}
    for name, values in ratios.items():
        out[name] = values.replace([np.inf, -np.inf], np.nan).clip(-CLIP, CLIP)
    return out


def variant(data, replace):
    spec_v = copy.deepcopy(data["spec_base"])
    kept = [f for f in spec_v["numeric_features"] if not (replace and f in REPLACED)]
    spec_v["numeric_features"] = kept + RELATIVE
    return spec_v


def fit_scores(data, spec_v, fit_rows, rows):
    columns = sum(feature_columns(spec_v), [])
    pipeline = make_pipeline_for(data, spec_v)
    pipeline.fit(fit_rows[columns], fit_rows[LABEL])
    return pipeline, pipeline.predict_proba(rows[columns])[:, 1]


def run():
    data = load_data(add_relative_inputs)
    print(data["train"][RELATIVE].describe().round(3).to_string(), flush=True)
    specs = {CANDIDATE: variant(data, True), ADDED: variant(data, False)}
    windows = validation_rows(data["train"], data["folds"])
    valid_scores = {name: [fit_scores(data, spec_v, fit_rows, rows)[1] for fit_rows, rows in windows] for name, spec_v in specs.items()}
    pipeline, test_scores = fit_scores(data, specs[CANDIDATE], data["train"], data["test"])
    joblib.dump(pipeline, OUT / f"{CANDIDATE}.joblib")
    evaluate(OUT, data, valid_scores, {CANDIDATE: test_scores}, {CANDIDATE: "relative_replaced"})


if __name__ == "__main__":
    run()
