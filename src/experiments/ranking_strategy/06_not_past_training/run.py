"""Train the late-warning model only on orders not past their promised date at handover.

Run from the repository root:
    python src/experiments/ranking_strategy/06_not_past_training/run.py

Orders already past their date always rank above all others, so the model only has to order those still in contention.
Writes the candidate model and CSVs next to this file.
"""

from __future__ import annotations

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

CANDIDATE = "not_past_training"
PAST_SCORE = 2.0  # above any probability


def fit_scores(data, fit_rows, rows):
    columns = sum(feature_columns(data["spec_base"]), [])
    keep = fit_rows[fit_rows["past_promise_at_handover"].ne(1)]
    pipeline = make_pipeline_for(data)
    pipeline.fit(keep[columns], keep[LABEL])
    return pipeline, np.where(rows["past_promise_at_handover"].eq(1), PAST_SCORE, pipeline.predict_proba(rows[columns])[:, 1])


def run():
    data = load_data()
    valid_scores = {CANDIDATE: [fit_scores(data, fit_rows, rows)[1] for fit_rows, rows in validation_rows(data["train"], data["folds"])]}
    pipeline, test_scores = fit_scores(data, data["train"], data["test"])
    joblib.dump(pipeline, OUT / f"{CANDIDATE}.joblib")
    evaluate(OUT, data, valid_scores, {CANDIDATE: test_scores}, {CANDIDATE: "not_past_training"})


if __name__ == "__main__":
    run()
