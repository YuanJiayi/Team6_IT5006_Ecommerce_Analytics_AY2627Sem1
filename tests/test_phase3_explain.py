"""Exact linear contribution checks for the late-risk champion.

Run from the repository root:
    python -m pytest tests/test_phase3_explain.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist.classification import feature_columns, make_pipeline  # noqa: E402
from olist.paths import PROCESSED_DIR  # noqa: E402
from phase3 import explain  # noqa: E402


def training_frame(spec, size=320, seed=7):
    rng = np.random.default_rng(seed)
    data = {}
    for column in spec["numeric_features"]:
        data[column] = rng.normal(12.0, 3.0, size)
    data["same_state"] = rng.integers(0, 2, size)
    data["purchase_hour"] = rng.integers(0, 24, size)
    categories = {
        "customer_state": ["SP", "RJ", "MG", "BA", "RS"],
        "seller_state": ["SP", "PR", "SC", "GO"],
        "product_category": ["bed_bath_table", "health_beauty", "sports_leisure"],
        "payment_type": ["credit_card", "boleto", "voucher"],
        "purchase_dayofweek": ["Mon", "Tue", "Wed", "Thu", "Fri"],
    }
    for column in spec["categorical_features"]:
        data[column] = rng.choice(categories[column], size)
    frame = pd.DataFrame(data)
    signal = (0.35 * frame["promised_days"] - 0.18 * frame["route_typical_days"]
              + 0.22 * frame["seller_ship_days"] + 0.03 * frame["distance_km"]
              - 1.1 * frame["same_state"])
    probability = 1.0 / (1.0 + np.exp(-(signal - signal.mean())))
    frame["is_late"] = (rng.random(size) < probability).astype(int)
    if frame["is_late"].nunique() != 2:
        frame.loc[frame.index[:2], "is_late"] = [0, 1]
    return frame


class LinearExplainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((PROCESSED_DIR / "phase2_feature_spec.json").read_text())
        cls.frame = training_frame(cls.spec)
        cls.columns = sum(feature_columns(cls.spec), [])
        cls.pipeline = make_pipeline(
            "linear", {"C": 1.0}, cls.spec,
            log_features=cls.spec["linear_log_candidates"],
        )
        cls.pipeline.fit(cls.frame[cls.columns], cls.frame["is_late"])
        cls.explainer = explain.LinearExplainer(cls.pipeline, cls.columns, cls.frame)

    def test_contributions_are_additive_on_the_logit_scale(self):
        rows = self.frame.iloc[:20]
        contributions, logits = self.explainer.explain(rows)
        combined = self.explainer.base_value + contributions.sum(axis=1).to_numpy()
        expected = self.pipeline.decision_function(rows[self.columns])
        np.testing.assert_allclose(combined, expected, atol=1e-9)
        np.testing.assert_allclose(logits, expected, atol=1e-9)

    def test_contributions_aggregate_to_original_inputs(self):
        contributions, _ = self.explainer.explain(self.frame.iloc[:5])
        self.assertEqual(set(contributions.columns), set(self.columns))
        self.assertIn("purchase_hour", contributions.columns)
        self.assertIn("product_category", contributions.columns)
        self.assertNotIn("hour_sin", contributions.columns)

    def test_top_inputs_are_positive_and_limited(self):
        contributions, _ = self.explainer.explain(self.frame.iloc[:5])
        for position in range(len(contributions)):
            top = self.explainer.top(contributions.iloc[position])
            self.assertLessEqual(len(top), 3)
            self.assertTrue(all(item["contribution"] > 0 for item in top))
            self.assertTrue(all(item["input"] in contributions.columns for item in top))
            values = [item["contribution"] for item in top]
            self.assertEqual(values, sorted(values, reverse=True))

    def test_input_labels_are_plain_language(self):
        self.assertEqual(explain.input_label("promise_slack"), "Promise slack (days)")
        self.assertEqual(explain.input_label("remaining_slack"), "Spare time")
        self.assertEqual(explain.input_label("slack_shortfall"), "Spare time below zero")
        self.assertEqual(explain.input_label("past_promise_at_handover"), "Already past promise")
        self.assertEqual(explain.input_label("handover_days"), "Days from purchase to handover")
        self.assertEqual(explain.input_label("state_backlog_recent_vs_8wk"), "Regional backlog")
        self.assertNotIn("slack", explain.input_label("slack_shortfall").lower())

    def test_every_model_input_has_an_explicit_label(self):
        for column in self.columns:
            self.assertIn(
                column,
                explain.INPUT_LABELS,
                "{} has no explicit human label".format(column),
            )

    def test_non_linear_champion_fails_loudly(self):
        dummy = make_pipeline("dummy", {}, self.spec)
        dummy.fit(self.frame[self.columns], self.frame["is_late"])
        with self.assertRaises(ValueError):
            explain.LinearExplainer(dummy, self.columns, self.frame)


if __name__ == "__main__":
    unittest.main()
