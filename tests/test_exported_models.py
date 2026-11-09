"""The saved deployment models load and predict from raw input columns, as a caller would."""

import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from olist.paths import MODELS_DIR  # noqa: E402


class ExportedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())
        cls.stats = json.loads((MODELS_DIR / "feature_stats.json").read_text())

    def test_saved_with_this_sklearn_version(self):
        self.assertEqual(self.metadata["environment"]["sklearn"], sklearn.__version__)

    def test_each_model_predicts_its_example_input(self):
        for name, info in self.metadata["models"].items():
            with self.subTest(model=name):
                pipeline = joblib.load(MODELS_DIR / info["file"])
                row = pd.DataFrame([info["example_input"]])[info["feature_columns"]].astype(
                    {c: float for c in info["numeric_columns"]})
                if info["task"] == "regression":
                    days = pipeline.predict(row)[0]
                    self.assertTrue(np.isfinite(days) and 0 < days < 200)
                else:
                    probability = pipeline.predict_proba(row)[0, 1]
                    self.assertTrue(0.0 <= probability <= 1.0)

    def test_unknown_category_and_missing_history_are_handled(self):
        info = self.metadata["models"]["late_risk_handover"]
        pipeline = joblib.load(MODELS_DIR / info["file"])
        row = pd.DataFrame([info["example_input"]])[info["feature_columns"]].astype(
            {c: float for c in info["numeric_columns"]})
        row.loc[0, "product_category"] = "not_a_category"
        row.loc[0, "route_transit_90d_at_handover"] = np.nan
        self.assertTrue(np.isfinite(pipeline.predict_proba(row)[0, 1]))

    def test_feature_stats_cover_every_numeric_input(self):
        numeric = {c for info in self.metadata["models"].values() for c in info["numeric_columns"]}
        self.assertEqual(numeric, set(self.stats))
        for column, described in self.stats.items():
            self.assertIn("mean", described, column)
            self.assertIn("std", described, column)


if __name__ == "__main__":
    unittest.main()
