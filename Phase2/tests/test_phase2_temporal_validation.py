"""Check that saved validation folds respect the primary evaluation timeline."""

import json
import unittest
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class Phase2TemporalValidationTest(unittest.TestCase):
    def test_folds_have_mature_validation_windows_and_as_of_training(self):
        spec = json.loads((DATA_DIR / "phase2_feature_spec.json").read_text())
        table = pd.read_csv(
            DATA_DIR / "phase2_order_table.csv",
            usecols=["order_purchase_timestamp", "outcome_available_at", "split", "cv_fold"],
            parse_dates=["order_purchase_timestamp", "outcome_available_at"],
        )
        train = table.loc[table["split"].eq("train")]
        validation_end = pd.Timestamp(spec["primary_cutoff"]) - pd.Timedelta(
            days=spec["validation_maturity_days"]
        )

        self.assertTrue(train.loc[train["cv_fold"].ge(0), "order_purchase_timestamp"].lt(validation_end).all())
        self.assertTrue(table.loc[~table["split"].eq("train"), "cv_fold"].isna().all())
        for fold in range(spec["n_folds"]):
            validation = train.loc[train["cv_fold"].eq(fold)]
            self.assertGreater(len(validation), 0)
            start = validation["order_purchase_timestamp"].min()
            fit = train.loc[
                train["order_purchase_timestamp"].lt(start)
                & train["outcome_available_at"].lt(start)
            ]
            self.assertGreater(len(fit), 0)
            self.assertTrue(fit["outcome_available_at"].lt(start).all())


if __name__ == "__main__":
    unittest.main()
