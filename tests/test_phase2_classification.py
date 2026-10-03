"""Checks for fold boundaries, preprocessing isolation, and alert threshold selection."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from phase2_classification import (
    feature_columns, feature_stages, load_primary_training, make_pipeline, mcc_threshold, metrics, risk_ranking,
    temporal_folds,
)


class ClassificationWorkflowTest(unittest.TestCase):
    def test_threshold_matches_brute_force_with_tied_scores(self):
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        probability = np.array([0.2, 0.8, 0.2, 0.7, 0.7, 0.7, 0.1, 0.3])
        choices = [np.nextafter(probability.max(), np.inf), *sorted(set(probability), reverse=True)]
        expected = max(choices, key=lambda t: matthews_corrcoef(y, probability >= t))
        self.assertEqual(mcc_threshold(y, probability), expected)
        constant = np.full(len(y), 0.4)
        self.assertGreater(mcc_threshold(y, constant), 0.4)

    def test_payment_sensitivity_and_fold_fitted_preparation(self):
        spec = {
            "numeric_features": ["distance_km", "purchase_hour", "max_installments"],
            "categorical_features": ["customer_state", "payment_type"],
            "payment_sensitivity_features": ["payment_type", "max_installments"],
            "random_state": 42,
        }
        training = pd.DataFrame({
            "distance_km": [10.0, 20.0, np.nan, 30.0], "purchase_hour": [0, 6, 12, 18],
            "max_installments": [1, 2, 1, 2], "customer_state": ["SP", "SP", "RJ", "RJ"],
            "payment_type": ["credit_card", "boleto", "credit_card", "boleto"],
        })
        validation = training.iloc[[0, 1]].copy()
        validation["distance_km"] = [100000.0, np.nan]
        validation["customer_state"] = "unseen"
        pipeline = make_pipeline("linear", {"C": 1.0}, spec, include_payment=False)
        pipeline.fit(training, [0, 1, 0, 1])
        imputer = pipeline.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
        self.assertEqual(imputer.statistics_.tolist(), [20.0])
        probability = pipeline.predict_proba(validation)
        self.assertTrue(np.isfinite(probability).all())
        self.assertEqual(imputer.statistics_.tolist(), [20.0])
        numeric, categorical = feature_columns(spec, False)
        self.assertEqual(numeric, ["distance_km", "purchase_hour"])
        self.assertEqual(categorical, ["customer_state"])

    def test_log_branch_is_fitted_on_training_rows_and_only_for_linear_models(self):
        spec = {
            "numeric_features": ["total_price", "purchase_hour"], "categorical_features": ["customer_state"],
            "payment_sensitivity_features": [], "random_state": 42,
        }
        training = pd.DataFrame({"total_price": [10.0, 100.0, np.nan, 1000.0], "purchase_hour": [1, 5, 9, 13],
                                 "customer_state": ["SP", "SP", "RJ", "RJ"]})
        validation = pd.DataFrame({"total_price": [np.nan, 1e6], "purchase_hour": [0, 23], "customer_state": ["x", "SP"]})
        pipeline = make_pipeline("linear", {"C": 1.0}, spec, log_features=["total_price"])
        pipeline.fit(training, [0, 1, 0, 1])
        logged = pipeline.named_steps["prepare"].named_transformers_["logged"]
        self.assertEqual(logged.named_steps["impute"].statistics_.tolist(), [100.0])  # median of training rows only
        self.assertTrue(np.isfinite(pipeline.predict_proba(validation)).all())
        names = pipeline.named_steps["prepare"].get_feature_names_out().tolist()
        self.assertIn("logged__total_price", names)
        self.assertNotIn("numeric__total_price", names)
        forest = make_pipeline("forest", {"max_depth": 2}, spec, log_features=["total_price"])
        self.assertNotIn("logged", [name for name, _, _ in forest.named_steps["prepare"].transformers])

    def test_feature_stages_build_cumulatively_to_the_saved_contract(self):
        _, spec = load_primary_training()
        stages = feature_stages(spec)
        self.assertIn("purchase_month", stages["original"]["categorical"])
        self.assertNotIn("purchase_month", stages["no_month"]["categorical"])
        self.assertTrue(set(spec["history_features"]["features"]).isdisjoint(stages["no_month"]["numeric"]))
        self.assertTrue({"route_typical_days", "promise_slack"}.issubset(stages["route"]["numeric"]))
        self.assertNotIn("seller_ship_days", stages["route"]["numeric"])
        self.assertEqual(stages["seller"], {"numeric": spec["numeric_features"],
                                            "categorical": spec["categorical_features"]})

    def test_risk_ranking_counts_and_lift(self):
        predictions = pd.DataFrame({
            "fold": [0] * 10, "is_late": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "probability": [0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05]})
        top = risk_ranking(predictions, shares=(0.2,)).query("scope == 'all_folds'").iloc[0]
        self.assertEqual((top["flagged"], top["caught"], top["late"]), (2, 1, 2))  # riskiest two: 0.9 (late), 0.8 (not)
        self.assertAlmostEqual(top["recall"], 0.5)
        self.assertAlmostEqual(top["lift"], 0.5 / 0.2)

    def test_actual_fold_indices_exclude_unavailable_labels_and_holdout(self):
        train, spec = load_primary_training()
        self.assertTrue(train["split"].eq("train").all())
        folds = temporal_folds(train, spec)
        seen = set()
        for fit, valid in folds:
            start = train.iloc[valid]["order_purchase_timestamp"].min()
            self.assertTrue(train.iloc[fit]["order_purchase_timestamp"].lt(start).all())
            self.assertTrue(train.iloc[fit]["outcome_available_at"].lt(start).all())
            self.assertFalse(set(valid) & seen)
            seen.update(valid)
            unavailable = train.index[train["order_purchase_timestamp"].lt(start)
                                      & train["outcome_available_at"].ge(start)]
            self.assertFalse(set(fit) & set(unavailable))
        self.assertEqual(len(seen), int(train["cv_fold"].ge(0).sum()))

    def test_saved_predictions_and_threshold_reproduce_without_holdout_rows(self):
        result = Path(__file__).resolve().parents[1] / "results/phase2/classification"
        selection = json.loads((result / "selection.json").read_text())
        oof = pd.read_csv(result / "selected_validation_predictions.csv", float_precision="round_trip")
        recorded = pd.read_csv(result / "threshold_metrics.csv", float_precision="round_trip").iloc[1]
        train, spec = load_primary_training()
        eligible = train.loc[train[spec["primary_fold"]].ge(0)]
        self.assertEqual(set(oof["order_id"]), set(eligible["order_id"]))
        self.assertTrue(oof["order_id"].is_unique)
        self.assertTrue(oof["probability"].between(0, 1).all())
        self.assertFalse(selection["primary_test_evaluated"])
        actual = metrics(oof["is_late"], oof["probability"], selection["selected_threshold"])
        for name, value in actual.items():
            self.assertAlmostEqual(value, recorded[name], places=12)
        self.assertTrue(set(selection["feature_stages"]["seller"]["numeric"]).issuperset(
            spec["history_features"]["features"]))
        self.assertIn("log_transform", selection)
        ladder = pd.read_csv(result / "ladder_summary.csv")
        self.assertEqual(set(ladder["step"]), set(range(1, 13)))
        self.assertTrue(ladder.loc[ladder["step"].eq(1), "mean_average_precision"].iloc[0]
                        < ladder["mean_average_precision"].max())
        risk = pd.read_csv(result / "risk_ranking.csv").query("scope == 'all_folds'")
        self.assertTrue((risk["orders"] == len(oof)).all())
        self.assertTrue((risk["lift"] > 1).all())
        by_fold = pd.read_csv(result / "selected_threshold_by_fold.csv")
        for name in ("true_negative", "false_positive", "false_negative", "true_positive"):
            self.assertEqual(by_fold[name].sum(), recorded[name])
        joined = oof.merge(eligible[["order_id", "is_late", "cv_fold"]], on="order_id", validate="one_to_one")
        self.assertTrue(joined["is_late_x"].eq(joined["is_late_y"]).all())
        self.assertTrue(joined["fold"].eq(joined["cv_fold"]).all())


if __name__ == "__main__":
    unittest.main()
