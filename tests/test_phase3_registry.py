"""End-to-end checks that the local MLflow registry reproduces the Phase 2 models."""

import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist.classification import feature_columns  # noqa: E402
from olist.eta import CANDIDATES as ETA_CANDIDATES  # noqa: E402
from olist.eta import load_stage_table, stage_spec  # noqa: E402
from olist.handover_classifier import (  # noqa: E402
    LABEL, action_metrics, baseline_scores, load_cohort, monthly_flags, variant_spec)
from olist.regression import regression_metrics  # noqa: E402
from phase3 import drift as drift_module  # noqa: E402
from phase3 import registry  # noqa: E402

TOLERANCE = 1e-9


class PairedWeekGapTests(unittest.TestCase):
    def test_mae_level_when_the_pooled_gap_interval_contains_zero(self):
        weeks = np.repeat(np.arange(4), 10)
        y = np.zeros(40)
        pred_c = np.ones(40)
        pred_r = np.concatenate([np.full(20, 1.2), np.full(20, 0.8)])
        result = registry.paired_week_gap(y, pred_c, pred_r, weeks, "mae")
        self.assertAlmostEqual(result["gap"], 0.0, places=12)
        self.assertLess(result["low"], result["high"])
        self.assertEqual(result["verdict"], "level")

    def test_mae_clear_better_and_worse(self):
        weeks = np.repeat(np.arange(4), 6)
        y = np.zeros(24)
        pred_c = np.zeros(24)
        pred_r = np.array([0.5, 1.0, 1.5, 2.0]).repeat(6)
        better = registry.paired_week_gap(y, pred_c, pred_r, weeks, "mae")
        worse = registry.paired_week_gap(y, pred_r, pred_c, weeks, "mae")
        self.assertAlmostEqual(better["gap"], 1.25, places=12)
        self.assertLess(better["low"], better["high"])
        self.assertGreater(better["low"], 0.0)
        self.assertEqual(better["verdict"], "better")
        self.assertAlmostEqual(worse["gap"], -1.25, places=12)
        self.assertLess(worse["low"], worse["high"])
        self.assertLess(worse["high"], 0.0)
        self.assertEqual(worse["verdict"], "worse")

    def test_pr_auc_clear_better(self):
        weeks = np.repeat(np.arange(4), 8)
        y = np.concatenate([np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                            np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
        pred_c = 0.1 + y
        pred_r = 0.1 + (1.0 - y)
        result = registry.paired_week_gap(y, pred_c, pred_r, weeks, "pr_auc")
        self.assertGreater(result["gap"], 0.0)
        self.assertLess(result["low"], result["high"])
        self.assertGreater(result["low"], 0.0)
        self.assertEqual(result["verdict"], "better")

    def test_interval_is_the_percentile_of_seeded_bootstrap_gaps(self):
        weeks = np.repeat(np.arange(6), 5)
        y = np.linspace(0.0, 3.0, 30)
        pred_c = np.linspace(0.2, 2.4, 30)
        pred_r = np.linspace(0.9, 3.6, 30)
        result = registry.paired_week_gap(y, pred_c, pred_r, weeks, "mae")
        rng = np.random.default_rng(42)
        draws = rng.integers(0, 6, size=(registry.BOOTSTRAP_RESAMPLES, 6))
        error_c = np.abs(y - pred_c)
        error_r = np.abs(y - pred_r)
        groups = [np.flatnonzero(weeks == index) for index in range(6)]
        sums_c = np.array([error_c[group].sum() for group in groups])
        sums_r = np.array([error_r[group].sum() for group in groups])
        counts = np.array([len(group) for group in groups])
        sampled = counts[draws].sum(axis=1)
        gaps = (sums_r[draws].sum(axis=1) / sampled
                - sums_c[draws].sum(axis=1) / sampled)
        low, high = np.percentile(gaps, [2.5, 97.5])
        self.assertAlmostEqual(result["low"], low, places=12)
        self.assertAlmostEqual(result["high"], high, places=12)
        self.assertLess(result["low"], result["high"])

    def test_verdict_follows_the_interval_sign(self):
        self.assertEqual(registry._gap_verdict(-0.13, 0.20), "level")
        self.assertEqual(registry._gap_verdict(0.01, 0.30), "better")
        self.assertEqual(registry._gap_verdict(-0.30, -0.01), "worse")
        self.assertEqual(registry._gap_verdict(0.0, 0.13), "level")
        self.assertEqual(registry._gap_verdict(-0.13, 0.0), "level")

    def test_seed_makes_the_bootstrap_deterministic(self):
        weeks = np.repeat(np.arange(5), 8)
        y = np.linspace(0.0, 4.0, 40)
        pred_c = np.linspace(0.4, 3.2, 40)
        pred_r = np.linspace(1.1, 4.4, 40)
        first = registry.paired_week_gap(y, pred_c, pred_r, weeks, "mae")
        second = registry.paired_week_gap(y, pred_c, pred_r, weeks, "mae")
        self.assertEqual(first, second)


class ValidationImportanceTests(unittest.TestCase):
    def test_each_validation_row_is_scored_by_its_own_window_model(self):
        class Constant:
            def __init__(self, value):
                self.value = value

            def predict(self, frame):
                return np.full(len(frame), self.value, dtype=float)

        frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        result = registry._fold_predictions(
            [Constant(1.0), Constant(2.0)], np.array([0, 1, 0, 1]), frame, "regression")
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 1.0, 2.0]))


class ValidationReproductionGuardTests(unittest.TestCase):
    def test_guard_accepts_a_reproduced_score(self):
        registry._check_metric("rung window 0 pr_auc", 0.5, 0.5 + 1e-12)

    def test_guard_rejects_a_mismatched_score(self):
        with self.assertRaises(AssertionError):
            registry._check_metric("rung window 0 pr_auc", 0.5, 0.5001)


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        cls.store = root / "store"
        cls.results = root / "results"
        cls.summary = registry.run(store=cls.store, results=cls.results)
        cls.uri = f"sqlite:///{(cls.store / 'mlflow.db').resolve()}"
        cls.client = MlflowClient(tracking_uri=cls.uri, registry_uri=cls.uri)
        mlflow.set_tracking_uri(cls.uri)
        mlflow.set_registry_uri(cls.uri)
        cls.addClassCleanup(cls._temporary.cleanup)

    def test_three_registered_models_with_champion_aliases(self):
        registered = {model.name for model in self.client.search_registered_models()}
        self.assertEqual(registered, set(registry.MODELS))
        for name in registry.MODELS:
            versions = self.client.search_model_versions(f"name = '{name}'")
            if name == "late_risk":
                # The evaluation model plus one servable refit per month, May-October.
                self.assertEqual(sorted(int(version.version) for version in versions),
                                 list(range(1, 8)))
                monthly = [version for version in versions if version.tags.get("source") == "monthly_refit"]
                self.assertEqual(sorted(version.tags["refit_date"] for version in monthly),
                                 ["2018-05-01", "2018-06-01", "2018-07-01",
                                  "2018-08-01", "2018-09-01", "2018-10-01"])
                for version in monthly:
                    self.assertGreater(int(version.tags["fit_rows"]), 0)
                    self.assertEqual(version.tags["candidate"], self.summary[name]["champion_candidate"])
            else:
                self.assertEqual([int(version.version) for version in versions], [1])
            champion = self.client.get_model_version_by_alias(name, "champion")
            self.assertEqual(int(champion.version), 1)
            self.assertEqual(champion.tags["source"], "phase2_selection")
            self.assertEqual(champion.tags["candidate"], self.summary[name]["champion_candidate"])

    def test_every_run_has_a_plain_english_note_and_the_final_is_named_for_the_chosen_model(self):
        for name in registry.MODELS:
            experiment = self.client.get_experiment_by_name(name)
            runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
            self.assertTrue(runs, name)
            for run in runs:
                note = run.data.tags.get(registry.NOTE_TAG)
                self.assertTrue(note, f"{name} {run.data.tags.get('run_kind')} has no note")
                self.assertEqual(note.count("\n"), 0, note)
            final = next(run for run in runs if run.data.tags.get("run_kind") == "final")
            self.assertEqual(final.data.tags.get("mlflow.runName"),
                             f"chosen model \u2013 {self.summary[name]['champion_candidate']}")

    def test_eta_models_reproduce_saved_test_metrics(self):
        table, spec, _ = load_stage_table(ROOT)
        test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
        saved = pd.read_csv(registry.ETA_RESULTS / "test_metrics.csv")
        month = test["order_purchase_timestamp"].dt.to_period("M").astype(str)
        for stage, name in (("checkout", "checkout_eta"), ("handover", "handover_eta")):
            champion = self.summary[name]["champion_candidate"]
            model = mlflow.sklearn.load_model(f"models:/{name}@champion")
            columns = sum(feature_columns(stage_spec(spec, stage)), [])
            predicted = model.predict(test[columns])
            live = {"overall": regression_metrics(test["delivery_days"], predicted)}
            for value, index in test.groupby(month).groups.items():
                live[value] = regression_metrics(test.loc[index, "delivery_days"], predicted[index])
            expected = saved[(saved["stage"] == stage) & (saved["candidate"] == champion)]
            for scope, values in live.items():
                row = expected[expected["scope"] == scope].iloc[0]
                for field in registry.REGRESSION_FIELDS:
                    self.assertAlmostEqual(float(values[field]), float(row[field]), delta=TOLERANCE,
                                           msg=f"{name} {scope} {field}")
            self.assertAlmostEqual(live["overall"]["mae"], 5.114008700075504 if stage == "checkout"
                                   else 4.199441288329846, delta=TOLERANCE)

    def test_late_risk_model_and_slack_rule_reproduce_saved_test_metrics(self):
        table, spec, _ = load_cohort(ROOT)
        train = table[table[spec["primary_split"]].eq("train")]
        test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
        spec_v = variant_spec(spec)
        columns = sum(feature_columns(spec_v), [])
        model = mlflow.sklearn.load_model("models:/late_risk@champion")
        scores = model.predict_proba(test[columns])[:, 1]

        eta_selection = json.loads(registry.ETA_SELECTION.read_text())
        eta_handover = eta_selection["selected_model"]["handover"]["candidate"]
        regression_family, regression_params = next(
            (family, params) for candidate, family, params in ETA_CANDIDATES
            if candidate == eta_handover)
        base = baseline_scores(train, test, stage_spec(spec, "handover"), regression_family, regression_params)
        y = test[LABEL].to_numpy()
        months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
        saved = pd.read_csv(registry.HANDOVER_RESULTS / "test_metrics.csv")
        for ranking, values in (("model", scores), ("slack_rule", base["slack_rule"])):
            row = saved[(saved["variant"] == "primary") & (saved["ranking"] == ranking)].iloc[0]
            flags = monthly_flags(values, months, float(row["k"]))  # the share Phase 2 saved its test metrics at
            live = {**action_metrics(y, flags),
                    "roc_auc": float(roc_auc_score(y, values)),
                    "pr_auc": float(average_precision_score(y, values))}
            for field in registry.CLASSIFIER_FIELDS:
                self.assertAlmostEqual(float(live[field]), float(row[field]), delta=TOLERANCE,
                                       msg=f"late_risk {ranking} {field}")
        self.assertAlmostEqual(
            self.summary["late_risk"]["test_metrics"]["model"]["pr_auc"], 0.4886741903468081, delta=TOLERANCE)
        self.assertAlmostEqual(
            self.summary["late_risk"]["test_metrics"]["slack_rule"]["pr_auc"], 0.5193804905926744, delta=TOLERANCE)
        self.assertAlmostEqual(
            self.summary["late_risk"]["test_metrics"]["blend_rule"]["pr_auc"], 0.5219604493187078, delta=TOLERANCE)
        self.assertEqual(self.summary["late_risk"]["test_counts"]["model_caught_top10"], 377)
        self.assertEqual(self.summary["late_risk"]["test_counts"]["blend_caught_top10"], 411)
        daily = self.summary["late_risk"]["test_daily"]
        self.assertAlmostEqual(daily["blend_mean_warning_days"], 6.734375, places=6)
        self.assertAlmostEqual(daily["slack_mean_warning_days"], 4.371666666666667, places=6)
        self.assertEqual(daily["blend_caught"], 256)
        self.assertEqual(daily["blend_caught_3plus"], 152)

    def test_benchmarks_match_the_phase2_reference_rules(self):
        cv = pd.read_csv(registry.ETA_RESULTS / "cv_summary.csv")
        checkout = self.summary["checkout_eta"]["benchmark"]
        handover = self.summary["handover_eta"]["benchmark"]
        late = self.summary["late_risk"]["benchmark"]

        self.assertEqual(checkout["name"], "route_typical_days")
        self.assertEqual(checkout["label"], "Route's average so far")
        self.assertEqual(handover["name"], "elapsed_plus_route_baseline")
        self.assertEqual(late["name"], "slack_rule")
        for benchmark in (checkout, handover):
            self.assertTrue(math.isfinite(benchmark["test"]["mae"]))
            self.assertTrue(math.isfinite(benchmark["validation"]["mean"]["mae"]))
            self.assertEqual(len(benchmark["validation"]["windows"]), 5)
            self.assertTrue(all(math.isfinite(row["mae"]) for row in benchmark["validation"]["windows"]))
        self.assertLessEqual(abs(checkout["test"]["mae"] - 5.21), 0.2)

        saved = cv[(cv["stage"] == "handover")
                   & (cv["candidate"] == "elapsed_plus_route_baseline")].iloc[0]
        self.assertAlmostEqual(handover["validation"]["mean"]["mae"], float(saved["mean_mae"]),
                               delta=TOLERANCE)

        windows = [row["pr_auc"] for row in late["validation"]["windows"]]
        self.assertEqual(len(windows), 5)
        self.assertTrue(all(math.isfinite(value) for value in windows))
        self.assertAlmostEqual(late["validation"]["mean"]["pr_auc"], sum(windows) / len(windows),
                               delta=TOLERANCE)
        self.assertAlmostEqual(late["test"]["pr_auc"], 0.5193804905926744, places=3)

        for name, expected, metric in (
                ("checkout_eta", checkout["test"]["mae"], "reference_route_typical_days_test_mae"),
                ("handover_eta", handover["test"]["mae"],
                 "reference_elapsed_plus_route_baseline_test_mae"),
                ("late_risk", late["validation"]["mean"]["pr_auc"],
                 "reference_slack_rule_validation_mean_pr_auc")):
            experiment = self.client.get_experiment_by_name(name)
            runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
            final = next(run for run in runs if run.data.tags.get("run_kind") == "final")
            self.assertAlmostEqual(final.data.metrics[metric], expected, delta=TOLERANCE)
        experiment = self.client.get_experiment_by_name("late_risk")
        runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
        final = next(run for run in runs if run.data.tags.get("run_kind") == "final")
        self.assertEqual(
            len(self.client.get_metric_history(
                final.info.run_id, "reference_slack_rule_validation_window_pr_auc")), 5)

    def test_reference_rules_cover_the_expected_wording_and_numbers(self):
        checkout = self.summary["checkout_eta"]
        handover = self.summary["handover_eta"]
        late = self.summary["late_risk"]
        self.assertEqual([row["key"] for row in checkout["references"]],
                         ["route_typical_days", "mean_baseline"])
        self.assertEqual([row["key"] for row in handover["references"]],
                         ["elapsed_plus_route_baseline", "mean_baseline"])
        self.assertEqual([row["key"] for row in late["references"]],
                         ["deadline_rule", "slack_rule", "blend_rule", "random_rule"])
        roles = {row["key"]: row["role"] for row in late["references"]}
        self.assertEqual(roles["slack_rule"], "headline")
        self.assertEqual(roles["deadline_rule"], "context")
        self.assertEqual(roles["blend_rule"], "context")
        self.assertEqual(roles["random_rule"], "floor")
        for row in checkout["references"] + handover["references"] + late["references"]:
            self.assertTrue(row["label"] and row["short"] and row["hover"], row["key"])
            self.assertEqual(len(row["validation"]["windows"]), 5, row["key"])

        by_key = {row["key"]: row for row in late["references"]}
        self.assertAlmostEqual(by_key["deadline_rule"]["test"]["pr_auc"], 0.48693453580031665, places=3)
        self.assertAlmostEqual(by_key["slack_rule"]["test"]["pr_auc"], 0.5193804905926744, places=3)
        self.assertAlmostEqual(by_key["blend_rule"]["test"]["pr_auc"], 0.5219604493187078, places=3)
        self.assertAlmostEqual(by_key["random_rule"]["test"]["pr_auc"], 0.03494539781591264, places=3)
        self.assertAlmostEqual(by_key["blend_rule"]["validation"]["mean"]["pr_auc"], 0.383056, places=3)
        self.assertAlmostEqual(by_key["random_rule"]["validation"]["mean"]["pr_auc"], 0.1067, places=3)

        checkout_floor = checkout["references"][1]
        self.assertAlmostEqual(checkout_floor["validation"]["mean"]["mae"], 7.214693687035944, places=6)
        self.assertAlmostEqual(checkout_floor["test"]["mae"], 6.2673, places=3)
        self.assertAlmostEqual(handover["references"][0]["test"]["mae"], 3.7617, places=3)
        self.assertEqual(checkout["references"][0]["short"],
                         "Operations rule: the route's average delivery time.")
        self.assertEqual(handover["references"][0]["short"],
                         "Operations rule: days so far + the route's average carrier time.")
        self.assertEqual(late["references"][0]["short"],
                         "Operations rule: parcels with the fewest days left go first.")
        self.assertEqual(by_key["slack_rule"]["label"], "Remaining slack")
        self.assertEqual(by_key["slack_rule"]["short"],
                         "Operations rule: parcels with the least remaining slack go first.")
        for key in ("deadline_rule", "blend_rule"):
            self.assertIn("test_gap_vs_headline", by_key[key])
            self.assertIn("test_gap_low_vs_headline", by_key[key])
            self.assertIn("test_gap_high_vs_headline", by_key[key])
            self.assertIn(by_key[key]["test_gap_verdict"], {"better", "level", "worse"})
        for key in ("slack_rule", "random_rule"):
            self.assertNotIn("test_gap_vs_headline", by_key[key])
        self.assertAlmostEqual(by_key["blend_rule"]["test_gap_vs_headline"], 0.0026, places=3)
        self.assertEqual(by_key["blend_rule"]["test_gap_verdict"], "level")
        champion_rung = next(row for rows in late["ladders"].values() for row in rows
                             if row["candidate"] == late["champion_candidate"])
        self.assertAlmostEqual(champion_rung["test_gap_vs_headline"], -0.0307, places=3)
        self.assertEqual(champion_rung["test_gap_verdict"], "worse")

    def test_late_risk_benchmark_follows_the_headline_role(self):
        late = self.summary["late_risk"]
        roles = {row["key"]: row["role"] for row in late["references"]}
        headline = next(row for row in late["references"] if row["role"] == "headline")
        self.assertEqual(headline["key"], "slack_rule")
        self.assertEqual(late["benchmark"]["name"], headline["key"])
        self.assertEqual(late["benchmark"]["test"], headline["test"])
        self.assertEqual(roles["deadline_rule"], "context")
        self.assertEqual(roles["blend_rule"], "context")

    def test_ladder_shared_view_state_points_at_the_ladder_runs(self):
        cases = {
            "checkout_eta": ("mae", True),
            "handover_eta": ("mae", True),
            "late_risk": ("pr_auc", False),
        }
        for name, (metric, ascending) in cases.items():
            experiment = self.client.get_experiment_by_name(name)
            saved = json.loads(experiment.tags[registry.SHARED_VIEW_STATE_TAG])
            self.assertEqual(saved, {
                "searchFilter": "tags.run_kind = 'candidate'",
                "orderByKey": f"metrics.`validation_mean_{metric}`",
                "orderByAsc": ascending,
                "selectedColumns": [
                    f"metrics.`validation_mean_{metric}`",
                    f"metrics.`test_{metric}`",
                    "metrics.`gap_to_best`",
                    "metrics.`se_of_gap`",
                    "metrics.`within_one_se`",
                ],
            }, name)

    def test_classifier_ladder_validation_mcc_matches_mlflow(self):
        late = self.summary["late_risk"]
        runs = {(run.data.tags["ladder_family"], int(run.data.tags["ladder_rung"])): run
                for run in self._ladder_runs("late_risk")}
        for family, rows in late["ladders"].items():
            for row in rows:
                run = runs[(family, row["rung"])]
                history = self.client.get_metric_history(
                    run.info.run_id, "validation_window_mcc_top10pct")
                self.assertEqual(len(history), 5, row["candidate"])
                values = {point.step: float(point.value) for point in history}
                for window in row["validation"]["windows"]:
                    self.assertAlmostEqual(window["mcc"], values[window["window"]],
                                           delta=1e-12, msg=row["candidate"])
                self.assertAlmostEqual(row["validation"]["mean"]["mcc"],
                                       statistics.mean(values.values()), delta=1e-12,
                                       msg=row["candidate"])
                self.assertAlmostEqual(run.data.metrics["validation_mean_mcc_top10pct"],
                                       statistics.mean(values.values()), delta=1e-12,
                                       msg=row["candidate"])

    def test_champion_candidate_and_hyperparameters_match_the_selection_json(self):
        eta_selection = json.loads(registry.ETA_SELECTION.read_text())
        for stage, name in (("checkout", "checkout_eta"), ("handover", "handover_eta")):
            selected = eta_selection["selected_model"][stage]
            self.assertEqual(self.summary[name]["champion_candidate"], selected["candidate"])
            self.assertEqual(self.summary[name]["champion_params"], selected["parameters"])
        classifier_selection = json.loads(registry.CLASSIFIER_SELECTION.read_text())
        selected = classifier_selection["variants"]["primary"]["selected_model"]
        self.assertEqual(self.summary["late_risk"]["champion_candidate"], selected["candidate"])
        self.assertEqual(self.summary["late_risk"]["champion_params"], selected["parameters"])

    def _ladder_runs(self, name):
        experiment = self.client.get_experiment_by_name(name)
        runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
        return [run for run in runs if "ladder_family" in run.data.tags]

    def test_every_ladder_rung_has_train_validation_and_test_metrics(self):
        fields = {
            "regression": ("mae", "rmse", "r2"),
            "classification": ("pr_auc", "roc_auc", "mcc", "precision", "recall"),
        }
        validation_sources = {
            "regression": ("validation_window_mae", "validation_window_rmse",
                           "validation_window_r2"),
            "classification": ("validation_window_pr_auc", "validation_window_roc_auc",
                               "validation_window_recall_top10pct",
                               "validation_window_precision_top10pct",
                               "validation_window_mcc_top10pct"),
        }
        selection_mean = {"regression": "validation_mean_mae",
                          "classification": "validation_mean_pr_auc"}
        for name in registry.MODELS:
            kind = "classification" if name == "late_risk" else "regression"
            champion = self.summary[name]["champion_candidate"]
            runs = self._ladder_runs(name)
            self.assertTrue(runs, name)
            by_family = {}
            for run in runs:
                family = run.data.tags["ladder_family"]
                by_family.setdefault(family, []).append(run)
                for field in fields[kind]:
                    metric = field if kind == "regression" else (
                        field if field in ("pr_auc", "roc_auc") else f"{field}_top10pct")
                    self.assertIn(f"train_{metric}", run.data.metrics, f"{name} {run.data.tags['candidate']} train {field}")
                    self.assertIn(f"test_{metric}", run.data.metrics, f"{name} {run.data.tags['candidate']} test {field}")
                self.assertIn(selection_mean[kind], run.data.metrics)
                for field in ("mae", "rmse", "r2", "pr_auc", "roc_auc", "precision", "recall"):
                    self.assertNotIn(field, run.data.metrics, f"{name} bare {field} metric")
                self.assertIn("test_gap_vs_headline", run.data.metrics)
                self.assertIn("test_gap_low_vs_headline", run.data.metrics)
                self.assertIn("test_gap_high_vs_headline", run.data.metrics)
                self.assertIn(run.data.tags.get("test_gap_verdict"), {"better", "level", "worse"})
                self.assertIn("test_gap_method", run.data.params)
                for source in validation_sources[kind]:
                    self.assertEqual(len(self.client.get_metric_history(run.info.run_id, source)), 5,
                                     f"{name} {run.data.tags['candidate']} window {source}")
                if kind == "classification":
                    self.assertIn("validation_mean_mcc_top10pct", run.data.metrics)
                if run.data.tags["candidate"] != champion:
                    self.assertEqual(run.data.tags.get("test_use"),
                                     "reference_only_selection_on_validation")
            self.assertEqual(set(by_family), {"Linear", "Tree"})
            for family, family_runs in by_family.items():
                rungs = sorted(int(run.data.tags["ladder_rung"]) for run in family_runs)
                self.assertEqual(rungs, list(range(len(family_runs))), f"{name} {family}")
                summary_rungs = [row["rung"] for row in self.summary[name]["ladders"][family]]
                self.assertEqual(summary_rungs, rungs)
                for row in self.summary[name]["ladders"][family]:
                    self.assertIn("test_gap_vs_headline", row)
                    self.assertIn("test_gap_low_vs_headline", row)
                    self.assertIn("test_gap_high_vs_headline", row)
                    self.assertIn(row["test_gap_verdict"], {"better", "level", "worse"})
                    if kind == "classification":
                        self.assertIn("mcc", row["validation"]["mean"])
                        self.assertTrue(all("mcc" in window for window in row["validation"]["windows"]))
            self.assertEqual(sum(1 for run in runs
                                 if run.data.tags["candidate"] == champion), 1, name)

    def test_ladder_values_reproduce_the_saved_phase2_numbers(self):
        eta_saved = pd.read_csv(registry.ETA_RESULTS / "test_metrics.csv")
        apparent = pd.read_csv(registry.FINAL_RESULTS / "train_apparent_metrics.csv")
        for name, stage in (("checkout_eta", "checkout"), ("handover_eta", "handover")):
            for rows in self.summary[name]["ladders"].values():
                for rung in rows:
                    saved = eta_saved[(eta_saved["stage"] == stage) & (eta_saved["candidate"] == rung["candidate"])
                                      & (eta_saved["scope"] == "overall")]
                    if len(saved):
                        for field in registry.REGRESSION_LADDER_FIELDS:
                            self.assertAlmostEqual(rung["test"][field], float(saved.iloc[0][field]),
                                                   delta=TOLERANCE, msg=f"{name} {rung['candidate']} test {field}")
                    if rung["candidate"] == self.summary[name]["champion_candidate"]:
                        train = apparent[(apparent["task"] == "regression") & (apparent["stage"] == stage)
                                         & (apparent["model"] == rung["candidate"])].iloc[0]
                        for field in registry.REGRESSION_LADDER_FIELDS:
                            self.assertAlmostEqual(rung["train"][field], float(train[field]),
                                                   delta=TOLERANCE, msg=f"{name} train {field}")

        classifier_saved = pd.read_csv(registry.HANDOVER_RESULTS / "test_metrics.csv")
        champion = self.summary["late_risk"]["champion_candidate"]
        for rows in self.summary["late_risk"]["ladders"].values():
            for rung in rows:
                if rung["candidate"] != champion:
                    continue
                saved = classifier_saved[(classifier_saved["variant"] == "primary")
                                         & (classifier_saved["ranking"] == "model")].iloc[0]
                # Phase 2 saved its action metrics at its 14% cost-ratio share; the registry shows the
                # report's 10%. Score-based fields must match the saved files; 10% action metrics must
                # match the final run and the report's 377 of 672 late test orders caught.
                for field in ("roc_auc", "pr_auc"):
                    self.assertAlmostEqual(rung["test"][field], float(saved[field]), delta=TOLERANCE,
                                           msg=f"late_risk test {field}")
                final = self.summary["late_risk"]["test_metrics"]["model"]
                for field in registry.CLASSIFIER_FIELDS:
                    self.assertAlmostEqual(rung["test"][field], final[field], delta=TOLERANCE,
                                           msg=f"late_risk test {field} vs final run")
                self.assertAlmostEqual(rung["test"]["recall"], 377 / 672, delta=TOLERANCE)
                train = apparent[(apparent["task"] == "classification")
                                 & (apparent["stage"] == "handover")
                                 & (apparent["model"] == champion)].iloc[0]
                for field in ("roc_auc", "pr_auc"):
                    self.assertAlmostEqual(rung["train"][field], float(train[field]), delta=TOLERANCE,
                                           msg=f"late_risk train {field}")

    def test_classifier_train_apparent_metrics_use_the_ten_percent_share(self):
        apparent = pd.read_csv(registry.FINAL_RESULTS / "train_apparent_metrics.csv")
        champion = self.summary["late_risk"]["champion_candidate"]
        saved = apparent[(apparent["task"] == "classification")
                         & (apparent["stage"] == "handover")
                         & (apparent["model"] == champion)].iloc[0]
        rung = next(rung for rows in self.summary["late_risk"]["ladders"].values()
                    for rung in rows if rung["candidate"] == champion)
        experiment = self.client.get_experiment_by_name("late_risk")
        runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
        final = next(run for run in runs if run.data.tags.get("run_kind") == "final")
        for field in ("precision", "recall", "f1", "mcc"):
            self.assertIn(f"train_apparent_{field}_top10pct", final.data.metrics)
            self.assertNotIn(f"train_apparent_{field}_top14pct", final.data.metrics)
            self.assertAlmostEqual(final.data.metrics[f"train_apparent_{field}_top10pct"],
                                   rung["train"][field], delta=TOLERANCE,
                                   msg=f"late_risk train {field}")
        self.assertFalse([metric for metric in final.data.metrics if metric.endswith("_top14pct")])
        for retired in ("train_apparent_review_share", "train_apparent_flagged_orders",
                        "train_apparent_caught_late_orders"):
            self.assertNotIn(retired, final.data.metrics)
        for field in ("orders", "late_orders", "late_rate", "pr_auc", "roc_auc"):
            self.assertAlmostEqual(final.data.metrics[f"train_apparent_{field}"],
                                   float(saved[field]), delta=TOLERANCE,
                                   msg=f"late_risk train {field}")

    def test_every_candidate_validation_mean_matches_the_selection_stage2(self):
        eta_selection = json.loads(registry.ETA_SELECTION.read_text())
        classifier_selection = json.loads(registry.CLASSIFIER_SELECTION.read_text())
        validation = pd.read_csv(registry.HANDOVER_RESULTS / "validation_windows.csv")
        available = set(validation[validation["variant"] == "primary"]["candidate"])
        classifier_names = [name for group in classifier_selection["groups"]
                            for name in group["configs"] if name in available]

        expectations = {
            "checkout_eta": [row["candidate"] for row in eta_selection["stage2"]["checkout"]],
            "handover_eta": [row["candidate"] for row in eta_selection["stage2"]["handover"]],
            "late_risk": classifier_names,
        }
        stage2 = {
            "checkout_eta": {row["candidate"]: row for row in eta_selection["stage2"]["checkout"]},
            "handover_eta": {row["candidate"]: row for row in eta_selection["stage2"]["handover"]},
            "late_risk": {row["candidate"]: row for row in classifier_selection["variants"]["primary"]["stage2"]},
        }
        mean_keys = {"checkout_eta": "validation_mean_mae",
                     "handover_eta": "validation_mean_mae",
                     "late_risk": "validation_mean_pr_auc"}
        for name, expected_candidates in expectations.items():
            experiment = self.client.get_experiment_by_name(name)
            runs = self.client.search_runs([experiment.experiment_id], max_results=1000)
            parent = next(run for run in runs if run.data.tags.get("run_kind") == "selection")
            children = {run.data.tags["candidate"]: run for run in runs
                        if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
                        and run.data.tags.get("run_kind") == "candidate"}
            self.assertEqual(set(children), set(expected_candidates))
            for candidate, row in stage2[name].items():
                self.assertAlmostEqual(children[candidate].data.metrics[mean_keys[name]],
                                       float(row["mean"]), delta=1e-12,
                                       msg=f"{name} {candidate}")

    def test_every_candidate_run_trained_its_validation_windows(self):
        for name in registry.MODELS:
            runs = self._candidate_runs(name)
            self.assertTrue(runs, name)
            for run in runs:
                candidate = run.data.tags["candidate"]
                history = self.client.get_metric_history(run.info.run_id, "fit_seconds")
                self.assertEqual(len(history), 5, f"{name} {candidate} fit seconds")
                self.assertGreater(run.data.metrics["fit_seconds_total"], 0.0, f"{name} {candidate}")
                orders = self.client.get_metric_history(run.info.run_id, "window_fit_orders")
                evaluated = self.client.get_metric_history(run.info.run_id, "window_validation_orders")
                self.assertEqual(len(orders), 5, f"{name} {candidate} fit orders")
                self.assertEqual(len(evaluated), 5, f"{name} {candidate} validation orders")
                self.assertTrue(all(point.value > 0 for point in orders), f"{name} {candidate}")

    def test_permutation_importance_artifact_is_on_every_final_run(self):
        columns = ["input", "key", "members", "importance_mean", "importance_std", "direction"]
        allowed = {"mixed/overlapping", "higher \u2192 more likely late",
                   "higher \u2192 less likely late", "higher \u2192 more days",
                   "higher \u2192 fewer days"}
        for name in registry.MODELS:
            version = self.client.get_model_version_by_alias(name, "champion")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(self.client.download_artifacts(
                    version.run_id, "permutation_importance.csv", tmp))
                frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns), columns, name)
            self.assertGreater(len(frame), 5, name)
            self.assertTrue(frame["importance_std"].ge(0).all(), name)
            self.assertTrue(set(frame["direction"].fillna("")) <= allowed, name)
            self.assertTrue((frame["direction"].fillna("") != "mixed/overlapping").any(), name)
            self.assertTrue(frame["members"].fillna("").str.len().gt(0).all(), name)
            grouped = frame["members"].fillna("").str.contains(",")
            self.assertTrue(grouped.any(), name)
            self.assertTrue((frame.loc[grouped, "direction"] == "mixed/overlapping").all(), name)

    def test_champion_runs_log_drift_thresholds_for_the_selected_inputs(self):
        eta_table, eta_spec, _ = load_stage_table(ROOT)
        eta_train = eta_table[eta_table[eta_spec["primary_split"]].eq("train")].reset_index(drop=True)
        cohort, classifier_spec, _ = load_cohort(ROOT)
        classifier_train = cohort[
            cohort[classifier_spec["primary_split"]].eq("train")].reset_index(drop=True)
        sources = {
            "checkout_eta": (eta_train, eta_spec),
            "handover_eta": (eta_train, eta_spec),
            "late_risk": (classifier_train, classifier_spec),
        }
        for name in registry.MODELS:
            version = self.client.get_model_version_by_alias(name, "champion")
            final = self.client.get_run(version.run_id)
            self.assertEqual(final.data.tags.get("drift_thresholds"), "drift_thresholds.json")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(self.client.download_artifacts(
                    version.run_id, "drift_thresholds.json", tmp))
                payload = json.loads(path.read_text())
            self.assertEqual(payload["model"], name)
            self.assertEqual(payload["reference"], "champion training data")
            self.assertEqual(payload["readings"], "monthly")
            self.assertEqual(payload["n_readings"], 11)
            expected = self._current_thresholds(*sources[name], payload["inputs"])
            self.assertEqual(set(payload["inputs"]), set(expected), name)
            for feature, entry in payload["inputs"].items():
                self.assertEqual(entry["threshold"], expected[feature]["threshold"], feature)
                self.assertEqual(entry["validation"], expected[feature]["validation"], feature)
                self.assertNotIn("false_alarms", entry, feature)
                self.assertEqual(entry["readings"], expected[feature]["readings"], feature)
                self.assertIn(entry["statistic"], ("ks", "tv"), feature)

    @staticmethod
    def _current_thresholds(train, spec, entries):
        """Recompute the deployed monthly thresholds."""
        features = [{"feature": feature, "type": "numeric" if entry["statistic"] == "ks"
                     else "categorical"}
                    for feature, entry in entries.items()]
        return drift_module.input_threshold_table(train, spec, features)["inputs"]

    def test_the_final_run_records_the_candidates_selection_compared(self):
        for name in registry.MODELS:
            version = self.client.get_model_version_by_alias(name, "champion")
            final = self.client.get_run(version.run_id)
            pool = [key for key in (final.data.tags.get("selection_pool") or "").split(",") if key]
            self.assertGreater(len(pool), 1, name)
            self.assertIn(self.summary[name]["champion_candidate"], pool, name)

    def test_late_risk_rule_validation_metrics_are_logged_for_every_measure(self):
        version = self.client.get_model_version_by_alias("late_risk", "champion")
        final = self.client.get_run(version.run_id)
        for field in ("pr_auc", "roc_auc", "mcc", "precision", "recall"):
            history = self.client.get_metric_history(
                final.info.run_id, f"reference_slack_rule_validation_window_{field}")
            self.assertEqual(len(history), 5, field)
            mean = final.data.metrics[f"reference_slack_rule_validation_mean_{field}"]
            self.assertAlmostEqual(mean, sum(point.value for point in history) / 5, delta=1e-12,
                                   msg=field)

    def _candidate_runs(self, name):
        experiment = self.client.get_experiment_by_name(name)
        runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
        parent = next(run for run in runs if run.data.tags.get("run_kind") == "selection")
        return [run for run in runs
                if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
                and run.data.tags.get("run_kind") == "candidate"]


if __name__ == "__main__":
    unittest.main()
