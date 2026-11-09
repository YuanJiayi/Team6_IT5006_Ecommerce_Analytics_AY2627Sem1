"""Tests for phase3.monitor: Evidently reproduction, maturity, model checks and API paths."""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")

from mlflow.tracking import MlflowClient  # noqa: E402

from olist.paths import DASHBOARD_PHASE2, PROCESSED_DIR  # noqa: E402
from phase3 import monitor, serving  # noqa: E402
from phase3.mlflow_server import mlflow_server  # noqa: E402

drift = monitor.drift
PHASE2_DATA = PROCESSED_DIR
PHASE2_RESULTS = DASHBOARD_PHASE2
PAYLOADS = {}

DATES = ("2018-06-30", "2018-08-22")


def payload(date):
    if date not in PAYLOADS:
        PAYLOADS[date] = monitor.build_monitor(date)
    return PAYLOADS[date]


class StatTestTests(unittest.TestCase):
    def test_ks_distance_stat_equals_scipy(self):
        rng = np.random.default_rng(7)
        reference = pd.Series(rng.normal(size=200))
        current = pd.Series(rng.normal(0.4, 1.1, size=250))
        distance, drifted = monitor._ks_distance_stat(reference, current, None, 0.1)
        self.assertEqual(distance, float(ks_2samp(reference, current).statistic))
        self.assertEqual(drifted, distance > 0.1)
        with_nan = current.copy()
        with_nan.iloc[:10] = np.nan
        distance, _ = monitor._ks_distance_stat(reference, with_nan, None, 0.1)
        self.assertEqual(distance, float(ks_2samp(reference, with_nan.dropna()).statistic))

    def test_evidently_uses_the_ks_distance_and_tv_distance(self):
        rng = np.random.default_rng(11)
        reference = pd.DataFrame({
            "x": rng.normal(size=300),
            "colour": ["a"] * 150 + ["b"] * 150,
        })
        current = pd.DataFrame({
            "x": rng.normal(0.7, size=300),
            "colour": ["a"] * 240 + ["b"] * 60,
        })
        scores = monitor.evidently_drift_scores(reference, current, ["x"], ["colour"])
        self.assertAlmostEqual(scores["x"],
                               float(ks_2samp(reference["x"], current["x"]).statistic),
                               places=12)
        self.assertAlmostEqual(scores["colour"], 0.3, places=12)

    def test_evidently_verdict_uses_the_given_per_input_threshold(self):
        rng = np.random.default_rng(12)
        reference = pd.DataFrame({"x": rng.normal(size=400)})
        current = pd.DataFrame({"x": rng.normal(0.3, size=400)})
        low = monitor.evidently_value_drift(reference, current, ["x"], [], {"x": 0.01})
        high = monitor.evidently_value_drift(reference, current, ["x"], [], {"x": 0.99})
        self.assertTrue(low["x"]["drifted"])
        self.assertFalse(high["x"]["drifted"])
        self.assertEqual(low["x"]["score"], high["x"]["score"])

    def test_tv_distance_treats_missing_as_a_category(self):
        reference = pd.Series(["a", "a", None, None])
        current = pd.Series(["a", "b", "b", None])
        distance, drifted = monitor._tv_distance_stat(reference, current, None, 0.1)
        self.assertAlmostEqual(distance, 0.5, places=12)
        self.assertTrue(drifted)


class AlertRuleTests(unittest.TestCase):
    def test_input_levels_use_the_validation_thresholds(self):
        self.assertEqual(monitor._input_level(0.05, 0.1), "normal")
        self.assertEqual(monitor._input_level(0.1, 0.1), "normal")
        self.assertEqual(monitor._input_level(0.19, 0.1), "unusual")
        self.assertEqual(monitor._input_level(None, 0.1), "normal")
        self.assertEqual(monitor._input_level(0.2, None), "normal")

    def test_bias_bounds_and_sign(self):
        bounds = monitor._bias_bounds({"signed": {"min": -4.0, "max": -1.0, "sd": 1.0}})
        self.assertEqual(bounds, {"min": -4.0, "max": -1.0, "sd": 1.0,
                                  "act_min": -5.0, "act_max": 0.0})
        self.assertEqual(monitor.bias_level(-2.0, bounds), "none")
        self.assertEqual(monitor.bias_level(-0.5, bounds), "watch")
        self.assertEqual(monitor.bias_level(0.5, bounds), "act")
        self.assertEqual(monitor.bias_level(-6.0, bounds), "act")
        self.assertEqual(monitor.bias_level(None, bounds), "none")

    def test_performance_level(self):
        validation = {"mean": 5.0, "sd": 0.5}
        self.assertEqual(monitor.performance_level(5.9, validation, higher_is_worse=True), "none")
        self.assertEqual(monitor.performance_level(6.1, validation, higher_is_worse=True), "watch")
        self.assertEqual(monitor.performance_level(0.50, {"mean": 0.60, "sd": 0.02},
                                                    higher_is_worse=False), "watch")
        self.assertEqual(monitor.performance_level(0.57, {"mean": 0.60, "sd": 0.02},
                                                    higher_is_worse=False), "none")

    def test_status_card_does_not_say_back_to_normal_while_at_watch(self):
        model = {
            "name": "late_risk",
            "title": "Late risk",
            "stage": "late_warning",
            "kind": "classification",
            "rule_label": "Remaining slack",
            "months": [
                {"month": "2018-06", "level": "watch",
                 "levels": {"rule": "none", "performance": "watch", "overall": "watch"},
                 "reason": "Late-risk blend outside its validation range."},
                {"month": "2018-07", "level": "none",
                 "levels": {"rule": "none", "performance": "none", "overall": "none"},
                 "reason": "within the validation range"},
            ],
        }
        card = monitor._status_card(model, {"windows": []})
        self.assertEqual(card["level"], "watch")
        self.assertNotIn("back to normal", card["reason"].lower())
        self.assertIn("status stays Watch", card["reason"])
        self.assertTrue(card["reason"].endswith("."))

    def test_rule_level_only_acts_on_a_clear_loss(self):
        self.assertEqual(monitor._rule_level({"verdict": "worse"}), "act")
        self.assertEqual(monitor._rule_level({"verdict": "level"}), "none")
        self.assertEqual(monitor._rule_level({"verdict": "better"}), "none")

    def test_build_alerts_one_line_per_model_and_month_newest_first(self):
        model_checks = {
            "models": [
                {
                    "name": "checkout_eta", "title": "Checkout", "kind": "regression",
                    "rule_label": "Route's average", "months": [
                        {"month": "2018-05", "level": "act", "too_few": False,
                         "levels": {"rule": "act", "bias": "act", "performance": "none"},
                         "model": {"mae": 4.5, "signed_error": 2.7},
                         "rule": {"mae": 4.1}, "rule_gap": {"gap": -0.4}, "reason": "x"},
                        {"month": "2018-06", "level": "watch", "too_few": False,
                         "levels": {"rule": "none", "bias": "watch", "performance": "none"},
                         "model": {"mae": 5.0, "signed_error": 1.0},
                         "rule": {"mae": 5.1}, "reason": "x"},
                    ],
                },
                {
                    "name": "late_risk", "title": "Late risk", "kind": "classification",
                    "rule_label": "Remaining slack", "months": [
                        {"month": "2018-06", "level": "act", "too_few": False,
                         "levels": {"rule": "act", "performance": "none"},
                         "model": {"pr_auc": 0.2}, "rule": {"pr_auc": 0.4},
                         "rule_gap": {"gap": -0.2}, "reason": "x"},
                    ],
                },
            ],
        }
        alerts = monitor.build_alerts(model_checks)
        self.assertEqual(len(alerts), 3)
        self.assertEqual([alert["month"] for alert in alerts],
                         ["2018-06", "2018-06", "2018-05"])
        self.assertTrue(all(alert["sentence"].endswith(".") for alert in alerts))
        self.assertEqual({alert["kind"] for alert in alerts}, {"model"})


class MaturityRuleTests(unittest.TestCase):
    @staticmethod
    def _frame(promised_days, purchase, outcome, late):
        return pd.DataFrame({
            "order_purchase_timestamp": pd.to_datetime(purchase),
            "promised_days": promised_days,
            "outcome_available_at": pd.to_datetime(outcome),
            "delivery_days": [10.0] * len(purchase),
            "is_late": late,
        })

    def test_late_rate_is_ready_at_95_percent_delivered(self):
        purchase = ["2018-01-01"] * 20
        promised = [10.0] * 20
        cutoff = pd.Timestamp("2018-02-01 09:00")
        delivered = ["2018-01-05"] * 19 + ["2018-03-01"]
        frame = self._frame(promised, purchase, delivered, [0] * 19 + [1])
        at_95 = monitor.month_outcomes(frame, cutoff)[0]
        self.assertAlmostEqual(at_95["share_matured"], 0.95)
        self.assertTrue(at_95["late_mature"])
        self.assertEqual(at_95["open_orders"], 1)
        self.assertEqual(at_95["open_past_promise"], 1)
        self.assertEqual(at_95["unknown_orders"], 0)
        self.assertEqual(at_95["late_orders"], 1)
        self.assertEqual(at_95["known_orders"], 20)
        self.assertAlmostEqual(at_95["late_rate"], 0.05)

        at_90 = monitor.month_outcomes(
            self._frame(promised, purchase, delivered[:18] + ["2018-03-01"] * 2, [0] * 20),
            cutoff)[0]
        self.assertAlmostEqual(at_90["share_matured"], 0.9)
        self.assertFalse(at_90["late_mature"])
        self.assertIsNone(at_90["late_rate"])

    def test_open_past_promise_counts_late_and_open_before_promise_is_unknown(self):
        purchase = ["2018-01-01"] * 40
        promised = [10.0] * 39 + [30.0]
        outcome = ["2018-01-05"] * 38 + ["2018-03-01", "2018-03-01"]
        late = [0] * 37 + [1] + [0, 0]
        row = monitor.month_outcomes(
            self._frame(promised, purchase, outcome, late),
            pd.Timestamp("2018-01-15 09:00"))[0]
        self.assertTrue(row["late_mature"])
        self.assertEqual(row["orders"], 40)
        self.assertEqual(row["matured_orders"], 38)
        self.assertEqual(row["open_orders"], 2)
        self.assertEqual(row["open_past_promise"], 1)
        self.assertEqual(row["unknown_orders"], 1)
        self.assertEqual(row["known_orders"], 39)
        self.assertEqual(row["late_orders"], 2)
        self.assertAlmostEqual(row["late_rate"], round(2 / 39, 4))


@unittest.skipUnless((PHASE2_DATA / "phase2_order_table.csv").is_file(),
                     "the Phase 2 data is not available")
class RealDataMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls._previous_ui = os.environ.get("PRAZO_MLFLOW_UI")
        uri = cls._stack.enter_context(mlflow_server(serving.DEFAULT_STORE))
        os.environ["PRAZO_MLFLOW_UI"] = uri
        cls.addClassCleanup(cls._close_server)

        cls.state = monitor.load_state()
        cls.payloads = {date: payload(date) for date in DATES}

    @classmethod
    def _close_server(cls):
        if cls._previous_ui is None:
            os.environ.pop("PRAZO_MLFLOW_UI", None)
        else:
            os.environ["PRAZO_MLFLOW_UI"] = cls._previous_ui
        cls._stack.close()

    def visible(self, date):
        cutoff = monitor.cutoff_for(date)
        return self.state["cohort"][self.state["cohort"]["order_purchase_timestamp"] < cutoff]

    def test_loaded_champions_match_the_refit_path_on_the_test_rows(self):
        refit_models, _, _ = drift.fit_selected_models(
            self.state["cohort"], self.state["spec"], self.state["selected"])
        test = self.state["cohort"][self.state["cohort"]["split"].eq("test")].reset_index(drop=True)
        for stage in ("checkout", "handover"):
            columns = sum(drift.feature_columns(self.state["contracts"][stage]), [])
            np.testing.assert_allclose(
                self.state["models"][stage].predict(test[columns]),
                refit_models[stage].predict(test[columns]),
                rtol=0, atol=1e-9, err_msg=stage)
        columns = sum(drift.feature_columns(self.state["contracts"]["late_warning"]), [])
        np.testing.assert_allclose(
            self.state["models"]["late_warning"].predict_proba(test[columns])[:, 1],
            refit_models["late_warning"].predict_proba(test[columns])[:, 1],
            rtol=0, atol=1e-9, err_msg="late_warning")

    def test_payload_shape_and_notes(self):
        for date, result in self.payloads.items():
            self.assertEqual(result["as_of"], date)
            self.assertEqual(result["note"], monitor.NOTE)
            for key in ("inputs", "outcomes", "model_checks", "alerts", "status_cards"):
                self.assertIn(key, result)
            self.assertIsInstance(result["first_load_ms"], int)
            self.assertEqual(result["mismatches"], [])
            self.assertEqual(len(result["status_cards"]), 3)
            self.assertEqual([card["name"] for card in result["status_cards"]],
                             ["checkout_eta", "handover_eta", "late_risk"])
            self.assertTrue(all(card["reason"].endswith(".") for card in result["status_cards"]))
            self.assertTrue(all(card["action"] for card in result["status_cards"]))
            self.assertEqual(result["model_checks"]["test_start"],
                             self.state["spec"]["primary_cutoff"])
            self.assertIn(self.state["spec"]["primary_cutoff"],
                          result["model_checks"]["test_start_note"])
            self.assertIn("May 2018 is partial", result["model_checks"]["test_start_note"])
            self.assertNotIn("PR-AUC is computed", json.dumps(result))

    def test_every_input_number_matches_drift(self):
        for date, result in self.payloads.items():
            visible = self.visible(date)
            windows = monitor._windows(visible, result["inputs"]["windows"])
            expected = drift.x_drift(visible, self.state["spec"], windows,
                                     cutoff=monitor.cutoff_for(date))
            features = {item["feature"]: item for item in result["inputs"]["features"]}
            for stage in ("checkout", "handover", "late_warning"):
                for month in result["inputs"]["windows"]:
                    for feature, source in expected[stage]["windows"][month].items():
                        if feature not in features:
                            continue
                        value = source["ks"] if features[feature]["type"] == "numeric" else source["total_variation"]
                        self.assertAlmostEqual(features[feature]["months"][month]["score"], value,
                                               places=12,
                                               msg=f"{date} {stage} {month} {feature}")

    def test_handover_inputs_ignore_orders_not_yet_handed_over(self):
        date = "2018-08-22"
        result = self.payloads[date]
        visible = self.visible(date)
        cutoff = monitor.cutoff_for(date)
        month = drift.month_label(visible["order_purchase_timestamp"])
        august = visible[(visible["split"].eq("test")) & (month == "2018-08")]
        self.assertEqual(int((~august["handover_time"].le(cutoff)).sum()), 558)
        features = {item["feature"]: item for item in result["inputs"]["features"]}
        for feature, before, after in (
                ("route_transit_90d_at_handover", 0.3764, 0.3673),
                ("state_backlog_recent_vs_8wk", 0.2599, 0.305)):
            if feature not in features:
                continue
            known = august.loc[august["handover_time"].le(cutoff), feature]
            self.assertAlmostEqual(
                features[feature]["months"]["2018-08"]["score"],
                drift.ks_distance(self.state["train"][feature], known), places=4)
            self.assertAlmostEqual(
                drift.ks_distance(self.state["train"][feature], august[feature]), before, places=4)
            self.assertAlmostEqual(
                features[feature]["months"]["2018-08"]["score"], after, places=4)

    def test_august_inputs_still_cross_their_new_bars(self):
        result = self.payloads["2018-08-22"]
        august = [item for item in result["inputs"]["features"]
                  if item["months"]["2018-08"]["level"] == "unusual"]
        self.assertTrue(august)
        for item in august:
            self.assertGreater(item["months"]["2018-08"]["score"], item["threshold"],
                               msg=item["feature"])

    def test_input_thresholds_come_from_the_validation_months(self):
        method, readings = drift.validation_readings(self.state["train"], self.state["spec"])
        self.assertEqual(method, "monthly")
        self.assertEqual(len(readings), 11)
        for date, result in self.payloads.items():
            for item in result["inputs"]["features"]:
                values = []
                for _, fit, positions, _ in readings:
                    if item["type"] == "numeric":
                        value = drift.ks_distance(self.state["train"].iloc[fit][item["feature"]],
                                                  self.state["train"].iloc[positions][item["feature"]])
                    else:
                        value = drift.total_variation(
                            self.state["train"].iloc[fit][item["feature"]],
                            self.state["train"].iloc[positions][item["feature"]])[0]
                    values.append(value)
                self.assertEqual(item["threshold"], round(max(values), 4))
                self.assertEqual(len(item["validation"]), 11)
                for stored, value in zip(item["validation"], values):
                    self.assertAlmostEqual(stored, value, places=4)

    def test_evidently_verdicts_match_dashboard_colours(self):
        for date, result in self.payloads.items():
            visible = self.visible(date)
            windows = monitor._windows(visible, result["inputs"]["windows"])
            reference = visible.loc[visible["split"].eq("train").to_numpy()]
            features = result["inputs"]["features"]
            numeric = [item["feature"] for item in features if item["type"] == "numeric"]
            categorical = [item["feature"] for item in features if item["type"] == "categorical"]
            thresholds = {item["feature"]: item["threshold"] for item in features}
            for month in result["inputs"]["windows"]:
                current = visible.loc[windows[month].to_numpy()]
                current = monitor._mask_unavailable_inputs(
                    current, self.state["input_availability"], monitor.cutoff_for(date))
                verdicts = monitor.evidently_value_drift(reference, current, numeric, categorical,
                                                         thresholds)
                for item in features:
                    stored = item["months"][month]
                    self.assertEqual(stored["evidently_drifted"], verdicts[item["feature"]]["drifted"],
                                     msg=f"{date} {month} {item['feature']}")
                    self.assertEqual(stored["level"] == "unusual",
                                     verdicts[item["feature"]]["drifted"],
                                     msg=f"{date} {month} {item['feature']} colour")

    def test_rule_gaps_reproduce_the_models_tab_method_on_the_full_test_period(self):
        registry = monitor.registry
        self.assertEqual(registry.BOOTSTRAP_RESAMPLES, 2000)
        self.assertEqual(registry.BOOTSTRAP_SEED, 42)
        test = self.state["cohort"][self.state["cohort"]["split"].eq("test")].reset_index(drop=True)
        weeks = test["order_purchase_timestamp"].dt.to_period("W").astype(str).to_numpy()
        for stage in ("checkout", "handover"):
            columns = sum(drift.feature_columns(self.state["contracts"][stage]), [])
            predicted = self.state["models"][stage].predict(test[columns])
            if stage == "checkout":
                rule = test["route_typical_days"].fillna(
                    self.state["train"]["route_typical_days"].median()).to_numpy(dtype=float)
            else:
                rule = self.state["handover_rule"].predict(test[self.state["handover_columns"]])
            expected = registry.paired_week_gap(test[drift.TARGET].to_numpy(dtype=float),
                                                predicted, rule, weeks, "mae")
            got = monitor._paired_gap(test[drift.TARGET].to_numpy(dtype=float),
                                      predicted, rule, weeks, "mae")
            self.assertEqual(got, expected)
            self.assertLess(got["low"], got["high"])
        columns = sum(drift.feature_columns(self.state["contracts"]["late_warning"]), [])
        scores = self.state["models"]["late_warning"].predict_proba(test[columns])[:, 1]
        median = self.state["train"]["route_transit_90d_at_handover"].median()
        rule_scores = -(test["promised_days"] - test["handover_days"]
                        - test["route_transit_90d_at_handover"].fillna(median)).to_numpy(dtype=float)
        expected = registry.paired_week_gap(test[drift.LABEL].to_numpy(dtype=float),
                                            scores, rule_scores, weeks, "pr_auc")
        got = monitor._paired_gap(test[drift.LABEL].to_numpy(dtype=float),
                                  scores, rule_scores, weeks, "pr_auc")
        self.assertEqual(got, expected)

    def test_eta_edges_are_pooled_and_reproduce_the_registry_gap(self):
        registry = monitor.registry
        for date, result in self.payloads.items():
            cutoff = monitor.cutoff_for(date)
            test = self.state["cohort"][self.state["cohort"]["split"].eq("test")].reset_index(drop=True)
            later = test[test["delivery_days"].notna() & test["outcome_available_at"].le(cutoff)]
            later_month = drift.month_label(later["order_purchase_timestamp"])
            weeks = later["order_purchase_timestamp"].dt.to_period("W").astype(str).to_numpy()
            for stage in ("checkout", "handover"):
                columns = sum(drift.feature_columns(self.state["contracts"][stage]), [])
                predicted = self.state["models"][stage].predict(later[columns])
                if stage == "checkout":
                    rule = later["route_typical_days"].fillna(
                        self.state["train"]["route_typical_days"].median()).to_numpy(dtype=float)
                else:
                    rule = self.state["handover_rule"].predict(later[self.state["handover_columns"]])
                model = next(item for item in result["model_checks"]["models"]
                             if item["stage"] == stage)
                for month in model["months"]:
                    if month.get("too_few"):
                        continue
                    pooled = np.asarray(later_month <= month["month"])
                    expected = registry.paired_week_gap(
                        later.loc[pooled, drift.TARGET].to_numpy(dtype=float),
                        predicted[pooled], rule[pooled], weeks[pooled], "mae")
                    stored = month["rule_gap"]
                    self.assertEqual(stored["gap"], monitor._round(expected["gap"]),
                                     msg=f"{date} {stage} {month['month']}")
                    self.assertEqual(stored["low"], monitor._round(expected["low"]),
                                     msg=f"{date} {stage} {month['month']}")
                    self.assertEqual(stored["high"], monitor._round(expected["high"]),
                                     msg=f"{date} {stage} {month['month']}")
                    self.assertEqual(stored["verdict"], expected["verdict"])
                    self.assertEqual(stored["weeks"], len(pd.unique(weeks[pooled])))

    def test_outcome_numbers_follow_the_new_maturity_rule(self):
        for date, result in self.payloads.items():
            cutoff = monitor.cutoff_for(date)
            visible = self.visible(date)
            month = pd.Series(drift.month_label(visible["order_purchase_timestamp"]),
                              index=visible.index)
            promised = visible["order_purchase_timestamp"] + pd.to_timedelta(
                visible["promised_days"], unit="D")
            delivered = (visible["delivery_days"].notna()
                         & visible["outcome_available_at"].le(cutoff))
            for row in result["outcomes"]["months"]:
                mask = month.eq(row["month"])
                sub, done = visible.loc[mask], delivered.loc[mask]
                share = int(done.sum()) / len(sub)
                open_past = (~done) & promised.loc[mask].le(cutoff)
                unknown = (~done) & promised.loc[mask].gt(cutoff)
                self.assertEqual(row["orders"], len(sub), msg=f"{date} {row['month']}")
                self.assertEqual(row["matured_orders"], int(done.sum()))
                self.assertEqual(row["open_orders"], int((~done).sum()))
                self.assertEqual(row["open_past_promise"], int(open_past.sum()))
                self.assertEqual(row["unknown_orders"], int(unknown.sum()))
                self.assertEqual(row["share_matured"], round(share, 4))
                self.assertEqual(row["late_mature"], share >= monitor.DELIVERY_MATURE_SHARE)
                self.assertEqual(row["delivery_mature"], share >= monitor.DELIVERY_MATURE_SHARE)
                if row["late_mature"]:
                    late = int(sub.loc[done, "is_late"].sum()) + int(open_past.sum())
                    known = int(done.sum()) + int(open_past.sum())
                    self.assertEqual(row["late_orders"], late)
                    self.assertEqual(row["known_orders"], known)
                    self.assertEqual(row["late_rate"], round(late / known, 4))
                    delivered_rate = sub.loc[done, "is_late"].mean()
                    self.assertEqual(row["late_rate_delivered_only"], round(delivered_rate, 4))
                else:
                    self.assertIsNone(row["late_orders"])
                    self.assertIsNone(row["late_rate"])
                    self.assertIsNone(row["known_orders"])
                self.assertEqual(row["mean_delivery_days"],
                                 monitor._round(sub.loc[done, "delivery_days"].mean())
                                 if done.any() else None)
                self.assertEqual(row["mean_promised_days"],
                                 monitor._round(sub["promised_days"].mean()))

    def test_reference_ranges_only_use_months_mature_at_the_cutoff(self):
        expected = {
            "mean_delivery_days": (10.9021, 16.8313),
            "late_rate": (0.028, 0.1886),
            "mean_promised_days": (22.0254, 39.4564),
        }
        for date, result in self.payloads.items():
            outcomes = result["outcomes"]
            self.assertEqual(outcomes["reference_min_orders"], 500)
            self.assertEqual(outcomes["reference_months"], 16, date)
            for metric, (low, high) in expected.items():
                bounds = outcomes["reference_ranges"][metric]
                self.assertIsNotNone(bounds, msg=f"{date} {metric}")
                self.assertEqual(bounds["min"], low, msg=f"{date} {metric}")
                self.assertEqual(bounds["max"], high, msg=f"{date} {metric}")
                self.assertGreater(bounds["width"], 0)
        # May 2018 (purchases to 25 May) was only 68.5% delivered at the 26 May cutoff.
        visible = self.visible("2018-10-18")
        cutoff = monitor.cutoff_for("2018-10-18")
        primary = pd.Timestamp(self.state["spec"]["primary_cutoff"])
        universe = visible[visible["order_purchase_timestamp"].lt(primary)]
        may = next(row for row in monitor.month_outcomes(universe, primary)
                   if row["month"] == "2018-05")
        self.assertLess(may["share_matured"], monitor.DELIVERY_MATURE_SHARE)
        reference, months = monitor.reference_ranges(
            visible, cutoff, self.state["spec"]["primary_cutoff"])
        self.assertEqual(months, 16)
        self.assertEqual(reference["mean_delivery_days"]["min"], 10.9021)

    def test_every_model_number_matches_drift(self):
        for date, result in self.payloads.items():
            cutoff = monitor.cutoff_for(date)
            visible = self.visible(date)
            windows = monitor._windows(visible, result["inputs"]["windows"])
            regressions = drift.xy_regressions(
                visible, self.state["models"], self.state["contracts"],
                self.state["train"], cutoff, windows)
            test = visible[visible["split"].eq("test")]
            test_month = drift.month_label(test["order_purchase_timestamp"])
            known, labels = drift.late_labels(test, cutoff)
            scores = monitor._late_risk_scores(self.state, test, cutoff)
            known = known & np.isfinite(scores)
            for model in result["model_checks"]["models"]:
                stage = model["stage"]
                for month in model["months"]:
                    if month.get("too_few"):
                        continue
                    if stage in ("checkout", "handover"):
                        expected = regressions[stage]["monthly"][month["month"]]
                        self.assertAlmostEqual(month["model"]["mae"], expected["mae"], places=12)
                        self.assertAlmostEqual(month["model"]["signed_error"],
                                               expected["signed_error"], places=12)
                    else:
                        mask = np.asarray(test_month == month["month"]) & known
                        expected = float(drift.average_precision_score(labels[mask], scores[mask]))
                        self.assertAlmostEqual(month["model"]["pr_auc"], monitor._round(expected),
                                               places=12)

    def test_late_risk_check_scores_the_monthly_refit_blend(self):
        result = self.payloads["2018-08-22"]
        late = next(model for model in result["model_checks"]["models"]
                    if model["stage"] == "late_warning")
        self.assertEqual(late["ranking"], "50/50 blend of the model and remaining slack")
        months = {month["month"]: month for month in late["months"] if not month.get("too_few")}
        self.assertEqual(sorted(months), ["2018-06", "2018-07"])
        self.assertAlmostEqual(months["2018-06"]["model"]["pr_auc"], 0.2003, places=4)
        self.assertAlmostEqual(months["2018-07"]["model"]["pr_auc"], 0.4468, places=4)
        self.assertAlmostEqual(months["2018-06"]["model"]["roc_auc"], 0.7922, places=4)
        self.assertEqual(months["2018-06"]["levels"]["performance"], "none")
        self.assertEqual(late["validation"]["roc_auc"]["n_windows"], 5)
        # The blend is not just the model: the model-alone PR-AUC is different.
        cutoff = monitor.cutoff_for("2018-08-22")
        test = self.visible("2018-08-22")
        test = test[test["split"].eq("test")]
        test_month = drift.month_label(test["order_purchase_timestamp"])
        known, labels = drift.late_labels(test, cutoff)
        scores = monitor._late_risk_scores(self.state, test, cutoff)
        known = known & np.isfinite(scores)
        columns = sum(drift.feature_columns(self.state["contracts"]["late_warning"]), [])
        model_scores = self.state["models"]["late_warning"].predict_proba(test[columns])[:, 1]
        for name in months:
            mask = np.asarray(test_month == name) & known
            model_ap = float(drift.average_precision_score(labels[mask], model_scores[mask]))
            self.assertNotAlmostEqual(months[name]["model"]["pr_auc"], monitor._round(model_ap),
                                      places=3, msg=name)
            expected_roc = float(roc_auc_score(labels[mask], scores[mask]))
            self.assertAlmostEqual(months[name]["model"]["roc_auc"],
                                   monitor._round(expected_roc), places=12, msg=name)

    def test_operation_rule_numbers_match_phase2_formulas(self):
        cutoff = monitor.cutoff_for("2018-08-22")
        visible = self.visible("2018-08-22")
        result = self.payloads["2018-08-22"]
        test = visible[visible["split"].eq("test")]
        later = test[test["delivery_days"].notna() & test["outcome_available_at"].le(cutoff)]
        later_month = drift.month_label(later["order_purchase_timestamp"])
        test_month = drift.month_label(test["order_purchase_timestamp"])
        train = self.state["train"]

        checkout_rule = later["route_typical_days"].fillna(train["route_typical_days"].median())
        handover_rule = self.state["handover_rule"].predict(
            later[self.state["handover_columns"]])
        median = train["route_transit_90d_at_handover"].median()
        slack = (test["promised_days"] - test["handover_days"]
                 - test["route_transit_90d_at_handover"].fillna(median))
        late_rule = -slack.to_numpy(dtype=float)
        known_labels, labels = drift.late_labels(test, cutoff)
        scores = monitor._late_risk_scores(self.state, test, cutoff)
        scored = known_labels & np.isfinite(scores)

        for model in result["model_checks"]["models"]:
            stage = model["stage"]
            for month in model["months"]:
                if month.get("too_few"):
                    continue
                if stage in ("checkout", "handover"):
                    mask = np.asarray(later_month == month["month"])
                    target = later.loc[mask, drift.TARGET]
                    if stage == "checkout":
                        metrics = drift.regression_metrics(
                            target, checkout_rule[mask].to_numpy(dtype=float))
                    else:
                        metrics = drift.regression_metrics(target, handover_rule[mask])
                    self.assertAlmostEqual(month["rule"]["mae"], monitor._round(metrics["mae"]), places=12)
                else:
                    mask = np.asarray(test_month == month["month"]) & scored
                    if labels[mask].sum():
                        expected = float(drift.average_precision_score(labels[mask], late_rule[mask]))
                    else:
                        expected = None
                    self.assertEqual(month["rule"]["pr_auc"], monitor._round(expected))

    def test_validation_numbers_match_the_selection_files(self):
        eta = json.loads((PHASE2_RESULTS / "eta" / "selection.json").read_text())
        result = self.payloads["2018-08-22"]
        means = {model["stage"]: model["validation"]["mean"]
                 for model in result["model_checks"]["models"]}
        for stage in ("checkout", "handover"):
            candidate = self.state["selected"][stage]["candidate"]
            stage2 = {row["candidate"]: row["mean"] for row in eta["stage2"][stage]}
            self.assertAlmostEqual(means[stage], round(stage2[candidate], 4), places=12)
        # The late-risk check now reports the blend, not the selected model alone.
        self.assertEqual(means["late_warning"], monitor._late_risk_validation(self.state)["mean"])
        self.assertNotEqual(means["late_warning"], 0.3923)

    def test_validation_sd_and_bias_range_come_from_the_phase2_windows(self):
        result = self.payloads["2018-08-22"]
        eta = pd.read_csv(PHASE2_RESULTS / "eta" / "ladder_windows.csv")
        expected = {}
        for stage in ("checkout", "handover"):
            candidate = self.state["selected"][stage]["candidate"]
            rows = eta[(eta["stage"] == stage) & (eta["model"] == candidate)]
            expected[stage] = (float(rows["mae"].mean()), float(statistics.stdev(rows["mae"])),
                               float(rows["mean_signed_error"].min()),
                               float(rows["mean_signed_error"].max()))
        expected["late_warning"] = (monitor._late_risk_validation(self.state)["mean"],
                                     monitor._late_risk_validation(self.state)["sd"])
        self.assertEqual(monitor._late_risk_validation(self.state)["n_windows"], 5)
        roc = monitor._late_risk_validation(self.state)["roc_auc"]
        self.assertEqual(roc["n_windows"], 5)
        self.assertEqual(len(roc["values"]), 5)
        self.assertGreater(roc["mean"], 0.5)
        self.assertGreater(roc["sd"], 0)
        for model in result["model_checks"]["models"]:
            mean, sd = expected[model["stage"]][:2]
            self.assertAlmostEqual(model["validation"]["mean"], round(mean, 4), places=12)
            self.assertAlmostEqual(model["validation"]["sd"], round(sd, 4), places=12)
            if model["stage"] == "late_warning":
                self.assertEqual(model["validation"]["roc_auc"], roc)
            if model["kind"] == "regression":
                low, high = expected[model["stage"]][2:]
                self.assertAlmostEqual(model["bias_bounds"]["min"], round(low, 4), places=12)
                self.assertAlmostEqual(model["bias_bounds"]["max"], round(high, 4), places=12)
                bounds = model["bias_bounds"]
                self.assertAlmostEqual(bounds["act_min"], round(bounds["min"] - bounds["sd"], 4),
                                       places=12)
                self.assertAlmostEqual(bounds["act_max"], round(bounds["max"] + bounds["sd"], 4),
                                       places=12)

    def test_maturity_of_the_two_dates(self):
        june = self.payloads["2018-06-30"]
        status = {month["month"]: (month["late_mature"], month["delivery_mature"])
                  for month in june["outcomes"]["months"]}
        self.assertEqual(status["2018-05"], (True, True))
        self.assertEqual(status["2018-06"], (False, False))
        late = next(model for model in june["model_checks"]["models"]
                    if model["stage"] == "late_warning")
        self.assertTrue(all(month.get("too_few") for month in late["months"]))
        card = next(card for card in june["status_cards"] if card["name"] == "late_risk")
        self.assertEqual(card["level"], "none")
        self.assertIn("Not enough mature data", card["reason"])

        august = self.payloads["2018-08-22"]
        status = {month["month"]: (month["late_mature"], month["delivery_mature"])
                  for month in august["outcomes"]["months"]}
        self.assertEqual(status["2018-05"], (True, True))
        self.assertEqual(status["2018-06"], (True, True))
        self.assertEqual(status["2018-07"], (True, True))
        self.assertEqual(status["2018-08"], (False, False))
        late = next(model for model in august["model_checks"]["models"]
                    if model["stage"] == "late_warning")
        self.assertEqual([month["month"] for month in late["months"]], ["2018-05", "2018-06", "2018-07"])
        self.assertTrue(late["months"][0]["too_few"])
        self.assertFalse(late["months"][1].get("too_few", False))
        self.assertFalse(late["months"][2].get("too_few", False))

    def test_unmature_month_marks_delivery_days_and_late_rate_not_ready(self):
        august = self.payloads["2018-08-22"]
        month = next(row for row in august["outcomes"]["months"] if row["month"] == "2018-08")
        self.assertFalse(month["delivery_mature"])
        self.assertFalse(month["late_mature"])
        self.assertIsNotNone(month["mean_delivery_days"])

    def test_model_checks_use_the_test_split_and_need_enough_orders(self):
        august = self.payloads["2018-08-22"]
        checks = {model["stage"]: model for model in august["model_checks"]["models"]}
        self.assertIn("test split", august["model_checks"]["cohort_note"].lower())
        self.assertEqual(monitor.MIN_POOLED_WEEKS, 4)
        for stage in ("checkout", "handover"):
            for month in checks[stage]["months"]:
                if month["month"] == "2018-05":
                    self.assertTrue(month["too_few"], msg=f"{stage} May")
                    self.assertIn("purchase weeks", month["reason"])
                    self.assertIsNone(month["rule_gap"])
                else:
                    self.assertFalse(month.get("too_few", False), msg=f"{stage} {month['month']}")
                    self.assertGreaterEqual(month["orders"], 500)
                    self.assertIsNotNone(month["rule_gap"])
        late_may = checks["late_warning"]["months"][0]
        self.assertEqual(late_may["month"], "2018-05")
        self.assertTrue(late_may["too_few"])
        self.assertEqual(late_may["orders"], 755)
        self.assertEqual(late_may["late_orders"], 2)
        self.assertEqual(monitor.MIN_LATE_ORDERS, 50)
        self.assertIn("Not enough late orders", late_may["reason"])

    def test_status_cards_come_from_the_model_checks(self):
        august = self.payloads["2018-08-22"]
        cards = {card["name"]: card for card in august["status_cards"]}
        self.assertEqual(cards["checkout_eta"]["level"], "act")
        self.assertEqual(cards["checkout_eta"]["reason"], "Predicting about 4 days too long.")
        self.assertNotIn("Jun 2018", cards["checkout_eta"]["reason"])
        self.assertEqual(cards["checkout_eta"]["month"], "2018-07")
        self.assertEqual(cards["handover_eta"]["level"], "act")
        self.assertEqual(cards["handover_eta"]["reason"], "Predicting about 3 days too long.")
        self.assertEqual(cards["handover_eta"]["month"], "2018-07")
        self.assertEqual(cards["late_risk"]["level"], "none")
        self.assertEqual(cards["late_risk"]["reason"],
                         "Late-risk blend within its validation range in Jul 2018.")
        self.assertEqual(cards["late_risk"]["month"], "2018-07")
        models = {model["stage"]: model for model in august["model_checks"]["models"]}
        for card in cards.values():
            stage = {"checkout_eta": "checkout", "handover_eta": "handover",
                     "late_risk": "late_warning"}[card["name"]]
            overall = [month["levels"]["overall"] for month in models[stage]["months"]
                       if not month.get("too_few")]
            self.assertIn(card["level"], overall)

    def test_alerts_are_short_newest_first_and_plain(self):
        august = self.payloads["2018-08-22"]
        self.assertTrue(august["alerts"])
        self.assertEqual({alert["kind"] for alert in august["alerts"]}, {"model"})
        seen = set()
        for alert in august["alerts"]:
            self.assertIn(alert["level"], ("watch", "act"))
            self.assertTrue(alert["sentence"].endswith("."))
            self.assertGreater(len(alert["sentence"]), 20)
            self.assertNotIn("_", alert["sentence"])
            key = (alert["model"], alert["month"])
            self.assertNotIn(key, seen)
            seen.add(key)
        months = [alert["month"] for alert in august["alerts"]]
        self.assertEqual(months, sorted(months, reverse=True))
        for model in august["model_checks"]["models"]:
            for month in model["months"]:
                if month.get("too_few") or month["level"] == "none":
                    self.assertNotIn((model["name"], month["month"]), seen)

    def test_input_labels_are_plain_names(self):
        self.assertEqual(monitor._feature_label("remaining_slack"), "Remaining slack")
        self.assertEqual(monitor._feature_label("past_promise_at_handover"), "Already past promise")
        self.assertEqual(monitor._feature_label("state_backlog_recent_vs_8wk"), "Regional backlog")
        self.assertEqual(monitor._feature_label("route_typical_days"),
                         "Route's usual delivery time")
        self.assertEqual(monitor._feature_label("handover_days"),
                         "Days from purchase to handover")
        self.assertEqual(monitor._feature_label("promise_slack"), "Promise slack")

    def test_selected_inputs_use_the_importance_cutoff(self):
        result = self.payloads["2018-08-22"]
        self.assertEqual(result["inputs"]["importance_cutoff"], monitor.IMPORTANCE_CUTOFF)
        features = result["inputs"]["features"]
        self.assertTrue(features)
        self.assertEqual(len({item["feature"] for item in features}), len(features))
        for item in features:
            self.assertGreaterEqual(item["threshold"], 0.0)
            self.assertEqual(len(item["validation"]), 11)
            self.assertTrue(item["used_by"])
        labels = {item["label"] for item in features}
        self.assertIn("Remaining slack", labels)
        self.assertIn("Promised delivery time", labels)
        self.assertTrue(all("_" not in label for label in labels))

    def test_inputs_have_only_normal_and_unusual_states(self):
        for date, result in self.payloads.items():
            for item in result["inputs"]["features"]:
                for month in result["inputs"]["windows"]:
                    self.assertIn(item["months"][month]["level"], ("normal", "unusual"),
                                  msg=f"{date} {month} {item['feature']}")
            self.assertIn("Unusual", result["inputs"]["threshold_note"])
            self.assertIn("stored with that model version in MLflow",
                          result["inputs"]["recalibration_note"])

    def test_validation_psi_is_reported_without_false_alarms(self):
        for date, result in self.payloads.items():
            inputs = result["inputs"]
            validation = inputs["validation"]
            self.assertIn(validation["reading"], ("monthly", "window"))
            self.assertEqual(validation["reading_count"], 11 if validation["reading"] == "monthly" else 5)
            features = inputs["features"]
            self.assertNotIn("false_alarm", validation)
            psi = validation["psi"]
            self.assertEqual(psi["bins"], 10)
            self.assertEqual(psi["epsilon"], 0.0001)
            self.assertEqual(psi["readings"], validation["reading_count"] * len(features))
            for cutoff in ("0.1", "0.25"):
                self.assertGreaterEqual(psi["flagged"][cutoff], 0)
                self.assertLessEqual(psi["flagged"][cutoff], psi["readings"])
            self.assertNotIn("text", psi)
            for row in psi["features"]:
                self.assertEqual(row["readings"], validation["reading_count"])

    def test_status_cards_count_shifted_inputs_without_changing_status(self):
        result = self.payloads["2018-08-22"]
        cards = {card["name"]: card for card in result["status_cards"]}
        self.assertEqual(cards["checkout_eta"]["level"], "act")
        self.assertEqual(cards["handover_eta"]["level"], "act")
        self.assertEqual(cards["late_risk"]["level"], "none")
        latest = result["inputs"]["windows"][-1]
        for model in result["model_checks"]["models"]:
            short = {"checkout": "Checkout", "handover": "Handover",
                     "late_warning": "Late risk"}[model["stage"]]
            count = sum(1 for item in result["inputs"]["features"]
                        if short in item["used_by"]
                        and item["months"][latest]["level"] == "unusual")
            noun = "input" if count == 1 else "inputs"
            expected = f"{count} {noun} shifted" if count else ""
            card = cards[model["name"]]
            self.assertEqual(card["input_note"], expected)

    def test_caching_reuses_the_first_load(self):
        first = monitor.build_monitor("2018-06-30")
        second = monitor.build_monitor("2018-06-30")
        self.assertTrue(second["cached"])
        self.assertEqual(second["first_load_ms"], first["first_load_ms"])
        self.assertEqual(second["alerts"], first["alerts"])

    def test_html_report_is_generated_and_cached(self):
        html = monitor.build_report_html("2018-06-30")
        self.assertTrue(html.lstrip().startswith("<!DOCTYPE html>"))
        self.assertIn("KS distance", html)
        self.assertIn("Total variation distance", html)
        self.assertIn("Scores computed with Evidently", html)
        self.assertIs(monitor.build_report_html("2018-06-30"), html)

    def test_html_report_verdicts_agree_with_the_dashboard(self):
        result = self.payloads["2018-06-30"]
        html = monitor.build_report_html("2018-06-30")
        latest = result["inputs"]["windows"][-1]
        for item in result["inputs"]["features"]:
            start = html.find("column '{}'".format(item["feature"]))
            self.assertGreaterEqual(start, 0, item["feature"])
            snippet = html[start:start + 260]
            detected = "Data drift detected" in snippet
            self.assertEqual(detected, item["months"][latest]["level"] == "unusual",
                             msg=item["feature"])


class DriftThresholdLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        uri = "sqlite:///{}".format((Path(self.tmp.name) / "mlflow.db").resolve())
        self.client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        self.experiment = self.client.create_experiment(
            "thresholds", artifact_location=(Path(self.tmp.name) / "artifacts").resolve().as_uri())

    def _log_model(self, name, features, log_artifact=True, raw=None):
        run_id = self.client.create_run(self.experiment).info.run_id
        payload = {
            "model": name,
            "reference": "champion training data",
            "readings": "monthly",
            "n_readings": 11,
            "inputs": {
                feature: {
                    "threshold": threshold,
                    "statistic": "ks",
                    "readings": 11,
                    "validation": [threshold],
                }
                for feature, threshold in features.items()
            },
        }
        if log_artifact:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / monitor.DRIFT_THRESHOLDS_ARTIFACT
                path.write_text(raw if raw is not None else json.dumps(payload))
                self.client.log_artifact(run_id, str(path))
        self.client.create_registered_model(name)
        version = self.client.create_model_version(name, source="model", run_id=run_id)
        self.client.set_registered_model_alias(name, "champion", str(version.version))

    def test_champion_thresholds_use_the_lower_value_for_a_shared_input(self):
        self._log_model("checkout_eta", {"shared": 0.4, "a": 0.1})
        self._log_model("handover_eta", {"shared": 0.2})
        self._log_model("late_risk", {"shared": 0.3, "b": 0.2})
        merged, reading, n_readings = monitor._champion_thresholds(self.client)
        self.assertEqual((reading, n_readings), ("monthly", 11))
        self.assertEqual(merged["shared"]["threshold"], 0.2)
        self.assertTrue({"a", "b"} <= set(merged))

    def test_missing_artifact_is_a_loud_error_naming_the_model(self):
        self._log_model("checkout_eta", {}, log_artifact=False)
        with self.assertRaises(monitor.MonitorUnavailable) as caught:
            monitor._load_model_drift_thresholds(self.client, "checkout_eta")
        message = str(caught.exception)
        self.assertIn("checkout_eta", message)
        self.assertIn("python -m phase3.registry", message)

    def test_malformed_artifact_is_a_loud_error_naming_the_model(self):
        self._log_model("checkout_eta", {}, raw="{not json")
        with self.assertRaises(monitor.MonitorUnavailable) as caught:
            monitor._load_model_drift_thresholds(self.client, "checkout_eta")
        message = str(caught.exception)
        self.assertIn("checkout_eta", message)
        self.assertIn("python -m phase3.registry", message)


class MissingDataTests(unittest.TestCase):
    def test_missing_root_raises_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(monitor.MonitorUnavailable):
                monitor.build_monitor("2018-06-30", root=Path(tmp))

    def test_invalid_dates_raise_value_error(self):
        for value in (None, "", "not-a-date", "2018-13-01", "2018-08-22T00:00:00"):
            with self.assertRaises(ValueError):
                monitor.validate_date(value)


if __name__ == "__main__":
    unittest.main()


class RegistryLoadingTests(unittest.TestCase):
    def test_registry_and_serving_are_the_package_modules(self):
        from phase3 import registry, serving

        self.assertIs(monitor.registry, registry)
        self.assertIs(monitor.serving, serving)
        self.assertTrue(callable(monitor.registry.paired_week_gap))
