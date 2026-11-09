"""Checks for the Models API and the read-only MLflow proxy.

Run from the repository root with the project environment:
    python deployment/olist_dashboard/verify_models.py
"""

from __future__ import annotations

import contextlib
import csv
import datetime
import http.client
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "deployment" / "olist_dashboard"
for entry in (str(ROOT / "src"), str(APP)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import server  # noqa: E402
from models_api import MODEL_SETTING_KEYS  # noqa: E402
from phase3.mlflow_server import mlflow_server  # noqa: E402

from olist.paths import DASHBOARD_PHASE2, PHASE3_OUTPUTS

SUMMARY = PHASE3_OUTPUTS / "registry" / "summary.json"
ETA_TEST = DASHBOARD_PHASE2 / "eta" / "test_metrics.csv"
ETA_CV_SUMMARY = DASHBOARD_PHASE2 / "eta" / "cv_summary.csv"
LATE_TEST = DASHBOARD_PHASE2 / "handover" / "test_metrics.csv"
LATE_MONTHS = DASHBOARD_PHASE2 / "handover" / "test_monthly_counts.csv"
REG_FIELDS = ("mae", "rmse", "r2", "mean_signed_error")
CLS_FIELDS = ("pr_auc", "mcc", "roc_auc", "precision", "recall")
TOLERANCE = 1e-9


class DummyAsk:
    store = None
    model = None


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, path, method="GET", payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request("http://127.0.0.1:{}{}".format(port, path),
                                 data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            return response.status, response.headers.get("Content-Type"), body
    except urllib.error.HTTPError as error:
        try:
            body = error.read()
            return error.code, error.headers.get("Content-Type"), body
        finally:
            error.close()


def json_body(body):
    return json.loads(body.decode("utf-8"))


class ModelsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        cls.store = root / "store"
        cls.results = root / "results"
        env = dict(os.environ)
        env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(
            [sys.executable, "-m", "phase3.registry",
             "--store", str(cls.store), "--results", str(cls.results)],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError("phase3.registry failed:\n{}\n{}".format(completed.stdout, completed.stderr))
        cls.db = cls.store / "mlflow.db"
        cls.previous_db = os.environ.get("PRAZO_MLFLOW_DB")
        cls.previous_ui = os.environ.get("PRAZO_MLFLOW_UI")
        cls._stack = contextlib.ExitStack()
        cls.uri = cls._stack.enter_context(mlflow_server(cls.store))
        os.environ["PRAZO_MLFLOW_DB"] = str(cls.db)
        os.environ["PRAZO_MLFLOW_UI"] = cls.uri
        cls.summary = json.loads((cls.results / "summary.json").read_text())
        cls.server = server.create_server(host="127.0.0.1", port=0, ask_service=DummyAsk())
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = MlflowClient(tracking_uri=cls.uri, registry_uri=cls.uri)
        cls.addClassCleanup(cls._temporary.cleanup)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls._stack.close()
        if cls.previous_db is None:
            os.environ.pop("PRAZO_MLFLOW_DB", None)
        else:
            os.environ["PRAZO_MLFLOW_DB"] = cls.previous_db
        if cls.previous_ui is None:
            os.environ.pop("PRAZO_MLFLOW_UI", None)
        else:
            os.environ["PRAZO_MLFLOW_UI"] = cls.previous_ui

    def models(self):
        status, content_type, body = request(self.port, "/api/models")
        self.assertEqual(status, 200, body)
        self.assertTrue(content_type.startswith("application/json"))
        return json_body(body)

    def test_three_models_match_the_saved_summary_and_phase2_eta_output(self):
        models = {model["name"]: model for model in self.models()}
        self.assertEqual(set(models), {"checkout_eta", "handover_eta", "late_risk"})
        for name in ("checkout_eta", "handover_eta"):
            overall = models[name]["metrics"]["test"]["overall"]
            expected = self.summary[name]["test_metrics"]["overall"]
            for field in REG_FIELDS:
                self.assertAlmostEqual(overall[field], expected[field], delta=TOLERANCE,
                                       msg="{} test {}".format(name, field))
            for month in models[name]["metrics"]["test"]["months"]:
                expected_month = self.summary[name]["test_metrics"][month["month"]]
                for field in REG_FIELDS:
                    self.assertAlmostEqual(month["metrics"][field], expected_month[field], delta=TOLERANCE,
                                           msg="{} {} {}".format(name, month["month"], field))
        with ETA_TEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for stage, name in (("checkout", "checkout_eta"), ("handover", "handover_eta")):
            champion = self.summary[name]["champion_candidate"]
            row = next(item for item in rows
                       if item["stage"] == stage and item["candidate"] == champion and item["scope"] == "overall")
            self.assertAlmostEqual(models[name]["metrics"]["test"]["overall"]["mae"],
                                   float(row["mae"]), delta=TOLERANCE)
        classifier = models["late_risk"]
        for ranking in ("model", "slack_rule"):
            actual = classifier["metrics"]["test"][ranking]
            expected = self.summary["late_risk"]["test_metrics"][ranking]
            for field in CLS_FIELDS:
                self.assertAlmostEqual(actual[field], expected[field], delta=TOLERANCE,
                                       msg="late_risk {} {}".format(ranking, field))
        self.assertEqual(classifier["review_share_percent"], 10.0)

    def test_late_risk_live_model_follows_the_selected_date(self):
        cases = (("2018-05-10", "2018-05-01", "Live model: retrained 1 May 2018"),
                 ("2018-08-22", "2018-08-01", "Live model: retrained 1 Aug 2018"),
                 ("2018-10-18", "2018-10-01", "Live model: retrained 1 Oct 2018"))
        for date, refit, label in cases:
            status, _, body = request(self.port, "/api/models?date={}".format(date))
            self.assertEqual(status, 200, body)
            late = {model["name"]: model for model in json_body(body)}["late_risk"]
            self.assertEqual(late["live_model"]["refit_date"], refit, date)
            self.assertEqual(late["live_model"]["label"], label, date)
        late = {model["name"]: model for model in self.models()}["late_risk"]
        self.assertIsNone(late["live_model"])
        self.assertAlmostEqual(late["warning_days"]["blend"], 6.7, places=1)
        self.assertAlmostEqual(late["warning_days"]["slack"], 4.4, places=1)
        versions = self.client.search_model_versions("name = 'late_risk'")
        refits = sorted(version.tags["refit_date"] for version in versions
                        if version.tags.get("source") == "monthly_refit")
        self.assertEqual(refits, ["2018-05-01", "2018-06-01", "2018-07-01",
                                  "2018-08-01", "2018-09-01", "2018-10-01"])
        self.assertEqual(self.summary["late_risk"]["test_counts"]["model_caught_top10"], 377)
        self.assertEqual(self.summary["late_risk"]["test_counts"]["blend_caught_top10"], 411)

    def test_late_risk_train_metrics_use_the_ten_percent_share(self):
        classifier = {model["name"]: model for model in self.models()}["late_risk"]
        version = self.client.get_model_version_by_alias("late_risk", "champion")
        final = self.client.get_run(version.run_id)
        for field in ("precision", "recall", "f1", "mcc"):
            self.assertIn("train_apparent_{}_top10pct".format(field), final.data.metrics)
            self.assertNotIn("train_apparent_{}_top14pct".format(field), final.data.metrics)
        self.assertFalse([metric for metric in final.data.metrics if metric.endswith("_top14pct")])
        for retired in ("train_apparent_review_share", "train_apparent_flagged_orders",
                        "train_apparent_caught_late_orders"):
            self.assertNotIn(retired, final.data.metrics)
        for field in ("precision", "recall", "mcc"):
            self.assertAlmostEqual(classifier["metrics"]["train"][field],
                                   final.data.metrics["train_apparent_{}_top10pct".format(field)],
                                   delta=TOLERANCE, msg=field)
        self.assertEqual(classifier["review_share_percent"], 10.0)
        self.assertNotIn("train_review_share_percent", classifier)

    def test_grid_includes_every_candidate_and_matches_validation_means(self):
        models = {model["name"]: model for model in self.models()}
        for name, model in models.items():
            experiment = self.client.get_experiment_by_name(name)
            runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
            parent = next(run for run in runs if run.data.tags.get("run_kind") == "selection")
            candidates = {
                run.data.tags["candidate"]: run for run in runs
                if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
                and run.data.tags.get("run_kind") == "candidate"
            }
            grid = {row["name"]: row for row in model["grid"]}
            self.assertEqual(set(grid), set(candidates), name)
            self.assertEqual(sum(1 for row in model["grid"] if row["selected"]), 1, name)
            self.assertTrue(grid[model["best_candidate"]]["selected"], name)
            mean_key = "validation_mean_{}".format(
                "mae" if model["kind"] == "regression" else "pr_auc")
            for candidate, run in candidates.items():
                self.assertAlmostEqual(grid[candidate]["validation_mean"],
                                       float(run.data.metrics[mean_key]), delta=1e-12,
                                       msg="{} {}".format(name, candidate))
                self.assertIn("params", grid[candidate])
            self.assertTrue(any(row["grid_edge"] for row in model["grid"]), name)

    def test_ladders_match_mlflow_runs_and_champion_line(self):
        models = {model["name"]: model for model in self.models()}
        ladder_fields = {
            "regression": ("mae", "rmse", "r2"),
            "classification": ("pr_auc", "roc_auc", "mcc", "precision", "recall"),
        }
        window_fields = {
            "regression": (("mae", "validation_window_mae"), ("rmse", "validation_window_rmse"),
                           ("r2", "validation_window_r2")),
            "classification": (("pr_auc", "validation_window_pr_auc"),
                               ("roc_auc", "validation_window_roc_auc"),
                               ("mcc", "validation_window_mcc_top10pct"),
                               ("precision", "validation_window_precision_top10pct"),
                               ("recall", "validation_window_recall_top10pct")),
        }
        for name, model in models.items():
            experiment = self.client.get_experiment_by_name(name)
            runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
            parent = next(run for run in runs if run.data.tags.get("run_kind") == "selection")
            children = [run for run in runs
                        if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id]
            ladder_runs = [run for run in children if "ladder_family" in run.data.tags]
            api_rows = {(family, row["rung"]): row
                        for family, rows in model["ladders"].items() for row in rows}
            self.assertEqual(set(api_rows), {(run.data.tags["ladder_family"],
                                              int(run.data.tags["ladder_rung"]))
                                             for run in ladder_runs}, name)
            for run in ladder_runs:
                family = run.data.tags["ladder_family"]
                rung = int(run.data.tags["ladder_rung"])
                row = api_rows[(family, rung)]
                self.assertEqual(row["candidate"], run.data.tags["candidate"])
                self.assertEqual(row["params"],
                                 {key: value for key, value in run.data.params.items()
                                  if key in MODEL_SETTING_KEYS})
                self.assertTrue(row["label"] and row["description"] and row["short_label"])
                for field in ladder_fields[model["kind"]]:
                    metric = field if model["kind"] == "regression" else (
                        field if field in ("pr_auc", "roc_auc") else field + "_top10pct")
                    self.assertAlmostEqual(row["train"][field],
                                           float(run.data.metrics["train_{}".format(metric)]),
                                           delta=TOLERANCE, msg="{} {} train {}".format(name, rung, field))
                    self.assertAlmostEqual(row["test"][field],
                                           float(run.data.metrics["test_{}".format(metric)]),
                                           delta=TOLERANCE, msg="{} {} test {}".format(name, rung, field))
                for field, source in window_fields[model["kind"]]:
                    values = [float(point.value)
                              for point in self.client.get_metric_history(run.info.run_id, source)]
                    self.assertEqual(len(values), 5, "{} {} {}".format(name, rung, source))
                    expected_mean = statistics.mean(values)
                    expected_se = statistics.stdev(values) / math.sqrt(len(values))
                    self.assertAlmostEqual(row["validation"]["mean"][field], expected_mean,
                                           delta=TOLERANCE, msg="{} {} val {}".format(name, rung, field))
                    self.assertAlmostEqual(row["validation"]["se"][field], expected_se,
                                           delta=TOLERANCE, msg="{} {} se {}".format(name, rung, field))
                    self.assertEqual(len(row["validation"]["windows"]), 5)
                if row["candidate"] != model["candidate"]:
                    self.assertEqual(run.data.tags.get("test_use"),
                                     "reference_only_selection_on_validation")

            version = self.client.get_model_version_by_alias(name, "champion")
            final_run = self.client.get_run(version.run_id)
            line = model["champion_line"]
            self.assertEqual(line["registered_model"], name)
            self.assertEqual(line["champion_version"], int(version.version))
            self.assertEqual(line["total_versions"],
                             len(self.client.search_model_versions("name = '{}'".format(name))))
            tracked = {run.data.tags["candidate"] for run in children
                       if run.data.tags.get("run_kind") in ("candidate", "ladder")}
            self.assertEqual(line["candidate_count"], len(tracked))
            expected_start = datetime.datetime.fromtimestamp(
                final_run.info.start_time / 1000.0, tz=datetime.timezone.utc
            ).isoformat().replace("+00:00", "Z")
            self.assertEqual(line["final_run_started_at"], expected_start)

    def test_run_counts_sum_to_the_tracked_total(self):
        models = {model["name"]: model for model in self.models()}
        for name, model in models.items():
            counts = model["run_counts"]
            self.assertTrue(counts, name)
            self.assertEqual(sum(counts.values()), model["tracked_runs"], name)
            self.assertEqual(model["tracked_runs"], model["champion_line"]["candidate_count"], name)
            for family, rows in model["ladders"].items():
                for row in rows:
                    self.assertGreaterEqual(
                        counts.get(row["type"], 0), 1,
                        "{} {} {} has no tracked runs".format(name, family, row["candidate"]),
                    )

    def test_comparison_rows_are_slimmed(self):
        expected = {
            "regression": [("train", "mae"), ("validation", "mae"),
                           ("test", "mae"), ("test", "r2")],
            "classification": [("train", "pr_auc"), ("validation", "pr_auc"),
                               ("test", "pr_auc"), ("test", "mcc")],
        }
        moved_to_details = {"rmse", "roc_auc", "precision", "recall", "mean_signed_error"}
        models = {model["name"]: model for model in self.models()}
        for name, model in models.items():
            rows = model["comparison_rows"]
            self.assertEqual([(row["split"], row["metric"]) for row in rows],
                             expected[model["kind"]], name)
            self.assertEqual(model["has_slack_rule"], model["kind"] == "classification", name)
            self.assertFalse(moved_to_details & {row["metric"] for row in rows}, name)
            labels = [row["label"] for row in rows]
            if model["kind"] == "regression":
                self.assertIn("Test R\u00b2", labels, name)
            else:
                self.assertIn("Test MCC", labels, name)

    def test_benchmarks_come_from_the_phase2_reference_rules(self):
        models = {model["name"]: model for model in self.models()}
        checkout = models["checkout_eta"]["benchmark"]
        handover = models["handover_eta"]["benchmark"]
        late = models["late_risk"]["benchmark"]
        self.assertEqual(checkout["name"], "route_typical_days")
        self.assertEqual(checkout["label"], "Route's average so far")
        self.assertEqual(handover["name"], "elapsed_plus_route_baseline")
        self.assertEqual(late["name"], "slack_rule")
        with ETA_CV_SUMMARY.open(newline="", encoding="utf-8") as handle:
            expected = float(next(
                row["mean_mae"] for row in csv.DictReader(handle)
                if row["stage"] == "handover" and row["candidate"] == "elapsed_plus_route_baseline"
            ))
        self.assertAlmostEqual(handover["validation"]["mean"]["mae"], expected, delta=TOLERANCE)
        self.assertTrue(math.isfinite(handover["test"]["mae"]))
        self.assertTrue(math.isfinite(checkout["test"]["mae"]))
        self.assertLessEqual(abs(checkout["test"]["mae"] - 5.21), 0.2)
        windows = late["validation"]["windows"]
        self.assertEqual(len(windows), 5)
        self.assertTrue(all(math.isfinite(row["pr_auc"]) for row in windows))
        self.assertAlmostEqual(late["test"]["pr_auc"], 0.519, places=3)

    def test_references_carry_wording_and_metrics_for_every_rule(self):
        models = {model["name"]: model for model in self.models()}
        self.assertEqual([row["key"] for row in models["checkout_eta"]["references"]],
                         ["route_typical_days", "mean_baseline"])
        self.assertEqual([row["key"] for row in models["handover_eta"]["references"]],
                         ["elapsed_plus_route_baseline", "mean_baseline"])
        late = {row["key"]: row for row in models["late_risk"]["references"]}
        self.assertEqual(list(late), ["deadline_rule", "slack_rule", "blend_rule", "random_rule"])
        self.assertEqual(late["slack_rule"]["role"], "headline")
        self.assertEqual(late["deadline_rule"]["role"], "context")
        self.assertEqual(late["blend_rule"]["role"], "context")
        self.assertEqual(late["random_rule"]["role"], "floor")
        self.assertEqual(late["deadline_rule"]["short"],
                         "Operations rule: parcels with the fewest days left go first.")
        self.assertEqual(late["slack_rule"]["label"], "Remaining slack")
        self.assertEqual(late["slack_rule"]["short"],
                         "Operations rule: parcels with the least remaining slack go first.")
        for reference in late.values():
            self.assertTrue(reference["label"] and reference["hover"], reference["key"])
            self.assertEqual(len(reference["validation"]["windows"]), 5, reference["key"])
        self.assertAlmostEqual(late["slack_rule"]["test"]["pr_auc"], 0.519, places=3)
        self.assertAlmostEqual(late["blend_rule"]["test"]["pr_auc"], 0.522, places=3)  # monthly-refit blend
        self.assertAlmostEqual(late["random_rule"]["test"]["pr_auc"], 0.035, places=3)
        floor = models["checkout_eta"]["references"][1]
        self.assertEqual(floor["label"], "Predict the average for everyone")
        self.assertAlmostEqual(floor["validation"]["mean"]["mae"], 7.215, places=3)
        self.assertAlmostEqual(floor["test"]["mae"], 6.267, places=3)

    def test_late_risk_rate_and_months_come_from_the_saved_test_outputs(self):
        late = {model["name"]: model for model in self.models()}["late_risk"]
        with LATE_TEST.open(newline="", encoding="utf-8") as handle:
            expected = float(next(
                row for row in csv.DictReader(handle)
                if row["variant"] == "primary" and row["ranking"] == "model"
            )["late_rate"])
        self.assertAlmostEqual(late["late_rate"], expected, delta=TOLERANCE)
        with LATE_MONTHS.open(newline="", encoding="utf-8") as handle:
            months = sorted({row["month"] for row in csv.DictReader(handle)})
        self.assertEqual(late["test_months"], months)

    def test_validation_windows_and_permutation_importance(self):
        models = {model["name"]: model for model in self.models()}
        for name, model in models.items():
            self.assertEqual(len(model["metrics"]["validation"]["windows"]), 5, name)
            mean = model["metrics"]["validation"]["mean"]
            self.assertTrue(mean, name)
            version = self.client.get_model_version_by_alias(name, "champion")
            status, _, body = request(self.port, "/api/models/{}/importance".format(name))
            self.assertEqual(status, 200, body)
            payload = json_body(body)
            self.assertEqual(payload["model"], name)
            self.assertLessEqual(len(payload["inputs"]), 15)
            values = [row["importance_mean"] for row in payload["inputs"]]
            self.assertEqual(values, sorted(values, reverse=True))
            allowed = {"mixed/overlapping", "higher \u2192 more likely late",
                       "higher \u2192 less likely late", "higher \u2192 more days",
                       "higher \u2192 fewer days"}
            self.assertTrue(all(row["input"] and row["direction"] in allowed
                                for row in payload["inputs"]))
            self.assertTrue(any(row["members"].find(",") >= 0 for row in payload["inputs"]))
            self.assertTrue(all(row["importance_std"] >= 0 for row in payload["inputs"]))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(self.client.download_artifacts(version.run_id, "permutation_importance.csv", tmp))
                with path.open(newline="", encoding="utf-8") as handle:
                    total = sum(1 for _ in csv.DictReader(handle))
            self.assertEqual(payload["total"], total)
            self.assertGreater(payload["total"], 5)

    def test_full_hyperparameters_contain_only_model_settings(self):
        models = {model["name"]: model for model in self.models()}
        for name, model in models.items():
            self.assertTrue(set(model["params"]) <= set(MODEL_SETTING_KEYS), name)
            self.assertNotIn("test_gap_method", model["params"], name)
            self.assertFalse(any(key.startswith("reference_") for key in model["params"]), name)
            for row in model["grid"]:
                self.assertTrue(set(row["params"]) <= set(MODEL_SETTING_KEYS), name)
            for rows in model["ladders"].values():
                for row in rows:
                    self.assertTrue(set(row["params"]) <= set(MODEL_SETTING_KEYS), name)

    def test_late_risk_rule_validation_measures_match_the_store(self):
        model = {item["name"]: item for item in self.models()}["late_risk"]
        version = self.client.get_model_version_by_alias("late_risk", "champion")
        final = self.client.get_run(version.run_id)
        rule = next(reference for reference in model["references"]
                    if reference["role"] == "headline")
        for field in ("pr_auc", "roc_auc", "mcc", "precision", "recall"):
            self.assertAlmostEqual(
                rule["validation"]["mean"][field],
                final.data.metrics["reference_slack_rule_validation_mean_{}".format(field)],
                delta=TOLERANCE, msg=field)
            history = {point.step: float(point.value) for point in self.client.get_metric_history(
                final.info.run_id, "reference_slack_rule_validation_window_{}".format(field))}
            for window in rule["validation"]["windows"]:
                self.assertAlmostEqual(window[field], history[window["window"]],
                                       delta=TOLERANCE, msg="{} {}".format(field, window["window"]))
        self.assertEqual(len(rule["validation"]["windows"]), 5)
        for field in ("pr_auc", "roc_auc", "mcc", "precision", "recall"):
            self.assertIn(field, rule["test"])

    def test_tuning_rows_use_paired_gaps_to_the_chosen_setting(self):
        models = {model["name"]: model for model in self.models()}
        window_key = {"regression": "validation_window_mae", "classification": "validation_window_pr_auc"}
        for name, model in models.items():
            experiment = self.client.get_experiment_by_name(name)
            runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
            candidate_runs = {run.data.tags["candidate"]: run for run in runs
                              if run.data.tags.get("run_kind") == "candidate"}
            histories = {
                candidate: {point.step: float(point.value) for point in self.client.get_metric_history(
                    run.info.run_id, window_key[model["kind"]])}
                for candidate, run in candidate_runs.items()
            }
            chosen = next(row for row in model["grid"] if row["selected"])
            chosen_history = histories[model["candidate"]]
            for row in model["grid"]:
                self.assertAlmostEqual(row["gap_signed"],
                                       row["validation_mean"] - chosen["validation_mean"],
                                       delta=1e-12, msg=name)
                history = histories[row["name"]]
                differences = [history[step] - chosen_history[step] for step in sorted(history)]
                expected = (statistics.stdev(differences) / math.sqrt(len(differences))
                            if len(differences) > 1 else 0.0)
                self.assertAlmostEqual(row["se_of_gap"], expected, delta=1e-12,
                                       msg="{} {}".format(name, row["name"]))

    def test_proxy_refuses_writes_and_reports_a_missing_upstream(self):
        status, _, body = request(self.port, "/mlflow/api/2.0/mlflow/runs/delete", "POST", {})
        self.assertEqual(status, 403)
        self.assertEqual(json_body(body)["error"]["code"], "forbidden_proxy_request")
        status, _, body = request(self.port, "/mlflow/api/2.0/mlflow/runs/delete", "DELETE")
        self.assertEqual(status, 403)
        self.assertEqual(json_body(body)["error"]["code"], "forbidden_proxy_request")
        previous = os.environ.get("PRAZO_MLFLOW_UI")
        os.environ["PRAZO_MLFLOW_UI"] = "http://127.0.0.1:{}".format(free_port())
        try:
            status, _, body = request(self.port, "/mlflow/")
        finally:
            if previous is None:
                os.environ.pop("PRAZO_MLFLOW_UI", None)
            else:
                os.environ["PRAZO_MLFLOW_UI"] = previous
        self.assertEqual(status, 502)
        message = json_body(body)["error"]["message"]
        self.assertIn("run_local.sh", message)

    def test_refused_post_body_does_not_corrupt_the_next_request(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            body = json.dumps({"experiment_ids": ["1"]})
            connection.request("POST", "/mlflow/api/2.0/mlflow/runs/delete", body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 403)
            self.assertEqual(response.getheader("Connection"), "close")
            connection.request("GET", "/api/models")
            follow = connection.getresponse()
            follow.read()
            self.assertEqual(follow.status, 200)
        finally:
            connection.close()

    def test_missing_store_returns_503_with_the_build_command(self):
        previous = os.environ.get("PRAZO_MLFLOW_DB")
        previous_ui = os.environ.get("PRAZO_MLFLOW_UI")
        missing = Path(self._temporary.name) / "missing" / "mlflow.db"
        os.environ["PRAZO_MLFLOW_DB"] = str(missing)
        os.environ.pop("PRAZO_MLFLOW_UI", None)
        try:
            status, _, body = request(self.port, "/api/models")
        finally:
            if previous is None:
                os.environ.pop("PRAZO_MLFLOW_DB", None)
            else:
                os.environ["PRAZO_MLFLOW_DB"] = previous
            if previous_ui is None:
                os.environ.pop("PRAZO_MLFLOW_UI", None)
            else:
                os.environ["PRAZO_MLFLOW_UI"] = previous_ui
        self.assertEqual(status, 503)
        error = json_body(body)["error"]
        self.assertEqual(error["code"], "models_unavailable")
        self.assertIn("python -m phase3.registry", error["message"])

    def test_renamed_validation_metric_keys_are_read_by_the_api(self):
        models = {model["name"]: model for model in self.models()}
        window_fields = {
            "regression": (("mae", "validation_window_mae"), ("rmse", "validation_window_rmse"),
                           ("r2", "validation_window_r2")),
            "classification": (("pr_auc", "validation_window_pr_auc"),
                               ("roc_auc", "validation_window_roc_auc"),
                               ("mcc", "validation_window_mcc_top10pct"),
                               ("precision", "validation_window_precision_top10pct"),
                               ("recall", "validation_window_recall_top10pct")),
        }
        bare_fields = ("mae", "rmse", "r2", "pr_auc", "roc_auc",
                       "precision_top10", "recall_top10", "precision", "recall", "mcc")
        for name, model in models.items():
            experiment = self.client.get_experiment_by_name(name)
            runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
            parent = next(run for run in runs if run.data.tags.get("run_kind") == "selection")
            ladder_runs = [run for run in runs
                           if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
                           and "ladder_family" in run.data.tags]
            api_rows = {(family, row["rung"]): row
                        for family, rows in model["ladders"].items() for row in rows}
            mean_key = "validation_mean_{}".format(
                "mae" if model["kind"] == "regression" else "pr_auc")
            for run in ladder_runs:
                self.assertIn(mean_key, run.data.metrics, name)
                for bare in bare_fields:
                    self.assertNotIn(bare, run.data.metrics, "{} {}".format(name, bare))
                row = api_rows[(run.data.tags["ladder_family"],
                                int(run.data.tags["ladder_rung"]))]
                for field, source in window_fields[model["kind"]]:
                    history = {point.step: float(point.value)
                               for point in self.client.get_metric_history(run.info.run_id, source)}
                    self.assertEqual(len(history), 5, "{} {} {}".format(name, row["candidate"], source))
                    for window in row["validation"]["windows"]:
                        self.assertAlmostEqual(window[field], history[window["window"]],
                                               delta=TOLERANCE,
                                               msg="{} {} {}".format(name, row["candidate"], field))

    def test_test_gap_verdicts_pass_through_the_api(self):
        models = {model["name"]: model for model in self.models()}
        for name, model in models.items():
            experiment = self.client.get_experiment_by_name(name)
            runs = self.client.search_runs([experiment.experiment_id], max_results=5000)
            parent = next(run for run in runs if run.data.tags.get("run_kind") == "selection")
            children = [run for run in runs
                        if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id]
            ladder_runs = [run for run in children if "ladder_family" in run.data.tags]
            api_rows = {(family, row["rung"]): row
                        for family, rows in model["ladders"].items() for row in rows}
            for run in ladder_runs:
                row = api_rows[(run.data.tags["ladder_family"],
                                int(run.data.tags["ladder_rung"]))]
                self.assertAlmostEqual(row["test_gap_vs_headline"],
                                       float(run.data.metrics["test_gap_vs_headline"]),
                                       delta=TOLERANCE, msg=name)
                self.assertAlmostEqual(row["test_gap_low_vs_headline"],
                                       float(run.data.metrics["test_gap_low_vs_headline"]),
                                       delta=TOLERANCE, msg=name)
                self.assertAlmostEqual(row["test_gap_high_vs_headline"],
                                       float(run.data.metrics["test_gap_high_vs_headline"]),
                                       delta=TOLERANCE, msg=name)
                self.assertEqual(row["test_gap_verdict"], run.data.tags["test_gap_verdict"])
                self.assertTrue(row["test_gap_method"], name)

            version = self.client.get_model_version_by_alias(name, "champion")
            final = self.client.get_run(version.run_id)
            for reference in model["references"]:
                prefix = "reference_{}_".format(reference["key"])
                if reference["role"] in ("headline", "floor"):
                    self.assertIsNone(reference["test_gap_verdict"], reference["key"])
                    continue
                self.assertAlmostEqual(reference["test_gap_vs_headline"],
                                       float(final.data.metrics[prefix + "test_gap_vs_headline"]),
                                       delta=TOLERANCE, msg=reference["key"])
                self.assertAlmostEqual(reference["test_gap_low_vs_headline"],
                                       float(final.data.metrics[prefix + "test_gap_low_vs_headline"]),
                                       delta=TOLERANCE, msg=reference["key"])
                self.assertAlmostEqual(reference["test_gap_high_vs_headline"],
                                       float(final.data.metrics[prefix + "test_gap_high_vs_headline"]),
                                       delta=TOLERANCE, msg=reference["key"])
                self.assertEqual(reference["test_gap_verdict"],
                                 final.data.tags[prefix + "test_gap_verdict"])
                self.assertTrue(reference["test_gap_method"], reference["key"])

    def test_models_js_verdict_mapping(self):
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        script = r'''
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const code = fs.readFileSync(path.join(process.cwd(), "deployment", "olist_dashboard", "models.js"), "utf8");
const sandbox = {
  window: {
    location: { hash: "" },
    addEventListener() {},
    matchMedia() { return { matches: false }; },
    setTimeout() {}
  },
  document: {
    getElementById() { return null; },
    querySelectorAll() { return []; },
    querySelector() { return null; }
  },
  console: console
};
vm.runInNewContext(code, sandbox);
const api = sandbox.window.PrazoModels;
assert.strictEqual(api.gapDescriptor({kind: "regression"},
  {gap: 0.09, low: -0.17, high: 0.35, verdict: "level"}).tone, "same");
assert.strictEqual(api.gapDescriptor({kind: "regression"},
  {gap: 0.30, low: 0.04, high: 0.56, verdict: "better"}).tone, "good");
assert.strictEqual(api.gapDescriptor({kind: "classification"},
  {gap: -0.30, low: -0.56, high: -0.04, verdict: "worse"}).tone, "amber");
assert.strictEqual(api.gapDescriptor({kind: "regression"},
  {gap: 0.30, low: 0.04, high: 0.56, verdict: "better"}).text, "\u25B2 Model better");
assert.strictEqual(api.gapDescriptor({kind: "classification"},
  {gap: -0.30, low: -0.56, high: -0.04, verdict: "worse"}).text, "\u25BC Rule better");
assert.strictEqual(api.gapDescriptor({kind: "regression"},
  {gap: 0.09, low: -0.17, high: 0.35, verdict: "level"}).text, "\u2248 About the same");
const shown = api.visibleReferences({references: [
  {key: "deadline_rule", role: "context", label: "Closest to deadline"},
  {key: "slack_rule", role: "headline", label: "Remaining slack"},
  {key: "blend_rule", role: "context", label: "50/50 blend"},
  {key: "random_rule", role: "floor", label: "Random order"}
]});
assert.strictEqual(shown.map(function (row) { return row.key; }).join(","), "slack_rule");
assert.strictEqual(api.visibleReferences({references: []}).length, 0);
'''
        completed = subprocess.run(["node", "-e", script], cwd=str(ROOT),
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
