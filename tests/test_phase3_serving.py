"""Serving checks: champion equivalence, monthly recalibration and the calibration report."""

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase3 import queue, registry, serving  # noqa: E402
from phase3.mlflow_server import mlflow_server  # noqa: E402


def _store():
    """Use PRAZO_MLFLOW_DB or the local store when present; otherwise build a temporary one."""
    export = os.environ.get("PRAZO_MLFLOW_DB")
    if export:
        return serving.database_path(export).parent, None
    default = serving.DEFAULT_DB
    if default.is_file():
        return default.parent, None
    temporary = tempfile.TemporaryDirectory()
    store = Path(temporary.name) / "store"
    registry.run(store=store, results=Path(temporary.name) / "results")
    return store, temporary


class ChampionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store, cls.temporary = _store()
        cls._stack = contextlib.ExitStack()
        cls.uri = cls._stack.enter_context(mlflow_server(cls.store))
        cls.table, cls.spec, _ = queue.load_queue_table()
        cls.table = cls.table.reset_index(drop=True)
        cls.champions = serving.load_champions(uri=cls.uri)

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def test_three_champions_are_loaded_by_alias_with_versions(self):
        self.assertEqual(set(serving.model_versions(self.champions)), set(serving.MODELS))
        for name, version in serving.model_versions(self.champions).items():
            self.assertIsInstance(version, int)
            self.assertGreaterEqual(version, 1)
            self.assertEqual(set(self.champions.models), set(serving.MODELS))

    def test_late_risk_version_is_the_first_of_the_month(self):
        for date, refit in (("2018-05-10", "2018-05-01"), ("2018-08-22", "2018-08-01"),
                            ("2018-10-18", "2018-10-01")):
            version = serving.late_risk_version_for_date(date, store=self.uri)
            self.assertEqual(version.tags["refit_date"], refit)
            self.assertEqual(version.tags["source"], "monthly_refit")
        with self.assertRaises(RuntimeError):
            serving.late_risk_version_for_date("2018-11-01", store=self.uri)
        model, version = serving.load_late_risk_for_date("2018-08-22", store=self.uri)
        self.assertEqual(version.tags["refit_date"], "2018-08-01")
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_champion_path_reproduces_the_refit_for_three_test_dates(self):
        report = serving.equivalence_check(
            self.champions, self.table, self.spec,
            ["2018-06-15", "2018-07-15", "2018-08-22"])
        self.assertEqual(len(report), 3)
        self.assertTrue(report["same_order"].all(), report)
        self.assertTrue((report["max_risk_diff"] < 1e-9).all(), report)
        self.assertTrue((report["max_eta_diff"] < 1e-9).all(), report)

    def test_champion_queue_carries_eta_days(self):
        context = serving.champion_context(self.champions, self.table, self.spec)
        frame = queue.queue("2018-08-22", context=context)
        self.assertIn("eta_days", frame.columns)
        self.assertTrue(np.isfinite(frame["eta_days"]).all())


class RecalibrationTests(unittest.TestCase):
    def test_fewer_than_50_late_orders_falls_back_to_raw_and_is_flagged(self):
        raw = np.linspace(0.01, 0.9, 100)
        late = np.zeros(100, dtype=bool)
        late[:10] = True
        recalibration = serving.fit_recalibration(raw, late)
        self.assertTrue(recalibration.fallback)
        self.assertEqual(recalibration.pool_late, 10)
        self.assertEqual(recalibration.slope, 1.0)
        self.assertEqual(recalibration.intercept, 0.0)
        np.testing.assert_allclose(serving.apply_recalibration(raw, recalibration), raw)

    def test_logistic_fit_has_positive_slope_and_preserves_order(self):
        rng = np.random.default_rng(7)
        raw = rng.uniform(0.001, 0.999, 4000)
        late = rng.random(4000) < raw
        recalibration = serving.fit_recalibration(raw, late)
        self.assertFalse(recalibration.fallback)
        self.assertGreater(recalibration.slope, 0.0)
        recalibrated = serving.apply_recalibration(raw, recalibration)
        order = np.argsort(raw, kind="stable")
        self.assertTrue(np.all(np.diff(recalibrated[order]) >= -1e-12))

    def test_pool_is_month_minus_two_with_outcome_known_before_month_start(self):
        table = pd.DataFrame({
            "purchase_month": ["2018-06", "2018-06", "2018-06", "2018-07", "2018-08"],
            "outcome_available_at": pd.to_datetime([
                "2018-07-01 08:00", "2018-08-01 08:00", "2018-06-20 08:00",
                "2018-07-15 08:00", "2018-08-10 08:00"]),
            "late": [False, True, False, False, True],
        })
        months = serving.month_periods(table)
        index = serving.calibration_pool_index(table, months, "2018-08")
        self.assertEqual(index.tolist(), [0, 2])

    def test_test_month_calibration_keeps_the_ranking(self):
        table, spec, _ = queue.load_queue_table()
        table = table.reset_index(drop=True)
        store, temporary = _store()
        try:
            with mlflow_server(store) as uri:
                champions = serving.load_champions(uri=uri)
                context = serving.champion_context(champions, table, spec)
            raw = context.table["risk"].to_numpy(dtype=float)
            status = serving.test_calibration_status(table, raw, champions)
            self.assertEqual([row["month"] for row in status["months"]], list(serving.TEST_MONTHS))
            self.assertIsInstance(status["calibration_ok"], bool)
            for month in serving.TEST_MONTHS:
                mask = table["split"].eq("test").to_numpy() & np.asarray(
                    serving.month_periods(table) == pd.Period(month, freq="M"), dtype=bool)
                recalibration = serving.month_calibration(table, serving.month_periods(table), raw, month)
                recalibrated = serving.apply_recalibration(raw[mask], recalibration)
                order = np.argsort(raw[mask], kind="stable")
                self.assertTrue(np.all(np.diff(recalibrated[order]) >= -1e-12), month)
        finally:
            if temporary is not None:
                temporary.cleanup()


class CalibrationReportTests(unittest.TestCase):
    def test_report_has_month_summaries_bins_and_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "calibration_check.csv"
            store, temporary = _store()
            try:
                with mlflow_server(store) as uri:
                    summary = serving.build_calibration_check(store=uri, out=out)
            finally:
                if temporary is not None:
                    temporary.cleanup()
            self.assertTrue(out.is_file())
            frame = pd.read_csv(out)
            expected = {"scope", "month", "calibration", "orders", "late", "actual_late_rate",
                        "mean_predicted", "brier", "fallback", "pool_orders", "pool_late",
                        "slope", "intercept", "rank_preserved", "bin", "bin_lower", "bin_upper",
                        "bin_orders", "bin_mean_predicted", "bin_actual_late_rate"}
            self.assertTrue(expected <= set(frame.columns), set(frame.columns))
            self.assertEqual(set(frame["month"]), set(serving.VALIDATION_MONTHS + serving.TEST_MONTHS))
            self.assertEqual(frame.groupby(["scope", "month", "calibration"])["bin"].count().min(), 10)
            self.assertTrue(frame["rank_preserved"].all())
            self.assertEqual(summary["calibration_ok"], frame.query(
                "scope == 'test' and calibration == 'recalibrated' and bin == 0"
            ).apply(lambda row: abs(row["mean_predicted"] - row["actual_late_rate"]) <= 0.02, axis=1).all())


if __name__ == "__main__":
    unittest.main()
