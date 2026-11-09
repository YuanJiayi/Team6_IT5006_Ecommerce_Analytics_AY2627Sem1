"""Checks for the Models-tab monitoring API.

Run from the repository root with the project environment:
    python deployment/olist_dashboard/test_monitor_api.py
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "deployment" / "olist_dashboard"
for entry in (str(ROOT / "src"), str(APP)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import monitor_api  # noqa: E402


class ValidationTests(unittest.TestCase):
    def test_missing_date_is_a_400(self):
        status, payload = monitor_api.monitor_response(None)
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "bad_request")
        self.assertIn("date", payload["error"]["message"].lower())

    def test_invalid_dates_are_400s(self):
        for value in ("", "not-a-date", "2018-13-01", "2018-02-30", "2018-08-22T00:00:00"):
            status, payload = monitor_api.monitor_response(value)
            self.assertEqual(status, 400, value)
            self.assertEqual(payload["error"]["code"], "bad_request")

    def test_report_missing_date_is_a_400(self):
        status, body, content_type = monitor_api.report_response(None)
        self.assertEqual(status, 400)
        self.assertIn("application/json", content_type)
        self.assertIn(b"bad_request", body)


class UnavailableTests(unittest.TestCase):
    def test_missing_phase2_root_is_a_503(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, payload = monitor_api.monitor_response("2018-06-30", root=Path(tmp))
            self.assertEqual(status, 503)
            self.assertEqual(payload["error"]["code"], "monitor_unavailable")
            self.assertTrue(payload["error"]["message"])

    def test_missing_phase2_root_report_is_a_503(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, body, content_type = monitor_api.report_response("2018-06-30", root=Path(tmp))
            self.assertEqual(status, 503)
            self.assertIn("application/json", content_type)
            self.assertIn(b"monitor_unavailable", body)


@unittest.skipUnless((ROOT / "data" / "processed" / "phase2_order_table.csv").is_file(),
                     "the Phase 2 data is not available")
class RealApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from phase3 import serving
        from phase3.mlflow_server import mlflow_server

        cls.root = ROOT
        cls._stack = contextlib.ExitStack()
        cls._previous_ui = os.environ.get("PRAZO_MLFLOW_UI")
        os.environ["PRAZO_MLFLOW_UI"] = cls._stack.enter_context(
            mlflow_server(serving.DEFAULT_STORE))
        cls.addClassCleanup(cls._restore)

    @classmethod
    def _restore(cls):
        if cls._previous_ui is None:
            os.environ.pop("PRAZO_MLFLOW_UI", None)
        else:
            os.environ["PRAZO_MLFLOW_UI"] = cls._previous_ui
        cls._stack.close()

    def test_monitor_returns_a_payload(self):
        status, payload = monitor_api.monitor_response("2018-06-30", root=self.root)
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["as_of"], "2018-06-30")
        self.assertIn("inputs", payload)
        self.assertIn("outcomes", payload)
        self.assertIn("model_checks", payload)
        self.assertIsInstance(payload["alerts"], list)

    def test_report_returns_html(self):
        status, body, content_type = monitor_api.report_response("2018-06-30", root=self.root)
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertTrue(body.lstrip().startswith(b"<!DOCTYPE html>"))
        self.assertIn(b"KS distance", body)


if __name__ == "__main__":
    unittest.main()
