"""Checks for the Actions queue API.

Run from the repository root with the project environment:
    python deployment/olist_dashboard/test_actions_api.py
or with pytest: python -m pytest deployment/olist_dashboard/test_actions_api.py
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "deployment" / "olist_dashboard"
for entry in (str(ROOT / "src"), str(APP)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import actions_api  # noqa: E402
import server  # noqa: E402


class DummyAsk:
    store = None
    model = None


def _store():
    from phase3 import serving

    export = os.environ.get("PRAZO_MLFLOW_DB")
    if export:
        return serving.database_path(export), None
    default = serving.DEFAULT_DB
    if default.is_file():
        return default, None
    from phase3 import registry

    temporary = tempfile.TemporaryDirectory()
    store = Path(temporary.name) / "store"
    registry.run(store=store, results=Path(temporary.name) / "results")
    return store / "mlflow.db", temporary


def _request(port, path):
    request = urllib.request.Request("http://127.0.0.1:{}{}".format(port, path))
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.headers.get("Content-Type"), response.read()
    except urllib.error.HTTPError as error:
        try:
            return error.code, error.headers.get("Content-Type"), error.read()
        finally:
            error.close()


class ActionsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from phase3.mlflow_server import mlflow_server

        cls.db, cls.temporary = _store()
        cls.mlflow = contextlib.ExitStack()
        cls.store_uri = cls.mlflow.enter_context(mlflow_server(Path(cls.db).parent))
        cls.log_dir = tempfile.TemporaryDirectory()
        cls.log_path = Path(cls.log_dir.name) / "prediction_log.jsonl"
        cls.service = actions_api.ActionsService(store=cls.store_uri, prediction_log=cls.log_path)
        if cls.service.error:
            raise RuntimeError(cls.service.error)
        cls.server = server.create_server(host="127.0.0.1", port=0, ask_service=DummyAsk(),
                                          actions_service=cls.service)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls._stop)

    @classmethod
    def _stop(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.log_dir.cleanup()
        cls.mlflow.close()
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def queue(self, date):
        status, content_type, body = _request(self.port, "/api/actions/queue?date={}".format(date))
        self.assertEqual(status, 200, body)
        self.assertTrue(content_type.startswith("application/json"))
        return json.loads(body.decode("utf-8"))

    def test_queue_for_2018_08_22_has_tiers_and_versions(self):
        payload = self.queue("2018-08-22")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["date"], "2018-08-22")
        self.assertEqual(payload["method"], "blend")
        self.assertEqual(payload["capacity"], 44)
        self.assertIsNone(payload["capacity_note"])
        self.assertEqual(set(payload["model_versions"]), {"checkout_eta", "handover_eta", "late_risk"})
        self.assertEqual(set(payload["tiers"]), {"act", "watch", "none"})
        self.assertEqual(payload["tiers"]["act"], 44)
        self.assertEqual(payload["tiers"]["watch"], 44)
        self.assertEqual(payload["tiers"]["act"] + payload["tiers"]["watch"] + payload["tiers"]["none"]
                         + payload["customer_update"], len(payload["rows"]))
        self.assertEqual(payload["customer_update"],
                         sum(1 for row in payload["rows"] if row["past_promise"]))
        first = payload["rows"][0]
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["tier"], "act")
        self.assertIsNotNone(first["predicted_delivery_date"])
        self.assertLessEqual(first["spare_time"], first["days_left"] + 1.0)
        self.assertGreaterEqual(first["late_risk_percentile"], 0.0)
        self.assertLessEqual(first["late_risk_percentile"], 1.0)
        self.assertGreaterEqual(first["spare_time_percentile"], 0.0)
        self.assertLessEqual(first["spare_time_percentile"], 1.0)
        self.assertGreaterEqual(first["model_rank"], 1)
        self.assertLessEqual(first["model_rank"], first["open_orders"])
        for row in payload["rows"]:
            for field in ("order_id", "rank", "tier",
                          "predicted_delivery_date", "promised_date", "days_left", "spare_time",
                          "late_risk_percentile", "spare_time_percentile", "model_rank", "open_orders",
                          "model_versions"):
                self.assertIn(field, row)
            self.assertIn(row["tier"], {"act", "watch", "none"})
        updates = [row for row in payload["rows"] if row["past_promise"]]
        self.assertTrue(updates)
        self.assertTrue(all(row["suggested_action"] == "customer_update" for row in updates))

    def test_never_delivered_parcel_is_on_the_customer_update_list(self):
        payload = self.queue("2018-08-22")
        order_id = "ee64d42b8cf066f35eac1cf57de1aa85"
        row = next(item for item in payload["rows"] if item["order_id"] == order_id)
        self.assertTrue(row["past_promise"])
        self.assertTrue(row["customer_update"])
        self.assertEqual(row["suggested_action"], "customer_update")
        self.assertIsNone(row["rank"])

    def test_no_queue_row_exposes_a_probability(self):
        for date in ("2018-06-15", "2018-07-16", "2018-08-22"):
            payload = self.queue(date)
            self.assertNotIn("probability_shown", payload)
            self.assertNotIn("calibration", payload)
            for row in payload["rows"]:
                self.assertNotIn("risk", row)
                for field in row:
                    self.assertNotIn("probab", field.lower(),
                                     "{} exposes a probability field".format(date))

    def test_tiers_are_the_capacity_blocks_in_rank_order(self):
        payload = self.queue("2018-06-15")
        self.assertEqual(payload["capacity"], 30)
        ranked = [row for row in payload["rows"] if not row["past_promise"]]
        self.assertEqual([row["rank"] for row in ranked], list(range(1, len(ranked) + 1)))
        self.assertTrue(all(row["tier"] == "act" for row in ranked[:30]))
        self.assertTrue(all(row["tier"] == "watch" for row in ranked[30:60]))
        self.assertTrue(all(row["tier"] == "none" for row in ranked[60:]))

    def test_missing_and_invalid_dates_are_rejected(self):
        status, _, body = _request(self.port, "/api/actions/queue")
        self.assertEqual(status, 400, body)
        status, _, body = _request(self.port, "/api/actions/queue?date=not-a-date")
        self.assertEqual(status, 400, body)

    def test_queue_matches_the_dashboard_winnie_queue(self):
        import pandas as pd
        from phase3 import queue

        payload = self.queue("2018-08-22")
        state = self.service._date_state(pd.Timestamp("2018-08-22"))
        expected = queue.winnie_queue("2018-08-22", context=state["context"],
                                      capacity=payload["capacity"])
        expected = expected[~expected["past_promise"]].sort_values("rank", kind="stable")
        got = [row["order_id"] for row in payload["rows"] if not row["past_promise"]]
        self.assertEqual(got, expected["order_id"].tolist())

    def test_whole_test_replay_is_winnie_b_mix_with_the_champion_scores(self):
        import pandas as pd
        from phase3 import queue

        table = self.service.table
        spec = self.service.spec
        start = pd.Timestamp(spec["primary_cutoff"])
        test = table[table[spec["primary_split"]].eq("test")]
        end = test["order_purchase_timestamp"].max().normalize()
        train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
        orders = table[table["handover_time"].ge(start)
                       & table["handover_time"].lt(end + pd.Timedelta(days=1))
                       & table["late"].notna()]
        orders = orders.sort_values(["handover_time", "order_id"], kind="stable").reset_index(drop=True)
        scored = orders.merge(self.service.context.table[["order_id", "risk"]],
                              on="order_id", how="left", validate="one_to_one")
        model_scores = scored["risk"].to_numpy(dtype=float)
        rule_scores = queue.winnie_rule_scores(train, orders)
        model_ref = self.service.context.table[
            self.service.context.table[spec["primary_split"]].eq("train")]["risk"].to_numpy(dtype=float)
        rule_ref = queue.winnie_rule_scores(train, train)

        reference = queue._winnie_single_daily("test", orders, start, end, model_scores,
                                               rule_scores, model_ref, rule_ref)
        reference = reference[reference["version"].eq("B_mix")].reset_index(drop=True)
        picks, _ = queue.winnie_daily_selection("test", orders, start, end, model_scores,
                                                rule_scores, model_ref, rule_ref,
                                                versions=("B_mix",))

        detail, _ = queue.winnie_daily_replay(self.service.context, start=start, end=end)
        late = detail[detail["late"].astype(bool)]
        caught = int(len(late))
        self.assertEqual(int(reference["caught_before_date"].sum()), caught)
        self.assertEqual(int(reference["caught_3plus_days"].sum()),
                         int((late["days_left"] >= 3).sum()))
        self.assertAlmostEqual(
            float(reference["days_left_total"].sum() / max(caught, 1)),
            float(late["days_left"].mean()),
        )
        expected_ids = {str(morning.date()): sorted(part["order_id"])
                        for morning, part in picks.groupby("morning")}
        got_ids = {str(morning.date()): sorted(part["order_id"])
                   for morning, part in detail.groupby("morning")}
        self.assertEqual(got_ids, expected_ids)


    def promises(self, date):
        status, content_type, body = _request(self.port, "/api/actions/promises?date={}".format(date))
        self.assertEqual(status, 200, body)
        self.assertTrue(content_type.startswith("application/json"))
        return json.loads(body.decode("utf-8"))

    def test_queue_rows_carry_why_contributions_and_performance(self):
        payload = self.queue("2018-08-22")
        self.assertIn("why_base_value", payload)
        self.assertEqual(payload["why_scale"], "logit")
        for row in payload["rows"]:
            self.assertIsInstance(row["why"], list)
            self.assertLessEqual(len(row["why"]), 3)
            for item in row["why"]:
                self.assertIn("input", item)
                self.assertIn("label", item)
                self.assertGreater(item["contribution"], 0.0)
        performance = payload["performance"]
        self.assertEqual(performance["weeks"], 4)
        self.assertLess(performance["start"], performance["end"])
        for method in ("blend", "slack"):
            metrics = performance[method]
            self.assertIn("caught", metrics)
            self.assertIn("caught_3plus_days", metrics)
            self.assertIn("mean_days_left", metrics)
            self.assertLessEqual(metrics["caught_3plus_days"], metrics["caught"])
            if metrics["caught"]:
                self.assertIsNotNone(metrics["mean_days_left"])
        self.assertGreater(performance["blend"]["caught_3plus_days"], 0)
        self.assertGreater(performance["slack"]["caught_3plus_days"], 0)
        self.assertGreaterEqual(payload["elapsed_ms"], 0.0)

    def test_queue_performance_uses_the_winnie_rescue_metrics(self):
        payload = self.queue("2018-06-15")
        performance = payload["performance"]
        self.assertEqual(performance["start"], "2018-05-18")
        self.assertEqual(performance["end"], "2018-06-14")
        self.assertGreater(performance["blend"]["caught_3plus_days"], 0)

        # A parcel due soon is no longer counted as a useful 3+ day rescue.
        later = self.queue("2018-08-22")["performance"]
        self.assertGreater(later["blend"]["caught_3plus_days"],
                           later["slack"]["caught_3plus_days"])
        self.assertGreater(later["blend"]["mean_days_left"],
                           later["slack"]["mean_days_left"])

    def test_why_contributions_are_additive_on_the_logit_scale(self):
        import numpy as np
        import pandas as pd

        state = self.service._date_state(pd.Timestamp("2018-08-22"))
        explainer = self.service._month_explainer(state)
        as_of = pd.Timestamp("2018-08-22")
        open_time = as_of + pd.Timedelta(hours=9)
        rows = state["context"].table[
            state["context"].table["handover_time"].lt(open_time)
            & state["context"].table["outcome_available_at"].gt(open_time)
        ].iloc[:25]
        contributions, logits = explainer.explain(rows)
        combined = explainer.base_value + contributions.sum(axis=1).to_numpy()
        np.testing.assert_allclose(combined, logits, atol=1e-9)
        self.assertEqual(set(contributions.columns), set(self.service.champions.classifier_columns))

    def test_promises_endpoint_uses_only_orders_known_by_the_as_of_morning(self):
        import pandas as pd

        payload = self.promises("2018-08-22")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["run"], "B")
        self.assertEqual(payload["model_label"], "Checkout delivery-time model without Olist's promise as an input")
        self.assertNotIn("model_versions", payload)
        self.assertEqual(len(payload["regions"]), 6)
        for region in payload["regions"]:
            for field in ("region", "buffer_days", "our_mean_promise_days",
                          "olist_mean_promise_days", "n_orders"):
                self.assertIn(field, region)
            self.assertGreater(region["n_orders"], 0)
            self.assertIsNotNone(region["buffer_days"])
            self.assertIsNotNone(region["our_mean_promise_days"])
            self.assertIsNotNone(region["olist_mean_promise_days"])
        table = self.service._policy_table
        month = pd.Period("2018-08")
        cutoff = pd.Timestamp("2018-08-22") + pd.Timedelta(hours=9)
        expected_orders = int(
            (table["purchase_month"].eq(month)
             & table["order_purchase_timestamp"].lt(cutoff)).sum())
        orders = sum(region["n_orders"] for region in payload["regions"])
        self.assertEqual(orders, expected_orders)
        our = sum(region["our_mean_promise_days"] * region["n_orders"]
                  for region in payload["regions"]) / orders
        olist = sum(region["olist_mean_promise_days"] * region["n_orders"]
                    for region in payload["regions"]) / orders
        self.assertAlmostEqual(our, 17.2031, delta=0.01)
        self.assertAlmostEqual(olist, 16.1619, delta=0.01)
        performance = payload["performance"]
        self.assertIsNotNone(performance)
        self.assertTrue(performance["enough"])
        self.assertGreaterEqual(performance["orders"], actions_api.PROMISE_MIN_ORDERS)
        self.assertEqual((performance["start"], performance["end"]), ("2018-06-19", "2018-07-17"))
        self.assertGreater(performance["not_yet_due"], 0)
        self.assertGreater(performance["our_mean_promise_days"], 0.0)
        self.assertGreater(performance["olist_mean_promise_days"], 0.0)
        # The same resolved purchases show our promises are shorter overall.
        self.assertLess(performance["our_mean_promise_days"],
                        performance["olist_mean_promise_days"])
        self.assertGreaterEqual(performance["our_late_rate"], 0.0)
        self.assertLessEqual(performance["our_late_rate"], 1.0)
        self.assertGreaterEqual(performance["olist_late_rate"], 0.0)
        self.assertLessEqual(performance["olist_late_rate"], 1.0)

    def test_promises_on_the_first_morning_use_only_early_orders(self):
        import pandas as pd

        payload = self.promises("2018-07-01")
        table = self.service._policy_table
        month = pd.Period("2018-07")
        cutoff = pd.Timestamp("2018-07-01") + pd.Timedelta(hours=9)
        expected_orders = int(
            (table["purchase_month"].eq(month)
             & table["order_purchase_timestamp"].lt(cutoff)).sum())
        self.assertEqual(sum(region["n_orders"] for region in payload["regions"]),
                         expected_orders)
        self.assertEqual(len(payload["regions"]), 6)
        empty = [region for region in payload["regions"] if region["n_orders"] == 0]
        self.assertTrue(empty)
        for region in empty:
            self.assertIsNone(region["our_mean_promise_days"])
            self.assertIsNone(region["olist_mean_promise_days"])
            self.assertIsNotNone(region["buffer_days"])

    def test_promise_cohort_drops_orders_whose_promises_have_not_passed(self):
        import pandas as pd

        as_of = pd.Timestamp("2018-08-22")
        policy_start = pd.Timestamp("2018-06-01")
        purchased = pd.Timestamp("2018-06-01 12:00")
        frame = pd.DataFrame({
            "order_id": [str(index) for index in range(300)],
            "order_purchase_timestamp": purchased,
            "our_promise_days": 10.0,
            "olist_promise_days": 20.0,
            "delivery_days": 5.0,
            "is_late": False,
            "outcome_available_at": purchased + pd.Timedelta(days=5),
        })
        not_due = pd.DataFrame([{
            "order_id": "not-due",
            "order_purchase_timestamp": purchased,
            "our_promise_days": 100.0,
            "olist_promise_days": 1.0,
            "delivery_days": 5.0,
            "is_late": False,
            "outcome_available_at": purchased + pd.Timedelta(days=5),
        }])
        frame = pd.concat([frame, not_due], ignore_index=True)

        cohort = actions_api.promise_cohort(frame, as_of, policy_start)
        self.assertIsNotNone(cohort)
        self.assertEqual(cohort["window_orders"], len(frame))
        self.assertEqual(len(cohort["due"]), 300)
        self.assertNotIn("not-due", set(cohort["due"]["order_id"]))
        metrics = actions_api.promise_metrics(cohort["due"], as_of + pd.Timedelta(hours=9))
        self.assertAlmostEqual(metrics["our_mean_promise_days"], 10.0)
        self.assertAlmostEqual(metrics["olist_mean_promise_days"], 20.0)
        self.assertLess(metrics["our_mean_promise_days"], metrics["olist_mean_promise_days"])

    def test_promise_performance_needs_a_window_inside_policy_coverage(self):
        performance = self.promises("2018-06-15")["performance"]
        self.assertFalse(performance["enough"])
        self.assertEqual(performance["orders"], 0)
        self.assertIsNone(performance["start"])

    def test_promises_for_a_month_without_scored_orders_is_empty_not_an_error(self):
        for date in ("2018-09-01", "2018-09-15", "2018-10-01", "2018-10-18"):
            payload = self.promises(date)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["regions"], [])
            performance = payload["performance"]
            self.assertIsNotNone(performance)
            self.assertIsInstance(performance["enough"], bool)
        self.assertTrue(self.promises("2018-09-01")["performance"]["enough"])
        self.assertFalse(self.promises("2018-10-18")["performance"]["enough"])

    def test_promises_before_the_phase2_cutoff_are_rejected(self):
        status, _, body = _request(self.port, "/api/actions/promises?date=2018-05-15")
        self.assertEqual(status, 400, body)
        status, _, body = _request(self.port, "/api/actions/promises?date=not-a-date")
        self.assertEqual(status, 400, body)

    def test_missing_store_answers_503_for_both_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            service = actions_api.ActionsService(store=Path(directory) / "missing")
            self.assertIsNotNone(service.error)
            instance = server.create_server(host="127.0.0.1", port=0, ask_service=DummyAsk(),
                                            actions_service=service)
            port = instance.server_address[1]
            thread = threading.Thread(target=instance.serve_forever, daemon=True)
            thread.start()
            try:
                for path in ("/api/actions/queue?date=2018-08-22",
                             "/api/actions/promises?date=2018-08-22"):
                    status, _, body = _request(port, path)
                    self.assertEqual(status, 503, body)
                    payload = json.loads(body.decode("utf-8"))
                    self.assertEqual(payload["error"]["code"], "actions_unavailable")
            finally:
                instance.shutdown()
                instance.server_close()
                thread.join(timeout=5)

    def test_repeated_date_is_served_from_cache(self):
        first = self.queue("2018-07-15")
        second = self.queue("2018-07-15")
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["performance"], second["performance"])


    def test_zero_capacity_morning_explains_and_keeps_the_ranked_list(self):
        payload = self.queue("2018-07-16")
        self.assertEqual(payload["capacity"], 0)
        self.assertEqual(
            payload["capacity_note"],
            "No parcels to act on this morning: the daily list is 10% of yesterday's "
            "handovers, and there were none.")
        ranked = [row for row in payload["rows"] if not row["past_promise"]]
        self.assertTrue(ranked)
        self.assertTrue(all(row["tier"] == "none" for row in ranked))
        self.assertTrue(all(row["model_rank"] is not None for row in ranked))
        self.assertIsNone(self.queue("2018-08-22")["capacity_note"])

    def test_may_dates_before_the_review_period_are_honest_empty_lists(self):
        for date in ("2018-05-01", "2018-05-10", "2018-05-26"):
            payload = self.queue(date)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["rows"], [], date)
            self.assertEqual(payload["customer_update"], 0, date)
            self.assertEqual(payload["tiers"], {"act": 0, "watch": 0, "none": 0}, date)
            self.assertIsInstance(payload["why_base_value"], float)
            for method in ("blend", "slack"):
                metrics = payload["performance"][method]
                self.assertEqual(metrics["caught"], 0)
                self.assertEqual(metrics["caught_3plus_days"], 0)
                self.assertIsNone(metrics["mean_days_left"])

    def test_order_and_predict_do_not_crash_when_the_queue_is_empty(self):
        order_id = "e481f51cbdc54678b7cc49136f2d6af7"
        self.assertEqual(self.service.order_lookup("2018-10-18", order_id)["state"], "delivered")
        prediction = self.predict("2018-10-18", order_id)
        self.assertTrue(prediction["ok"])
        self.assertEqual(prediction["state"], "delivered")
        self.assertFalse(prediction["on_list"])
        unknown = self.predict("2018-10-18", "0" * 32)
        self.assertEqual(unknown["state"], "unknown")

    def test_spare_time_is_the_smaller_of_handover_spare_time_and_days_left(self):
        import numpy as np
        import pandas as pd

        payload = self.queue("2018-08-22")
        rows = [row for row in payload["rows"] if not row["past_promise"]]
        ids = [row["order_id"] for row in rows]
        table = self.service.context.table.set_index("order_id").loc[ids]
        train = self.service.context.table[
            self.service.context.table[self.service.spec["primary_split"]].eq("train")]
        median = train["route_transit_90d_at_handover"].median()
        at_handover = (table["promised_days"] - table["handover_days"]
                       - table["route_transit_90d_at_handover"].fillna(median))
        review = pd.Timestamp("2018-08-22 09:00")
        days_left = (table["promised_days"]
                     - (review - table["order_purchase_timestamp"]).dt.total_seconds() / 86400.0)
        expected = np.minimum(at_handover.to_numpy(dtype=float),
                              days_left.to_numpy(dtype=float))
        for row, value in zip(rows, expected):
            self.assertAlmostEqual(row["spare_time"], float(value), places=9)

    def test_order_lookup_finds_open_orders_and_explains_the_rest(self):
        queue = self.queue("2018-08-22")
        first = next(row for row in queue["rows"] if not row["past_promise"])
        status, _, body = _request(
            self.port,
            "/api/actions/order?date=2018-08-22&order_id={}".format(first["order_id"]))
        self.assertEqual(status, 200, body)
        found = json.loads(body.decode("utf-8"))
        self.assertEqual(found["state"], "open")
        self.assertEqual(found["rank"], first["rank"])

        status, _, body = _request(
            self.port, "/api/actions/order?date=2018-08-22&order_id=not-an-order-id")
        self.assertEqual(status, 400, body)

        delivered = self.service.order_lookup("2018-08-22", "e481f51cbdc54678b7cc49136f2d6af7")
        self.assertEqual(delivered["state"], "delivered")
        not_bought = self.service.order_lookup("2018-08-22", "6d25592267349b322799e2beb687871e")
        self.assertEqual(not_bought["state"], "not_bought")
        not_handed = self.service.order_lookup("2018-08-22", "7d2286ec219f58023352077d938dee48")
        self.assertEqual(not_handed["state"], "not_handed_over")

    def predict(self, date, order_id):
        status, content_type, body = _request(
            self.port, "/api/actions/predict?date={}&order_id={}".format(date, order_id))
        self.assertEqual(status, 200, body)
        self.assertTrue(content_type.startswith("application/json"))
        return json.loads(body.decode("utf-8"))

    def test_predict_equals_the_queue_row_for_an_on_list_order(self):
        import pandas as pd

        date = "2018-08-22"
        order_id = "d3310038bbf991ce02501a51022f5283"
        queue = self.queue(date)
        row = next(item for item in queue["rows"] if item["order_id"] == order_id)
        prediction = self.predict(date, order_id)
        self.assertTrue(prediction["ok"])
        self.assertTrue(prediction["found"])
        self.assertEqual(prediction["state"], "open")
        self.assertTrue(prediction["on_list"])
        for field in ("rank", "open_orders", "spare_time", "late_risk_percentile",
                      "spare_time_percentile", "predicted_delivery_date", "promised_date",
                      "days_left", "tier", "why", "model_versions"):
            self.assertEqual(prediction[field], row[field], field)
        self.assertEqual(prediction["model_versions"],
                         {name: int(version) for name, version in
                          self.service._state_versions(
                              self.service._date_state(pd.Timestamp(date))).items()})

    def test_predict_scores_an_open_off_list_order(self):
        date = "2018-08-22"
        order_id = "d74d2985a4c187334dc59722076a2e5f"
        prediction = self.predict(date, order_id)
        self.assertTrue(prediction["found"])
        self.assertFalse(prediction["on_list"])
        self.assertFalse(prediction["customer_update"])
        self.assertIn(prediction["state"], {"reviewed_earlier", "before_review_period"})
        if prediction["state"] == "reviewed_earlier":
            self.assertIn("picked on an earlier morning", prediction["message"])
        self.assertIsNotNone(prediction["rank"])
        self.assertGreaterEqual(prediction["rank"], 1)
        self.assertGreaterEqual(prediction["open_orders"], prediction["rank"])
        self.assertIsNotNone(prediction["late_risk_percentile"])
        self.assertGreaterEqual(prediction["late_risk_percentile"], 0.0)
        self.assertLessEqual(prediction["late_risk_percentile"], 1.0)
        self.assertIsNotNone(prediction["spare_time"])
        self.assertIsNotNone(prediction["days_left"])
        self.assertRegex(prediction["predicted_delivery_date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertTrue(prediction["why"])
        self.assertEqual(set(prediction["model_versions"]),
                         {"checkout_eta", "handover_eta", "late_risk"})

    def test_predict_keeps_the_status_messages_for_ineligible_orders(self):
        expected = {
            "e481f51cbdc54678b7cc49136f2d6af7": "delivered",
            "6d25592267349b322799e2beb687871e": "not_bought",
            "7d2286ec219f58023352077d938dee48": "not_handed_over",
            "0" * 32: "unknown",
        }
        for order_id, state in expected.items():
            prediction = self.predict("2018-08-22", order_id)
            self.assertEqual(prediction["state"], state)
            self.assertFalse(prediction["on_list"])
            self.assertIn("message", prediction)
            self.assertTrue(prediction["message"])
            self.assertIsNone(prediction.get("rank"))

    def test_predict_says_when_an_order_is_outside_the_scored_population(self):
        import pandas as pd

        order_id = "8e24261a7e58791d10cb1bf9da94df5c"
        raw = pd.read_csv(
            ROOT / "data" / "raw" / "olist_orders_dataset.csv",
            usecols=["order_id", "order_status", "order_delivered_carrier_date",
                     "order_estimated_delivery_date"],
            parse_dates=["order_delivered_carrier_date", "order_estimated_delivery_date"],
        )
        record = raw[raw["order_id"].eq(order_id)].iloc[0]
        self.assertEqual(record["order_status"], "unavailable")
        self.assertTrue(pd.isna(record["order_delivered_carrier_date"]))
        self.assertLess(record["order_estimated_delivery_date"], pd.Timestamp("2018-08-22"))

        prediction = self.predict("2018-08-22", order_id)
        self.assertTrue(prediction["ok"])
        self.assertFalse(prediction["found"])
        self.assertEqual(prediction["state"], "not_scored")
        self.assertEqual(
            prediction["message"],
            "This order is not scored: it has no usable carrier handover, so it never appears "
            "on a morning list.")
        self.assertFalse(prediction["on_list"])
        self.assertIsNone(prediction.get("rank"))
        lookup = self.service.order_lookup("2018-08-22", order_id)
        self.assertEqual(lookup["state"], "not_scored")
        self.assertEqual(lookup["message"], prediction["message"])

        unknown = self.predict("2018-08-22", "0" * 32)
        self.assertEqual(unknown["state"], "unknown")
        self.assertEqual(unknown["message"], "No order with that ID is in the order records.")

    def test_predict_writes_a_log_line_with_the_listed_fields(self):
        import pandas as pd

        order_id = "d3310038bbf991ce02501a51022f5283"
        self.predict("2018-08-22", order_id)
        lines = [json.loads(line) for line in self.log_path.read_text().splitlines() if line.strip()]
        line = next(item for item in reversed(lines)
                    if item["order_id"] == order_id and item["as_of_date"] == "2018-08-22")
        for field in ("timestamp", "as_of_date", "order_id", "model_versions",
                      "late_risk_score", "late_risk_percentile", "predicted_delivery_date",
                      "spare_time", "rank", "latency_ms"):
            self.assertIn(field, line)
        self.assertIsInstance(line["late_risk_score"], float)
        self.assertIsInstance(line["latency_ms"], float)
        self.assertEqual(line["model_versions"],
                         {name: int(version) for name, version in
                          self.service._state_versions(
                              self.service._date_state(pd.Timestamp("2018-08-22"))).items()})

    def test_ranking_and_performance_are_pinned_to_the_monthly_refit_values(self):
        import hashlib

        expected = {
            "2018-06-15": {
                "count": 1443,
                "ids_sha256": "c3843bac681889b67d8504ab17b9ee895bbdeff58fa6ba3c9332ecfd1fb4d00f",
                "capacity": 30,
                "blend": {"caught": 16, "caught_3plus_days": 8, "mean_days_left": None},
                "slack": {"caught": 16, "caught_3plus_days": 8, "mean_days_left": None},
            },
            "2018-07-16": {
                "count": 362,
                "ids_sha256": "708e634762a3f33cd28393ad278307b47c4d9e9e04dfadec40f63a9ae48eda0e",
                "capacity": 0,
                "blend": {"caught": 7, "caught_3plus_days": 5, "mean_days_left": None},
                "slack": {"caught": 14, "caught_3plus_days": 10, "mean_days_left": None},
            },
            "2018-08-22": {
                "count": 1396,
                "ids_sha256": "5a9c364c00efed63a651a9a0360a57a1d112ac5c330f3addd27d5eeba265f223",
                "capacity": 44,
                "blend": {"caught": 97, "caught_3plus_days": 43, "mean_days_left": 3.9342783505154637},
                "slack": {"caught": 126, "caught_3plus_days": 27, "mean_days_left": 1.9583333333333333},
            },
        }
        for date, before in expected.items():
            payload = self.queue(date)
            ids = [row["order_id"] for row in payload["rows"] if not row["past_promise"]]
            self.assertEqual(len(ids), before["count"])
            self.assertEqual(hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                             before["ids_sha256"])
            self.assertEqual(payload["capacity"], before["capacity"])
            for method in ("blend", "slack"):
                metrics = payload["performance"][method]
                self.assertEqual(metrics["caught"], before[method]["caught"])
                self.assertEqual(metrics["caught_3plus_days"],
                                 before[method]["caught_3plus_days"])
                if before[method]["mean_days_left"] is None:
                    self.assertIsNone(metrics["mean_days_left"])
                else:
                    self.assertAlmostEqual(metrics["mean_days_left"],
                                           before[method]["mean_days_left"], places=9)


    def test_full_queue_payload_matches_the_monthly_refit_payload(self):
        import hashlib

        expected = {
            "2018-08-22": "553b96d82194423eda97d96e48b2f9cd76b5e0015ce3a93e3ab9b4baedf1d935",
            "2018-06-15": "8677697d5e0e093b9fb714386f36f4483e1af88bcf415c6b8fd5c51db024fd92",
        }
        for date, digest in expected.items():
            payload = self.queue(date)
            stable = {key: value for key, value in payload.items()
                      if key not in ("elapsed_ms", "cached")}
            got = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
            self.assertEqual(got, digest, date)


if __name__ == "__main__":
    unittest.main()
