"""Point-in-time, ranking and replay checks for the Phase 3 daily rescue queue."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase3.queue import (CLASSIFIER_MODEL, CLASSIFIER_PARAMS, CLASSIFIER_SELECTION,  # noqa: E402
                          ETA_MODEL, ETA_PARAMS, ETA_SELECTION, RESULT_DIR, QueueContext,
                          build_context, chronological_folds, daily_capacity, load_queue_table,
                          queue, replay, replay_metrics, reproduce_winnie, rescue_metrics,
                          selection_block, static_test_metrics, verify_winnie, winnie_daily_replay,
                          winnie_daily_selection, winnie_percentile, winnie_queue, winnie_rule_scores)


def synthetic_context():
    rows = pd.DataFrame({
        "order_id": ["a", "b", "c", "d", "e", "f"],
        "order_purchase_timestamp": pd.to_datetime([
            "2018-05-28 10:00", "2018-05-29 10:00", "2018-05-30 10:00",
            "2018-05-31 10:00", "2018-06-01 10:00", "2018-06-02 10:00"]),
        "handover_time": pd.to_datetime([
            "2018-06-01 08:00", "2018-06-01 10:00", "2018-06-02 08:00",
            "2018-06-02 10:00", "2018-06-03 08:00", "2018-06-04 08:00"]),
        "outcome_available_at": pd.to_datetime([
            "2018-06-05 08:00", "2018-06-06 08:00", "2018-06-07 08:00",
            "2018-06-08 08:00", "2018-06-09 08:00", "2018-06-10 08:00"]),
        "promise_date": pd.to_datetime([
            "2018-06-05", "2018-06-06", "2018-06-07",
            "2018-06-08", "2018-06-09", "2018-06-10"]),
        "promised_days": [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
        "handover_days": [4.0, 3.0, 3.0, 2.0, 2.0, 2.0],
        "route": ["SP->SP"] * 6,
        "route_transit_90d_at_handover": [2.0, 6.0, 1.0, 5.0, 1.0, 1.0],
        "eta_days": [9.0, 10.0, 8.0, 9.0, 8.0, 8.0],
        "risk": [0.1, 0.9, 0.2, 0.8, 0.3, 0.4],
        "late": [False, True, False, True, False, False],
        "is_late": [0, 1, 0, 1, 0, 0],
        "purchase_month": ["2018-06"] * 6,
        "handover_day": pd.to_datetime([
            "2018-06-01", "2018-06-01", "2018-06-02",
            "2018-06-02", "2018-06-03", "2018-06-04"]),
    })
    return QueueContext(table=rows, history=rows, spec={}, median_transit=2.0)


class SelectionTests(unittest.TestCase):
    def test_model_parameters_come_from_the_phase2_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            path.write_text(json.dumps({"selected_model": {"handover": {
                "candidate": "ridge_7", "family": "ridge", "parameters": {"alpha": 7.0}}}}))
            self.assertEqual(selection_block(path, "selected_model", "handover")["parameters"],
                             {"alpha": 7.0})
            with self.assertRaises(FileNotFoundError):
                selection_block(Path(tmp) / "missing.json", "selected_model")
            with self.assertRaises(KeyError):
                selection_block(path, "selected_model", "checkout")
        classifier = json.loads(CLASSIFIER_SELECTION.read_text())["variants"]["primary"]["selected_model"]
        eta = json.loads(ETA_SELECTION.read_text())["selected_model"]["handover"]
        self.assertEqual(CLASSIFIER_MODEL["family"], classifier["family"])
        self.assertEqual(CLASSIFIER_PARAMS, classifier["parameters"])
        self.assertEqual(ETA_MODEL["family"], eta["family"])
        self.assertEqual(ETA_PARAMS, eta["parameters"])


class QueueShapeTests(unittest.TestCase):
    def setUp(self):
        self.context = synthetic_context()

    def test_open_orders_are_handed_before_0900_and_not_yet_delivered(self):
        frame = queue("2018-06-04", capacity=10, method="deadline",
                      context=self.context, transit_mode="handover")
        # a-c are handed before 09:00 on 4 June; d (10:00 on 2 June) and e/f are open;
        # every synthetic delivery is after the review time.
        self.assertEqual(set(frame["order_id"]), {"a", "b", "c", "d", "e", "f"})
        # f was handed at 08:00 on 4 June, so it is in the same morning queue.
        self.assertIn("f", set(frame["order_id"]))

    def test_order_handed_after_0900_enters_the_next_morning(self):
        frame = queue("2018-06-01", capacity=10, method="deadline",
                      context=self.context, transit_mode="handover")
        self.assertIn("a", set(frame["order_id"]))
        self.assertNotIn("b", set(frame["order_id"]))
        frame = queue("2018-06-02", capacity=10, method="deadline",
                      context=self.context, transit_mode="handover")
        self.assertIn("b", set(frame["order_id"]))

    def test_promise_date_is_the_end_of_that_day(self):
        rows = self.context.table.copy()
        rows.loc[rows["order_id"].eq("a"), "promise_date"] = pd.Timestamp("2018-06-03")
        context = QueueContext(table=rows, history=rows, spec={}, median_transit=2.0)
        frame = queue("2018-06-04", capacity=10, method="deadline",
                      context=context, transit_mode="handover").set_index("order_id")
        self.assertTrue(bool(frame.loc["a", "past_promise"]))
        self.assertEqual(frame.loc["a", "suggested_action"], "customer_update")
        self.assertFalse(bool(frame.loc["a", "in_capacity"]))
        self.assertTrue(np.isnan(frame.loc["a", "rank"]))

    def test_deadline_ranks_the_fewest_days_left_first(self):
        frame = queue("2018-06-04", capacity=2, method="deadline",
                      context=self.context, transit_mode="handover")
        ranked = frame[~frame["past_promise"]].sort_values("rank")
        self.assertEqual(ranked["order_id"].tolist()[:2], ["a", "b"])
        self.assertEqual(ranked["days_until_promise"].tolist()[:2], [1, 2])
        self.assertTrue(ranked["in_capacity"].iloc[:2].all())
        self.assertFalse(ranked["in_capacity"].iloc[2:].any())

    def test_slack_rule_uses_the_handover_remaining_slack(self):
        frame = queue("2018-06-04", capacity=3, method="slack",
                      context=self.context, transit_mode="handover").set_index("order_id")
        # slack_at_handover = promised_days - handover_days - transit; smaller goes first.
        expected = (self.context.table.set_index("order_id")["promised_days"]
                    - self.context.table.set_index("order_id")["handover_days"]
                    - self.context.table.set_index("order_id")["route_transit_90d_at_handover"])
        order = expected.sort_values(kind="stable").index.tolist()
        ranked = frame[~frame["past_promise"]].sort_values("rank").index.tolist()
        self.assertEqual(ranked, order[: len(ranked)])

    def test_blend_is_the_mean_of_risk_and_slack_percentiles_that_morning(self):
        frame = queue("2018-06-04", capacity=10, method="blend",
                      context=self.context, transit_mode="handover")
        stage2 = frame[~frame["past_promise"]].sort_values("order_id")
        n = len(stage2)
        from scipy.stats import rankdata
        risk_pct = rankdata(stage2["risk"].to_numpy(), method="average") / n
        slack_pct = rankdata(-stage2["slack_at_handover"].to_numpy(), method="average") / n
        np.testing.assert_allclose(stage2["blend_score"].to_numpy(), (risk_pct + slack_pct) / 2)

    def test_expected_days_remaining_subtracts_elapsed_time(self):
        frame = queue("2018-06-04", capacity=10, method="deadline",
                      context=self.context, transit_mode="handover").set_index("order_id")
        row = self.context.table.set_index("order_id").loc["a"]
        elapsed = (pd.Timestamp("2018-06-04 09:00") - row["order_purchase_timestamp"]).total_seconds() / 86400
        self.assertAlmostEqual(frame.loc["a", "expected_days_remaining"], row["eta_days"] - elapsed)

    def test_daily_capacity_is_ten_percent_of_previous_day_handovers(self):
        rows = pd.DataFrame({
            "handover_time": pd.to_datetime(
                ["2018-06-10 08:00"] * 11 + ["2018-06-11 12:00"] * 3),
        })
        self.assertEqual(daily_capacity(rows, "2018-06-11"), 2)
        self.assertEqual(daily_capacity(rows, "2018-06-12"), 1)
        self.assertEqual(daily_capacity(rows, "2018-06-13"), 0)

    def test_queue_default_capacity_is_the_daily_rule(self):
        # Two orders were handed over on 3 June, so 4 June's Act block is one.
        frame = queue("2018-06-04", method="deadline",
                      context=self.context, transit_mode="handover")
        self.assertEqual(int(frame["in_capacity"].sum()), 1)


class ReplayTests(unittest.TestCase):
    def test_per_day_selects_the_top_capacity_once_and_records_warning(self):
        context = synthetic_context()
        detail, daily = replay(context, method="slack", policy="per_day", capacity=1,
                               transit_mode="handover")
        # Only one order is selected per day. a is the only order open on 1 June;
        # b has the smallest handover slack among the 2 June queue and is chosen then.
        first = detail.set_index("order_id")["first_selected"]
        self.assertEqual(pd.Timestamp(first.loc["a"]), pd.Timestamp("2018-06-01"))
        self.assertEqual(pd.Timestamp(first.loc["b"]), pd.Timestamp("2018-06-02"))

    def test_per_day_without_capacity_uses_the_daily_rule(self):
        detail, daily = replay(synthetic_context(), method="slack", policy="per_day",
                               capacity=None, transit_mode="handover")
        self.assertLessEqual(int(daily["capacity"].max()), 1)
        second = daily.set_index("as_of").loc[pd.Timestamp("2018-06-02").date()]
        self.assertEqual(int(second["capacity"]), 1)

    def test_per_month_quota_limits_the_month(self):
        context = synthetic_context()
        detail, daily = replay(context, method="deadline", policy="per_month", capacity=10,
                               share=0.1, transit_mode="handover")
        # Six June handovers -> quota ceil(0.6) = 1.
        self.assertEqual(len(detail), 1)
        self.assertEqual(int(daily["selected"].sum()), 1)

    def test_replay_detail_carries_the_0900_review_warning(self):
        detail, _ = replay(synthetic_context(), method="slack", policy="per_day", capacity=1,
                           transit_mode="handover")
        row = detail.set_index("order_id").loc["a"]
        elapsed = (pd.Timestamp("2018-06-01 09:00") - row["order_purchase_timestamp"]).total_seconds() / 86400
        self.assertAlmostEqual(row["days_left"], row["promised_days"] - elapsed)

    def test_rescue_metrics_count_only_three_plus_day_late_catches(self):
        detail = pd.DataFrame({
            "late": [True, True, True, False],
            "days_left": [4.0, 3.0, 2.9, 8.0],
        })
        metrics = rescue_metrics(detail)
        self.assertEqual(metrics["caught"], 3)
        self.assertEqual(metrics["caught_3plus_days"], 2)
        self.assertAlmostEqual(metrics["mean_days_left"], (4.0 + 3.0 + 2.9) / 3)
        self.assertEqual(rescue_metrics(pd.DataFrame({"late": [], "days_left": []})),
                         {"caught": 0, "caught_3plus_days": 0, "mean_days_left": None})

    def test_warning_threshold_metrics(self):
        detail = pd.DataFrame({
            "order_id": ["a", "b", "c"],
            "late": [True, True, False],
            "warning_days": [5, 1, 4],
            "purchase_month": ["2018-08", "2018-08", "2018-06"],
        })
        daily = pd.DataFrame({"selected": [3], "selected_late": [2]})
        metrics = replay_metrics(detail, daily, 3, 3)
        self.assertEqual(metrics["late_found_total"], 2)
        self.assertEqual(metrics["late_found_3plus"], 1)
        self.assertEqual(metrics["august_3plus"], 1)
        self.assertAlmostEqual(metrics["mean_warning_days"], 3.0)


class EndToEndTests(unittest.TestCase):
    def test_phase2_catch_counts_reproduce_exactly(self):
        table, spec, _ = load_queue_table()
        train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
        test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
        context = build_context(train, test, table, spec)
        metrics, _, _ = static_test_metrics(context)
        metrics = metrics.set_index("ranking")
        self.assertEqual(int(metrics.loc["slack", "caught_top10_monthly"]), 398)
        self.assertEqual(int(metrics.loc["model", "caught_top10_monthly"]), 377)
        self.assertAlmostEqual(metrics.loc["deadline", "pr_auc"], 0.487, places=3)
        self.assertAlmostEqual(metrics.loc["slack", "pr_auc"], 0.519, places=3)

    def test_whole_test_daily_size_matches_winnie_saved_mornings(self):
        table, spec, _ = load_queue_table()
        train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
        test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
        context = build_context(train, test, table, spec)
        _, daily = replay(context, method="blend", policy="per_day", capacity=None)
        daily = daily.assign(morning=daily["as_of"].astype(str))
        saved = pd.read_csv(RESULT_DIR / "winnie_daily_test_mornings.csv")
        saved = saved[saved["version"].eq("A_mix")][["morning", "capacity"]]
        merged = saved.merge(daily[["morning", "capacity"]], on="morning",
                             suffixes=("_saved", "_ours"))
        self.assertEqual(len(merged), len(saved))
        self.assertTrue((merged["capacity_saved"] == merged["capacity_ours"]).all(), merged)


class WinnieDashboardTests(unittest.TestCase):
    def test_queue_before_the_review_start_is_empty_with_typed_columns(self):
        frame = winnie_queue("2018-05-10", context=synthetic_context(), capacity=1)
        self.assertTrue(frame.empty)
        self.assertEqual(frame["past_promise"].dtype, bool)
        self.assertEqual(frame["in_capacity"].dtype, bool)
        self.assertEqual(len(frame[frame["past_promise"]]), 0)

    def test_open_parcels_are_in_the_queue_table_without_a_split_or_label(self):
        table, _, _ = load_queue_table()
        open_rows = table[table["split"].isna()]
        saved = pd.read_csv(ROOT / "data" / "processed" / "phase2_open_orders.csv",
                            usecols=["order_id"])
        self.assertEqual(set(open_rows["order_id"]), set(saved["order_id"]))
        self.assertTrue(open_rows["is_late"].isna().all())
        self.assertTrue(open_rows["late"].isna().all())

    def test_live_queue_is_the_reviewed_b_mix_list(self):
        context = synthetic_context()
        frame = winnie_queue("2018-06-04", context=context, capacity=1)
        ranked = frame[~frame["past_promise"]].sort_values("rank")
        self.assertEqual(ranked["rank"].tolist(), list(range(1, len(ranked) + 1)))
        self.assertEqual(int(frame["in_capacity"].sum()), 1)
        self.assertAlmostEqual(float(ranked["blend_score"].iloc[0]),
                               float(ranked["blend_score"].max()))
        # The live list must be exactly the morning's B_mix picks plus the
        # still-unreviewed candidates behind them.
        start = context.table["handover_time"].min().normalize()
        orders = context.table.sort_values(["handover_time", "order_id"], kind="stable").reset_index(drop=True)
        model_scores = orders["risk"].to_numpy(dtype=float)
        model_ref = model_scores
        rule_ref = winnie_rule_scores(orders, orders)
        picks, _ = winnie_daily_selection("dashboard", orders, start, pd.Timestamp("2018-06-03"),
                                          model_scores, winnie_rule_scores(orders, orders),
                                          model_ref, rule_ref, versions=("B_mix",))
        morning = picks[picks["morning"].eq(pd.Timestamp("2018-06-04"))]
        self.assertEqual(set(morning["order_id"]),
                         set(ranked.loc[ranked["in_capacity"], "order_id"]))

    def test_replay_matches_winnie_single_daily_exactly(self):
        context = synthetic_context()
        start = context.table["handover_time"].min().normalize()
        end = context.table["handover_time"].max().normalize()
        orders = context.table.sort_values(["handover_time", "order_id"], kind="stable").reset_index(drop=True)
        model_scores = orders["risk"].to_numpy(dtype=float)
        rule_scores = winnie_rule_scores(orders, orders)
        rule_ref = rule_scores
        reference = winnie_daily_selection("test", orders, start, end, model_scores, rule_scores,
                                           model_scores, rule_ref, versions=("B_mix",))
        detail, daily = winnie_daily_replay(context, start=start, end=end)
        self.assertEqual(sorted(detail["order_id"]), sorted(reference[0]["order_id"]))
        caught = detail[detail["late"].astype(bool)]
        ref_daily = reference[1]
        self.assertEqual(len(caught), int(ref_daily["caught_before_date"].sum()))
        self.assertEqual(int((caught["days_left"] >= 3).sum()),
                         int(ref_daily["caught_3plus_days"].sum()))
        self.assertEqual(len(daily), len(reference[1]))
        self.assertEqual(daily["capacity"].tolist(), reference[1]["capacity"].tolist())

    def test_each_order_is_reviewed_once_with_the_daily_size(self):
        context = synthetic_context()
        detail, daily = winnie_daily_replay(context)
        self.assertFalse(detail["order_id"].duplicated().any())
        self.assertTrue((daily["reviewed"] <= daily["capacity"]).all())


class WinnieReproductionTests(unittest.TestCase):
    def test_percentile_is_the_share_of_fitting_orders_at_or_below(self):
        reference = np.array([1.0, 2.0, 2.0, 4.0])
        got = winnie_percentile(reference, np.array([0.0, 2.0, 3.0, 5.0]))
        np.testing.assert_allclose(got, [0.0, 0.75, 0.75, 1.0])

    def test_rule_score_is_minus_the_handover_slack(self):
        fit = pd.DataFrame({"route_transit_90d_at_handover": [2.0, 4.0]})
        rows = pd.DataFrame({"promised_days": [10.0, 10.0],
                             "handover_days": [3.0, 5.0],
                             "route_transit_90d_at_handover": [3.0, np.nan]})
        np.testing.assert_allclose(winnie_rule_scores(fit, rows), [-4.0, -2.0])

    def test_task4_headline_results_match_the_rerun(self):
        table, spec, _ = load_queue_table()
        train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
        folds = chronological_folds(train, spec)
        verify_winnie(reproduce_winnie(table, spec, folds))


if __name__ == "__main__":
    unittest.main()
