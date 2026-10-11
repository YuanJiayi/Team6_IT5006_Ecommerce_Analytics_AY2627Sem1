"""Validation-only as-of-handover backlog feature experiment.

Run from Phase2: python experiments/backlog_feature/run.py
The existing selected logistic fit and five chronological windows remain fixed.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from phase2_eta import chronological_folds  # noqa: E402
from phase2_handover_classifier import (CANDIDATES, LABEL, RELATIVE_BACKLOG_INPUT, action_metrics,  # noqa: E402
                                        fit_scores, load_cohort, monthly_flags, ranking_metrics, variant_spec)


def open_before_handover(events: pd.DataFrame, handovers: pd.DataFrame,
                         group: str | None = None, entry: str = "order_purchase_timestamp",
                         subtract_target: bool = True) -> np.ndarray:
    """Count orders entered into a stage but not yet delivered at each handover.

    For a past purchase, its delivery timestamp is used only to decide whether
    the delivery event had already happened by the handover. Future deliveries
    are never used as known labels or predictors. This is an event replay.
    """
    result = np.zeros(len(handovers), dtype=np.int64)
    event_groups = [(None, events)] if group is None else events.groupby(group, dropna=False)
    for key, part in event_groups:
        queries = handovers if group is None else handovers.loc[handovers[group].eq(key)]
        if len(queries) == 0:
            continue
        purchases = np.sort(part[entry].dropna().to_numpy(dtype="datetime64[ns]"))
        deliveries = np.sort(part["order_delivered_customer_date"].dropna().to_numpy(dtype="datetime64[ns]"))
        t = queries["handover_time"].to_numpy(dtype="datetime64[ns]")
        n_purchased = np.searchsorted(purchases, t, side="right")
        n_delivered = np.searchsorted(deliveries, t, side="right")
        result[queries.index.to_numpy()] = n_purchased - n_delivered
    # The target order is itself already purchased and still open at handover.
    return np.maximum(result - int(subtract_target), 0)


def weekly_history(events: pd.DataFrame, table: pd.DataFrame, group: str | None = None) -> dict[str, np.ndarray]:
    """Weekly as-of snapshots from one to eight weeks before each handover.

    These are approximations to trailing 2/4/8-week mean backlog. The target
    order is removed from historical snapshots when it was already purchased.
    """
    snapshots = []
    for week in range(1, 9):
        query = table.copy()
        query["handover_time"] = query["handover_time"] - pd.Timedelta(days=7 * week)
        count = open_before_handover(events, query, group=group, subtract_target=False)
        own_open = table["order_purchase_timestamp"].le(query["handover_time"]).to_numpy(dtype=int)
        snapshots.append(np.maximum(count - own_open, 0))
    array = np.stack(snapshots)
    return {f"mean_{weeks}wk": array[:weeks].mean(axis=0) for weeks in (2, 4, 8)}


def main() -> None:
    table, spec, _ = load_cohort(ROOT)
    events = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                         usecols=["order_id", "customer_id", "order_purchase_timestamp", "order_delivered_carrier_date",
                                  "order_delivered_customer_date"],
                         parse_dates=["order_purchase_timestamp", "order_delivered_carrier_date",
                                      "order_delivered_customer_date"])
    customers = pd.read_csv(ROOT / "data/olist_customers_dataset.csv",
                            usecols=["customer_id", "customer_state"])
    events = events.merge(customers, on="customer_id", how="left", validate="many_to_one")
    table = table.merge(events[["order_id", "customer_state"]].rename(columns={"customer_state": "backlog_customer_state"}),
                        on="order_id", how="left", validate="one_to_one")
    table["open_orders_at_handover"] = open_before_handover(events, table)
    table["state_open_orders_at_handover"] = open_before_handover(
        events.rename(columns={"customer_state": "backlog_customer_state"}), table, "backlog_customer_state")
    table["carrier_open_orders_at_handover"] = open_before_handover(
        events, table, entry="order_delivered_carrier_date")
    table["state_carrier_open_orders_at_handover"] = open_before_handover(
        events.rename(columns={"customer_state": "backlog_customer_state"}), table,
        "backlog_customer_state", entry="order_delivered_carrier_date")
    # Log scale prevents busy periods from dominating an otherwise linear classifier.
    table["log_open_orders_at_handover"] = np.log1p(table["open_orders_at_handover"])
    table["log_state_open_orders_at_handover"] = np.log1p(table["state_open_orders_at_handover"])
    table["log_carrier_open_orders_at_handover"] = np.log1p(table["carrier_open_orders_at_handover"])
    table["log_state_carrier_open_orders_at_handover"] = np.log1p(table["state_carrier_open_orders_at_handover"])
    for name, values in weekly_history(events, table).items():
        table[f"log_global_backlog_{name}"] = np.log1p(values)
    state_events = events.rename(columns={"customer_state": "backlog_customer_state"})
    for name, values in weekly_history(state_events, table, "backlog_customer_state").items():
        table[f"log_state_backlog_{name}"] = np.log1p(values)
    table["global_backlog_recent_vs_8wk"] = (table["log_open_orders_at_handover"]
                                              - table["log_global_backlog_mean_8wk"])
    table["state_backlog_recent_vs_8wk"] = (table["log_state_open_orders_at_handover"]
                                             - table["log_state_backlog_mean_8wk"])
    assert table[["open_orders_at_handover", "state_open_orders_at_handover"]].ge(0).all().all()

    train = table.loc[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    assert len(folds) == 5
    selection = json.loads((ROOT / "results/phase2/handover/selection.json").read_text())
    selected = selection["variants"]["primary"]["selected"]
    assert selected == "logistic_c0.01"
    family, params = next((f, p) for n, f, p in CANDIDATES if n == selected)

    features = {
        "baseline": (),
        "global_backlog": ("log_open_orders_at_handover",),
        "state_backlog": ("log_state_open_orders_at_handover",),
        "both_backlogs": ("log_open_orders_at_handover", "log_state_open_orders_at_handover"),
        "carrier_backlog": ("log_carrier_open_orders_at_handover",),
        "state_carrier_backlog": ("log_state_carrier_open_orders_at_handover",),
        "both_carrier_backlogs": ("log_carrier_open_orders_at_handover", "log_state_carrier_open_orders_at_handover"),
        "global_mean_2wk": ("log_global_backlog_mean_2wk",),
        "global_mean_4wk": ("log_global_backlog_mean_4wk",),
        "global_mean_8wk": ("log_global_backlog_mean_8wk",),
        "state_mean_2wk": ("log_state_backlog_mean_2wk",),
        "state_mean_4wk": ("log_state_backlog_mean_4wk",),
        "state_mean_8wk": ("log_state_backlog_mean_8wk",),
        "global_recent_vs_8wk": ("global_backlog_recent_vs_8wk",),
        "state_recent_vs_8wk": ("state_backlog_recent_vs_8wk",),
    }
    former_spec = variant_spec(spec)
    former_spec["numeric_features"] = [f for f in former_spec["numeric_features"]
                                       if f != RELATIVE_BACKLOG_INPUT]
    assert RELATIVE_BACKLOG_INPUT not in former_spec["numeric_features"]

    def candidate_spec(extra):
        out = copy.deepcopy(former_spec)
        out["numeric_features"] += list(extra)
        return out

    rows = []
    for name, extra in features.items():
        current_spec = candidate_spec(extra)
        for window, (fit, valid) in enumerate(folds, start=1):
            y = train.iloc[valid][LABEL].to_numpy()
            score = fit_scores(family, params, current_spec, train.iloc[fit], train.iloc[valid],
                               spec["linear_log_candidates"])
            rows.append({"variant": name, "window": window, "fit_orders": len(fit),
                         "validation_orders": len(valid), **ranking_metrics(y, score)})
        current = pd.DataFrame(rows).query("variant == @name")
        print(name, current[["pr_auc", "recall_top10", "precision_top10"]].mean().to_dict(), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "validation_results.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "definition": "Open-order backlog counts purchases by handover minus deliveries completed by handover; carrier backlog counts carrier handovers minus completed deliveries. Each excludes the target order, is either global or within buyer state, and is log1p transformed.",
        "selected_model_unchanged": selected,
        "validation_mean": result.groupby("variant")[["pr_auc", "recall_top10", "precision_top10"]].mean().to_dict("index"),
    }, indent=2) + "\n")
    print(result.pivot(index="window", columns="variant", values="pr_auc").to_string(), flush=True)

    # Descriptive diagnostic only: choose the state-relative feature using
    # validation above, then refit each fixed specification on all train rows.
    test = table.loc[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    y = test[LABEL].to_numpy()
    test_rows = []
    for name in ("baseline", "state_recent_vs_8wk"):
        current_spec = candidate_spec(features[name])
        score = fit_scores(family, params, current_spec, train, test, spec["linear_log_candidates"])
        for month in ["all", *np.unique(months)]:
            mask = np.ones(len(test), dtype=bool) if month == "all" else months == month
            metric = ranking_metrics(y[mask], score[mask])
            flags = monthly_flags(score, months, 0.10)
            month_action = action_metrics(y[mask], flags[mask])
            test_rows.append({"variant": name, "month": month, "orders": int(mask.sum()),
                              "pr_auc": metric["pr_auc"], "roc_auc": metric["roc_auc"],
                              "recall_top10_within_month": month_action["recall"],
                              "precision_top10_within_month": month_action["precision"]})
    test_result = pd.DataFrame(test_rows)
    test_result.to_csv(OUT / "test_diagnostic.csv", index=False)
    print("Test diagnostic, not for model selection:", flush=True)
    print(test_result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
