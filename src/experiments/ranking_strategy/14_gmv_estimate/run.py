"""Estimate the yearly GMV of late parcels that the recommended ranking would put in front of staff.

Run from the repository root:
    python src/experiments/ranking_strategy/14_gmv_estimate/run.py

Annual base: delivered orders purchased from 1 September 2017 to 31 August 2018. GMV is the order's item prices, with
freight shown separately. Catch rates come from the test set, using the 50/50 blend with monthly refits and the top 10%
of each purchase month (experiment 10), counted by orders and by GMV. These figures show value reaching the review list,
not value saved. The script stops if the test counts do not reproduce experiment 10. Writes CSVs to data/outputs/experiments/ranking_strategy/14_gmv_estimate/.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE.parents[2]))

from olist.paths import EXPERIMENT_OUTPUTS, RAW_DIR  # noqa: E402

OUT = EXPERIMENT_OUTPUTS / "ranking_strategy" / HERE.name
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EXPERIMENTS))

from task4_common import LABEL, TOP_SHARE, load_data, monthly_flags  # noqa: E402

_spec = importlib.util.spec_from_file_location("experiment_10", EXPERIMENTS / "10_mix_monthly_refits" / "run.py")
exp10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp10)

YEAR = (pd.Timestamp("2017-09-01"), pd.Timestamp("2018-09-01"))
EXPECTED = {"caught": 411, "caught_not_past": 213, "late": 672, "late_not_past": 474}


def order_values():
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv", usecols=["order_id", "price", "freight_value"])
    return items.groupby("order_id").agg(gmv=("price", "sum"), freight=("freight_value", "sum")).reset_index()


def all_delivered_orders(values):
    """Every delivered order with delivery and promise dates, as a cross-check on the model cohort."""
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv",
                         parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    orders = orders[orders["order_status"].eq("delivered") & orders["order_delivered_customer_date"].notna()
                    & orders["order_estimated_delivery_date"].notna()]
    orders = orders.merge(values, on="order_id", how="left")
    orders["late"] = orders["order_delivered_customer_date"].dt.normalize() > orders["order_estimated_delivery_date"].dt.normalize()
    return orders


def run():
    data = load_data()
    values = order_values()
    table = data["table"].merge(values, on="order_id", how="left", validate="one_to_one")
    if table["gmv"].isna().any():
        raise AssertionError("Some cohort orders have no items")

    # Catch rates on test, by orders and by GMV
    test = data["test"].merge(values, on="order_id", how="left", validate="one_to_one")
    scores = exp10.monthly_scores(data, data["test"], exp10.TEST_START)[exp10.REFRESHED]
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    flags = monthly_flags(scores, months, TOP_SHARE)
    late = test[LABEL].to_numpy().astype(bool)
    not_past = test["past_promise_at_handover"].ne(1).to_numpy()
    gmv = test["gmv"].to_numpy()
    got = {"caught": int((late & flags).sum()), "caught_not_past": int((late & flags & not_past).sum()),
           "late": int(late.sum()), "late_not_past": int((late & not_past).sum())}
    if got != EXPECTED:
        raise AssertionError(f"Test counts do not reproduce experiment 10: {got}")
    rates = pd.DataFrame([
        {"group": "all_late", "orders": late.sum(), "caught": (late & flags).sum(), "gmv": gmv[late].sum(), "gmv_caught": gmv[late & flags].sum()},
        {"group": "late_before_date", "orders": (late & not_past).sum(), "caught": (late & flags & not_past).sum(),
         "gmv": gmv[late & not_past].sum(), "gmv_caught": gmv[late & flags & not_past].sum()},
        {"group": "late_past_date", "orders": (late & ~not_past).sum(), "caught": (late & flags & ~not_past).sum(),
         "gmv": gmv[late & ~not_past].sum(), "gmv_caught": gmv[late & flags & ~not_past].sum()},
    ])
    rates["order_rate"] = rates["caught"] / rates["orders"]
    rates["gmv_rate"] = rates["gmv_caught"] / rates["gmv"]
    rates["gmv_per_order"] = rates["gmv"] / rates["orders"]

    # Annual base
    year = table[table["order_purchase_timestamp"].ge(YEAR[0]) & table["order_purchase_timestamp"].lt(YEAR[1])]
    year_late = year[year[LABEL].astype(bool)]
    year_before = year_late[year_late["past_promise_at_handover"].ne(1)]
    broad = all_delivered_orders(values)
    broad_year = broad[broad["order_purchase_timestamp"].ge(YEAR[0]) & broad["order_purchase_timestamp"].lt(YEAR[1])]
    base = pd.DataFrame([
        {"base": "model cohort", "orders": len(year), "late": len(year_late), "late_rate": len(year_late) / len(year),
         "late_gmv": year_late["gmv"].sum(), "late_freight": year_late["freight"].sum(),
         "late_before_date": len(year_before), "late_before_date_gmv": year_before["gmv"].sum()},
        {"base": "all delivered orders", "orders": len(broad_year), "late": int(broad_year["late"].sum()),
         "late_rate": broad_year["late"].mean(), "late_gmv": broad_year.loc[broad_year["late"], "gmv"].sum(),
         "late_freight": broad_year.loc[broad_year["late"], "freight"].sum(), "late_before_date": np.nan, "late_before_date_gmv": np.nan},
    ])

    r = rates.set_index("group")
    c = base.set_index("base").loc["model cohort"]
    estimates = pd.DataFrame([
        {"estimate": "Late parcels reaching the list (any action)", "orders": c["late"] * r.loc["all_late", "order_rate"],
         "gmv": c["late_gmv"] * r.loc["all_late", "gmv_rate"]},
        {"estimate": "Late parcels reaching the list before their date (could still be rescued)",
         "orders": c["late_before_date"] * r.loc["late_before_date", "order_rate"],
         "gmv": c["late_before_date_gmv"] * r.loc["late_before_date", "gmv_rate"]},
        {"estimate": "Teammate's method: all late orders x 61%", "orders": c["late"] * 0.61, "gmv": c["late_gmv"] * 0.61},
    ])

    rates.to_csv(OUT / "test_catch_rates.csv", index=False)
    base.to_csv(OUT / "annual_base.csv", index=False)
    estimates.to_csv(OUT / "estimates.csv", index=False)
    pd.set_option("display.width", 200)
    print(rates.round(3).to_string(index=False))
    print(base.round(3).to_string(index=False))
    print(estimates.round(0).to_string(index=False))


if __name__ == "__main__":
    run()
