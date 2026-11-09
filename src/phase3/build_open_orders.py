"""Write data/processed/phase2_open_orders.csv: model inputs for parcels handed over but never delivered.

Run from the repository root:
    python -m phase3.build_open_orders
"""

from __future__ import annotations

from olist.paths import RAW_DIR, REPO_ROOT

from phase3.data_prep import load_raw
from phase3.open_orders import snapshot_counts, write_open_orders


def main():
    table, path = write_open_orders(REPO_ROOT)
    print(f"Saved {path} ({len(table):,} rows)")
    print("Rows by order_status:")
    print(table["order_status"].value_counts().to_string())

    orders_raw = load_raw(RAW_DIR)["orders"]
    check = snapshot_counts(orders_raw)
    print(f"\nWith the carrier at 2018-08-22 09:00 and not delivered by then: {check['open']:,} open")
    print("By final status:", check["by_status"])
    print(f"Never delivered (no customer delivery date): {check['never_delivered']:,}")
    print("Never delivered by final status:", check["never_delivered_by_status"])
    print("Scoring table rows open at that morning:",
          int(((table["handover_time"] < "2018-08-22 09:00")
               & (table["outcome_available_at"].isna()
                  | (table["outcome_available_at"] >= "2018-08-22 09:00"))).sum()))


if __name__ == "__main__":
    main()
