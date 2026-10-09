import pandas as pd, sys
D = sys.argv[1]
o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_delivered_customer_date","order_estimated_delivery_date"])
rv = pd.read_csv(f"{D}/olist_order_reviews_dataset.csv")
r = rv.groupby("order_id").agg(score=("review_score","min"), has_text=("review_comment_message", lambda s: s.notna().any()))
d = o.merge(r, left_on="order_id", right_index=True, how="left")
print("order status:", (d.order_status.value_counts(normalize=True)*100).round(2).to_dict())
d["bad"] = d.score <= 2
d["late"] = d.order_delivered_customer_date.dt.normalize() > d.order_estimated_delivery_date
d["grp"] = d.order_status.where(d.order_status != "delivered", d.late.map({True:"delivered-late", False:"delivered-on-time"}))
g = d.groupby("grp").agg(orders=("order_id","size"), bad_rate=("bad","mean"), bad_n=("bad","sum"))
g["share_of_all_bad"] = g.bad_n / g.bad_n.sum()
print(g.round(3).to_string())
it = pd.read_csv(f"{D}/olist_order_items_dataset.csv")
m = it.groupby("order_id").agg(n_sellers=("seller_id","nunique"), n_items=("order_item_id","size"))
x = d[d.grp=="delivered-on-time"].join(m, on="order_id")
print("on-time bad rate by #sellers:", x.groupby(x.n_sellers.clip(upper=2)).bad.mean().round(3).to_dict(), " by #items:", x.groupby(x.n_items.clip(upper=3)).bad.mean().round(3).to_dict())
