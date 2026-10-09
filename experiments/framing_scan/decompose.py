import numpy as np, pandas as pd, sys
D = sys.argv[1]; DAY = pd.Timedelta(days=1)
o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"])
it = pd.read_csv(f"{D}/olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
lim = it.groupby("order_id").shipping_limit_date.max()
d = o[(o.order_status=="delivered") & o.order_delivered_customer_date.notna() & o.order_delivered_carrier_date.notna()].copy()
d["lim"] = d.order_id.map(lim)
d["hand"] = (d.order_delivered_carrier_date - d.order_purchase_timestamp)/DAY
d["leg"] = (d.order_delivered_customer_date - d.order_delivered_carrier_date)/DAY
d["dur"] = d.hand + d.leg
d = d[(d.hand >= 0) & (d.leg >= 0)]
d["late"] = d.order_delivered_customer_date.dt.normalize() > d.order_estimated_delivery_date
d["slack"] = (d.order_estimated_delivery_date + DAY - d.order_purchase_timestamp)/DAY  # days available
print(f"n={len(d)}  mean handover {d.hand.mean():.1f}d (median {d.hand.median():.1f}), mean carrier leg {d.leg.mean():.1f}d (median {d.leg.median():.1f})")
v = d.dur.var(); print(f"variance share of delivery time: handover {d.hand.var()/v:.0%}, carrier {d.leg.var()/v:.0%}, 2*cov {2*np.cov(d.hand,d.leg)[0,1]/v:.0%}")
L = d[d.late]
print(f"late orders n={len(L)}  mean handover {L.hand.mean():.1f}d vs on-time {d[~d.late].hand.mean():.1f}d; carrier {L.leg.mean():.1f}d vs {d[~d.late].leg.mean():.1f}d")
# counterfactuals: would the late order be on time if one stage had been typical (overall median)?
mh, ml = d.hand.median(), d.leg.median()
fix_h = (np.minimum(L.hand, mh) + L.leg) <= L.slack
fix_l = (L.hand + np.minimum(L.leg, ml)) <= L.slack
fix_lim = (np.minimum(L.order_delivered_carrier_date, L.lim) - L.order_purchase_timestamp)/DAY + L.leg <= L.slack
print(f"of late orders, on time if: handover were median -> {fix_h.mean():.0%}; carrier leg were median -> {fix_l.mean():.0%}; seller met shipping limit -> {fix_lim.mean():.0%}")
print(f"  seller-only fixable {(fix_h & ~fix_l).mean():.0%}, carrier-only {(fix_l & ~fix_h).mean():.0%}, either {(fix_h & fix_l).mean():.0%}, neither {(~fix_h & ~fix_l).mean():.0%}")
# predictability of each stage from its own entity (between-group share of variance)
se = it.groupby("order_id").seller_id.first(); d["seller"] = d.order_id.map(se)
cu = pd.read_csv(f"{D}/olist_customers_dataset.csv").set_index("customer_id").customer_state; d["cst"] = d.customer_id.map(cu)
ss = pd.read_csv(f"{D}/olist_sellers_dataset.csv").set_index("seller_id").seller_state; d["route"] = d.seller.map(ss) + ">" + d.cst
for col, grp in [("hand","seller"),("leg","route"),("leg","seller")]:
    g = d.groupby(grp)[col]; between = (g.transform("mean")-d[col].mean()).var()/d[col].var()
    print(f"share of {col} variance explained by {grp} averages (in-sample, upper bound): {between:.0%}")
