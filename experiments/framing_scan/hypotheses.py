"""Quick tests of alternative business framings on raw Olist CSVs. Chronological split at 2018-05-26."""
import numpy as np, pandas as pd, sys
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

D = sys.argv[1]
BOUND = pd.Timestamp("2018-05-26")
DAY = pd.Timedelta(days=1)
o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"])
it = pd.read_csv(f"{D}/olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
pr = pd.read_csv(f"{D}/olist_products_dataset.csv")
se = pd.read_csv(f"{D}/olist_sellers_dataset.csv")
cu = pd.read_csv(f"{D}/olist_customers_dataset.csv")
rv = pd.read_csv(f"{D}/olist_order_reviews_dataset.csv")

it = it.merge(pr[["product_id", "product_weight_g"]], on="product_id", how="left").merge(se[["seller_id", "seller_state"]], on="seller_id")
agg = it.groupby("order_id").agg(n_items=("order_item_id", "size"), price=("price", "sum"), freight=("freight_value", "sum"),
    weight=("product_weight_g", "sum"), n_sellers=("seller_id", "nunique"), seller_id=("seller_id", "first"),
    seller_state=("seller_state", "first"), ship_limit=("shipping_limit_date", "max"))
df = o.merge(agg, on="order_id").merge(cu[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id")

# ---------- H5: repeat purchase (CLV / retention framing)
oc = df.groupby("customer_unique_id").size()
print(f"H5 repeat customers: {(oc > 1).mean():.1%} of customers have >1 order")

d = df[(df.order_status == "delivered") & df.order_delivered_customer_date.notna()].copy()
d["pday"] = d.order_purchase_timestamp.dt.normalize()
d["dur"] = (d.order_delivered_customer_date - d.order_purchase_timestamp) / DAY
d["need"] = (d.order_delivered_customer_date.dt.normalize() - d.pday).dt.days   # promise days needed to be on time
d["olist_promise"] = (d.order_estimated_delivery_date.dt.normalize() - d.pday).dt.days
d["late"] = (d.need > d.olist_promise).astype(int)
d["route"] = d.seller_state + ">" + d.customer_state
d = d.sort_values("order_purchase_timestamp").reset_index(drop=True)

# ---------- H1: promise padding and whether earliness buys satisfaction
rs = rv.groupby("order_id").review_score.min()
d["review"] = d.order_id.map(rs)
d["early_by"] = d.olist_promise - d.need
print("\nH1 promised vs needed days, by period")
for name, m in [("train", d.order_purchase_timestamp < BOUND), ("test", d.order_purchase_timestamp >= BOUND)]:
    x = d[m]
    print(f"  {name}: promise mean {x.olist_promise.mean():.1f}d, actual need mean {x.need.mean():.1f}d, "
          f"median slack {x.early_by.median():.0f}d, on-time {1 - x.late.mean():.1%}")
bins = [-99, -8, -1, 0, 3, 7, 14, 21, 99]
lab = ["late>7", "late1-7", "on date", "early1-3", "early4-7", "early8-14", "early15-21", "early>21"]
g = d.groupby(pd.cut(d.early_by, bins, labels=lab), observed=True).agg(n=("review", "size"), mean_review=("review", "mean"),
    pct_1_2=("review", lambda s: (s <= 2).mean()))
print("  review score by days early (+) / late (-)\n" + g.round(3).to_string())

# ---------- H4: seller shipping-limit breaches
s = d[d.order_delivered_carrier_date.notna()]
miss = s.order_delivered_carrier_date > s.ship_limit
print(f"\nH4 seller missed shipping limit: {miss.mean():.1%}; late rate if missed {s.late[miss].mean():.1%} vs "
      f"{s.late[~miss].mean():.1%}; share of late orders with a seller miss {miss[s.late == 1].mean():.1%}")


# ---------- as-of history features (only outcomes completed before the query time)
def asof_mean(keys, ev_time, ev_val, q_keys, q_time, window_days=None):
    out = np.full(len(q_keys), np.nan)
    ev = pd.DataFrame({"k": keys, "t": ev_time, "v": ev_val}).dropna().sort_values("t")
    q = pd.DataFrame({"k": q_keys, "t": q_time, "i": np.arange(len(q_keys))})
    for k, qg in q.groupby("k"):
        eg = ev[ev.k == k]
        if eg.empty:
            continue
        t = eg.t.values; cs = np.concatenate([[0], np.cumsum(eg.v.values)])
        hi = np.searchsorted(t, qg.t.values, side="left")
        lo = np.searchsorted(t, (qg.t - pd.Timedelta(days=window_days)).values, "left") if window_days else np.zeros_like(hi)
        n = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            out[qg.i.values] = np.where(n >= 5, (cs[hi] - cs[lo]) / n, np.nan)
    return out

d["handover"] = (d.order_delivered_carrier_date - d.order_purchase_timestamp) / DAY
d["carrier_leg"] = (d.order_delivered_customer_date - d.order_delivered_carrier_date) / DAY
t_q = d.order_purchase_timestamp
d["route_dur_90"] = asof_mean(d.route, d.order_delivered_customer_date, d.dur, d.route, t_q, 90)
d["route_dur_all"] = asof_mean(d.route, d.order_delivered_customer_date, d.dur, d.route, t_q)
d["route_leg_90"] = asof_mean(d.route, d.order_delivered_customer_date, d.carrier_leg, d.route, t_q, 90)
d["seller_hand"] = asof_mean(d.seller_id, d.order_delivered_carrier_date, d.handover, d.seller_id, t_q)
d["cust_dur_90"] = asof_mean(d.customer_state, d.order_delivered_customer_date, d.dur, d.customer_state, t_q, 90)
d["dow"] = t_q.dt.dayofweek
for c in ["seller_state", "customer_state"]:
    d[c + "_c"] = d[c].astype("category").cat.codes
F = ["route_dur_90", "route_dur_all", "route_leg_90", "seller_hand", "cust_dur_90", "n_items", "price", "freight",
     "weight", "n_sellers", "dow", "seller_state_c", "customer_state_c"]
train = (t_q < BOUND) & (d.order_delivered_customer_date < BOUND)
test = t_q >= BOUND

# ---------- H2: promise setting at a matched on-time rate (purchase-time info only)
gb = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=300, learning_rate=0.05, random_state=42,
                                   categorical_features=[F.index("seller_state_c"), F.index("customer_state_c")])
gb.fit(d.loc[train, F], d.loc[train, "need"])
d["mu_gb"] = gb.predict(d[F])
d["mu_route"] = d.route_dur_90.fillna(d.cust_dur_90).fillna(d.loc[train, "dur"].mean())


def rolling_promise(mu, target, mult):
    """Each test day: offset calibrated on purchases made 90-45 days earlier and delivered before that day."""
    prom = pd.Series(np.nan, index=d.index)
    for day in d.loc[test, "pday"].unique():
        cal = (d.pday >= day - 90 * DAY) & (d.pday < day - 45 * DAY) & (d.order_delivered_customer_date < day)
        r = (d.need[cal] / mu[cal].clip(lower=1)) if mult else (d.need[cal] - mu[cal])
        off = np.quantile(r, target)
        m = test & (d.pday == day)
        prom[m] = np.ceil(mu[m].clip(lower=1) * off) if mult else np.ceil(mu[m] + off)
    return prom


x = d[test]
print(f"\nH2 promise setting on later orders (n={test.sum()}); Olist: mean {x.olist_promise.mean():.1f}d, on-time {1 - x.late.mean():.1%}")
for target in [0.90, 0.95, 1 - x.late.mean()]:
    for name, mu, mult in [("route-90d mean", d.mu_route, False), ("GBM q90 additive", d.mu_gb, False),
                           ("GBM q90 multiplic.", d.mu_gb, True)]:
        p = rolling_promise(mu, target, mult)[test]
        ot = (p >= x.need).mean()
        print(f"  target {target:.3f} {name:20s}: mean promise {p.mean():5.1f}d (Olist {x.olist_promise.mean():.1f}), "
              f"on-time {ot:.1%}, worst month on-time {(p >= x.need).groupby(x.pday.dt.month).mean().min():.1%}")
# post-hoc upper bound: same on-time as Olist, offset chosen on test itself
for name, mu in [("route", d.mu_route), ("GBM", d.mu_gb)]:
    r = (x.need - mu[test]); off = np.quantile(r, 1 - x.late.mean())
    print(f"  oracle-matched {name}: mean promise {np.ceil(mu[test] + off).mean():.1f}d at on-time {(np.ceil(mu[test] + off) >= x.need).mean():.1%}")

# ---------- H3: late risk at carrier handover vs at purchase
d["slack_hand"] = d.olist_promise - d.handover - d.route_leg_90
d["slack_purch"] = d.olist_promise - d.route_dur_90
FH = F + ["olist_promise", "handover", "slack_hand"]
FP = F + ["olist_promise", "slack_purch"]
ok = d.handover.notna()
for name, feats in [("purchase", FP), ("handover", FH)]:
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=42)
    clf.fit(d.loc[train & ok, feats], d.loc[train & ok, "late"])
    pp = clf.predict_proba(d.loc[test & ok, feats])[:, 1]; y = d.loc[test & ok, "late"]
    k = int(0.10 * len(y)); top = np.argsort(-pp)[:k]
    print(f"\nH3 {name}-time late risk: PR-AUC {average_precision_score(y, pp):.3f} (base {y.mean():.3f}), "
          f"ROC {roc_auc_score(y, pp):.3f}, top-10% precision {y.values[top].mean():.1%}, recall {y.values[top].sum() / y.sum():.1%}")
    for mth, g2 in d.loc[test & ok].assign(p=pp).groupby(lambda i: d.pday[i].month):
        print(f"    month {mth}: PR-AUC {average_precision_score(g2.late, g2.p):.3f} base {g2.late.mean():.3f} ROC {roc_auc_score(g2.late, g2.p):.3f}")
# how much warning time does a handover alert leave?
lt = d[test & ok & (d.late == 1)]
print(f"  late orders: median days from handover to promised date {((lt.order_estimated_delivery_date - lt.order_delivered_carrier_date) / DAY).median():.1f}")
z = d[test & ok & d.slack_hand.notna()]
print(f"\nRULE handover slack (promise - elapsed - route 90d carrier leg): PR-AUC {average_precision_score(z.late, -z.slack_hand):.3f}, "
      f"ROC {roc_auc_score(z.late, -z.slack_hand):.3f}, n={len(z)}")
z2 = d[test & ok]
print(f"RULE handover elapsed alone: PR-AUC {average_precision_score(z2.late, z2.handover):.3f}")
