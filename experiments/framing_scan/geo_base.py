"""Fair learned-vs-rule comparison on Olist. Task 1: promise setting. Task 2: handover late alert."""
import numpy as np, pandas as pd, sys, warnings
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as APS, roc_auc_score as AUC
warnings.filterwarnings("ignore")
D = "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"
DAY = pd.Timedelta(days=1); T = pd.Timestamp
o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date"])
it = pd.read_csv(f"{D}/olist_order_items_dataset.csv")
pr = pd.read_csv(f"{D}/olist_products_dataset.csv"); se = pd.read_csv(f"{D}/olist_sellers_dataset.csv"); cu = pd.read_csv(f"{D}/olist_customers_dataset.csv")
it = it.merge(pr[["product_id", "product_weight_g"]], on="product_id", how="left").merge(se[["seller_id", "seller_state"]], on="seller_id")
agg = it.groupby("order_id").agg(n_items=("order_item_id", "size"), price=("price", "sum"), freight=("freight_value", "sum"),
    weight=("product_weight_g", "sum"), n_sellers=("seller_id", "nunique"), seller_id=("seller_id", "first"), seller_state=("seller_state", "first"))
df = o.merge(agg, on="order_id").merge(cu[["customer_id", "customer_state"]], on="customer_id")
# month-to-date handover volume across ALL orders (known at handover time)
h = df.dropna(subset=["order_delivered_carrier_date"]).sort_values("order_delivered_carrier_date")
h["mtd"] = h.groupby(h.order_delivered_carrier_date.dt.to_period("M")).cumcount()
df = df.merge(h[["order_id", "mtd"]], on="order_id", how="left")
d = df[(df.order_status == "delivered") & df.order_delivered_customer_date.notna()].copy()
d["tq"] = d.order_purchase_timestamp; d["th"] = d.order_delivered_carrier_date; d["deliv"] = d.order_delivered_customer_date
d["pday"] = d.tq.dt.normalize()
d["dur"] = (d.deliv - d.tq) / DAY
d["need"] = (d.deliv.dt.normalize() - d.pday).dt.days
d["promise"] = (d.order_estimated_delivery_date.dt.normalize() - d.pday).dt.days
d["late"] = (d.need > d.promise).astype(int)
d["route"] = d.seller_state + ">" + d.customer_state
d["same_state"] = (d.seller_state == d.customer_state).astype(int)
d["handover"] = (d.th - d.tq) / DAY
d["leg"] = (d.deliv - d.th) / DAY
d["dow"] = d.tq.dt.dayofweek; d["h_dow"] = d.th.dt.dayofweek; d["h_hour"] = d.th.dt.hour
for c in ["seller_state", "customer_state"]: d[c + "_c"] = d[c].astype("category").cat.codes
d = d.sort_values("tq").reset_index(drop=True)

def asof(ev_k, ev_t, ev_v, q_k, q_t, win=None, quant=None):
    """Mean/std/count(/quantile) of event values completed strictly before query time (within win days)."""
    ev = pd.DataFrame({"k": ev_k.values, "t": ev_t.values, "v": ev_v.values}).dropna().sort_values("t")
    q = pd.DataFrame({"k": q_k.values, "t": q_t.values, "i": np.arange(len(q_k))})
    m = np.full(len(q), np.nan); s = m.copy(); n = np.zeros(len(q)); qq = m.copy()
    egs = {k: g for k, g in ev.groupby("k")}
    for k, qg in q.groupby("k"):
        if k not in egs: continue
        eg = egs[k]; t = eg.t.values; v = eg.v.values.astype(float)
        cs = np.r_[0, np.cumsum(v)]; cs2 = np.r_[0, np.cumsum(v * v)]
        hi = np.searchsorted(t, qg.t.values, "left")
        lo = np.searchsorted(t, qg.t.values - np.timedelta64(win, "D"), "left") if win else np.zeros_like(hi)
        c = hi - lo; ok = c >= 5; i = qg.i.values
        n[i] = c
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = (cs[hi] - cs[lo]) / c; var = (cs2[hi] - cs2[lo]) / c - mu ** 2
        m[i[ok]] = mu[ok]; s[i[ok]] = np.sqrt(np.clip(var[ok], 0, None))
        if quant:
            for j in np.where(ok)[0]: qq[i[j]] = np.quantile(v[lo[j]:hi[j]], quant)
    return m, s, n, qq

