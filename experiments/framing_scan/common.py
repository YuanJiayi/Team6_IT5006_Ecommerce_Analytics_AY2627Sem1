"""Shared loaders and as-of helpers for fail.py (task 1) and carrier.py (task 2)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
D = "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad"
DAY = pd.Timedelta(days=1); T = pd.Timestamp; K = 20
VAL = [(T(f"2018-0{m}-01"), T(f"2018-0{m+1}-01") if m < 5 else T("2018-05-26")) for m in range(1, 6)]
BOUND = T("2018-05-26")

def asof(ek, et, ev, qk, qt, win=None, quants=()):
    """as-of stats: events with time strictly < query time (optionally within win days). returns n, mean, [quantiles]."""
    ev_ = pd.DataFrame({"k": np.asarray(ek), "t": np.asarray(et), "v": np.asarray(ev, float)}).dropna(subset=["k", "t"]).sort_values("t")
    q = pd.DataFrame({"k": np.asarray(qk), "t": np.asarray(qt), "i": np.arange(len(qk))})
    n = np.zeros(len(q)); m = np.full(len(q), np.nan); qs = [np.full(len(q), np.nan) for _ in quants]
    egs = {k: g for k, g in ev_.groupby("k")}
    for k, qg in q.groupby("k"):
        eg = egs.get(k)
        if eg is None: continue
        t = eg.t.values; v = eg.v.values; cs = np.r_[0, np.cumsum(np.nan_to_num(v))]; i = qg.i.values
        hi = np.searchsorted(t, qg.t.values, "left")
        lo = np.searchsorted(t, qg.t.values - np.timedelta64(win, "D"), "left") if win else np.zeros_like(hi)
        c = hi - lo; n[i] = c
        with np.errstate(invalid="ignore", divide="ignore"): m[i] = np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)
        if quants:
            for j in np.where(c >= 5)[0]:
                r = np.quantile(v[lo[j]:hi[j]], quants)
                for a, x in enumerate(np.atleast_1d(r)): qs[a][i[j]] = x
    return (n, m, *qs)
def shrink(n, m, parent, k=K): return np.where(n > 0, (np.nan_to_num(n * m) + k * parent) / (n + k), parent)

def load_base():
    """one row per order that has items. includes status, timestamps, composition, product, payment, geography."""
    o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date"])
    it = pd.read_csv(f"{D}/olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    pr = pd.read_csv(f"{D}/olist_products_dataset.csv"); se = pd.read_csv(f"{D}/olist_sellers_dataset.csv", dtype={"seller_zip_code_prefix": str})
    cu = pd.read_csv(f"{D}/olist_customers_dataset.csv", dtype={"customer_zip_code_prefix": str})
    pay = pd.read_csv(f"{D}/olist_order_payments_dataset.csv")
    geo = pd.read_csv(f"{D}/olist_geolocation_dataset.csv", dtype={"geolocation_zip_code_prefix": str}).groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
    pr["vol"] = pr.product_length_cm * pr.product_height_cm * pr.product_width_cm
    it = it.merge(pr[["product_id", "product_weight_g", "vol", "product_category_name", "product_photos_qty", "product_name_lenght", "product_description_lenght"]],
                  on="product_id", how="left").merge(se[["seller_id", "seller_state", "seller_zip_code_prefix"]], on="seller_id")
    it = it.sort_values(["order_id", "order_item_id"])
    agg = it.groupby("order_id").agg(n_items=("order_item_id", "size"), price=("price", "sum"), max_price=("price", "max"), freight=("freight_value", "sum"),
        weight=("product_weight_g", "sum"), vol=("vol", "sum"), photos=("product_photos_qty", "mean"), namelen=("product_name_lenght", "mean"),
        desclen=("product_description_lenght", "mean"), n_sellers=("seller_id", "nunique"), n_products=("product_id", "nunique"),
        seller_id=("seller_id", "first"), product_id=("product_id", "first"), cat=("product_category_name", "first"),
        seller_state=("seller_state", "first"), szip=("seller_zip_code_prefix", "first"), limit=("shipping_limit_date", "max"))
    pay["voucher_v"] = np.where(pay.payment_type == "voucher", pay.payment_value, 0)
    pg = pay.groupby("order_id").agg(n_pay=("payment_sequential", "size"), installments=("payment_installments", "max"), pay_value=("payment_value", "sum"), voucher_v=("voucher_v", "sum"))
    main = pay.sort_values("payment_value", ascending=False).drop_duplicates("order_id").set_index("order_id").payment_type.rename("pay_type")
    pg = pg.join(main); pg["voucher_share"] = pg.voucher_v / pg.pay_value.replace(0, np.nan)
    d = o.merge(agg, left_on="order_id", right_index=True).merge(pg, left_on="order_id", right_index=True, how="left") \
         .merge(cu[["customer_id", "customer_state", "customer_zip_code_prefix"]], on="customer_id")
    d = d.rename(columns={"customer_zip_code_prefix": "z5"})
    la, lo = geo.geolocation_lat, geo.geolocation_lng
    a1, o1 = d.szip.map(la).values, d.szip.map(lo).values; a2, o2 = d.z5.map(la).values, d.z5.map(lo).values
    r1, r2, dl_, dg_ = map(np.radians, (a1, a2, a2 - a1, o2 - o1))
    d["dist"] = 6371 * 2 * np.arcsin(np.sqrt(np.sin(dl_ / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dg_ / 2) ** 2))
    d["same_state"] = (d.seller_state == d.customer_state).astype(int)
    d["route"] = d.seller_state + ">" + d.customer_state
    d["z3"] = d.z5.str[:3]
    d["pdow"] = d.order_purchase_timestamp.dt.dayofweek; d["phour"] = d.order_purchase_timestamp.dt.hour; d["pdom"] = d.order_purchase_timestamp.dt.day
    d["cat"] = d["cat"].fillna("unknown")
    return d, o

def seller_backlog(d, t_col, all_orders):
    """backlog at t_col: that seller's orders (any status, first-item seller) approved in the prior 30d and not handed to carrier by t. null carrier=unshipped."""
    a = all_orders[all_orders.order_approved_at.notna()]
    G = {k: (g.order_approved_at.values, np.where(g.order_delivered_carrier_date.isna(), np.datetime64("2100-01-01"), g.order_delivered_carrier_date.values))
         for k, g in a.groupby("seller_id")}
    out = np.zeros(len(d)); tv = d[t_col].values
    for k, idx in d.groupby("seller_id").indices.items():
        aa, hh = G[k]
        out[idx] = [((aa >= ti - np.timedelta64(30, "D")) & (aa < ti) & (hh >= ti)).sum() for ti in tv[idx]]
    return out
