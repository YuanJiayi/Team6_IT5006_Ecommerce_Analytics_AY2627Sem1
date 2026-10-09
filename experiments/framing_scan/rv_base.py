"""Build order-level table with leak-free as-of history features. Output rv_base.pkl"""
import numpy as np, pandas as pd
D = "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"
DAY = pd.Timedelta(days=1)
PRIOR, K = 0.13, 10

def asof_sum_n(keys, ev_time, ev_val, q_keys, q_time):
    """as-of (strictly earlier events) sum and count per key (adapted from hypotheses.asof_mean)."""
    S = np.zeros(len(q_keys)); N = np.zeros(len(q_keys))
    ev = pd.DataFrame({"k": keys, "t": ev_time, "v": ev_val}).dropna().sort_values("t")
    q = pd.DataFrame({"k": np.asarray(q_keys), "t": np.asarray(q_time), "i": np.arange(len(q_keys))})
    egs = {k: g for k, g in ev.groupby("k")}
    for k, qg in q.groupby("k"):
        eg = egs.get(k)
        if eg is None: continue
        cs = np.concatenate([[0], np.cumsum(eg.v.values.astype(float))])
        hi = np.searchsorted(eg.t.values, qg.t.values, side="left")
        S[qg.i.values] = cs[hi]; N[qg.i.values] = hi
    return S, N

def build():
    rd = lambda n, **k: pd.read_csv(f"{D}/olist_{n}_dataset.csv", **k)
    o = rd("orders", parse_dates=["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"])
    it = rd("order_items", parse_dates=["shipping_limit_date"])
    pr, se, cu, pay = rd("products"), rd("sellers", dtype={"seller_zip_code_prefix":str}), rd("customers", dtype={"customer_zip_code_prefix":str}), rd("order_payments")
    rv = rd("order_reviews", parse_dates=["review_creation_date","review_answer_timestamp"])
    geo = rd("geolocation", dtype={"geolocation_zip_code_prefix":str}).groupby("geolocation_zip_code_prefix")[["geolocation_lat","geolocation_lng"]].median()
    cat = pd.read_csv(f"{D}/product_category_name_translation.csv").set_index("product_category_name").product_category_name_english
    # reviews per order
    r = rv.sort_values("review_score").groupby("order_id").agg(score=("review_score","min"), rv_time=("review_answer_timestamp","max"))
    txt = rv.assign(t=rv.review_comment_title.fillna("")+" "+rv.review_comment_message.fillna("")).groupby("order_id").t.apply(lambda s: " ".join(s).strip().lower())
    r["text"] = txt
    # items
    it = it.merge(pr, on="product_id", how="left").merge(se[["seller_id","seller_state"]], on="seller_id", how="left")
    it["cat"] = it.product_category_name.map(cat).fillna("unknown")
    it["vol"] = it.product_length_cm*it.product_height_cm*it.product_width_cm
    g = it.groupby("order_id")
    a = g.agg(n_items=("order_item_id","size"), n_sellers=("seller_id","nunique"), n_products=("product_id","nunique"), n_cats=("cat","nunique"),
              price=("price","sum"), freight=("freight_value","sum"), max_price=("price","max"),
              photos_min=("product_photos_qty","min"), photos_mean=("product_photos_qty","mean"),
              desc_mean=("product_description_lenght","mean"), desc_min=("product_description_lenght","min"), name_mean=("product_name_lenght","mean"),
              weight=("product_weight_g","sum"), vol=("vol","sum"), ship_limit=("shipping_limit_date","max"),
              seller_id=("seller_id","first"), seller_state=("seller_state","first"), cat=("cat","first"))
    a["freight_share"] = a.freight/(a.price+a.freight)
    p = pay.groupby("order_id").agg(pay_value=("payment_value","sum"), installments=("payment_installments","max"), n_pay=("payment_sequential","size"))
    p["pay_type"] = pay.sort_values("payment_value").groupby("order_id").payment_type.last()
    d = o.merge(a, on="order_id").join(p, on="order_id").merge(cu[["customer_id","customer_unique_id","customer_state","customer_zip_code_prefix"]], on="customer_id")
    d = d.join(r, on="order_id")
    # distance
    s1 = it.drop_duplicates("order_id")[["order_id","seller_id"]].merge(se[["seller_id","seller_zip_code_prefix"]], on="seller_id")
    d = d.merge(s1[["order_id","seller_zip_code_prefix"]], on="order_id", how="left")
    cl = geo.reindex(d.customer_zip_code_prefix); sl = geo.reindex(d.seller_zip_code_prefix)
    la1, lo1, la2, lo2 = map(np.radians, [cl.geolocation_lat.values, cl.geolocation_lng.values, sl.geolocation_lat.values, sl.geolocation_lng.values])
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    d["distance_km"] = 6371*2*np.arcsin(np.sqrt(h))
    d["same_state"] = (d.seller_state == d.customer_state).astype(int)
    # timing
    P = d.order_purchase_timestamp
    d["approve_h"] = (d.order_approved_at-P)/pd.Timedelta(hours=1)
    d["promise_days"] = (d.order_estimated_delivery_date-P)/DAY
    d["hour"], d["dow"], d["month"] = P.dt.hour, P.dt.dayofweek, P.dt.month
    d["delivered"] = d.order_status.eq("delivered") & d.order_delivered_customer_date.notna()
    d["days_late"] = (d.order_delivered_customer_date.dt.normalize()-d.order_estimated_delivery_date)/DAY   # >0 late
    d["late"] = (d.days_late > 0).astype(float).where(d.delivered)
    d["handover_d"] = (d.order_delivered_carrier_date-P)/DAY
    d["carrier_leg_d"] = (d.order_delivered_customer_date-d.order_delivered_carrier_date)/DAY
    d["ship_slack_d"] = (d.ship_limit-d.order_delivered_carrier_date)/DAY
    d["bad"] = (d.score <= 2).astype(float).where(d.score.notna())
    d = d.sort_values("order_purchase_timestamp").reset_index(drop=True)
    # ---- as-of history at the (order, seller/product/category) level, aggregated to order
    ii = it[["order_id","seller_id","product_id","cat"]].merge(d[["order_id","order_purchase_timestamp","bad","rv_time","late","order_delivered_customer_date"]], on="order_id")
    ii = ii.drop_duplicates(["order_id","seller_id"]).assign(dummy=0) if False else ii
    def hist(key, name, late=False):
        u = ii.drop_duplicates(["order_id", key])
        ev = u[u.bad.notna()]
        S, N = asof_sum_n(ev[key], ev.rv_time, ev.bad, u[key], u.order_purchase_timestamp)
        u = u.assign(**{name+"_n": N, name+"_bad": (S+K*PRIOR)/(N+K)})
        cols = [name+"_n", name+"_bad"]
        if late:
            ev = u[u.late.notna()]
            S2, N2 = asof_sum_n(ev[key], ev.order_delivered_customer_date, ev.late, u[key], u.order_purchase_timestamp)
            u[name+"_late"] = (S2+K*0.08)/(N2+K); u[name+"_nd"] = N2; cols += [name+"_late", name+"_nd"]
        ag = u.groupby("order_id")[cols].agg(["max","min","mean"] if False else ["max","min"])
        ag.columns = [f"{a}_{b}" for a, b in ag.columns]
        return ag
    for key, name, late in [("seller_id","sel",True), ("product_id","prod",False), ("cat","cat",False)]:
        d = d.join(hist(key, name, late), on="order_id")
    # drop redundant
    d = d.drop(columns=[c for c in d.columns if c.endswith("_nd_min") or c in ("cat_n_min","cat_n_max","prod_n_min")])
    d.to_pickle("rv_base.pkl")
    print(d.shape, d.bad.notna().sum(), d.delivered.sum())
    return d
if __name__ == "__main__": build()
