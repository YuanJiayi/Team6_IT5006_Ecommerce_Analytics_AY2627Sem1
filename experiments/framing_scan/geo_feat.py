"""Feature construction (all strictly as-of purchase time). Writes feat.pkl"""
exec(open("/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/geo_base.py").read())
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
# --- extra columns
cu2 = pd.read_csv(f"{D}/olist_customers_dataset.csv")[["customer_id", "customer_zip_code_prefix", "customer_city"]]
se2 = pd.read_csv(f"{D}/olist_sellers_dataset.csv")[["seller_id", "seller_zip_code_prefix"]]
first = it.sort_values(["order_id", "order_item_id"]).groupby("order_id").first().reset_index()[["order_id", "product_id"]]
first = first.merge(pr[["product_id", "product_category_name"]], on="product_id", how="left")
geo = pd.read_csv(f"{D}/olist_geolocation_dataset.csv").groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
def extra(x):
    x = x.merge(df[["order_id", "customer_id"]].drop_duplicates("order_id"), on="order_id", how="left") if "customer_id" not in x else x
    x = x.merge(cu2, on="customer_id", how="left").merge(se2, on="seller_id", how="left").merge(first[["order_id", "product_category_name"]], on="order_id", how="left")
    return x
d = extra(d)
d["cat_c"] = d.product_category_name.astype("category").cat.codes.astype(float).replace(-1, np.nan)
d["cz5"] = d.customer_zip_code_prefix; d["cz3"] = d.cz5 // 100
def ll(z):
    g = geo.reindex(z.values); return g.geolocation_lat.values, g.geolocation_lng.values
la1, lo1 = ll(d.seller_zip_code_prefix); la2, lo2 = ll(d.cz5)
a, b, c_, e = map(np.radians, (la1, lo1, la2, lo2))
d["dist"] = 6371 * 2 * np.arcsin(np.sqrt(np.sin((c_ - a) / 2) ** 2 + np.cos(a) * np.cos(c_) * np.sin((e - b) / 2) ** 2))
CAP = {"rio branco", "maceio", "macapa", "manaus", "salvador", "fortaleza", "brasilia", "vitoria", "goiania", "sao luis", "cuiaba", "campo grande", "belo horizonte",
       "belem", "joao pessoa", "curitiba", "recife", "teresina", "rio de janeiro", "natal", "porto alegre", "porto velho", "boa vista", "florianopolis", "sao paulo", "aracaju", "palmas"}
d["capital"] = d.customer_city.isin(CAP).astype(int)
d["month"] = d.tq.dt.month; d["leg_h"] = d.leg
def asof2(ev_k, ev_t, ev_v, q_k, q_t, win=None):
    """mean and count (no min count) of event values with event time strictly < query time, within win days."""
    ev = pd.DataFrame({"k": ev_k.values, "t": ev_t.values, "v": ev_v.values}).dropna().sort_values("t")
    q = pd.DataFrame({"k": q_k.values, "t": q_t.values, "i": np.arange(len(q_k))}).dropna(subset=["k"])
    m = np.full(len(q_k), np.nan); n = np.zeros(len(q_k)); egs = {k: g for k, g in ev.groupby("k")}
    for k, qg in q.groupby("k"):
        if k not in egs: continue
        t = egs[k].t.values; v = egs[k].v.values.astype(float); cs = np.r_[0, np.cumsum(v)]
        hi = np.searchsorted(t, qg.t.values, "left")
        lo = np.searchsorted(t, qg.t.values - np.timedelta64(win, "D"), "left") if win else np.zeros_like(hi)
        cnt = hi - lo; i = qg.i.values; n[i] = cnt
        with np.errstate(invalid="ignore", divide="ignore"): m[i] = np.where(cnt > 0, (cs[hi] - cs[lo]) / cnt, np.nan)
    return m, n
def shrink(m, n, parent, k): return (np.nan_to_num(m) * n + k * parent) / (n + k)
r = d.route; c = d.customer_state
d["rd90"], d["rd_std"], d["rd_n"], d["rd_p50"] = asof(r, d.deliv, d.dur, r, d.tq, 90, 0.5)
_, _, _, d["rd_p90"] = asof(r, d.deliv, d.dur, r, d.tq, 90, 0.9)
d["rt_spread"] = d.rd_p90 - d.rd_p50
d["cd90"], *_ = asof(c, d.deliv, d.dur, c, d.tq, 90)
d["mu_route"] = d.rd90.fillna(d.cd90).fillna(d.dur.mean())     # R0 point estimate (same as fair.py)
d["rl90"], _ = asof2(r, d.deliv, d.leg, r, d.tq, 90); d["rl90"] = d.rl90.fillna(d.leg.mean())
d["sell_hand"], d["sell_n"] = asof2(d.seller_id, d.th, d.handover, d.seller_id, d.tq)
d["sell_hand30"], _ = asof2(d.seller_id, d.th, d.handover, d.seller_id, d.tq, 30)
base0 = d.mu_route.values
RAW = {}   # raw (mean, n) per level/window for dur
for lv, key in [("z3", d.cz3), ("z5", d.cz5)]:
    for w in [180, None]: RAW[(lv, w)] = asof2(key, d.deliv, d.dur, key, d.tq, w)
for w in [30, 90]: RAW[("z3", w)] = asof2(d.cz3, d.deliv, d.dur, d.cz3, d.tq, w)
for k in [10, 20, 40]:
    for w in [180, None]:
        z3 = shrink(*RAW[("z3", w)], base0, k); d[f"r1_{w}_{k}"] = shrink(*RAW[("z5", w)], z3, k)
d["z5_dur"] = d["r1_None_20"]; d["z3_dur"] = shrink(*RAW[("z3", None)], base0, 20)
d["z3_dur30"] = shrink(*RAW[("z3", 30)], base0, 20); d["z3_dur90"] = shrink(*RAW[("z3", 90)], base0, 20)
d["z5_n"] = RAW[("z5", None)][1]
bl = d.rl90.values
for lv, key in [("z3", d.cz3), ("z5", d.cz5)]:
    m_, n_ = asof2(key, d.deliv, d.leg, key, d.tq, None)
    d[f"{lv}_leg"] = shrink(m_, n_, bl if lv == "z3" else d["z3_leg"].values, 20)
m_, n_ = asof2(d.cz3, d.deliv, d.leg, d.cz3, d.tq, 30); d["z3_leg30"] = shrink(m_, n_, bl, 20)
m_, n_ = asof2(d.customer_city, d.deliv, d.late, d.customer_city, d.tq, 180)
gl = d.late.mean(); d["city_late"] = shrink(m_, n_, gl, 20)
# seller backlog at purchase time: ALL orders (any status) purchased in previous 30d and not yet handed to carrier at t
al = df[["seller_id", "order_purchase_timestamp", "order_delivered_carrier_date"]].dropna(subset=["seller_id"])
bk = np.zeros(len(d)); dq = d[["seller_id", "tq"]].reset_index(); dq["i"] = dq["index"]
for s_, g in dq.groupby("seller_id"):
    a_ = al[al.seller_id == s_]; tp = a_.order_purchase_timestamp.values; th2 = a_.order_delivered_carrier_date.values
    th2 = np.where(np.isnat(th2), np.datetime64("2100-01-01"), th2)
    t = g.tq.values[:, None]
    bk[g.i.values] = ((tp[None, :] < t) & (tp[None, :] >= t - np.timedelta64(30, "D")) & (th2[None, :] > t)).sum(1)
d["backlog"] = bk
# route volume last 7d (all orders)
rv = np.zeros(len(d)); alr = df[["seller_state", "customer_state", "order_purchase_timestamp"]].dropna(); alr["route"] = alr.seller_state + ">" + alr.customer_state
ev = {k: np.sort(g.order_purchase_timestamp.values) for k, g in alr.groupby("route")}
for k, g in d.groupby("route"):
    t = ev[k]; q = g.tq.values; rv[g.index.values] = np.searchsorted(t, q, "left") - np.searchsorted(t, q - np.timedelta64(7, "D"), "left")
d["route_vol7"] = rv
d.to_pickle(SP + "feat.pkl")
print(d[["dist", "capital", "z5_dur", "z3_leg", "city_late", "backlog", "route_vol7", "rt_spread", "r1_180_20"]].describe().T)
