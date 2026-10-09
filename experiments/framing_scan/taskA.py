"""Task A: carrier delay alert at handover. Learned models vs best rule, as-of features, monthly refits."""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as APS, roc_auc_score as AUC
warnings.filterwarnings("ignore")
D = "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad"
DAY = pd.Timedelta(days=1); T = pd.Timestamp; K = 20
OUT = []
def P(s=""): print(s, flush=True); OUT.append(str(s))

o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date"])
it = pd.read_csv(f"{D}/olist_order_items_dataset.csv")
pr = pd.read_csv(f"{D}/olist_products_dataset.csv"); se = pd.read_csv(f"{D}/olist_sellers_dataset.csv", dtype={"seller_zip_code_prefix": str})
cu = pd.read_csv(f"{D}/olist_customers_dataset.csv", dtype={"customer_zip_code_prefix": str})
geo = pd.read_csv(f"{D}/olist_geolocation_dataset.csv", dtype={"geolocation_zip_code_prefix": str}).groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
it = it.merge(pr[["product_id", "product_weight_g"]], on="product_id", how="left").merge(se[["seller_id", "seller_state", "seller_zip_code_prefix"]], on="seller_id")
agg = it.groupby("order_id").agg(price=("price", "sum"), freight=("freight_value", "sum"), weight=("product_weight_g", "sum"),
    seller_id=("seller_id", "first"), seller_state=("seller_state", "first"), szip=("seller_zip_code_prefix", "first"))
df = o.merge(agg, on="order_id").merge(cu[["customer_id", "customer_state", "customer_zip_code_prefix"]], on="customer_id")
df["route"] = df.seller_state + ">" + df.customer_state
d = df[(df.order_status == "delivered") & df.order_delivered_customer_date.notna() & df.order_delivered_carrier_date.notna()].copy()
d["tq"] = d.order_purchase_timestamp; d["th"] = d.order_delivered_carrier_date; d["deliv"] = d.order_delivered_customer_date
d["pday"] = d.tq.dt.normalize()
d["need"] = (d.deliv.dt.normalize() - d.pday).dt.days
d["promise"] = (d.order_estimated_delivery_date.dt.normalize() - d.pday).dt.days
d["late"] = (d.need > d.promise).astype(int)
d["handover"] = (d.th - d.tq) / DAY; d["leg"] = (d.deliv - d.th) / DAY
d["h_dow"] = d.th.dt.dayofweek
d["rem"] = d.promise - d.handover                     # days remaining at handover
d["z5"] = d.customer_zip_code_prefix; d["z3"] = d.z5.str[:3]
la, lo = geo.geolocation_lat, geo.geolocation_lng
a1, o1 = d.szip.map(la).values, d.szip.map(lo).values; a2, o2 = d.z5.map(la).values, d.z5.map(lo).values
r1, r2, dl_, dg_ = map(np.radians, (a1, a2, a2 - a1, o2 - o1))
d["dist"] = 6371 * 2 * np.arcsin(np.sqrt(np.sin(dl_ / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dg_ / 2) ** 2))
d["same_state"] = (d.seller_state == d.customer_state).astype(int)
for c in ["seller_state", "customer_state"]: d[c + "_c"] = d[c].astype("category").cat.codes
d = d.sort_values("th").reset_index(drop=True)

def asof(ek, et, ev, qk, qt, win=None, quants=()):
    """as-of (events with time strictly < query time, optionally within win days). returns n, mean, [quantiles (need n>=5)]."""
    ev_ = pd.DataFrame({"k": np.asarray(ek), "t": np.asarray(et), "v": np.asarray(ev, float)}).sort_values("t")
    q = pd.DataFrame({"k": np.asarray(qk), "t": np.asarray(qt), "i": np.arange(len(qk))})
    n = np.zeros(len(q)); m = np.full(len(q), np.nan); qs = [np.full(len(q), np.nan) for _ in quants]
    egs = {k: g for k, g in ev_.groupby("k")}
    for k, qg in q.groupby("k"):
        eg = egs.get(k)
        if eg is None: continue
        t = eg.t.values; v = eg.v.values; cs = np.r_[0, np.cumsum(v)]; i = qg.i.values
        hi = np.searchsorted(t, qg.t.values, "left")
        lo = np.searchsorted(t, qg.t.values - np.timedelta64(win, "D"), "left") if win else np.zeros_like(hi)
        c = hi - lo; n[i] = c
        with np.errstate(invalid="ignore", divide="ignore"): m[i] = np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)
        if quants:
            for j in np.where(c >= 5)[0]:
                r = np.quantile(v[lo[j]:hi[j]], quants)
                for a, x in enumerate(np.atleast_1d(r)): qs[a][i[j]] = x
    return (n, m, *qs)
def shrink(n, m, parent): return np.where(n > 0, (np.nan_to_num(n * m) + K * parent) / (n + K), parent)

th, de = d.th, d.deliv
rt = d.route
n9, d["rl90"], q50, q90 = asof(rt, de, d.leg, rt, th, 90, (.5, .9))
d["spread"] = q90 - q50; d["rl_q90"] = q90
_, d["rl30"] = asof(rt, de, d.leg, rt, th, 30)[:2]
n7, *_ = asof(rt, th, np.ones(len(d)), rt, th, 7)   # NB: events = delivered orders only; see vol7 below for all handed orders
_, d["rl_all"] = asof(rt, de, d.leg, rt, th)[:2]
_, d["cl90"] = asof(d.customer_state, de, d.leg, d.customer_state, th, 90)[:2]
_, gl = asof(np.zeros(len(d)), de, d.leg, np.zeros(len(d)), th)[:2]
_, glate = asof(np.zeros(len(d)), de, d.late, np.zeros(len(d)), th)[:2]
gl = np.where(np.isnan(gl), d.leg.mean(), gl); glate = np.where(np.isnan(glate), d.late.mean(), glate)
d["leg_base"] = d.rl90.fillna(d.rl_all).fillna(d.cl90).fillna(pd.Series(gl, index=d.index))
d["trend"] = (d.rl30 - d.rl90).fillna(0); d["spread"] = d.spread.fillna(d.spread.median())
d["slack_rule"] = d.rem - d.leg_base
# volume handed over on the route in the previous 7 days (all handed-over orders, delivered or not)
allh = df[df.order_delivered_carrier_date.notna()]
d["vol7"] = asof(allh.route, allh.order_delivered_carrier_date, np.ones(len(allh)), rt, th, 7)[0]
# zip chains: zip5 -> zip3 -> customer state -> route (shrink child toward parent), for leg days and late share
def chain(target, route_parent):
    p = route_parent
    for key in ["customer_state", "z3", "z5"]:
        n, m = asof(d[key], de, d[target], d[key], th)[:2]
        p = shrink(n, m, p)
        if key == "z3": d["_p3_" + target] = p.copy()
    return p
d["leg_z5"] = chain("leg", d.leg_base.values); d["leg_z3"] = d["_p3_leg"]
rl_n, rl_m = asof(rt, de, d.late, rt, th)[:2]
d["late_route"] = shrink(rl_n, rl_m, glate)
d["late_z5"] = chain("late", d.late_route.values)
d["n_z5"] = asof(d.z5, de, d.leg, d.z5, th)[0]

# ---------- feature sets
for c in ["leg_z5", "leg_z3"]:
    d["slack_" + c] = d.rem - d[c]
d["slack_q"] = d.rem - d.rl_q90.fillna(d.leg_base + d.spread)
d["ratio_z5"] = d.leg_z5 / d.rem.clip(lower=0.5); d["ratio_base"] = d.leg_base / d.rem.clip(lower=0.5)
d["sp_rel"] = d.spread / d.rem.clip(lower=0.5)
GEO = ["dist", "leg_z5", "leg_z3", "leg_base", "late_z5", "late_route", "rl30", "rl_all", "rl_q90", "spread", "trend", "vol7", "n_z5",
       "rem", "weight", "freight", "price", "h_dow", "handover", "promise", "same_state", "seller_state_c", "customer_state_c"]
REL = ["rem", "slack_rule", "slack_leg_z5", "slack_leg_z3", "slack_q", "ratio_z5", "ratio_base", "sp_rel", "spread", "trend", "vol7", "dist", "weight", "freight", "h_dow"]
REG = [c for c in GEO if c not in ("late_z5", "late_route")]   # regression of leg residual: absolute features fine
cat_reg = [REG.index("seller_state_c"), REG.index("customer_state_c")]
cat_geo = [GEO.index("seller_state_c"), GEO.index("customer_state_c")]
P("n handed-over delivered orders: %d; dist NaN share %.3f; zip5 chain coverage n_z5>0: %.3f" % (len(d), d.dist.isna().mean(), (d.n_z5 > 0).mean()))

VAL = [(T(f"2018-0{m}-01"), T(f"2018-0{m+1}-01") if m < 5 else T("2018-05-26")) for m in range(1, 6)]
TST = [(T("2018-05-26"), T("2018-06-01")), (T("2018-06-01"), T("2018-07-01")), (T("2018-07-01"), T("2018-08-01")), (T("2018-08-01"), T("2018-09-01"))]
WINS = [("val", i, w) for i, w in enumerate(VAL)] + [("test", i, w) for i, w in enumerate(TST)]
thv, dlv, late = d.th.values, d.deliv.values, d.late.values
res = {}   # candidate -> {key: score}
widx = {}
resid = (d.leg - d.leg_z5).values
def lin(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
def logit(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
rule_defs = {}
for base in ["leg_base", "leg_z3", "leg_z5"]:
    for c in [0, .5, 1]:
        rule_defs[f"rule {base}+{c}*spread"] = (base, c)
Xr = d[REG].values; Xg = d[GEO].values; Xrel = d[REL].values
for split, i, (W, E) in WINS:
    key = (split, i); win = np.where((thv >= W.to_datetime64()) & (thv < E.to_datetime64()))[0]; widx[key] = win
    for nm, (b, c) in rule_defs.items():
        res.setdefault(nm, {})[key] = -(d.rem.values[win] - d[b].values[win] - c * d.spread.values[win])
    for scope in ["all", "180"]:
        tr = np.where((dlv < W.to_datetime64()) & ((thv >= (W - 180 * DAY).to_datetime64()) if scope == "180" else True))[0]
        rem_w, base_w = d.rem.values[win], d.leg_z5.values[win]
        for nm, mdl in [("reg ridge", lin()), ("reg gbm sq", HGR(max_iter=150, learning_rate=0.06, max_depth=4, random_state=0, categorical_features=cat_reg)),
                        ("reg gbm q80", HGR(loss="quantile", quantile=.8, max_iter=150, learning_rate=0.06, max_depth=4, random_state=0, categorical_features=cat_reg)),
                        ("reg gbm q90", HGR(loss="quantile", quantile=.9, max_iter=150, learning_rate=0.06, max_depth=4, random_state=0, categorical_features=cat_reg))]:
            p = mdl.fit(Xr[tr], resid[tr]).predict(Xr[win]); res.setdefault(f"{nm} [{scope}]", {})[key] = -(rem_w - base_w - p)
        for fs, X, cats in [("rel", Xrel, None), ("all", Xg, cat_geo)]:
            res.setdefault(f"clf logit {fs} [{scope}]", {})[key] = logit().fit(X[tr], late[tr]).predict_proba(X[win])[:, 1]
            res.setdefault(f"clf gbm d3 {fs} [{scope}]", {})[key] = HGC(max_iter=150, learning_rate=0.05, max_depth=3, random_state=0,
                categorical_features=cats).fit(X[tr], late[tr]).predict_proba(X[win])[:, 1]
    print("window done", key, flush=True)

def metr(keys, sc):
    pa, ro = [], []; h5, h10, n5, n10, pos, tot = [], [], [], [], [], []
    for k in keys:
        y = late[widx[k]]; s = sc[k]; pa.append(APS(y, s)); ro.append(AUC(y, s)); o_ = np.argsort(-s, kind="stable")
        h5.append(y[o_[:int(np.ceil(.05 * len(y)))]].sum()); h10.append(y[o_[:int(np.ceil(.10 * len(y)))]].sum()); pos.append(y.sum()); tot.append(len(y))
    return pa, ro, np.array(h5), np.array(h10), np.array(pos), np.array(tot)
vk = [("val", i) for i in range(5)]; tk = [("test", i) for i in range(4)]
vm = {nm: metr(vk, s) for nm, s in res.items()}
vpr = {nm: np.mean(m[0]) for nm, m in vm.items()}
P("\n## Validation mean PR-AUC, all candidates\n")
P("```\n" + pd.Series(vpr).sort_values(ascending=False).round(4).to_string() + "\n```")
fam = lambda pre: max([n for n in res if n.startswith(pre)], key=lambda n: vpr[n])
sel = {"(a) state-route slack rule": "rule leg_base+0*spread", "(b) best zip/spread slack rule": fam("rule"),
       "(c) best regression->slack": fam("reg"), "(d) best classifier": fam("clf")}
P("\nSelected on validation mean PR-AUC: " + str(sel))

def summarise(keys, nm, valmode):
    pa, ro, h5, h10, pos, tot = metr(keys, res[nm])
    if valmode: return [np.mean(pa), np.mean(ro), np.mean(h5 / (.05 * tot)), np.mean(h5 / pos), np.mean(h10 / (.1 * tot)), np.mean(h10 / pos)]
    y = np.concatenate([late[widx[k]] for k in keys]); s = np.concatenate([res[nm][k] for k in keys])
    return [APS(y, s), AUC(y, s), h5.sum() / (.05 * tot.sum()), h5.sum() / pos.sum(), h10.sum() / (.1 * tot.sum()), h10.sum() / pos.sum()]
cols = ["PR-AUC", "ROC-AUC", "top5% prec", "top5% recall", "top10% prec", "top10% recall"]
for title, keys, vm_ in [("VALIDATION (mean of 5 monthly windows)", vk, True), ("TEST (2018-05-26 on; scores pooled, top-k taken within each month)", tk, False)]:
    P(f"\n## {title}; base late rate {np.mean([late[widx[k]].mean() for k in keys]):.3f}")
    P(pd.DataFrame({lab: summarise(keys, nm, vm_) for lab, nm in sel.items()}, index=cols).T.round(3).to_string())
P("\n## TEST per month: PR-AUC / top-10% recall (n, late rate)")
tab = {}
for lab, nm in sel.items():
    r_ = []
    for k in tk:
        y = late[widx[k]]; s = res[nm][k]; top = np.argsort(-s, kind="stable")[:int(np.ceil(.1 * len(y)))]
        r_.append(f"{APS(y, s):.3f} / {y[top].sum() / y.sum():.1%}")
    tab[lab] = r_
P(pd.DataFrame(tab, index=[f"{m} (n={len(widx[k])}, {late[widx[k]].mean():.3f})" for m, k in zip(["May26-31", "Jun", "Jul", "Aug"], tk)]).T.to_string())
P("\n## VALIDATION per window top-10% recall (selected)")
P(pd.DataFrame({lab: [f"{h / p:.1%}" for h, p in zip(metr(vk, res[nm])[3], metr(vk, res[nm])[4])] for lab, nm in sel.items()},
               index=["Jan", "Feb", "Mar", "Apr", "May1-25"]).T.to_string())
open(f"{SP}/results_cls_A.md", "w").write("\n".join(OUT))
