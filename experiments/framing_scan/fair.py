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

# ---- purchase-clock features (Task 1)
r = d.route; c = d.customer_state
d["rd90"], d["rd_std"], d["rd_n"], _ = asof(r, d.deliv, d.dur, r, d.tq, 90)
d["rd30"], *_ = asof(r, d.deliv, d.dur, r, d.tq, 30)
d["rd_all"], *_ = asof(r, d.deliv, d.dur, r, d.tq)
d["cd90"], *_ = asof(c, d.deliv, d.dur, c, d.tq, 90)
d["sell_hand"], _, d["sell_n"], _ = asof(d.seller_id, d.th, d.handover, d.seller_id, d.tq)
d["mu_route"] = d.rd90.fillna(d.cd90).fillna(d.dur.mean())
F1 = ["rd90", "rd30", "rd_all", "cd90", "rd_std", "rd_n", "sell_hand", "sell_n", "n_items", "price", "freight", "weight",
      "n_sellers", "dow", "same_state", "seller_state_c", "customer_state_c"]
# ---- handover-clock features (Task 2)
hd = d.th.notna()
H = d[hd].copy()
r = H.route; c = H.customer_state
H["rl90"], H["rl_std"], H["rl_n"], H["rl_q90"] = asof(r, H.deliv, H.leg, r, H.th, 90, 0.9)
H["rl30"], *_ = asof(r, H.deliv, H.leg, r, H.th, 30)
H["rl_all"], *_ = asof(r, H.deliv, H.leg, r, H.th)
H["cl90"], *_ = asof(c, H.deliv, H.leg, c, H.th, 90)
H["sell_hand"], _, H["sell_n"], _ = asof(H.seller_id, H.th, H.handover, H.seller_id, H.th)
H["leg_base"] = H.rl90.fillna(H.rl_all).fillna(H.cl90).fillna(H.leg.mean())
H["slack_rule"] = H.promise - H.handover - H.leg_base
H = H.sort_values("th").reset_index(drop=True)
F2 = ["leg_base", "rl_std", "rl_q90", "rl_n", "rl30", "rl_all", "cl90", "sell_hand", "sell_n", "handover", "h_dow", "h_hour", "mtd",
      "weight", "freight", "price", "n_items", "promise", "same_state", "seller_state_c", "customer_state_c"]

VAL = [(T(f"2018-0{m}-01"), T(f"2018-0{m+1}-01") if m < 5 else T("2018-05-26")) for m in range(1, 6)]
TST = [(T("2018-05-26"), T("2018-06-01")), (T("2018-06-01"), T("2018-07-01")), (T("2018-07-01"), T("2018-08-01")), (T("2018-08-01"), T("2018-09-01"))]
WINS = [("val", i, w) for i, w in enumerate(VAL)] + [("test", i, w) for i, w in enumerate(TST)]
OUT = []
def P(s=""): print(s); OUT.append(s)

def mk(kind, loss="sq", cat_idx=()):
    if kind == "gbm": return HGR(loss="quantile", quantile=0.9, max_iter=150, learning_rate=0.08, random_state=0, categorical_features=list(cat_idx) or None) if loss == "q90" \
        else HGR(max_iter=150, learning_rate=0.08, random_state=0, categorical_features=list(cat_idx) or None)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))

# =============================== TASK 1
TG = np.array([.80, .85, .88, .90, .92, .93, .95, .96, .97, .98, .99]); REP = [.85, .90, .93, .95, .97]
cat1 = [F1.index("seller_state_c"), F1.index("customer_state_c")]
need = d.need.values; pday = d.pday.values; deliv = d.deliv.values; tq = d.tq.values
MODELS = [("gbm_sq", "gbm", "sq"), ("gbm_q90", "gbm", "q90"), ("ridge", "lin", "sq")]
CALS = [(90, 45), (60, 21)]
res = {}   # (method, cal) -> {winkey: array n x T}
widx = {}
base_mu = d.mu_route.values
def promises(mu_full, rng, win, cal):
    """mu_full: array over d (valid on rng). returns n_win x T promise array, daily recalibrated."""
    lo, hi = cal; out = np.full((len(win), len(TG)), np.nan); wd = pday[win]
    for day in np.unique(wd):
        cm = rng[(pday[rng] >= day - np.timedelta64(lo, "D")) & (pday[rng] < day - np.timedelta64(hi, "D")) & (deliv[rng] < day)]
        off = np.quantile(need[cm] - mu_full[cm], TG)
        m = wd == day
        out[m] = np.maximum(np.ceil(mu_full[win[m]][:, None] + off[None, :]), 1)
    return out
for split, i, (W, E) in WINS:
    key = (split, i); win = np.where((tq >= W.to_datetime64()) & (tq < E.to_datetime64()))[0]
    rng = np.where((tq >= (W - 100 * DAY).to_datetime64()) & (tq < E.to_datetime64()))[0]; widx[key] = win
    for cal in CALS: res.setdefault(("route", cal), {})[key] = promises(base_mu, rng, win, cal)
    for scope in ["all", "180"]:
        tr = np.where((deliv < W.to_datetime64()) & ((tq >= (W - 180 * DAY).to_datetime64()) if scope == "180" else True))[0]
        tr = tr[np.argsort(tq[tr])]; X = d[F1].values; y = need - base_mu
        for name, kind, loss in MODELS:
            if kind == "lin" and scope == "180": pass
            full = mk(kind, loss, cat1).fit(X[tr], y[tr]); mu_add = np.zeros(len(d)); mu_add[rng] = full.predict(X[rng])
            for f in np.array_split(np.arange(len(tr)), 4):   # cross-fit so calibration residuals are not in-sample
                trn = tr[np.setdiff1d(np.arange(len(tr)), f)]
                tgt = tr[f][np.isin(tr[f], rng)]
                if len(tgt): mu_add[tgt] = mk(kind, loss, cat1).fit(X[trn], y[trn]).predict(X[tgt])
            mu = base_mu + mu_add
            for cal in CALS: res.setdefault((f"{name}_{scope}", cal), {})[key] = promises(mu, rng, win, cal)
    print("t1 window done", key, flush=True)

def frontier(arrs, keys, mask_fn=None):
    A = np.concatenate([arrs[k] for k in keys]); w = np.concatenate([widx[k] for k in keys])
    if mask_fn is not None: s = mask_fn(w); A, w = A[s], w[s]
    ot = (A >= need[w][:, None]).mean(0); return ot, A.mean(0), A, w
def at(ot, mp, x):
    o_ = np.argsort(ot); return float(np.interp(x, ot[o_], mp[o_], left=np.nan, right=np.nan))
def olist(keys, mask_fn=None):
    w = np.concatenate([widx[k] for k in keys]);
    if mask_fn is not None: w = w[mask_fn(w)]
    return d.promise.values[w].mean(), 1 - d.late.values[w].mean()
vk = [("val", i) for i in range(5)]; tk = [("test", i) for i in range(4)]
P("# Task 1: promise setting\n")
for nm, ks in [("validation (pooled Jan-May 2018)", vk), ("test (2018-05-26 to 08-31)", tk)]:
    p, ot_ = olist(ks); P(f"Olist {nm}: mean promise {p:.2f}d, on-time {ot_:.1%}, n={sum(len(widx[k]) for k in ks)}")
P("\nValidation selection: mean promise at matched on-time 95% / 97% / Olist-val-realised (interp)")
_, olv = olist(vk); rows = []
for (m, cal), arrs in res.items():
    ot, mp, _, _ = frontier(arrs, vk); rows.append((m, cal, at(ot, mp, .95), at(ot, mp, .97), at(ot, mp, olv)))
sel = pd.DataFrame(rows, columns=["method", "cal", "d@95", "d@97", "d@olist"]).round(2); P(sel.to_string(index=False))
sc = sel.assign(s=sel["d@95"] + sel["d@97"] + sel["d@olist"])
best_route = sc[sc.method == "route"].sort_values("s").iloc[0]; best_model = sc[sc.method != "route"].sort_values("s").iloc[0]
BR = ("route", best_route.cal); BM = (best_model.method, best_model.cal)
P(f"\nChosen on validation: route cal={BR[1]}; model={BM[0]} cal={BM[1]}.  (existing baseline = route, (90, 45))")
meths = {"route(90,45)[existing]": ("route", (90, 45)), f"route{BR[1]}[val-best]": BR, f"{BM[0]}{BM[1]}[val-best model]": BM}
_, olt = olist(tk)
for nm, ks in [("VALIDATION", vk), ("TEST", tk)]:
    P(f"\n## Frontier {nm}: mean promise days / achieved on-time per target")
    t = {}
    for lab, mk_ in meths.items():
        ot, mp, _, _ = frontier(res[mk_], ks); t[lab] = [f"{mp[list(TG).index(x)]:.1f}d / {ot[list(TG).index(x)]:.1%}" for x in REP]
    P(pd.DataFrame(t, index=[f"target {x}" for x in REP]).T.to_string())
    ol = olv if nm == "VALIDATION" else olt
    t = {}
    for lab, mk_ in meths.items():
        ot, mp, _, _ = frontier(res[mk_], ks); t[lab] = [at(ot, mp, .95), at(ot, mp, .97), at(ot, mp, ol)]
    P(f"\nInterpolated mean promise days at matched on-time (Olist {nm.lower()} realised on-time = {ol:.1%}; Olist promise = {olist(ks)[0]:.1f}d)")
    P(pd.DataFrame(t, index=["on-time 95%", "on-time 97%", "Olist on-time"]).T.round(2).to_string())
# disagreement and segments (target 0.95)
j = list(TG).index(.95)
for nm, ks in [("VALIDATION", vk), ("TEST", tk)]:
    Ar = np.concatenate([res[BR][k] for k in ks])[:, j]; Am = np.concatenate([res[BM][k] for k in ks])[:, j]
    P(f"\n{nm}: at target .95 share of orders with |model - route promise| >= 3d: {np.mean(np.abs(Am - Ar) >= 3):.1%}; model shorter by >=3d: {np.mean(Ar - Am >= 3):.1%}; longer by >=3d: {np.mean(Am - Ar >= 3):.1%}")
allw = np.concatenate([widx[k] for k in vk]); q1, q2 = np.quantile(d.mu_route.values[allw], [1 / 3, 2 / 3])
segs = {"distance proxy: short route (route mean dur tercile 1)": lambda w: d.mu_route.values[w] <= q1,
        "distance proxy: mid": lambda w: (d.mu_route.values[w] > q1) & (d.mu_route.values[w] <= q2),
        "distance proxy: long": lambda w: d.mu_route.values[w] > q2,
        "same-state route": lambda w: d.same_state.values[w] == 1, "cross-state route": lambda w: d.same_state.values[w] == 0,
        "seller <20 prior deliveries": lambda w: d.sell_n.values[w] < 20, "seller >=20 prior": lambda w: d.sell_n.values[w] >= 20,
        "route history thin (<30 in 90d)": lambda w: d.rd_n.values[w] < 30, "route history rich": lambda w: d.rd_n.values[w] >= 30,
        "weight > 3kg": lambda w: d.weight.values[w] > 3000}
for nm, ks in [("VALIDATION", vk), ("TEST", tk)]:
    rows = []
    for sn, fn in segs.items():
        ot1, mp1, _, w = frontier(res[BR], ks, fn); ot2, mp2, _, _ = frontier(res[BM], ks, fn)
        rows.append((sn, len(w), at(ot1, mp1, .95), at(ot2, mp2, .95), at(ot2, mp2, .95) - at(ot1, mp1, .95)))
    P(f"\nSegments {nm}: mean promise days at matched 95% on-time (route-best vs model-best; negative diff = model shorter)")
    P(pd.DataFrame(rows, columns=["segment", "n", "route", "model", "diff"]).round(2).to_string(index=False))

# =============================== TASK 2
P("\n# Task 2: delay alert at carrier handover\n")
th_ = H.th.values; dl = H.deliv.values; yL = (H.leg - H.leg_base).values; late = H.late.values
cat2 = [F2.index("seller_state_c"), F2.index("customer_state_c")]
XA = H[F2].values; XC = H[F2 + ["slack_rule"]].values
sc2 = {}   # (model, scope) -> {winkey: score}
def clf_logit(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
def clf_gbm(): return HGC(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0, categorical_features=[len(F2) - 1 - 1 + 0 + 0] if False else cat2)
widx2 = {}
for split, i, (W, E) in WINS:
    key = (split, i); win = np.where((th_ >= W.to_datetime64()) & (th_ < E.to_datetime64()))[0]; widx2[key] = win
    sc2.setdefault(("rule", "-"), {})[key] = -H.slack_rule.values[win]
    for scope in ["all", "180"]:
        tr = np.where((dl < W.to_datetime64()) & ((th_ >= (W - 180 * DAY).to_datetime64()) if scope == "180" else True))[0]
        reg = HGR(max_iter=150, learning_rate=0.08, random_state=0, categorical_features=cat2).fit(XA[tr], yL[tr])
        slack_m = H.promise.values[win] - H.handover.values[win] - (H.leg_base.values[win] + reg.predict(XA[win]))
        sc2.setdefault(("leg_model", scope), {})[key] = -slack_m
        sc2.setdefault(("logit", scope), {})[key] = clf_logit().fit(XC[tr], late[tr]).predict_proba(XC[win])[:, 1]
        sc2.setdefault(("gbm_clf", scope), {})[key] = clf_gbm().fit(XC[tr], late[tr]).predict_proba(XC[win])[:, 1]
    print("t2 window done", key, flush=True)
def metr(keys, sc):
    pa, ro, rows = [], [], []; hit = {5: 0, 10: 0}; tot = 0; pos = 0
    for k in keys:
        y = late[widx2[k]]; s = sc[k]; pa.append(APS(y, s)); ro.append(AUC(y, s)); tot += len(y); pos += y.sum()
        for f in hit: hit[f] += y[np.argsort(-s)[:int(np.ceil(f / 100 * len(y)))]].sum()
    return pa, ro, hit, tot, pos
def pooled(keys, sc):
    y = np.concatenate([late[widx2[k]] for k in keys]); s = np.concatenate([sc[k] for k in keys]); return APS(y, s), AUC(y, s)
vk = [("val", i) for i in range(5)]; tk = [("test", i) for i in range(4)]
P("Validation scope selection (mean per-window PR-AUC): all-history vs trailing 180d")
pick = {}
for m in ["leg_model", "logit", "gbm_clf"]:
    v = {s: np.mean(metr(vk, sc2[(m, s)])[0]) for s in ["all", "180"]}; pick[m] = max(v, key=v.get)
    P(f"  {m}: all {v['all']:.4f}, 180d {v['180']:.4f} -> {pick[m]}")
rows = {"(a) slack rule": sc2[("rule", "-")], "(b) leg model -> slack": sc2[("leg_model", pick["leg_model"])],
        "(c1) logistic": sc2[("logit", pick["logit"])], "(c2) GBM classifier": sc2[("gbm_clf", pick["gbm_clf"])]}
for nm, ks in [("VALIDATION (mean over 5 monthly windows)", vk), ("TEST (4 months, scores pooled; top-k within each month)", tk)]:
    tab = {}
    for lab, s in rows.items():
        pa, ro, hit, tot, pos = metr(ks, s)
        if nm.startswith("VAL"): a, b = np.mean(pa), np.mean(ro)
        else: a, b = pooled(ks, s)
        tab[lab] = [a, b, hit[5] / (0.05 * tot), hit[5] / pos, hit[10] / (0.10 * tot), hit[10] / pos]
    base = np.mean([late[widx2[k]].mean() for k in ks]); P(f"\n## {nm}; base late rate ~{base:.3f}")
    P(pd.DataFrame(tab, index=["PR-AUC", "ROC-AUC", "top5% prec", "top5% recall", "top10% prec", "top10% recall"]).T.round(3).to_string())
P("\n## TEST by month (PR-AUC / top-10% recall)")
tab = {}
for lab, s in rows.items():
    r_ = []
    for k in tk:
        y = late[widx2[k]]; sk = s[k]; top = np.argsort(-sk)[:int(np.ceil(.1 * len(y)))]
        r_.append(f"{APS(y, sk):.3f} / {y[top].sum() / y.sum():.1%}")
    tab[lab] = r_
P(pd.DataFrame(tab, index=["May26-31", "Jun", "Jul", "Aug"]).T.to_string())
P("test months base late rate: " + ", ".join(f"{late[widx2[k]].mean():.3f}" for k in tk))
open("/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/results.md", "w").write("\n".join(OUT))
