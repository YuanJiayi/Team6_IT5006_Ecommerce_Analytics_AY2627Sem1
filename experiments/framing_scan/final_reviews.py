"""PART 2: timing-corrected bad-review recovery. Late rule + on-time classifier at delivery."""
import numpy as np, pandas as pd, warnings, sys
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
warnings.filterwarnings("ignore")
SPD = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
D = "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"; DAY = pd.Timedelta(days=1); T = pd.Timestamp
rd = lambda n, **k: pd.read_csv(f"{D}/olist_{n}_dataset.csv", **k)
o = rd("orders", parse_dates=["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"])
it = rd("order_items", parse_dates=["shipping_limit_date"]); pr, se, cu, pay = rd("products"), rd("sellers", dtype={"seller_zip_code_prefix":str}), rd("customers", dtype={"customer_zip_code_prefix":str}), rd("order_payments")
rv = rd("order_reviews", parse_dates=["review_creation_date","review_answer_timestamp"])
geo = rd("geolocation", dtype={"geolocation_zip_code_prefix":str}).groupby("geolocation_zip_code_prefix")[["geolocation_lat","geolocation_lng"]].median()
cat = pd.read_csv(f"{D}/product_category_name_translation.csv").set_index("product_category_name").product_category_name_english
# FIRST answered review per order
r = rv.sort_values(["review_answer_timestamp", "review_id"]).groupby("order_id").first()[["review_score", "review_answer_timestamp", "review_creation_date"]]
r.columns = ["score", "rv_time", "rv_create"]
it = it.merge(pr, on="product_id", how="left").merge(se[["seller_id","seller_state"]], on="seller_id", how="left")
it["cat"] = it.product_category_name.map(cat).fillna("unknown"); it["vol"] = it.product_length_cm*it.product_height_cm*it.product_width_cm
a = it.groupby("order_id").agg(n_items=("order_item_id","size"), n_sellers=("seller_id","nunique"), n_products=("product_id","nunique"), n_cats=("cat","nunique"),
    price=("price","sum"), freight=("freight_value","sum"), max_price=("price","max"), photos_min=("product_photos_qty","min"), photos_mean=("product_photos_qty","mean"),
    desc_mean=("product_description_lenght","mean"), desc_min=("product_description_lenght","min"), name_mean=("product_name_lenght","mean"),
    weight=("product_weight_g","sum"), vol=("vol","sum"), ship_limit=("shipping_limit_date","max"), seller_id=("seller_id","first"), seller_state=("seller_state","first"), cat=("cat","first"))
a["freight_share"] = a.freight/(a.price+a.freight)
p = pay.groupby("order_id").agg(pay_value=("payment_value","sum"), installments=("payment_installments","max"), n_pay=("payment_sequential","size"))
p["pay_type"] = pay.sort_values("payment_value").groupby("order_id").payment_type.last()
d = o.merge(a, on="order_id").join(p, on="order_id").merge(cu[["customer_id","customer_unique_id","customer_state","customer_zip_code_prefix"]], on="customer_id").join(r, on="order_id")
s1 = it.drop_duplicates("order_id")[["order_id","seller_id"]].merge(se[["seller_id","seller_zip_code_prefix"]], on="seller_id")
d = d.merge(s1[["order_id","seller_zip_code_prefix"]], on="order_id", how="left")
cl = geo.reindex(d.customer_zip_code_prefix); sl = geo.reindex(d.seller_zip_code_prefix)
la1, lo1, la2, lo2 = map(np.radians, [cl.geolocation_lat.values, cl.geolocation_lng.values, sl.geolocation_lat.values, sl.geolocation_lng.values])
d["distance_km"] = 6371*2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
d["same_state"] = (d.seller_state == d.customer_state).astype(int)
P0 = d.order_purchase_timestamp; DL = d.order_delivered_customer_date; EST = d.order_estimated_delivery_date
d["approve_h"] = (d.order_approved_at-P0)/pd.Timedelta(hours=1); d["promise_days"] = (EST-P0)/DAY
d["delivered"] = d.order_status.eq("delivered") & DL.notna()
d["days_late"] = (DL.dt.normalize()-EST)/DAY; d["late"] = (d.days_late > 0)
d["days_early"] = (EST - DL)/DAY            # fractional, known at delivery
d["handover_d"] = (d.order_delivered_carrier_date-P0)/DAY; d["carrier_leg_d"] = (DL-d.order_delivered_carrier_date)/DAY
d["ship_slack_d"] = (d.ship_limit-d.order_delivered_carrier_date)/DAY
d["deliv_hour"], d["deliv_dow"], d["pdow"] = DL.dt.hour, DL.dt.dayofweek, P0.dt.dayofweek
d["bad"] = (d.score <= 2).astype(float).where(d.score.notna())
d = d[d.delivered].sort_values("order_delivered_customer_date").reset_index(drop=True)
DL = d.order_delivered_customer_date
# ---- as-of history at delivery time t: only reviews ANSWERED strictly before t
def asof(keys, ev_t, ev_v, q_keys, q_t):
    S = np.zeros(len(q_keys)); N = np.zeros(len(q_keys))
    ev = pd.DataFrame({"k": keys, "t": ev_t, "v": ev_v}).dropna().sort_values("t")
    q = pd.DataFrame({"k": np.asarray(q_keys), "t": np.asarray(q_t), "i": np.arange(len(q_keys))}); egs = {k: g for k, g in ev.groupby("k")}
    for k, qg in q.groupby("k"):
        eg = egs.get(k)
        if eg is None: continue
        cs = np.r_[0, np.cumsum(eg.v.values.astype(float))]; hi = np.searchsorted(eg.t.values, qg.t.values, side="left"); S[qg.i.values] = cs[hi]; N[qg.i.values] = hi
    return S, N
K = 10
d["_g"] = 0; Sg, Ng = asof(d._g, d.rv_time, d.bad, d._g, DL); d["glob_bad"] = np.where(Ng > 50, Sg/np.maximum(Ng, 1), .10)
ii = it[["order_id","seller_id","product_id","cat"]].merge(d[["order_id","order_delivered_customer_date","bad","rv_time","late","glob_bad"]], on="order_id")
def hist(key, name, late=False):
    u = ii.drop_duplicates(["order_id", key]); ev = u[u.bad.notna()]
    S, N = asof(ev[key], ev.rv_time, ev.bad, u[key], u.order_delivered_customer_date)
    u = u.assign(**{name+"_n": N, name+"_bad": (S + K*u.glob_bad)/(N+K)}); cols = [name+"_n", name+"_bad"]
    if late:   # lateness of seller's earlier orders known by t (delivered before t)
        S2, N2 = asof(u[key], u.order_delivered_customer_date, u.late.astype(float), u[key], u.order_delivered_customer_date)
        u[name+"_late"] = (S2+K*.08)/(N2+K); cols += [name+"_late"]
    ag = u.groupby("order_id")[cols].agg(["max","min"]); ag.columns = [f"{a}_{b}" for a, b in ag.columns]; return ag
for key, name, late in [("seller_id","sel",True), ("product_id","prod",False), ("cat","cat",False)]: d = d.join(hist(key, name, late), on="order_id")
d = d.drop(columns=[c for c in d.columns if c in ("cat_n_min","cat_n_max","prod_n_min")])
Sc, Nc = asof(d.customer_unique_id, d.rv_time, d.bad, d.customer_unique_id, DL); d["cust_prev_bad"] = Sc; d["cust_prev_n"] = Nc
# ---- universes
BOUND = T("2018-05-26")
d["has_rv"] = d.score.notna()
d["ontime"] = ~d.late
# evaluable on-time population: reviewed, answered after delivery
d["pop"] = d.ontime & d.has_rv & (d.rv_time > d.order_delivered_customer_date)
for c in ["customer_state", "seller_state", "pay_type", "cat"]: d[c + "_c"] = d[c].astype("category").cat.codes
CAT_C = [c + "_c" for c in ["customer_state", "seller_state", "pay_type", "cat"]]
NUM = ["n_items","n_sellers","n_products","n_cats","price","freight","max_price","freight_share","photos_min","photos_mean","desc_mean","desc_min","name_mean","weight","vol",
 "pay_value","installments","n_pay","distance_km","same_state","approve_h","promise_days","sel_n_max","sel_n_min","sel_bad_max","sel_bad_min","sel_late_max","sel_late_min",
 "prod_n_max","prod_bad_max","prod_bad_min","cat_bad_max","cat_bad_min","cust_prev_bad","cust_prev_n","glob_bad",
 "days_early","handover_d","carrier_leg_d","ship_slack_d","deliv_hour","deliv_dow","pdow"]
FEATS = NUM + CAT_C
VAL = [(T(a), T(b)) for a, b in [("2018-01-01","2018-02-01"),("2018-02-01","2018-03-01"),("2018-03-01","2018-04-01"),("2018-04-01","2018-05-01"),("2018-05-01","2018-05-26")]]
TST = [(T("2018-05-26"), T("2018-06-01")), (T("2018-06-01"), T("2018-07-01")), (T("2018-07-01"), T("2018-08-01")), (T("2018-08-01"), T("2018-09-01"))]
def make(kind, cfg):
    cidx = [FEATS.index(c) for c in CAT_C]
    if kind == "logit":
        ct = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), QuantileTransformer(n_quantiles=200, output_distribution="normal")), NUM),
                                ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT_C)])
        return make_pipeline(ct, LogisticRegression(C=cfg["C"], max_iter=500))
    return HistGradientBoostingClassifier(random_state=0, categorical_features=cidx, **cfg)
CFGS = {"logit C=0.03": ("logit", {"C": .03}), "logit C=0.3": ("logit", {"C": .3}),
 "hgb d3 lr.05 it150": ("hgb", dict(max_depth=3, learning_rate=.05, max_iter=150, min_samples_leaf=100, l2_regularization=1)),
 "hgb leaf15 lr.05 it200": ("hgb", dict(max_leaf_nodes=15, learning_rate=.05, max_iter=200, min_samples_leaf=200, l2_regularization=5)),
 "hgb leaf31 lr.03 it300": ("hgb", dict(max_leaf_nodes=31, learning_rate=.03, max_iter=300, min_samples_leaf=50, l2_regularization=1))}
multi = lambda x: ((x.n_items >= 2) | (x.n_sellers >= 2)).astype(float).values
RULES = {"n_items>=2": lambda x: (x.n_items >= 2).astype(float).values, "n_sellers>=2": lambda x: (x.n_sellers >= 2).astype(float).values,
 "seller shrunk bad rate": lambda x: x.sel_bad_max.fillna(.10).values,
 "combined (multi flag, then seller rate)": lambda x: multi(x) * 10 + x.sel_bad_max.fillna(.10).values,
 "days early (less early = riskier)": lambda x: -x.days_early.values}
def tb(s, seed=0): return s + np.random.RandomState(seed).rand(len(s)) * 1e-9
def metrics(y, s):
    n = len(y); o_ = np.argsort(-s, kind="stable"); r = dict(n=n, base=y.mean(), PR=AP(y, s), ROC=AUC(y, s))
    for q in (.05, .10): k = int(round(n*q)); h = y[o_[:k]].sum(); r[f"p@{int(q*100)}"] = h/k; r[f"r@{int(q*100)}"] = h/y.sum()
    return r
O = []
def P(s=""): print(s, flush=True); O.append(s)
pop = d[d["pop"]].reset_index(drop=True); y = pop.bad.values.astype(int); pt = pop.order_purchase_timestamp
# ---------------- (i) late orders
tst_all = d[d.order_purchase_timestamp >= BOUND]
allbad = tst_all[tst_all.bad == 1]
late_t = tst_all[tst_all.late]; late_r = late_t[late_t.has_rv]
ct_late = late_t.order_estimated_delivery_date + DAY      # contact at END of promised day (conservative)
lb = late_t[late_t.bad == 1]; lb_in_time = (lb.rv_time > lb.order_estimated_delivery_date + DAY).sum()
lb_in_time_0 = (lb.rv_time > lb.order_estimated_delivery_date).sum()
P("# PART 2 results\n\n## (i) Late orders: rule 'not delivered by promised date' -> contact at promised date")
P(f"Test delivered orders (purchase>=2018-05-26): {len(tst_all)}; late: {len(late_t)} ({len(late_t)/len(tst_all):.1%}); reviewed late: {len(late_r)}; bad-review rate among reviewed late: {late_r.bad.mean():.1%} (on-time reviewed: {tst_all[tst_all.ontime&tst_all.has_rv].bad.mean():.1%})")
P(f"All test bad reviews (delivered, first review<=2): {len(allbad)}; late orders hold {len(lb)} ({len(lb)/len(allbad):.1%}); of those answered AFTER end of promised day: {lb_in_time} ({lb_in_time/len(allbad):.1%} of all bad); after 00:00 of promised day: {lb_in_time_0}")
ob = allbad[allbad.ontime]; P(f"On-time bad reviews: {len(ob)} ({len(ob)/len(allbad):.1%}); answered after delivery: {(ob.rv_time>ob.order_delivered_customer_date).sum()} ({(ob.rv_time>ob.order_delivered_customer_date).sum()/len(allbad):.1%} of all bad)")
P(f"Pop (on-time, answered after delivery) size in test: {(pop.order_purchase_timestamp>=BOUND).sum()}")
# ---------------- (ii) classifier
val = {}
for name, (kind, cfg) in CFGS.items():
    rs = []
    for A, B in VAL:
        tr = ((pt < A) & (pop.rv_time < A)).values; te = ((pt >= A) & (pt < B)).values
        m = make(kind, cfg).fit(pop[FEATS][tr], y[tr]); rs.append(metrics(y[te], m.predict_proba(pop[FEATS][te])[:, 1]))
    val[name] = pd.DataFrame(rs)
for name, f in RULES.items():
    rs = []
    for A, B in VAL:
        te = ((pt >= A) & (pt < B)).values; rs.append(metrics(y[te], tb(f(pop[te]))))
    val[name] = pd.DataFrame(rs)
V = pd.DataFrame({k: v.mean() for k, v in val.items()}).T.drop(columns="n")
bm = V.loc[list(CFGS)].PR.idxmax(); br = V.loc[list(RULES)].PR.idxmax()
P("\n## (ii) On-time orders, decision at delivery. Validation (mean of 5 windows)"); P(V.round(3).to_string()); P(f"chosen model: {bm}; chosen rule (val PR-AUC): {br}")
# test with monthly refit
def fit_pred(kind, cfg):
    sc = np.full(len(pop), np.nan); models = {}
    for A, B in TST:
        tr = ((pt < A) & (pop.rv_time < A)).values; te = ((pt >= A) & (pt < B)).values
        m = make(kind, cfg).fit(pop[FEATS][tr], y[tr]); sc[te] = m.predict_proba(pop[FEATS][te])[:, 1]; models[A] = m
    return sc, models
tm = (pt >= BOUND).values; Tp = pop[tm].reset_index(drop=True); yt = y[tm]
sm, models = fit_pred(*CFGS[bm]); sm = sm[tm]
fam_other = max([k for k in CFGS if CFGS[k][0] != CFGS[bm][0]], key=lambda k: V.PR[k]); so, _ = fit_pred(*CFGS[fam_other]); so = so[tm]
sc = {"MODEL " + bm: sm, "MODEL(other family) " + fam_other: so}
for n_, f in RULES.items(): sc["rule: " + n_] = tb(f(Tp))
P(f"\n## TEST (purchases >= 2018-05-26, once; monthly refit)"); P(pd.DataFrame({k: metrics(yt, s) for k, s in sc.items()}).T.round(3).to_string())
mk, rk = "MODEL " + bm, "rule: " + br
mon = Tp.order_purchase_timestamp.dt.to_period("M").values; rows = []
for pm in sorted(set(mon)):
    i = mon == pm; a_, b_ = metrics(yt[i], sm[i]), metrics(yt[i], sc[rk][i]); rows.append((str(pm), i.sum(), a_["base"], a_["PR"], b_["PR"], a_["r@10"], b_["r@10"]))
P("\nTest by month (model vs best rule):"); P(pd.DataFrame(rows, columns=["month","n","base","PR model","PR rule","r@10 model","r@10 rule"]).round(3).to_string(index=False))
rs_ = np.random.RandomState(1); bs = []
for _ in range(500):
    j = rs_.randint(0, len(yt), len(yt)); yy = yt[j]; ks = int(round(len(yy)*.1))
    f = lambda s: yy[np.argsort(-s[j], kind="stable")[:ks]].sum() / yy.sum()
    bs.append((f(sm) - f(sc[rk]), AP(yy, sm[j]) - AP(yy, sc[rk][j])))
bs = np.array(bs); P("Bootstrap (500) model minus best rule: recall@10 %.3f [%.3f, %.3f]; PR-AUC %.3f [%.3f, %.3f]" % (bs[:, 0].mean(), *np.percentile(bs[:, 0], [2.5, 97.5]), bs[:, 1].mean(), *np.percentile(bs[:, 1], [2.5, 97.5])))
# permutation importance using last-window model on the matching month only? use the Aug model on Aug+ rows; simpler: pooled, each row scored by its month's model
def pooled_pred(X_):
    out = np.zeros(len(X_)); mm = Tp.order_purchase_timestamp.values
    for A, B in TST:
        i = (mm >= A.to_datetime64()) & (mm < B.to_datetime64())
        if i.any(): out[i] = models[A].predict_proba(X_[i])[:, 1]
    return out
base = AP(yt, pooled_pred(Tp[FEATS])); pr_ = np.random.RandomState(2); imp = {}
for f in FEATS:
    ds = []
    for _ in range(3): Xp = Tp[FEATS].copy(); Xp[f] = pr_.permutation(Xp[f].values); ds.append(base - AP(yt, pooled_pred(Xp)))
    imp[f] = np.mean(ds)
P("\nPermutation importance (test PR-AUC drop), top 10:"); P(pd.Series(imp).sort_values(ascending=False).head(10).round(4).to_string())
# ---------------- combined system (operational: score ALL on-time delivered test orders at delivery)
P("\n## Combined system (i)+(ii), test: late orders contacted at end of promised day + top q% of ALL on-time delivered orders contacted at delivery")
ot_all = tst_all[tst_all.ontime].reset_index(drop=True)
# model scores via month models (trained on pop); rule scores
mm = ot_all.order_purchase_timestamp.values; s_model = np.zeros(len(ot_all))
for A, B in TST:
    i = (mm >= A.to_datetime64()) & (mm < B.to_datetime64())
    if i.any(): s_model[i] = models[A].predict_proba(ot_all[FEATS][i])[:, 1]
s_rule = tb(RULES[br](ot_all)); nb = len(allbad)
def reached(sel_idx):   # bad reviews reached in time among selected on-time orders
    z = ot_all.iloc[sel_idx]; return int(((z.bad == 1) & (z.rv_time > z.order_delivered_customer_date)).sum())
rows = []
for q in (0.0, .05, .10):
    k = int(round(len(ot_all) * q)); row = [f"late + top {int(q*100)}%" if q else "late only", len(late_t) + k]
    for s in (s_model, s_rule):
        row.append(lb_in_time + reached(np.argsort(-s, kind="stable")[:k]))
    rows.append(row)
tab = pd.DataFrame(rows, columns=["system", "contacts", "bad reached MODEL", "bad reached RULE"])
tab["MODEL share of all bad"] = tab["bad reached MODEL"] / nb; tab["RULE share of all bad"] = tab["bad reached RULE"] / nb
P(f"all test bad reviews = {nb}; on-time delivered test orders = {len(ot_all)}; late = {len(late_t)}; rule = {br}"); P(tab.round(3).to_string(index=False))
# same-budget random baseline for context
P(f"Random-contact baseline for top 10%: expected reached {lb_in_time + 0.10*(((ot_all.bad==1)&(ot_all.rv_time>ot_all.order_delivered_customer_date)).sum()):.0f}")
open(SPD + "results_final_part2.md", "w").write("\n".join(O))
