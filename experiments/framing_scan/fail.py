"""Task 1: fulfilment failure at approval."""
import sys; sys.path.insert(0, "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad")
from common import *
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score as APS, roc_auc_score as AUC
from sklearn.inspection import permutation_importance
OUT = []
def P(s=""): print(s, flush=True); OUT.append(str(s))
rng = np.random.RandomState(0)

d, o = load_base()
n_orders = len(o); n_noitem = (~o.order_id.isin(d.order_id)).sum()
noitem_status = o[~o.order_id.isin(d.order_id)].order_status.value_counts()
unapp = d.order_approved_at.isna().sum()
P(f"All orders {n_orders}; without items {n_noitem} ({dict(noitem_status)}); with items but not approved {unapp}")
fail_all = o.order_status.isin(["canceled", "unavailable"])
P(f"fail (canceled/unavailable) orders overall {fail_all.sum()}; of which have no items (no seller; cannot be scored) {(fail_all & ~o.order_id.isin(d.order_id)).sum()}; "
  f"never approved {(fail_all & o.order_approved_at.isna()).sum()}")
END = o.order_purchase_timestamp.max(); CUT = END - 60 * DAY
STUCK = ["shipped", "invoiced", "processing", "created", "approved"]
full = d.copy()
d = d[d.order_approved_at.notna()].copy()
P(f"Dataset end (last purchase) {END}; stuck cutoff {CUT}")
P(f"Scored population: approved orders with items = {len(d)}")
d["fail"] = d.order_status.isin(["canceled", "unavailable"]).astype(int)
d["stuck_old"] = (d.order_status.isin(STUCK) & (d.order_purchase_timestamp < CUT)).astype(int)
d["fail2"] = ((d.fail == 1) | (d.stuck_old == 1)).astype(int)
d["excl2"] = (d.order_status.isin(STUCK) & (d.order_purchase_timestamp >= CUT)).astype(int)   # variant undecided -> excluded from variant
P(d.order_status.value_counts().to_string()); P(f"fail in scored pop {d.fail.sum()} ({d.fail.mean():.4f}); fail2 {d.fail2.sum()}; excl2 {d.excl2.sum()}")
d = d.sort_values("order_approved_at").reset_index(drop=True)
A = d.order_approved_at; H = d.order_delivered_carrier_date
d["approval_delay"] = (A - d.order_purchase_timestamp) / DAY
d["hour"] = A.dt.hour; d["wday"] = A.dt.dayofweek
d["handover"] = (H - A) / DAY; d["miss"] = np.where(H.notna(), (H > d.limit).astype(float), np.nan)
d["allowed"] = (d.limit - A) / DAY
# ---- reviews
r = pd.read_csv(f"{D}/olist_order_reviews_dataset.csv", parse_dates=["review_answer_timestamp"]).sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
d["score"] = d.order_id.map(r.set_index("order_id").review_score); d["bad"] = (d.score <= 2).astype(float); d.loc[d.score.isna(), "bad"] = np.nan

# ---- as-of features
LAG = 30
t = A.values; ev_mat = (A + LAG * DAY).values       # failure outcome treated as known LAG days after approval
sid = d.seller_id
gn, gm = asof(np.zeros(len(d)), ev_mat, d.fail, np.zeros(len(d)), t)[:2]; g_fail = np.where(np.isnan(gm), d.fail.mean(), gm)
n_, m_ = asof(sid, ev_mat, d.fail, sid, t); d["s_fail"] = shrink(n_, m_, g_fail); d["s_nmature"] = n_
d["s_ord"] = asof(sid, t, np.ones(len(d)), sid, t)[0]
d["s_ord7"] = asof(sid, t, np.ones(len(d)), sid, t, 7)[0]
_, m30 = asof(sid, ev_mat, d.fail, sid, t, 90); d["s_fail90"] = m30
first = d.groupby("seller_id").order_approved_at.transform("min")
d["s_age"] = ((A - first) / DAY)                       # 0 for the seller's first order
_, g_miss = asof(np.zeros(len(d)), H.values, d.miss, np.zeros(len(d)), t)[:2]; g_miss = np.where(np.isnan(g_miss), 0.08, g_miss)
n_, m_ = asof(sid, H.values, d.miss, sid, t); d["s_miss"] = shrink(n_, m_, g_miss, 5)
_, gh = asof(np.zeros(len(d)), H.values, d.handover, np.zeros(len(d)), t)[:2]; gh = np.where(np.isnan(gh), 3, gh)
n_, m_ = asof(sid, H.values, d.handover, sid, t); d["s_hand"] = shrink(n_, m_, gh, 5)
d["backlog"] = seller_backlog(d, "order_approved_at", d)
d["new_seller"] = (d.s_ord == 0).astype(int)
n_, m_ = asof(d.cat, ev_mat, d.fail, d.cat, t); d["cat_fail"] = shrink(n_, m_, g_fail)
n_, m_ = asof(d.product_id, ev_mat, d.fail, d.product_id, t); d["prod_fail"] = shrink(n_, m_, d.cat_fail.values, 10); d["prod_n"] = n_
NUM = ["s_ord", "s_ord7", "s_fail", "s_nmature", "s_fail90", "s_miss", "s_hand", "backlog", "s_age", "cat_fail", "prod_fail", "prod_n", "weight", "vol", "photos",
       "namelen", "desclen", "price", "max_price", "freight", "n_items", "n_products", "n_sellers", "n_pay", "installments", "voucher_share", "pay_value",
       "approval_delay", "dist", "same_state", "hour", "wday", "pdom", "allowed"]
CAT = ["seller_state", "customer_state", "pay_type", "catg"]
topc = d[d.order_approved_at < BOUND].cat.value_counts().index[:30]; d["catg"] = d.cat.where(d.cat.isin(topc), "other")
for c in CAT: d[c] = d[c].fillna("na").astype(str)
X = d[NUM + CAT]
P(f"features: {len(NUM)} numeric + {len(CAT)} categorical")

def logit(C):
    pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CAT)])
    return make_pipeline(pre, LogisticRegression(C=C, max_iter=3000))
def hgb(depth, l2=1.0, lr=0.05, it=200):
    pre = ColumnTransformer([("n", "passthrough", NUM), ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT)])
    return make_pipeline(pre, HGC(max_depth=depth, learning_rate=lr, max_iter=it, l2_regularization=l2, min_samples_leaf=60,
        categorical_features=[len(NUM) + i for i in range(len(CAT))], random_state=0))
MODELS = {"logit C=0.01": lambda: logit(.01), "logit C=0.1": lambda: logit(.1), "logit C=1": lambda: logit(1),
          "hgb d2": lambda: hgb(2), "hgb d3": lambda: hgb(3), "hgb d5 l2=10": lambda: hgb(5, 10.)}
RULES = {"seller prior fail rate": X.s_fail.values, "new-seller flag": d.new_seller.values.astype(float), "seller backlog": d.backlog.values.astype(float),
         "seller prior miss rate": X.s_miss.values, "product prior fail rate": X.prod_fail.values, "approval delay": d.approval_delay.values}
def tb(s): return s + rng.rand(len(s)) * 1e-9   # random tie-break

def topk(y, s, f):
    k = int(np.ceil(f * len(y))); i = np.argsort(-s, kind="stable")[:k]; return y[i].mean(), y[i].sum() / max(y.sum(), 1)
def met(y, s):
    p5, r5 = topk(y, s, .05); p10, r10 = topk(y, s, .10)
    return dict(base=y.mean(), pr=APS(y, s), roc=AUC(y, s), p5=p5, r5=r5, p10=p10, r10=r10)
TST = [("May26-31", T("2018-05-26"), T("2018-06-01")), ("Jun", T("2018-06-01"), T("2018-07-01")), ("Jul", T("2018-07-01"), T("2018-08-01")),
       ("Aug", T("2018-08-01"), T("2018-09-01")), ("Sep-Oct", T("2018-09-01"), T("2019-01-01"))]

def run(label, lag, ok):
    """label: column name; lag: days for label maturity; ok: boolean mask of rows with a decided label."""
    y = d[label].values; Av = A.values
    P(f"\n# ===== Target `{label}` (positives in scored pop {int(y[ok].sum())}, rows used {ok.sum()}) =====")
    val = {}
    for (W, E) in VAL:
        tr = np.where(ok & (Av < (W - lag * DAY).to_datetime64()))[0]; ev = np.where(ok & (Av >= W.to_datetime64()) & (Av < E.to_datetime64()))[0]
        for nm, s in RULES.items(): val.setdefault(nm, []).append(met(y[ev], tb(s[ev])))
        for nm, f in MODELS.items(): val.setdefault(nm, []).append(met(y[ev], f().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[ev])[:, 1]))
        print(label, "val window", W.date(), len(tr), int(y[tr].sum()), len(ev), int(y[ev].sum()), flush=True)
    V = pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in val.items()}).T
    P("\n## Validation (mean of 5 monthly windows), all candidates\n```\n" + V.round(3).to_string() + "\n```")
    br = max(RULES, key=lambda k: V.pr[k]); bm = max(MODELS, key=lambda k: V.pr[k])
    P(f"Selected on validation PR-AUC: best rule = {br}; best model = {bm}")
    # test once
    tr = np.where(ok & (Av < (BOUND - lag * DAY).to_datetime64()))[0]; te = np.where(ok & (Av >= BOUND.to_datetime64()))[0]
    mdl = MODELS[bm]().fit(X.iloc[tr], y[tr]); sm = mdl.predict_proba(X.iloc[te])[:, 1]; sr = tb(RULES[br][te]); sn = tb(RULES["new-seller flag"][te])
    sc = {f"rule: {br}": sr, "rule: new-seller flag": sn, f"model: {bm}": sm}
    yt = y[te]
    rows = {k: met(yt, s) for k, s in sc.items()}
    P(f"\n## TEST (2018-05-26 on; n={len(te)}, positives={int(yt.sum())})\n```\n" + pd.DataFrame(rows).T.round(3).to_string() + "\n```")
    pm = {}
    for k, s in sc.items():
        r_ = []
        for nm, a, b in TST:
            msk = (Av[te] >= a.to_datetime64()) & (Av[te] < b.to_datetime64())
            if yt[msk].sum() == 0: r_.append(f"n={msk.sum()}, pos=0"); continue
            m = met(yt[msk], s[msk]); r_.append(f"PR {m['pr']:.3f} | R@5 {m['r5']:.2f} | R@10 {m['r10']:.2f} (n={msk.sum()}, pos={int(yt[msk].sum())})")
        pm[k] = r_
    P("\n## TEST per month\n```\n" + pd.DataFrame(pm, index=[x[0] for x in TST]).T.to_string() + "\n```")
    # permutation importance (test, AP drop), grouped by single column
    pi = permutation_importance(mdl, X.iloc[te], yt, scoring="average_precision", n_repeats=5, random_state=0)
    P("\n## Permutation importance (test, drop in PR-AUC), top 10\n```\n" + pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(10).round(4).to_string() + "\n```")
    return te, sc

# primary
okp = np.ones(len(d), bool)
te, sc = run("fail", 30, okp)
# ---- review impact
bad = d.bad.values; fl = d.fail.values
P("\n## Review impact")
rv = ~np.isnan(bad)
allr = pd.read_csv(f"{D}/olist_order_reviews_dataset.csv", parse_dates=["review_answer_timestamp"]).sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
allr["bad"] = allr.review_score <= 2; allr["fail"] = allr.order_id.map(o.set_index("order_id").order_status).isin(["canceled", "unavailable"])
P(f"Whole dataset: orders with a review {len(allr)}; bad (1-2 star) {allr.bad.sum()} ({allr.bad.mean():.3f}); bad reviews on fail orders {(allr.bad & allr.fail).sum()} "
  f"= {(allr.bad & allr.fail).sum() / allr.bad.sum():.3%} of all bad reviews; bad-review rate on fail orders {allr[allr.fail].bad.mean():.3f} (n={allr.fail.sum()}), on others {allr[~allr.fail].bad.mean():.3f}")
stk = o.order_status.isin(STUCK) & (o.order_purchase_timestamp < CUT); allr["stuck"] = allr.order_id.map(stk.set_axis(o.order_id))
P(f"Stuck-old orders with review {allr.stuck.sum()}: bad rate {allr[allr.stuck].bad.mean():.3f}; fail+stuck share of all bad reviews {(allr.bad & (allr.fail | allr.stuck)).sum() / allr.bad.sum():.3%}")
yt = d.fail.values[te]; bt = bad[te]; rt = ~np.isnan(bt)
P(f"Test pop: reviewed {rt.sum()}, bad {int(np.nansum(bt))}; bad on fail orders {int(np.nansum(bt * yt))} = {np.nansum(bt*yt)/np.nansum(bt):.2%} of test bad reviews")
k = int(np.ceil(.05 * len(te))); rows = {}
for nm, s in sc.items():
    q = np.argsort(-s, kind="stable")[:k]
    rows[nm] = dict(queue=k, fail_caught=int(yt[q].sum()), fail_recall=yt[q].sum() / yt.sum(), bad_caught=int(np.nansum(bt[q])), share_of_all_bad=np.nansum(bt[q]) / np.nansum(bt),
                    bad_rate_in_queue=np.nanmean(bt[q]), bad_rate_overall=np.nanmean(bt))
P("5% review queue on test (bad = score<=2 on reviewed orders)\n```\n" + pd.DataFrame(rows).T.round(3).to_string() + "\n```")
# variant
ok2 = (d.excl2 == 0).values
run("fail2", 60, ok2)
open(f"{SP}/results_fail.md", "w").write("# Task 1: fulfilment failure at approval\n\n" + "\n".join(OUT))
