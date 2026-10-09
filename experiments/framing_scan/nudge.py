"""Classifier vs reminder rule for seller shipping-deadline misses."""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")
D = "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"
DAY = pd.Timedelta(days=1); HR = pd.Timedelta(hours=1); BOUND = pd.Timestamp("2018-05-26")
o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"])
it = pd.read_csv(f"{D}/olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
pr = pd.read_csv(f"{D}/olist_products_dataset.csv"); se = pd.read_csv(f"{D}/olist_sellers_dataset.csv"); cu = pd.read_csv(f"{D}/olist_customers_dataset.csv")
it = it.merge(pr[["product_id", "product_weight_g", "product_category_name"]], on="product_id", how="left").merge(se[["seller_id", "seller_state"]], on="seller_id").sort_values(["order_id", "order_item_id"])
agg = it.groupby("order_id").agg(n_items=("order_item_id", "size"), price=("price", "sum"), freight=("freight_value", "sum"), weight=("product_weight_g", "sum"),
    n_sellers=("seller_id", "nunique"), seller_id=("seller_id", "first"), seller_state=("seller_state", "first"), cat=("product_category_name", "first"), limit=("shipping_limit_date", "max"))
d0 = o.merge(agg, on="order_id").merge(cu[["customer_id", "customer_state"]], on="customer_id")
d0 = d0[d0.order_approved_at.notna()]
ALL = {k: (g.order_approved_at.values, np.where(g.order_delivered_carrier_date.isna(), np.datetime64("2100-01-01"), g.order_delivered_carrier_date.values), g.limit.values)
       for k, g in d0.groupby("seller_id")}
d = d0[d0.order_delivered_carrier_date.notna()].sort_values("order_approved_at").reset_index(drop=True)
A, H, L = d.order_approved_at, d.order_delivered_carrier_date, d.limit
d["miss"] = (H > L).astype(int); d["hand_days"] = (H - A) / DAY
d["late"] = (d.order_delivered_customer_date.dt.normalize() > d.order_estimated_delivery_date.dt.normalize()).astype(float)
d.loc[d.order_delivered_customer_date.isna(), "late"] = np.nan
top = d[A < BOUND].cat.value_counts().index[:15]; d["cat"] = d.cat.where(d.cat.isin(top), "other").fillna("other")
d["hour"] = A.dt.hour; d["wday"] = A.dt.dayofweek
miss = d.miss.values; late = d.late.values; An, Hn, Ln = A.values, H.values, L.values
N = len(d)

# ---------- timeline ----------
tl = []
allowed = (L - A) / DAY
tl.append(f"Target population (approved, handed over, has items): {N}; miss rate {miss.mean():.3f}")
q = [.05, .1, .25, .5, .75, .9, .95]
tl.append("deadline - approval (days) quantiles " + str(q) + ": " + str(allowed.quantile(q).round(2).tolist()))
tl.append(f"share with deadline before approval: {(allowed<0).mean():.4f}; allowed<1d {(allowed<1).mean():.3f}; <2d {(allowed<2).mean():.3f}")
for lab, m in [("on-time", miss == 0), ("missed", miss == 1)]:
    tl.append(f"handover - approval (days), {lab} (n={m.sum()}) quantiles: " + str(d.hand_days[m].quantile(q).round(2).tolist()))
mm = miss == 1
tl.append(f"overshoot H-L for misses (hours) quantiles: " + str(((H - L)[mm] / HR).quantile(q).round(1).tolist()))
for x in (48, 24, 12):
    t = np.maximum(Ln - np.timedelta64(x, "h"), An); uns = Hn > t
    tl.append(f"-{x}h: misses unshipped at deadline-{x}h = {uns[mm].mean():.3f} (tautological: miss means H>deadline); misses whose deadline-{x}h falls after approval (full lead available) = {(Ln-np.timedelta64(x,'h')>An)[mm].mean():.3f}; "
              f"all orders still unshipped at nudge time = {uns.mean():.3f}; miss precision among them = {miss[uns].mean():.3f}")

# ---------- features as of arbitrary per-order time ----------
def raw_feats(t):
    cols = {k: np.full(N, np.nan) for k in ["hi", "cm", "ch", "s_m30", "s_m90", "backlog", "overdue", "ship24", "n7"]}
    for sid, g in d.groupby("seller_id"):
        idx = g.index.values; tt = t[idx]; aa, hh, ll = ALL[sid]
        ho = np.argsort(Hn[idx]); hs = Hn[idx][ho]; m = miss[idx][ho]; hd = d.hand_days.values[idx][ho]
        cm = np.concatenate([[0], np.cumsum(m)]); ch = np.concatenate([[0], np.cumsum(hd)])
        hi = np.searchsorted(hs, tt, "left"); cols["hi"][idx] = hi; cols["cm"][idx] = cm[hi]; cols["ch"][idx] = ch[hi]
        for w, k in [(30, "s_m30"), (90, "s_m90")]:
            lo = np.searchsorted(hs, tt - np.timedelta64(w, "D"), "left"); n = hi - lo
            cols[k][idx] = np.where(n >= 3, (cm[hi] - cm[lo]) / np.maximum(n, 1), np.nan)
        T = tt[:, None]; past = (aa[None, :] < T); unsh = past & (hh[None, :] > T)
        later = tt > An[idx]  # self is counted when t > approval -> remove
        cols["backlog"][idx] = (unsh & (aa[None, :] >= T - np.timedelta64(30, "D"))).sum(1) - later
        cols["overdue"][idx] = (unsh & (ll[None, :] <= T)).sum(1) - (later & (Ln[idx] <= tt))
        cols["n7"][idx] = (past & (aa[None, :] >= T - np.timedelta64(7, "D"))).sum(1) - later
        hsa = np.sort(hh); cols["ship24"][idx] = np.searchsorted(hsa, tt, "right") - np.searchsorted(hsa, tt - np.timedelta64(24, "h"), "right")
    return pd.DataFrame(cols)

TIMES = {"A": An}
for x in (48, 24, 12): TIMES[x] = np.maximum(Ln - np.timedelta64(x, "h"), An)
RAW = {k: raw_feats(v) for k, v in TIMES.items()}; print("features done", flush=True)
NUM = ["allowed", "elapsed", "remain", "s_n", "s_miss", "s_hand", "s_m30", "s_m90", "backlog", "overdue", "ship24", "n7", "n_items", "price", "freight", "weight", "n_sellers", "hour", "wday"]
CAT = ["cat", "seller_state", "customer_state"]
def build(key, gm, gh):
    r = RAW[key]; t = TIMES[key]
    X = d[["n_items", "price", "freight", "weight", "n_sellers", "hour", "wday"] + CAT].copy()
    X["allowed"] = allowed.values; X["elapsed"] = (t - An) / np.timedelta64(1, "D"); X["remain"] = (Ln - t) / np.timedelta64(1, "D")
    X["s_n"] = r.hi; X["s_miss"] = (r.cm + 5 * gm) / (r.hi + 5); X["s_hand"] = (r.ch + 5 * gh) / (r.hi + 5)
    for c in ["s_m30", "s_m90", "backlog", "overdue", "ship24", "n7"]: X[c] = r[c]
    return X[NUM + CAT]
def hgb():
    pre = ColumnTransformer([("n", "passthrough", NUM), ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan), CAT)])
    return make_pipeline(pre, HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, categorical_features=[len(NUM) + i for i in range(3)], random_state=0))

def topk(cand, s, k): return cand[np.argsort(-s, kind="stable")[:k]]
def summ(ev, nud, tn):
    n = len(ev); c = nud[miss[nud] == 1]
    return dict(vol=len(nud) / n, nudges=len(nud), recall=len(c) / miss[ev].sum(), prec=miss[nud].mean() if len(nud) else np.nan,
                lead=(Ln[c] - tn[c]) / HR, late=(late[nud] == 1).sum() / (late[ev] == 1).sum())

def run_fold(a, ev_mask):
    """a: refit time. returns {policy: (nud idx, tn)} for the fold"""
    ev = np.where(ev_mask)[0]; n = len(ev)
    known = ((An < np.datetime64(a)) & ((Hn < np.datetime64(a)) | (Ln < np.datetime64(a))))
    gm = miss[known].mean(); gh = d.hand_days.values[known].mean()
    out = {}; U = {}
    for x in (48, 24, 12):
        U[x] = ev[Hn[ev] > TIMES[x][ev]]
        out[f"P2 reminder -{x}h"] = (U[x], TIMES[x])
    V = {"V12": len(U[12]) / n, "V24": len(U[24]) / n, "V24/2": len(U[24]) / n / 2}
    Xs = {k: build(k, gm, gh) for k in ["A", 48, 24]}
    mods = {}
    for k in ["A", 48, 24]:
        tr = known & ((Hn > TIMES[k]) if k != "A" else True)
        mods[k] = hgb().fit(Xs[k][tr], miss[tr])
    sA = mods["A"].predict_proba(Xs["A"].iloc[ev])[:, 1]
    for bn, bv in list(V.items()) + [("5%", .05)]:
        out[f"P1 approval clf top {bn}"] = (topk(ev, sA, int(round(bv * n))), TIMES["A"])
    for x in (48, 24):
        s = mods[x].predict_proba(Xs[x].iloc[U[x]])[:, 1]; sr = Xs[x].s_miss.values[U[x]]
        for bn, bv in V.items():
            k = min(int(round(bv * n)), len(U[x]))
            out[f"P3 clf -{x}h top {bn}"] = (topk(U[x], s, k), TIMES[x])
            out[f"P3r prior-miss rule -{x}h top {bn}"] = (topk(U[x], sr, k), TIMES[x])
    # hybrid: P1 top5% at approval + P2 -24h reminder; earliest nudge per order
    n1 = out["P1 approval clf top 5%"][0]; tn = TIMES[24].copy(); tn[n1] = An[n1]
    out["P4 hybrid P1 5% + P2 -24h"] = (np.union1d(n1, U[24]), tn)
    out["_V"] = V
    return out

windows = [("2018-01-01", "2018-02-01"), ("2018-02-01", "2018-03-01"), ("2018-03-01", "2018-04-01"), ("2018-04-01", "2018-05-01"), ("2018-05-01", "2018-05-26")]
val = {}
for a, b in windows:
    ev_mask = ((A >= a) & (A < b)).values; o_ = run_fold(a, ev_mask); ev = np.where(ev_mask)[0]
    for k, v in o_.items():
        if k != "_V": val.setdefault(k, []).append(summ(ev, *v))
    print("val", a, flush=True)
ev_mask = (A >= BOUND).values; ev = np.where(ev_mask)[0]; o_ = run_fold(BOUND, ev_mask)
test = {k: summ(ev, *v) for k, v in o_.items() if k != "_V"}
def vagg(rs):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "lead"} for r in rs]).mean()
    df["lead"] = np.median(np.concatenate([r["lead"] for r in rs])); return df
V = {k: vagg(v) for k, v in val.items()}
md = ["# Nudge policies: classifier vs reminder rule\n", "## Timeline\n"] + ["- " + s for s in tl]
md += [f"\nTest n={len(ev)}, miss rate {miss[ev].mean():.3f}, late rate {np.nanmean(late[ev]):.3f}; test budgets (fraction of orders): " + str({k: round(v, 3) for k, v in o_['_V'].items()}),
       "\n## Policy table (test | validation mean of 5 windows). nudges% of orders; recall of misses; precision; median lead (h, deadline - nudge) for caught misses; late% = share of late deliveries among nudged\n",
       "| policy | test nudge% | recall | prec | lead h | late% | val nudge% | recall | prec | lead h | late% |", "|" + "---|" * 11]
for k in test:
    t, v = test[k], V[k]
    md.append(f"| {k} | {100*t['vol']:.1f} | {t['recall']:.3f} | {t['prec']:.3f} | {np.median(t['lead']):.1f} | {t['late']:.3f} | {100*v.vol:.1f} | {v.recall:.3f} | {v.prec:.3f} | {v.lead:.1f} | {v.late:.3f} |")
# by month
mo = A.dt.to_period("M").values
key = ["P1 approval clf top V24", "P2 reminder -24h", "P2 reminder -12h", "P3 clf -24h top V12", "P3 clf -24h top V24/2", "P3 clf -48h top V24", "P3r prior-miss rule -48h top V24", "P4 hybrid P1 5% + P2 -24h"]
md += ["\n## Test by month: nudge% / recall / precision / late% (policy budgets pooled)\n", "| policy | " + " | ".join(f"{m} (n={int((mo[ev]==m).sum())}, miss {miss[ev][mo[ev]==m].mean():.3f})" for m in sorted(set(mo[ev]))) + " |", "|" + "---|" * (1 + len(set(mo[ev])))]
for k in key:
    nud, tn = {kk: vv for kk, vv in [(k, None)]}, None
    nudg = o_[k][0]; row = []
    for m in sorted(set(mo[ev])):
        e = ev[mo[ev] == m]; nn = nudg[mo[nudg] == m]
        if len(e) == 0 or miss[e].sum() == 0: row.append("n/a"); continue
        s = summ(e, nn, o_[k][1]); row.append(f"{100*s['vol']:.1f} / {s['recall']:.2f} / {s['prec']:.2f} / {s['late']:.2f}")
    md.append(f"| {k} | " + " | ".join(row) + " |")
open("results_nudge.md", "w").write("\n".join(md)); print("\n".join(md))
