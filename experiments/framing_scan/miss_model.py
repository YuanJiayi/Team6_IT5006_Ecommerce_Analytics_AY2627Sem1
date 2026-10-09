"""Predict seller shipping-limit misses at payment approval. Assumes shipping_limit_date is known at approval."""
import numpy as np, pandas as pd, warnings, sys
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
D = "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"
OUT = sys.argv[1]
DAY = pd.Timedelta(days=1); BOUND = pd.Timestamp("2018-05-26")
o = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date"])
it = pd.read_csv(f"{D}/olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
pr = pd.read_csv(f"{D}/olist_products_dataset.csv"); se = pd.read_csv(f"{D}/olist_sellers_dataset.csv")
cu = pd.read_csv(f"{D}/olist_customers_dataset.csv")
it = it.merge(pr[["product_id", "product_weight_g", "product_category_name"]], on="product_id", how="left").merge(se[["seller_id", "seller_state"]], on="seller_id")
it = it.sort_values(["order_id", "order_item_id"])
agg = it.groupby("order_id").agg(n_items=("order_item_id", "size"), price=("price", "sum"), freight=("freight_value", "sum"),
    weight=("product_weight_g", "sum"), n_sellers=("seller_id", "nunique"), seller_id=("seller_id", "first"),
    seller_state=("seller_state", "first"), cat=("product_category_name", "first"), limit=("shipping_limit_date", "max"))
d = o.merge(agg, on="order_id").merge(cu[["customer_id", "customer_state"]], on="customer_id")
d = d[d.order_approved_at.notna() & d.order_delivered_carrier_date.notna()].sort_values("order_approved_at").reset_index(drop=True)
A, H = d.order_approved_at, d.order_delivered_carrier_date
d["miss"] = (H > d.limit).astype(int)
d["hand_days"] = (H - A) / DAY
d["allowed"] = (d.limit - A) / DAY
d["late"] = (d.order_delivered_customer_date.dt.normalize() > d.order_estimated_delivery_date.dt.normalize()).astype(float)
d.loc[d.order_delivered_customer_date.isna(), "late"] = np.nan
top = d[A < BOUND].cat.value_counts().index[:15]
d["cat"] = d.cat.where(d.cat.isin(top), "other").fillna("other")
d["hour"] = A.dt.hour; d["wday"] = A.dt.dayofweek

def history(gm, gh):
    """as-of features. gm global miss rate, gh global mean handover days (from the training fold)."""
    cols = {k: np.full(len(d), np.nan) for k in ["s_n", "s_miss", "s_hand", "s_m30", "s_m90", "backlog", "n7"]}
    for _, g in d.groupby("seller_id"):
        idx = g.index.values; t = A[idx].values
        e = np.maximum(A[idx].values, H[idx].values)       # time the order stops being backlog
        ho = np.argsort(H[idx].values); hs = H[idx].values[ho]
        m = g.miss.values[ho]; hd = g.hand_days.values[ho]
        cm = np.concatenate([[0], np.cumsum(m)]); ch = np.concatenate([[0], np.cumsum(hd)])
        hi = np.searchsorted(hs, t, "left")                # handovers strictly before t
        cols["s_n"][idx] = hi
        cols["s_miss"][idx] = (cm[hi] + 5 * gm) / (hi + 5)
        cols["s_hand"][idx] = (ch[hi] + 5 * gh) / (hi + 5)
        for w, k in [(30, "s_m30"), (90, "s_m90")]:
            lo = np.searchsorted(hs, t - np.timedelta64(w, "D"), "left"); n = hi - lo
            cols[k][idx] = np.where(n >= 3, (cm[hi] - cm[lo]) / np.maximum(n, 1), np.nan)
        ta = np.sort(t)
        cols["backlog"][idx] = np.searchsorted(ta, t, "left") - np.searchsorted(np.sort(e), t, "left")
        cols["n7"][idx] = np.searchsorted(ta, t, "left") - np.searchsorted(ta, t - np.timedelta64(7, "D"), "left")
    return pd.DataFrame(cols)

NUM = ["allowed", "s_n", "s_miss", "s_hand", "s_m30", "s_m90", "backlog", "n7", "n_items", "price", "freight", "weight", "n_sellers", "hour", "wday"]
CAT = ["cat", "seller_state", "customer_state"]
def feats(train_mask):
    gm = d.miss[train_mask].mean(); gh = d.hand_days[train_mask].mean()
    X = pd.concat([d[["allowed", "n_items", "price", "freight", "weight", "n_sellers", "hour", "wday"] + CAT], history(gm, gh)], axis=1)
    return X, gh

def lr():
    pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT)])
    return make_pipeline(pre, LogisticRegression(C=0.1, max_iter=2000))
def hgb(depth):
    pre = ColumnTransformer([("n", "passthrough", NUM),
        ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan), CAT)])
    return make_pipeline(pre, HistGradientBoostingClassifier(max_depth=depth, learning_rate=0.05, max_iter=300,
        categorical_features=[len(NUM) + i for i in range(3)], random_state=0))

def topk(y, s, f):
    k = int(round(f * len(y))); i = np.argsort(-s, kind="stable")[:k]
    return y[i].mean(), y[i].sum() / y.sum()
def metrics(y, s):
    p5, r5 = topk(y, s, .05); p10, r10 = topk(y, s, .10)
    return dict(base=y.mean(), prauc=average_precision_score(y, s), roc=roc_auc_score(y, s), p5=p5, r5=r5, p10=p10, r10=r10)

def scores(X, tr, ev, gh):
    """return dict method -> score array for ev rows, models fit on tr rows only"""
    y = d.miss.values
    s = {"A prior miss rate": X.s_miss.values[ev], "B time allowed (short=risky)": -X.allowed.values[ev],
         "C hand days - allowed": (X.s_hand - X.allowed).values[ev]}
    s["LogReg"] = lr().fit(X[tr], y[tr]).predict_proba(X[ev])[:, 1]
    for dep in (3, 6):
        s[f"HGB d{dep}"] = hgb(dep).fit(X[tr], y[tr]).predict_proba(X[ev])[:, 1]
    return s

wins = [("2018-01-01", "2018-02-01"), ("2018-02-01", "2018-03-01"), ("2018-03-01", "2018-04-01"), ("2018-04-01", "2018-05-01"), ("2018-05-01", "2018-05-26")]
val = {}
for a, b in wins:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    tr = ((A < a) & (H < a)).values; ev = ((A >= a) & (A < b)).values
    X, gh = feats(tr)
    for k, sc in scores(X, tr, ev, gh).items():
        val.setdefault(k, []).append(metrics(d.miss.values[ev], sc))
    print("fold", a.date(), tr.sum(), ev.sum(), flush=True)
V = {k: pd.DataFrame(v).mean() for k, v in val.items()}
hg = [k for k in V if k.startswith("HGB")]; best_hgb = max(hg, key=lambda k: V[k].prauc)
rules = [k for k in V if k[0] in "ABC"]; best_rule = max(rules, key=lambda k: V[k].prauc)
print("best HGB", best_hgb, "best rule", best_rule)

tr = ((A < BOUND) & (H < BOUND)).values; te = (A >= BOUND).values
X, gh = feats(tr); st = scores(X, tr, te, gh); yt = d.miss.values[te]
T = {k: pd.Series(metrics(yt, s)) for k, s in st.items() if k in V and (k not in hg or k == best_hgb)}
best_model = max(["LogReg", best_hgb], key=lambda k: V[k].prauc)
rows = []
for k in T:
    if k in V: rows.append(dict(method=k, **{f"val_{c}": V[k][c] for c in ["base", "prauc", "roc", "p5", "r5", "p10", "r10"]}, **{f"test_{c}": T[k][c] for c in ["base", "prauc", "roc", "p5", "r5", "p10", "r10"]}))
tab = pd.DataFrame(rows).set_index("method")

# monthly test PR-AUC
dt = d[te].copy(); mo = dt.order_approved_at.dt.to_period("M")
mt = pd.DataFrame({m: {k: (average_precision_score(yt[(mo == m).values], st[k][(mo == m).values]) if yt[(mo == m).values].sum() else np.nan) for k in [best_rule, best_model]}
    | {"n": (mo == m).sum(), "base": yt[(mo == m).values].mean()} for m in sorted(mo.unique())}).T
# lateness link
k = int(round(.10 * len(yt))); fl = np.argsort(-st[best_model], kind="stable")[:k]
dk = dt.reset_index(drop=True); lt = dk.late.notna()
late_all = dk.late[lt].mean(); late_flag = dk.late.iloc[fl].dropna().mean()
late_miss = dk.late[lt & (dk.miss == 1)].mean(); late_nomiss = dk.late[lt & (dk.miss == 0)].mean()
# permutation importance on test, best model refit
mdl = (lr() if best_model == "LogReg" else hgb(int(best_model[-1]))).fit(X[tr], d.miss.values[tr])
pi = permutation_importance(mdl, X[te], yt, scoring="average_precision", n_repeats=5, random_state=0)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(8)

f = lambda x: f"{x:.3f}"
md = [f"# Seller shipping-limit miss prediction\n",
 f"Population {len(d)} orders; train n={tr.sum()}, test n={te.sum()}. Assumption: shipping_limit_date (max over items) is known at approval. Seller = first item's seller. Backlog counts only orders in the population (cancelled/never-shipped orders are not visible).\n",
 f"Best rule (val PR-AUC): {best_rule}; best model: {best_model}\n",
 "| method | val base | val PR | val ROC | val P@5 | val R@5 | val P@10 | val R@10 | test base | test PR | test ROC | test P@5 | test R@5 | test P@10 | test R@10 |", "|" + "---|" * 15]
for m, r in tab.iterrows():
    md.append(f"| {m} | " + " | ".join(f(r[c]) for c in tab.columns) + " |")
md += ["\n## Test PR-AUC by month\n", "```\n"+mt.round(3).to_string()+"\n```", "\n## Lateness link (top-10% flagged by best model, test)\n",
 f"overall late rate {late_all:.3f}; flagged top-10% late rate {late_flag:.3f}; late rate if seller missed {late_miss:.3f} vs not missed {late_nomiss:.3f}",
 "\n## Permutation importance (test, drop in PR-AUC)\n", "```\n"+imp.round(4).to_string()+"\n```"]
open(f"{OUT}/results.md", "w").write("\n".join(md)); print("\n".join(md))
