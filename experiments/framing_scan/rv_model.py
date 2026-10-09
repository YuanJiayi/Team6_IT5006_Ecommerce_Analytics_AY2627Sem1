import numpy as np, pandas as pd, warnings, sys, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
warnings.filterwarnings("ignore")
d = pd.read_pickle("rv_base.pkl")
CATS = ["customer_state", "seller_state", "pay_type", "cat"]
for c in CATS:
    d[c + "_c"] = d[c].astype("category").cat.codes.replace(-1, np.nan) if False else d[c].astype("category").cat.codes
NUM_A = ["n_items","n_sellers","n_products","n_cats","price","freight","max_price","freight_share","photos_min","photos_mean","desc_mean","desc_min","name_mean",
 "weight","vol","pay_value","installments","n_pay","distance_km","same_state","approve_h","promise_days","hour","dow","month",
 "sel_n_max","sel_n_min","sel_bad_max","sel_bad_min","sel_late_max","sel_late_min","sel_nd_max","prod_n_max","prod_bad_max","prod_bad_min","cat_bad_max","cat_bad_min"]
NUM_B = ["days_late","late","handover_d","carrier_leg_d","ship_slack_d"]
CAT_C = [c + "_c" for c in CATS]
BOUND = pd.Timestamp("2018-05-26")
WINS = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in [("2018-01-01","2018-02-01"),("2018-02-01","2018-03-01"),("2018-03-01","2018-04-01"),("2018-04-01","2018-05-01"),("2018-05-01","2018-05-26")]]

def make(kind, cfg, feats):
    num = [f for f in feats if f not in CAT_C]; cat = [f for f in feats if f in CAT_C]
    if kind == "logit":
        ct = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), QuantileTransformer(n_quantiles=200, output_distribution="normal")), num),
                                ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat)])
        return make_pipeline(ct, LogisticRegression(C=cfg["C"], max_iter=500))
    return HistGradientBoostingClassifier(random_state=0, categorical_features=[feats.index(c) for c in cat], **cfg)

CFGS = {"logit C=0.03": ("logit", {"C": .03}), "logit C=0.3": ("logit", {"C": .3}),
 "hgb d3 lr.05 it150": ("hgb", dict(max_depth=3, learning_rate=.05, max_iter=150, min_samples_leaf=100, l2_regularization=1)),
 "hgb leaf15 lr.05 it200": ("hgb", dict(max_leaf_nodes=15, learning_rate=.05, max_iter=200, min_samples_leaf=200, l2_regularization=5)),
 "hgb leaf31 lr.03 it300": ("hgb", dict(max_leaf_nodes=31, learning_rate=.03, max_iter=300, min_samples_leaf=50, l2_regularization=1))}

rng = np.random.RandomState(0)
def tiebreak(s, aux=None): return s + (aux if aux is not None else 0) * 1e-6 + rng.rand(len(s)) * 1e-9
RULES = {"A": {"seller prior bad rate": lambda x: x.sel_bad_max.fillna(.13).values,
               "n_sellers>=2 flag": lambda x: (x.n_sellers >= 2).astype(float).values,
               "combined (multi-seller flag, then seller rate)": lambda x: (x.n_sellers >= 2).astype(float).values * 10 + x.sel_bad_max.fillna(.13).values},
         "B": {"days late": lambda x: x.days_late.values,
               "late flag (ties by days late)": lambda x: x.late.values + x.days_late.clip(-60, 60).values / 1000,
               "late + multi-seller": lambda x: 2 * x.late.values + (x.n_sellers >= 2).values + x.days_late.clip(-60, 60).values / 1000}}

def metrics(y, s):
    n = len(y); o = np.argsort(-s, kind="stable"); r = {"n": n, "base": y.mean(), "PR": AP(y, s), "ROC": AUC(y, s)}
    for q in (.05, .10):
        k = int(round(n * q)); h = y[o[:k]].sum(); r[f"p@{int(q*100)}"] = h / k; r[f"r@{int(q*100)}"] = h / y.sum()
    return r

def run(point):
    feats = NUM_A + CAT_C + (NUM_B if point == "B" else [])
    pop = d[d.score.notna() & (d.delivered if point == "B" else True)].reset_index(drop=True)
    y = pop.bad.values.astype(int); X = pop[feats]
    val = {}
    for name, (kind, cfg) in CFGS.items():
        rs = []
        for a, b in WINS:
            tr = ((pop.order_purchase_timestamp < a) & (pop.rv_time < a)).values; te = ((pop.order_purchase_timestamp >= a) & (pop.order_purchase_timestamp < b)).values
            m = make(kind, cfg, feats).fit(X[tr], y[tr]); rs.append(metrics(y[te], m.predict_proba(X[te])[:, 1]))
        val[name] = pd.DataFrame(rs)
    for name, f in RULES[point].items():
        rs = []
        for a, b in WINS:
            te = ((pop.order_purchase_timestamp >= a) & (pop.order_purchase_timestamp < b)).values; rs.append(metrics(y[te], tiebreak(f(pop[te]))))
        val[name] = pd.DataFrame(rs)
    V = pd.DataFrame({k: v.mean() for k, v in val.items()}).T.drop(columns="n")
    Vsd = pd.Series({k: v.PR.std() for k, v in val.items()})
    best_m = V.loc[list(CFGS)].PR.idxmax(); best_r = V.loc[list(RULES[point])].PR.idxmax()
    print(f"\n===== POINT {point}: n={len(pop)} base={y.mean():.3f}\nVALIDATION (mean of 5 windows)\n", V.round(3).to_string())
    print("PR-AUC std across windows:", Vsd.round(3).to_dict(), "\nchosen model:", best_m, " chosen rule:", best_r)
    # test, once
    tr = ((pop.order_purchase_timestamp < BOUND) & (pop.rv_time < BOUND)).values; te = (pop.order_purchase_timestamp >= BOUND).values
    kind, cfg = CFGS[best_m]; m = make(kind, cfg, feats).fit(X[tr], y[tr])
    sm = m.predict_proba(X[te])[:, 1]; T = pop[te].reset_index(drop=True); yt = y[te]
    scores = {"MODEL " + best_m: sm}
    for name, f in RULES[point].items(): scores["rule: " + name] = tiebreak(f(T))
    # also the other family's best for honesty
    other = max([k for k in CFGS if CFGS[k][0] != kind], key=lambda k: V.PR[k]); ko, co = CFGS[other]
    mo = make(ko, co, feats).fit(X[tr], y[tr]); scores["MODEL(other family) " + other] = mo.predict_proba(X[te])[:, 1]
    print(f"\nTEST (purchases >= {BOUND.date()}, once): train n={tr.sum()} test n={te.sum()}")
    print(pd.DataFrame({k: metrics(yt, s) for k, s in scores.items()}).T.drop(columns="n").round(3).to_string())
    mk = "MODEL " + best_m; rk = "rule: " + best_r
    mon = T.order_purchase_timestamp.dt.to_period("M").values
    rows = []
    for p in sorted(set(mon)):
        i = mon == p
        if i.sum() < 500: continue
        a_, b_ = metrics(yt[i], scores[mk][i]), metrics(yt[i], scores[rk][i]); rows.append((str(p), i.sum(), a_["base"], a_["PR"], b_["PR"], a_["r@10"], b_["r@10"], a_["p@10"], b_["p@10"]))
    print("\nTEST by month (model vs best rule):\n", pd.DataFrame(rows, columns=["month","n","base","PR model","PR rule","r@10 model","r@10 rule","p@10 model","p@10 rule"]).round(3).to_string(index=False))
    # bootstrap diff recall@10 and PR
    bs = []; b2 = np.random.RandomState(1)
    k = int(round(len(yt) * .1))
    for _ in range(200):
        j = b2.randint(0, len(yt), len(yt)); yy = yt[j]; ks = int(round(len(yy) * .1))
        rm = yy[np.argsort(-scores[mk][j], kind="stable")[:ks]].sum() / yy.sum(); rr = yy[np.argsort(-scores[rk][j], kind="stable")[:ks]].sum() / yy.sum()
        bs.append((rm - rr, AP(yy, scores[mk][j]) - AP(yy, scores[rk][j])))
    bs = np.array(bs); print("bootstrap model-rule diff: recall@10 %.3f [%.3f, %.3f]; PR-AUC %.3f [%.3f, %.3f]" % (bs[:, 0].mean(), *np.percentile(bs[:, 0], [2.5, 97.5]), bs[:, 1].mean(), *np.percentile(bs[:, 1], [2.5, 97.5])))
    # permutation importance on test (PR-AUC drop), best model
    base = AP(yt, sm); imp = {}; pr = np.random.RandomState(2)
    for f in feats:
        ds = []
        for _ in range(3):
            Xp = T[feats].copy(); Xp[f] = pr.permutation(Xp[f].values); ds.append(base - AP(yt, m.predict_proba(Xp)[:, 1]))
        imp[f] = np.mean(ds)
    print("\nPermutation importance (test PR-AUC drop), top 10:\n", pd.Series(imp).sort_values(ascending=False).head(10).round(4).to_string())
run(sys.argv[1])
