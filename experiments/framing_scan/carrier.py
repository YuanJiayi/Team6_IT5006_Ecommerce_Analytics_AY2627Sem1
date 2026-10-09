"""Task 2: carrier delay alert at handover, full as-of feature set, prevalence-robust options."""
import sys; sys.path.insert(0, "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad")
from common import *
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score as APS, roc_auc_score as AUC
OUT = []
def P(s=""): print(s, flush=True); OUT.append(str(s))
full, o = load_base()
d = full[(full.order_status == "delivered") & full.order_delivered_customer_date.notna() & full.order_delivered_carrier_date.notna()].copy()
d["tq"] = d.order_purchase_timestamp; d["th"] = d.order_delivered_carrier_date; d["deliv"] = d.order_delivered_customer_date
d["pday"] = d.tq.dt.normalize()
d["need"] = (d.deliv.dt.normalize() - d.pday).dt.days; d["promise"] = (d.order_estimated_delivery_date.dt.normalize() - d.pday).dt.days
d["late"] = (d.need > d.promise).astype(int)
d["days_late"] = (d.deliv.dt.normalize() - d.order_estimated_delivery_date.dt.normalize()).dt.days.astype(float)
d["handover"] = (d.th - d.tq) / DAY; d["leg"] = (d.deliv - d.th) / DAY
d["h_dow"] = d.th.dt.dayofweek; d["h_hour"] = d.th.dt.hour
d["rem"] = d.promise - d.handover
d["over_limit"] = (d.th - d.limit) / DAY            # known at handover
d["approve_to_hand"] = (d.th - d.order_approved_at) / DAY
d["approval_delay"] = (d.order_approved_at - d.tq) / DAY
d = d.sort_values("th").reset_index(drop=True)
th, de = d.th, d.deliv; tv = th.values; rt = d.route
P(f"n delivered+handed orders {len(d)}; late rate {d.late.mean():.4f}")
n9, d["rl90"], q50, q90 = asof(rt, de, d.leg, rt, th, 90, (.5, .9)); d["spread"] = q90 - q50; d["rl_q90"] = q90
d["rl30"] = asof(rt, de, d.leg, rt, th, 30)[1]; d["rl_all"] = asof(rt, de, d.leg, rt, th)[1]
d["cl90"] = asof(d.customer_state, de, d.leg, d.customer_state, th, 90)[1]
gl = asof(np.zeros(len(d)), de, d.leg, np.zeros(len(d)), th)[1]; glate = asof(np.zeros(len(d)), de, d.late, np.zeros(len(d)), th)[1]
gl = np.where(np.isnan(gl), d.leg.mean(), gl); glate = np.where(np.isnan(glate), d.late.mean(), glate)
d["leg_base"] = d.rl90.fillna(d.rl_all).fillna(d.cl90).fillna(pd.Series(gl, index=d.index))
d["trend"] = (d.rl30 - d.rl90).fillna(0); d["spread"] = d.spread.fillna(d.spread.median())
allh = full[full.order_delivered_carrier_date.notna()]
d["vol7"] = asof(allh.route, allh.order_delivered_carrier_date, np.ones(len(allh)), rt, th, 7)[0]
def chain(target, parent):
    p = parent
    for key in ["customer_state", "z3", "z5"]:
        n, m = asof(d[key], de, d[target], d[key], th)[:2]; p = shrink(n, m, p)
        if key == "z3": d["_p3_" + target] = p.copy()
    return p
d["leg_z5"] = chain("leg", d.leg_base.values); d["leg_z3"] = d["_p3_leg"]
rl_n, rl_m = asof(rt, de, d.late, rt, th)[:2]; d["late_route"] = shrink(rl_n, rl_m, glate); d["late_z5"] = chain("late", d.late_route.values)
d["n_z5"] = asof(d.z5, de, d.leg, d.z5, th)[0]
d["z3_late"] = d._p3_late if "_p3_late" in d else np.nan
# seller history (as of handover)
sid = d.seller_id
d["s_ord"] = asof(sid, th, np.ones(len(d)), sid, th)[0]
d["miss"] = (d.th > d.limit).astype(float)
n_, m_ = asof(sid, th, d.miss, sid, th); d["s_miss"] = shrink(n_, m_, np.nanmean(d.miss), 5)
n_, m_ = asof(sid, th, d.handover, sid, th); d["s_hand"] = shrink(n_, m_, d.handover.mean(), 5)
n_, m_ = asof(sid, de, d.leg - d.leg_z5, sid, th); d["s_leg_resid"] = shrink(n_, m_, 0.0, 10)   # seller's past leg minus route baseline
n_, m_ = asof(sid, de, d.late, sid, th); d["s_late"] = shrink(n_, m_, glate, 10)
d["s_age"] = (th - d.groupby("seller_id").th.transform("min")) / DAY
d["backlog"] = seller_backlog(d.assign(order_approved_at=d.order_approved_at), "th", full)
# product / category
n_, m_ = asof(d.cat, de, d.leg - d.leg_z5, d.cat, th); d["cat_leg_resid"] = shrink(n_, m_, 0.0, 10)
n_, m_ = asof(d.cat, de, d.late, d.cat, th); d["cat_late"] = shrink(n_, m_, glate, 10)
# slack / relative features (days remaining)
d["slack_rule"] = d.rem - d.leg_base; d["slack_leg_z5"] = d.rem - d.leg_z5; d["slack_leg_z3"] = d.rem - d.leg_z3
d["slack_q"] = d.rem - d.rl_q90.fillna(d.leg_base + d.spread); d["slack_sel"] = d.rem - d.leg_z5 - d.s_leg_resid - d.cat_leg_resid
d["ratio_z5"] = d.leg_z5 / d.rem.clip(lower=0.5); d["ratio_base"] = d.leg_base / d.rem.clip(lower=0.5); d["sp_rel"] = d.spread / d.rem.clip(lower=0.5)
d["hand_frac"] = d.handover / d.promise.clip(lower=1)
topc = d[d.th < BOUND].cat.value_counts().index[:30]; d["catg"] = d.cat.where(d.cat.isin(topc), "other")
for c in ["seller_state", "customer_state", "pay_type", "catg"]: d[c] = d[c].fillna("na").astype(str)
BASEN = ["dist", "leg_z5", "leg_z3", "leg_base", "late_z5", "late_route", "rl30", "rl_all", "rl_q90", "spread", "trend", "vol7", "n_z5", "rem", "weight", "freight", "price",
         "h_dow", "handover", "promise", "same_state"]
EXTRA = ["h_hour", "pdow", "phour", "pdom", "over_limit", "approve_to_hand", "approval_delay", "s_ord", "s_miss", "s_hand", "s_leg_resid", "s_late", "s_age", "backlog",
         "cat_leg_resid", "cat_late", "vol", "photos", "namelen", "desclen", "max_price", "n_items", "n_products", "n_sellers", "n_pay", "installments", "voucher_share", "pay_value"]
RELF = ["slack_rule", "slack_leg_z5", "slack_leg_z3", "slack_q", "slack_sel", "ratio_z5", "ratio_base", "sp_rel", "hand_frac"]
CATS = ["seller_state", "customer_state", "pay_type", "catg"]
FULL = BASEN + EXTRA; FULLREL = FULL + RELF
RELONLY = ["rem"] + RELF + ["spread", "trend", "vol7", "dist", "weight", "freight", "h_dow", "s_miss", "backlog", "over_limit"]
P(f"features: full {len(FULL)} numeric + {len(CATS)} categorical; full+rel {len(FULLREL)}")

def mk(kind, nums, cats, **kw):
    if kind in ("ridge", "logit"):
        pre = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums), ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=30), cats)])
        est = Ridge(alpha=30) if kind == "ridge" else LogisticRegression(C=0.1, max_iter=3000)
    else:
        pre = ColumnTransformer([("n", "passthrough", nums), ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats)])
        ci = [len(nums) + i for i in range(len(cats))]
        common = dict(max_iter=150, learning_rate=0.06, max_depth=4, random_state=0, categorical_features=ci, min_samples_leaf=50, l2_regularization=1.0)
        if kind == "gbm_clf": est = HGC(**{**common, "max_depth": 3, "learning_rate": 0.05})
        elif kind == "gbm_sq": est = HGR(**common)
        else: est = HGR(loss="quantile", quantile=kw["q"], **common)
    return make_pipeline(pre, est)

# candidates: name -> (target, kind, nums, scope(days or None), recency half-life or None, family, extra kw)
CAND = {}
for scope in (None, 180, 90):
    sn = "all" if scope is None else str(scope)
    for kind, kw in [("ridge", {}), ("gbm_sq", {}), ("gbm_q", {"q": .8}), ("gbm_q", {"q": .9})]:
        nm = f"(a) leg {kind}{kw.get('q', '')} [{sn}]"
        CAND[nm] = ("leg", kind, FULL, scope, None, "a", kw)
    for kind in ("ridge", "gbm_sq"):
        CAND[f"(c) days_late {kind} [{sn}]"] = ("days_late", kind, FULLREL, scope, None, "c", {})
    for fsn, fs in [("full", FULL), ("full+rel", FULLREL), ("rel-only", RELONLY)]:
        for kind in ("logit", "gbm_clf"):
            CAND[f"(b) clf {kind} {fsn} [{sn}]"] = ("late", kind, fs, scope, None, "b", {})
for fsn, fs in [("full", FULL), ("full+rel", FULLREL)]:
    for kind in ("logit", "gbm_clf"):
        CAND[f"(b) clf {kind} {fsn} [all,w60]"] = ("late", kind, fs, None, 60, "b", {})
CAND["(c) days_late gbm_sq [all,w60]"] = ("days_late", "gbm_sq", FULLREL, None, 60, "c", {})
CAND["(a) leg gbm_q0.8 [all,w60]"] = ("leg", "gbm_q", FULL, None, 60, "a", {"q": .8})
CAND["(a) leg gbm_q0.9 [all,w60]"] = ("leg", "gbm_q", FULL, None, 60, "a", {"q": .9})

def score(nm, W):
    tgt, kind, nums, scope, hl, fam, kw = CAND[nm]
    m = (d.deliv < W)
    if scope: m &= (d.th >= W - scope * DAY)
    tr = np.where(m)[0]
    y = d[tgt].values[tr].copy()
    if tgt == "days_late": y = np.clip(y, -20, 30)
    mdl = mk(kind, nums, CATS, **kw); fk = {}
    if hl:
        w = np.exp(-np.log(2) * ((W - d.th.iloc[tr]) / DAY).values / hl); fk = {mdl.steps[-1][0] + "__sample_weight": w}
    mdl.fit(d.iloc[tr][nums + CATS], y, **fk)
    return mdl
def predict(mdl, kind, tgt, ev):
    X = d.iloc[ev][mdl.steps[0][1].transformers[0][2] + CATS]
    if tgt == "late": return mdl.predict_proba(X)[:, 1]
    p = mdl.predict(X)
    return -(d.rem.values[ev] - p) if tgt == "leg" else p      # rank by risk (higher = worse)

TST = [(T("2018-05-26"), T("2018-06-01")), (T("2018-06-01"), T("2018-07-01")), (T("2018-07-01"), T("2018-08-01")), (T("2018-08-01"), T("2018-09-01"))]
thv = d.th.values; late = d.late.values
def win_idx(W, E): return np.where((thv >= W.to_datetime64()) & (thv < E.to_datetime64()))[0]
def topk(y, s, f):
    k = int(np.ceil(f * len(y))); i = np.argsort(-s, kind="stable")[:k]; return y[i].sum()
# ---- rules
RULE = {}
for base in ["leg_base", "leg_z3", "leg_z5"]:
    for c in [0, .5, 1]: RULE[f"rule slack {base}+{c}*spread"] = lambda ev, b=base, c=c: -(d.rem.values[ev] - d[b].values[ev] - c * d.spread.values[ev])
RULE["rule slack leg_z5+seller/cat resid"] = lambda ev: -d.slack_sel.values[ev]
res = {}; N = len(CAND)
for vi, (W, E) in enumerate(VAL):
    ev = win_idx(W, E)
    for nm, f in RULE.items(): res.setdefault(nm, {})[vi] = f(ev)
    for j, nm in enumerate(CAND):
        mdl = score(nm, W); res.setdefault(nm, {})[vi] = predict(mdl, CAND[nm][1], CAND[nm][0], ev)
    print("val window done", vi, flush=True)
def vmet(nm):
    pa, ro, r5, r10 = [], [], [], []
    for vi, (W, E) in enumerate(VAL):
        y = late[win_idx(W, E)]; s = res[nm][vi]; pa.append(APS(y, s)); ro.append(AUC(y, s)); r5.append(topk(y, s, .05) / y.sum()); r10.append(topk(y, s, .10) / y.sum())
    return dict(pr=np.mean(pa), roc=np.mean(ro), r5=np.mean(r5), r10=np.mean(r10), r10_min=np.min(r10))
V = pd.DataFrame({nm: vmet(nm) for nm in res}).T
P(f"\n## Validation (mean over 5 monthly windows; late rate {np.mean([late[win_idx(W,E)].mean() for W,E in VAL]):.3f}), all candidates, sorted by PR-AUC\n```\n" + V.sort_values('pr', ascending=False).round(3).to_string() + "\n```")
best_rule = max(RULE, key=lambda k: V.pr[k]); base_rule = "rule slack leg_base+0*spread"
sel = {"state-route slack rule (old)": base_rule, "best rule": best_rule}
for fam, lab in [("a", "(a) leg regression -> slack"), ("b", "(b) direct classifier"), ("c", "(c) days-late regression")]:
    sel[lab] = max([n for n in CAND if CAND[n][5] == fam], key=lambda n: V.pr[n])
sel["overall best model"] = max(CAND, key=lambda n: V.pr[n])
P("Selected on validation mean PR-AUC: " + str(sel))
# ---- test once for selected
tres = {nm: {} for nm in set(sel.values())}
for ti, (W, E) in enumerate(TST):
    ev = win_idx(W, E)
    for nm in tres:
        if nm in RULE: tres[nm][ti] = RULE[nm](ev)
        else: mdl = score(nm, W); tres[nm][ti] = predict(mdl, CAND[nm][1], CAND[nm][0], ev); tres[nm][("mdl", ti)] = mdl
ys = [late[win_idx(W, E)] for W, E in TST]
def agg(nm):
    y = np.concatenate(ys); s = np.concatenate([tres[nm][i] for i in range(4)])
    h5 = sum(topk(ys[i], tres[nm][i], .05) for i in range(4)); h10 = sum(topk(ys[i], tres[nm][i], .10) for i in range(4)); n = len(y)
    return dict(base=y.mean(), pr=APS(y, s), roc=AUC(y, s), p5=h5 / (.05 * n), r5=h5 / y.sum(), p10=h10 / (.10 * n), r10=h10 / y.sum())
P(f"\n## TEST (2018-05-26 on, pooled; top-k within each month)\n```\n" + pd.DataFrame({lab: agg(nm) for lab, nm in sel.items()}).T.round(3).to_string() + "\n```")
vv = pd.DataFrame({lab: {"val PR": V.pr[nm], "val R@5": V.r5[nm], "val R@10": V.r10[nm], "val min-window R@10": V.r10_min[nm]} for lab, nm in sel.items()}).T
P("\n## Validation for the selected\n```\n" + vv.round(3).to_string() + "\n```")
tab = {}
for lab, nm in sel.items():
    tab[lab] = [f"PR {APS(ys[i], tres[nm][i]):.3f} | R@10 {topk(ys[i], tres[nm][i], .1) / ys[i].sum():.1%}" for i in range(4)]
P("\n## TEST per month (n, late rate: " + ", ".join(f"{len(y)}/{y.mean():.3f}" for y in ys) + ")\n```\n" + pd.DataFrame(tab, index=["May26-31", "Jun", "Jul", "Aug"]).T.to_string() + "\n```")
P("\n## VALIDATION per window top-10% recall (selected)\n```\n" + pd.DataFrame({lab: [f"{topk(late[win_idx(*VAL[i])], res[nm][i], .1) / late[win_idx(*VAL[i])].sum():.1%}" for i in range(5)] for lab, nm in sel.items()},
    index=["Jan", "Feb", "Mar", "Apr", "May1-25"]).T.to_string() + "\n```")
# permutation importance (overall best, test month Jun-Aug pooled using the model fitted per month: use Jun model on Jun-Aug rows)
nm = sel["overall best model"]
if nm not in RULE:
    mdl = tres[nm][("mdl", 1)]; ev = np.concatenate([win_idx(*TST[i]) for i in (1, 2, 3)]); nums = mdl.steps[0][1].transformers[0][2]
    Xe = d.iloc[ev][nums + CATS].reset_index(drop=True); yv = late[ev]; r_ = np.random.RandomState(0)
    def sc_(X):
        if CAND[nm][0] == "late": return mdl.predict_proba(X)[:, 1]
        p = mdl.predict(X); return p - d.rem.values[ev] * 0 if CAND[nm][0] == "days_late" else p - d.rem.values[ev]
    base = APS(yv, sc_(Xe)); imp = {}
    for c in nums + CATS:
        dr = []
        for _ in range(3):
            X2 = Xe.copy(); X2[c] = r_.permutation(X2[c].values); dr.append(base - APS(yv, sc_(X2)))
        imp[c] = np.mean(dr)
    P(f"\n## Permutation importance of {nm} (Jun-Aug test, month-fitted-Jun model; drop in PR-AUC; base {base:.3f}), top 10\n```\n" + pd.Series(imp).sort_values(ascending=False).head(10).round(4).to_string() + "\n```")
open(f"{SP}/results_carrier.md", "w").write("# Task 2: carrier delay alert\n\n" + "\n".join(OUT))
