"""Tiered (Mondrian) conformal buffer vs single buffer, Pratik forest no-promise mu, scale 'mu'."""
import sys
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/d71361ad-8b17-4fcd-ac4f-b033089abeed/scratchpad/"
R = SP + "pratik"; sys.path.insert(0, R); sys.path.insert(0, R + "/experiments")
src = open(SP + "geo_run.py").read().split('if __name__ == "__main__":')[0]
g = {"__name__": "geo"}; exec(src, g)
import numpy as np, pandas as pd, multiprocessing as mp, json, copy
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.metrics import roc_auc_score, average_precision_score
from phase2_regression import make_regression_pipeline
from phase2_classification import feature_columns
d, X, WINS, widx, wrng, promises, front, at, need, tq, deliv, DAY, mu_r1, pday, TG, CAL, CATS, FALL = (g[k] for k in
    "d X WINS widx wrng promises front at need tq deliv DAY mu_r1 pday TG CAL CATS FALL".split())
spec = json.load(open(R + "/data/phase2_feature_spec.json"))
pt = pd.read_csv(R + "/data/phase2_order_table.csv").set_index("order_id")
spec_np = copy.deepcopy(spec)
spec_np["numeric_features"] = [f for f in spec["numeric_features"] if f not in ("promised_days", "promise_slack")]
cols = sum(feature_columns(spec), [])
common = d.order_id.isin(pt.index).values
PX = pt.reindex(d.order_id.values)[cols].reset_index(drop=True)
cols_np = sum(feature_columns(spec_np), [])
class Pratik:
    def fit(s, tr):
        tr = tr[common[tr]]
        s.m = make_regression_pipeline("forest", {"max_depth": None, "min_samples_leaf": 20}, spec_np).fit(PX.iloc[tr][cols_np], d.dur.values[tr]); return s
    def mu(s, Xf): return s.m.predict(PX.loc[Xf.index][cols_np])
spread = d.rt_spread.fillna(d.rt_spread.median()).values

IP = int(np.where(np.isclose(TG, .95))[0][0])
def rec_prec(score, late, k=.10):
    n = max(1, int(round(len(score) * k))); top = np.argsort(-score, kind="stable")[:n]
    return late[top].sum() / late.sum(), late[top].mean()
def do_window(wi):
    split, i, (W, E) = WINS[wi]; key = (split, i)
    rng = wrng[key]; rng = rng[common[rng]]; win = widx[key]; win = win[common[win]]
    tr = np.where(deliv < W.to_datetime64())[0]; tr = tr[np.argsort(tq[tr])]
    folds = np.array_split(np.arange(len(tr)), 3)
    allr = np.union1d(rng, win)
    mu = np.zeros(len(d)); mu[allr] = Pratik().fit(tr).mu(X.iloc[allr])
    trc = tr[common[tr]]; fo = {}
    for k, f in enumerate(folds):
        tgt = tr[f][common[tr[f]]]; fo[k] = tgt
        mu[tgt] = Pratik().fit(tr[np.setdiff1d(np.arange(len(tr)), f)]).mu(X.iloc[tgt])
    sc = np.clip(mu, 3, None); r = (need - mu) / sc
    P = promises(mu, "mu", rng, win)[:, IP]
    assert not np.isnan(P).any(), f"NaN promise in {key}"
    late = (need[win] > P).astype(int)
    ref = trc[tq[trc] >= (W - 90 * DAY).to_datetime64()]
    assert len(ref) > 500, (key, len(ref))
    thr = np.quantile(r[ref], .95); y = (r > thr).astype(int)
    mk = lambda: HGC(max_iter=120, learning_rate=0.08, random_state=0, categorical_features=[FALL.index(c) for c in CATS])
    XF = X[FALL].values
    prob = mk().fit(XF[trc], y[trc]).predict_proba(XF[win])[:, 1]
    rs = np.random.RandomState(0)
    nsel = d.n_sellers.values[win].astype(float)
    scores = {"classifier": prob, "mu": mu[win], "rt_spread": spread[win],
              "n_sellers,mu": nsel * 1e6 + mu[win], "random": rs.rand(len(win))}
    assert late.sum() > 0 and late.sum() < len(late)
    rows = []
    for m, sco in scores.items():
        rc, pr = rec_prec(sco, late); ap = average_precision_score(late, sco)
        rows.append(dict(split=split, win=i, method=m, n=len(win), late_rate=late.mean(), ref_rate=y[ref].mean(), auc=roc_auc_score(late, sco),
                         pr=ap, pr_rel=ap / late.mean(), rec10=rc, prec10=pr))
    print("window", key, flush=True)
    return rows
if __name__ == "__main__":
    with mp.get_context("fork").Pool(8) as pool: rs = pool.map(do_window, range(len(WINS)), chunksize=1)
    r = pd.DataFrame([x for rr in rs for x in rr]); r.to_csv(SP + "cmp_checkout_cls.csv", index=False)
    order = ["classifier", "mu", "rt_spread", "n_sellers,mu", "random"]
    print("LATE RATE per window"); print(r[r.method == "classifier"][["split", "win", "n", "late_rate", "ref_rate"]].round(4).to_string(index=False))
    for c in ["auc", "pr", "pr_rel", "rec10", "prec10"]:
        print(c); print(r.groupby(["method", "split"])[c].mean().unstack()[["val", "test"]].loc[order].round(4))
    print("val per-window rec10"); print(r.pivot_table(index="method", columns=["split", "win"], values="rec10").loc[order].round(3))
