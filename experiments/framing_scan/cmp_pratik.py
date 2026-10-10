"""Checkout promise at 95% on-time: our two-leg GBM vs Pratik's checkout ridge (c66c7be), same daily buffer, same orders.
Reuses geo_run.py's windows, cross-fitting and promises(); only the expected-time model differs."""
import sys, types
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/d71361ad-8b17-4fcd-ac4f-b033089abeed/scratchpad/"
R = SP + "pratik"; sys.path.insert(0, R); sys.path.insert(0, R + "/experiments")
src = open(SP + "geo_run.py").read().split('if __name__ == "__main__":')[0]
g = {"__name__": "geo"}; exec(src, g)
import numpy as np, pandas as pd, multiprocessing as mp
from phase2_regression import make_regression_pipeline
from phase2_classification import feature_columns
import json
d, X, WINS, widx, wrng, promises, front, at, need, tq, deliv, DAY, mu_r1 = (g[k] for k in
    "d X WINS widx wrng promises front at need tq deliv DAY mu_r1".split())
spec = json.load(open(R + "/data/phase2_feature_spec.json"))
pt = pd.read_csv(R + "/data/phase2_order_table.csv").set_index("order_id")
cols = sum(feature_columns(spec), [])
common = d.order_id.isin(pt.index).values
print("our rows", len(d), "with Pratik features", common.sum(), flush=True)
PX = pt.reindex(d.order_id.values)[cols].reset_index(drop=True)
class Pratik:   # frozen checkout selection: ridge alpha=100 on his 20 inputs, target delivery days
    def __init__(s, variant): s.variant = variant
    def fit(s, tr):
        tr = tr[common[tr]]
        s.m = make_regression_pipeline(*s.variant, spec).fit(PX.iloc[tr], d.dur.values[tr]); return s
    def mu(s, Xf, base): return s.m.predict(PX.loc[Xf.index])
MODELS = {"ours_two_leg": g["M2"], "pratik_ridge100": lambda: Pratik(("ridge", {"alpha": 100.0})),
          "pratik_forest_leaf20": lambda: Pratik(("forest", {"max_depth": None, "min_samples_leaf": 20}))}
def do_window(wi):
    split, i, (W, E) = WINS[wi]; key = (split, i)
    rng = wrng[key]; rng = rng[common[rng]]; win = widx[key]; win = win[common[win]]; out = {}
    tr = np.where(deliv < W.to_datetime64())[0]; tr = tr[np.argsort(tq[tr])]
    for name, mk in MODELS.items():
        mu = np.zeros(len(d)); mu[rng] = mk().fit(tr).mu(X.iloc[rng], mu_r1[rng])
        for f in np.array_split(np.arange(len(tr)), 3):   # out-of-fold predictions for calibration rows
            tgt = tr[f][np.isin(tr[f], rng)]
            if len(tgt): mu[tgt] = mk().fit(tr[np.setdiff1d(np.arange(len(tr)), f)]).mu(X.iloc[tgt], mu_r1[tgt])
        for sk in ["1", "mu", "sp"]:
            out[(name, sk)] = (promises(mu, sk, rng, win), win, np.abs(mu[win] - d.dur.values[win]).mean())
    print("window", key, flush=True); return key, out
if __name__ == "__main__":
    with mp.get_context("fork").Pool(8) as pool: rs = dict(pool.map(do_window, range(len(WINS)), chunksize=1))
    rows = []
    for key, out in rs.items():
        for (name, sk), (A, win, mae) in out.items():
            ot, mp_ = front(A, win); rows.append(dict(split=key[0], win=key[1], model=name, scale=sk, n=len(win),
                d95=at(ot, mp_, .95), mae=mae))
    r = pd.DataFrame(rows); r.to_csv(SP + "cmp_pratik.csv", index=False)
    best = r[r.split == "val"].groupby(["model", "scale"]).d95.mean().groupby("model").idxmin()
    print("scale chosen on validation:", dict(best))
    sel = pd.concat([r[(r.model == m) & (r.scale == s)] for m, s in best.values])
    print(sel.pivot_table(index=["split", "win"], columns="model", values="d95").round(2))
    print(sel.groupby(["split", "model"])[["d95", "mae"]].mean().round(3))
