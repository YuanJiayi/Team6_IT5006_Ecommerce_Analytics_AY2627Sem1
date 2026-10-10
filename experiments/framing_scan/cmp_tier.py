"""Tiered (Mondrian) conformal buffer vs single buffer, Pratik forest no-promise mu, scale 'mu'."""
import sys
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/d71361ad-8b17-4fcd-ac4f-b033089abeed/scratchpad/"
R = SP + "pratik"; sys.path.insert(0, R); sys.path.insert(0, R + "/experiments")
src = open(SP + "geo_run.py").read().split('if __name__ == "__main__":')[0]
g = {"__name__": "geo"}; exec(src, g)
import numpy as np, pandas as pd, multiprocessing as mp, json, copy
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.metrics import roc_auc_score
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
NT, MINN = 5, 30
def tiers_of(v, tr_c):
    cuts = np.quantile(v[tr_c], np.linspace(0, 1, NT + 1)[1:-1]); return np.searchsorted(cuts, v, side="right")
def tiered(mu, tier, rng, win, stats):
    lo, hi = CAL; sc = np.clip(mu, 3, None); out = np.full((len(win), len(TG)), np.nan); wd = pday[win]
    for day in np.unique(wd):
        cm = rng[(pday[rng] >= day - np.timedelta64(lo, "D")) & (pday[rng] < day - np.timedelta64(hi, "D")) & (deliv[rng] < day)]
        r = (need[cm] - mu[cm]) / sc[cm]; m = np.where(wd == day)[0]
        for t in range(NT):
            mt = m[tier[win[m]] == t]
            if not len(mt): continue
            ct = tier[cm] == t; stats[0] += 1
            if ct.sum() < MINN: ct = np.ones(len(cm), bool); stats[1] += 1; stats[3] += len(mt)
            stats[2] += len(mt)
            off = np.quantile(r[ct], TG)
            out[mt] = np.maximum(np.ceil(mu[win[mt]][:, None] + off[None, :] * sc[win[mt]][:, None]), 1)
    return out
def do_window(wi):
    split, i, (W, E) = WINS[wi]; key = (split, i)
    rng = wrng[key]; rng = rng[common[rng]]; win = widx[key]; win = win[common[win]]
    tr = np.where(deliv < W.to_datetime64())[0]; tr = tr[np.argsort(tq[tr])]
    folds = np.array_split(np.arange(len(tr)), 3)
    allr = np.union1d(rng, win)
    mu = np.zeros(len(d)); mu[allr] = Pratik().fit(tr).mu(X.iloc[allr])
    trc = tr[common[tr]]; fo = {}                       # fold id of each training row
    for k, f in enumerate(folds):
        tgt = tr[f][common[tr[f]]]; fo[k] = tgt
        mu[tgt] = Pratik().fit(tr[np.setdiff1d(np.arange(len(tr)), f)]).mu(X.iloc[tgt])
    sc = np.clip(mu, 3, None); r = (need - mu) / sc
    thr = np.quantile(r[trc], .8); y = (r > thr).astype(int)
    mk = lambda: HGC(max_iter=120, learning_rate=0.08, random_state=0, categorical_features=[FALL.index(c) for c in CATS])
    XF = X[FALL].values
    prob = np.zeros(len(d)); prob[allr] = mk().fit(XF[trc], y[trc]).predict_proba(XF[allr])[:, 1]
    for k, f in enumerate(folds):          # OOF probs for training rows (overrides full-fit on tr∩rng)
        tgt = fo[k]; oth = np.concatenate([fo[j] for j in fo if j != k])
        prob[tgt] = mk().fit(XF[oth], y[oth]).predict_proba(XF[tgt])[:, 1]
    auc = roc_auc_score(y[win], prob[win])
    tiersets = {"tier_classifier": tiers_of(prob, trc), "tier_rule": tiers_of(spread, trc), "tier_mu": tiers_of(mu, trc)}
    out = {"uniform": promises(mu, "mu", rng, win)}; fb = {}
    for n, tier in tiersets.items():
        st = [0, 0, 0, 0]; out[n] = tiered(mu, tier, rng, win, st); fb[n] = st
    print("window", key, flush=True)
    return key, dict(out=out, win=win, fb=fb, auc=auc, tier=tiersets["tier_classifier"][win], thr=thr, base=y[win].mean())
if __name__ == "__main__":
    with mp.get_context("fork").Pool(8) as pool: rs = dict(pool.map(do_window, range(len(WINS)), chunksize=1))
    rows = []
    for key, o in rs.items():
        for m, A in o["out"].items():
            ot, mp_ = front(A, o["win"]); rows.append(dict(split=key[0], win=key[1], method=m, n=len(o["win"]), d95=at(ot, mp_, .95),
                auc=o["auc"] if m == "tier_classifier" else np.nan, label_rate=o["base"], thr=o["thr"]))
    r = pd.DataFrame(rows); r.to_csv(SP + "cmp_tier.csv", index=False)
    order = ["uniform", "tier_classifier", "tier_rule", "tier_mu"]
    print(r.pivot_table(index=["split", "win"], columns="method", values="d95")[order].round(3))
    print(r.groupby(["split", "method"]).d95.mean().unstack()[order].round(3))
    print("AUC val per window", r[(r.method == "tier_classifier")].groupby(["split"]).auc.mean().round(4).to_dict(),
          r[(r.method == "tier_classifier") & (r.split == "val")].auc.round(4).tolist())
    print("label rate / thr", r[r.method=="uniform"][["split","win","label_rate","thr"]].round(3).values.tolist())
    for n in order[1:]:
        tot = np.sum([o["fb"][n] for o in rs.values()], 0)
        print(f"fallback {n}: day-tier cells {tot[0]}, fallback {tot[1]} ({tot[1]/tot[0]:.1%}); rows affected {tot[3]}/{tot[2]} ({tot[3]/tot[2]:.1%})")
    # per-tier on-time at the TG column chosen on pooled validation (smallest column with val on-time >= .95)
    for n in order:
        vk = [k for k in rs if k[0] == "val"]; tk = [k for k in rs if k[0] == "test"]
        cat = lambda ks, f: np.concatenate([f(rs[k]) for k in ks])
        vA = np.vstack([rs[k]["out"][n] for k in vk]); vn = np.concatenate([need[rs[k]["win"]] for k in vk])
        vot = (vA >= vn[:, None]).mean(0); c = int(np.argmax(vot >= .95)) if (vot >= .95).any() else None
        if c is None: print(n, "no column reaches 95% on val"); continue
        tA = np.vstack([rs[k]["out"][n] for k in tk])[:, c]; tn = np.concatenate([need[rs[k]["win"]] for k in tk])
        vA_c = vA[:, c]
        print(f"{n}: chosen TG={TG[c]} val on-time {vot[c]:.4f} val mean promise {vA_c.mean():.3f}; test on-time {(tA>=tn).mean():.4f} test mean promise {tA.mean():.3f}")
        if n == "tier_classifier":
            tt = np.concatenate([rs[k]["tier"] for k in tk]); vt = np.concatenate([rs[k]["tier"] for k in vk])
            for t in range(NT):
                print(f"  tier {t}: test n={int((tt==t).sum())} on-time {(tA[tt==t]>=tn[tt==t]).mean():.4f} mean promise {tA[tt==t].mean():.2f} | val on-time {(vA_c[vt==t]>=vn[vt==t]).mean():.4f} mean promise {vA_c[vt==t].mean():.2f}")
