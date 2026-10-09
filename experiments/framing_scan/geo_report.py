SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
src = open(SP + "geo_run.py").read()
exec(src[:src.index("# ---------- rules")])
a = src.index("CATS = ["); b = src.index("def do_window")
res, mus, widx = pickle.load(open(SP + "geo_res.pkl", "rb"))
BR1 = pick("R1"); w1, k1 = BR1[1][0], BR1[1][1]; d["mu_r1"] = d[f"r1_{w1}_{k1}"]; mu_r1 = d.mu_r1.values
exec(src[a:b])
OUT.clear()
P("# Geo experiment results (learned model vs geographic rule), promise days at matched on-time\n")
P(f"Olist val: promise {olist(vk)[0]:.2f}d on-time {olist(vk)[1]:.1%};  Olist test: promise {olist(tk)[0]:.2f}d on-time {olist(tk)[1]:.1%}\n")
names = {"R0": ["R0"], "R1": ["R1"], "M1": ["M1_sq", "M1_q90"], "M2": ["M2"], "M3": ["M3"]}
best = {}
for lab, ms in names.items():
    cands = [(k, *val_score(k)) for k in res if k[0] in ms]; cands = [c for c in cands if c[2] == 0] or cands
    best[lab] = min(cands, key=lambda r: r[1])[0]
P("Val-selected config (criterion: mean over 5 val windows of d@95): " + "; ".join(f"{l}={best[l]}" for l in best) + "\n")
def mae(k, keys): return np.mean(np.abs(np.concatenate([mus[k][x] for x in keys]) - np.concatenate([need[widx[x]] for x in keys])))
def tab(keys, per_window):
    rows = {}
    for lab, k in best.items():
        if per_window:
            v = []
            for key in keys:
                ot, mp = front(res[k][key], widx[key]); v.append([at(ot, mp, .95), at(ot, mp, .97), at(ot, mp, olist([key])[1])])
            v = np.nanmean(v, 0)
            mae_ = np.mean([np.mean(np.abs(mus[k][x] - need[widx[x]])) for x in keys])
        else:
            w = np.concatenate([widx[x] for x in keys]); A = np.concatenate([res[k][x] for x in keys]); ot, mp = front(A, w)
            v = [at(ot, mp, .95), at(ot, mp, .97), at(ot, mp, olist(keys)[1])]; mae_ = mae(k, keys)
        rows[lab] = [*v, mae_]
    rows["Olist"] = [olist(keys)[0]] * 3 + [np.nan]
    return pd.DataFrame(rows, index=["d@95%", "d@97%", "d@Olist-ontime", "MAE(point)"]).T.round(2)
P("## Validation (mean of 5 monthly windows)"); P(tab(vk, True).to_string())
P("\n## TEST 2018-05-26..08-31 (evaluated once, pooled)"); P(tab(tk, False).to_string())
bm = min(["M1", "M2", "M3"], key=lambda l: val_score(best[l])[0]); P(f"\nBest model on validation: {bm} {best[bm]}")
P("\n## Stability: promise days at 95% on-time per window (R0 / R1 / best model / difference model-R1)")
rows = []
for key in vk + tk:
    r_ = []
    for k in [best["R0"], best["R1"], best[bm]]:
        ot, mp = front(res[k][key], widx[key]); r_.append(at(ot, mp, .95))
    rows.append([f"{key[0]}{key[1]}", len(widx[key]), *r_, r_[2] - r_[1]])
P(pd.DataFrame(rows, columns=["win", "n", "R0", "R1", bm, "diff"]).round(2).to_string(index=False))
# test models side by side per month
P("\n## Test per-month d@95 for every method")
rows = {}
for lab, k in best.items():
    rows[lab] = [at(*front(res[k][key], widx[key]), .95) for key in tk]
P(pd.DataFrame(rows, index=["May26-31", "Jun", "Jul", "Aug"]).T.round(2).to_string())
# segments
def segs(keys):
    w = np.concatenate([widx[k] for k in keys]); q = np.quantile(d.dist.values[w][~np.isnan(d.dist.values[w])], [1 / 3, 2 / 3])
    f = lambda col, th, op: (lambda x: op(d[col].values[x], th))
    return {"dist short": lambda x: d.dist.values[x] <= q[0], "dist mid": lambda x: (d.dist.values[x] > q[0]) & (d.dist.values[x] <= q[1]),
            "dist long": lambda x: d.dist.values[x] > q[1], "capital dest": lambda x: d.capital.values[x] == 1, "interior dest": lambda x: d.capital.values[x] == 0,
            "same-state": lambda x: d.same_state.values[x] == 1, "cross-state": lambda x: d.same_state.values[x] == 0,
            "sparse zip5 (<5 prior)": lambda x: d.z5_n.values[x] < 5, "dense zip5 (>=5)": lambda x: d.z5_n.values[x] >= 5,
            "seller backlog >=8": lambda x: d.backlog.values[x] >= 8, "seller backlog <8": lambda x: d.backlog.values[x] < 8,
            "weight>3kg": lambda x: d.weight.values[x] > 3000}
for nm, keys in [("VALIDATION", vk), ("TEST", tk)]:
    A1 = np.concatenate([res[best["R1"]][x] for x in keys]); A2 = np.concatenate([res[best[bm]][x] for x in keys]); w = np.concatenate([widx[x] for x in keys]); rows = []
    for sn, fn in segs(keys).items():
        s = fn(w); o1, m1 = front(A1[s], w[s]); o2, m2 = front(A2[s], w[s]); a1, a2 = at(o1, m1, .95), at(o2, m2, .95)
        rows.append((sn, s.sum(), a1, a2, a2 - a1))
    P(f"\n## Segments {nm}: d@95 R1 vs {bm} (negative diff = model shorter)"); P(pd.DataFrame(rows, columns=["segment", "n", "R1", bm, "diff"]).round(2).to_string(index=False))
# permutation importance (refit on pre-test data, score on test)
k = best[bm]; scope = k[1][0]; Wt = TST[0][0]
tr = np.where((deliv < Wt.to_datetime64()) & ((tq >= (Wt - 180 * DAY).to_datetime64()) if scope == "180" else True))[0]
mkm = MODELS[k[0]]; mdl = mkm().fit(tr); tw = np.concatenate([widx[x] for x in tk])
Xt = X.iloc[tw].copy(); base = mu_r1[tw]; m0 = np.mean(np.abs(mdl.mu(Xt, base) - need[tw]))
cols = sorted(set(FH + FL)) if bm == "M2" else mdl.cols; rs = np.random.RandomState(0); imp = []
for c in cols:
    dl_ = []
    for _ in range(5):
        Xp = Xt.copy(); Xp[c] = rs.permutation(Xp[c].values); dl_.append(np.mean(np.abs(mdl.mu(Xp, base) - need[tw])) - m0)
    imp.append((c, np.mean(dl_)))
P(f"\n## Permutation importance ({bm}, increase in test MAE of point estimate; base MAE {m0:.3f}d)")
P(pd.DataFrame(sorted(imp, key=lambda t: -t[1])[:8], columns=["feature", "dMAE"]).round(3).to_string(index=False))
open(SP + "results_geo.md", "w").write("\n".join(OUT))
