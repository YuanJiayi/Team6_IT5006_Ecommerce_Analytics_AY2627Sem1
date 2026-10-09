SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
src = open(SP + "geo_run.py").read()
exec(src[:src.index("# ---------- rules")])
a = src.index("CATS = ["); b = src.index("class M1")
res, mus, widx = pickle.load(open(SP + "geo_res.pkl", "rb"))
BR1 = pick("R1"); d["mu_r1"] = d[f"r1_{BR1[1][0]}_{BR1[1][1]}"]; mu_r1 = d.mu_r1.values
exec(src[a:b])
W = T("2018-05-26"); win = np.concatenate([widx[k] for k in tk]); rng = np.where((tq >= (W - 100 * DAY).to_datetime64()) & (tq < T("2018-09-01").to_datetime64()))[0]
tr = np.where((deliv < W.to_datetime64()) & (tq >= (W - 180 * DAY).to_datetime64()))[0]; tr = tr[~np.isnan(d.handover.values[tr]) & ~np.isnan(d.leg.values[tr])]
def variant(drop):
    fh = [c for c in FH if c not in drop]; fl = [c for c in FL if c not in drop]
    def fit_pred(trn, idx):
        h = hgb(fh).fit(X.iloc[trn][fh].values, d.handover.values[trn]); l = hgb(fl).fit(X.iloc[trn][fl].values, d.leg.values[trn])
        return h.predict(X.iloc[idx][fh].values) + l.predict(X.iloc[idx][fl].values)
    mu = np.zeros(len(d)); mu[rng] = fit_pred(tr, rng)
    for f in np.array_split(np.arange(len(tr)), 3):
        tgt = tr[f][np.isin(tr[f], rng)]
        if len(tgt): mu[tgt] = fit_pred(tr[np.setdiff1d(np.arange(len(tr)), f)], tgt)
    A = promises(mu, "mu", rng, win); ot, mp = front(A, win)
    return drop, at(ot, mp, .95), np.mean(np.abs(mu[win] - need[win])), np.mean(mu[win] - need[win])
V = [[], ["month"], ["route_vol7"], ["backlog"], ["month", "route_vol7", "backlog"], ["month", "route_vol7", "backlog", "sell_hand", "sell_hand30", "sell_n"],
     ["month", "route_vol7", "backlog", "sell_hand", "sell_hand30", "sell_n", "dist", "capital", "z3_dur", "z5_dur", "z3_dur30", "z3_dur90", "z3_leg", "z5_leg", "z3_leg30", "z5_n", "city_late"]]
if __name__ == "__main__":
    with mp.get_context("fork").Pool(7) as pool: out = pool.map(variant, V)
    s = "# Ablation of M2 (static fit on <=2018-05-25, 180d scope, test pooled; scale=mu)\n" + "\n".join(f"drop={o[0]}: d@95={o[1]:.2f} MAE={o[2]:.2f} bias(mu-need)={o[3]:.2f}" for o in out)
    print(s); open(SP + "ablate.txt", "w").write(s)
