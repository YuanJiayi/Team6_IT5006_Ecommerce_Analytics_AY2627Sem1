"""PART 1: promise engine at pre-set conformal target (chosen on validation). Reuses geo_run.py machinery."""
SPD = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
src = open(SPD + "geo_run.py").read()
exec(src[:src.index("# ---------- rules")])
TG = np.round(np.arange(0.80, 0.9951, 0.0025), 4)
a = src.index("CATS = ["); b = src.index("def do_window")
w1, k1 = 180, 40; d["mu_r1"] = d[f"r1_{w1}_{k1}"]; mu_r1 = d.mu_r1.values
exec(src[a:b])
def m2_window(wi):
    split, i, (W, E) = WINS[wi]; key = (split, i); rng = wrng[key]; win = widx[key]
    tr = np.where(deliv < W.to_datetime64())[0]; tr = tr[np.argsort(tq[tr])]
    mu = np.zeros(len(d)); mu[rng] = M2().fit(tr).mu(X.iloc[rng], mu_r1[rng])
    for f in np.array_split(np.arange(len(tr)), 3):
        tgt = tr[f][np.isin(tr[f], rng)]
        if len(tgt): mu[tgt] = M2().fit(tr[np.setdiff1d(np.arange(len(tr)), f)]).mu(X.iloc[tgt], mu_r1[tgt])
    return key, promises(mu, "mu", rng, win), mu[win]
if __name__ == "__main__":
    with mp.get_context("fork").Pool(8) as pool: rs = pool.map(m2_window, range(len(WINS)))
    R = {"M2": {k: p for k, p, _ in rs}}; MU = {"M2": {k: m for k, _, m in rs}}
    mu0 = d.mu_route.values
    R["R0"] = {k: promises(mu0, "mu", wrng[k], widx[k]) for k in widx}; MU["R0"] = {k: mu0[widx[k]] for k in widx}
    R["R1"] = {k: promises(mu_r1, "sp", wrng[k], widx[k]) for k in widx}; MU["R1"] = {k: mu_r1[widx[k]] for k in widx}
    pickle.dump((R, MU, TG), open(SPD + "final_promise.pkl", "wb"))
    O = []
    def P(s=""): print(s, flush=True); O.append(s)
    ol_val = np.mean([1 - d.late.values[widx[k]].mean() for k in vk])
    P(f"Olist realised on-time, mean of 5 val windows: {ol_val:.1%}")
    ot_ = lambda m, k, j: (R[m][k][:, j] >= need[widx[k]]).mean()
    sel = {}
    for m in ["M2", "R0", "R1"]:
        va = np.array([[ot_(m, k, j) for j in range(len(TG))] for k in vk]).mean(0)
        for lab, goal in [("95", .95), ("olistval", ol_val)]:
            j = int(np.argmin(np.abs(va - goal))); sel[(m, lab)] = j
            P(f"{m} target for goal {lab}: conformal level {TG[j]} -> val mean on-time {va[j]:.2%}")
    def summ(m, j, keys):
        w = np.concatenate([widx[k] for k in keys]); p = np.concatenate([R[m][k][:, j] for k in keys]); n = need[w]
        return dict(n=len(w), mean=p.mean(), median=np.median(p), ontime=(p >= n).mean(), late=int((p < n).sum()))
    rows = []
    for (m, lab), j in sel.items():
        v = summ(m, j, vk); t = summ(m, j, tk)
        rows.append(dict(method=m, goal=lab, level=TG[j], val_mean=v["mean"], val_ontime=v["ontime"], test_mean=t["mean"], test_median=t["median"], test_ontime=t["ontime"], test_late=t["late"], n=t["n"]))
    tw = np.concatenate([widx[k] for k in tk]); ol = d.promise.values[tw]
    P("\n## TEST pooled (preset targets)"); T1 = pd.DataFrame(rows); P(T1.round(4).to_string(index=False))
    P(f"Olist actual on test: mean {ol.mean():.2f} median {np.median(ol):.0f} on-time {1-d.late.values[tw].mean():.2%} late {int(d.late.values[tw].sum())} of {len(tw)}")
    P("\n## TEST per month (on-time %, mean promise days)")
    names = ["May26-31", "Jun", "Jul", "Aug"]; rows = []
    for (m, lab), j in sel.items():
        rows.append([f"{m}/{lab}"] + [f"{summ(m, j, [k])['ontime']:.1%} / {summ(m, j, [k])['mean']:.1f}d (late {summ(m, j, [k])['late']})" for k in tk])
    rows.append(["Olist actual"] + [f"{1-d.late.values[widx[k]].mean():.1%} / {d.promise.values[widx[k]].mean():.1f}d (late {int(d.late.values[widx[k]].sum())})" for k in tk])
    P(pd.DataFrame(rows, columns=["cfg"] + names).to_string(index=False))
    P("\n## VAL per window on-time at preset target")
    rows = [[f"{m}/{lab}"] + [f"{ot_(m,k,j):.1%}" for k in vk] for (m, lab), j in sel.items()]
    P(pd.DataFrame(rows, columns=["cfg"] + [f"val{i}" for i in range(5)]).to_string(index=False))
    P("\n## Point-estimate error (days; target = needed whole days)")
    rows = []
    for m in ["M2", "R0", "R1"]:
        f = lambda keys: (lambda e: (np.abs(e).mean(), np.sqrt((e ** 2).mean())))(np.concatenate([MU[m][k] for k in keys]) - np.concatenate([need[widx[k]] for k in keys]))
        rows.append([m, *f(vk), *f(tk)])
    P(pd.DataFrame(rows, columns=["m", "val MAE", "val RMSE", "test MAE", "test RMSE"]).round(2).to_string(index=False))
    P("\n## Matched-on-time frontier numbers (from results_geo.md, test pooled, interpolated on test frontier)\n"
      "       d@95  d@97  d@Olist-ontime(96.5)  MAE\nR0     17.95 20.69 19.88 4.89\nR1     17.71 20.68 19.75 5.50\nM2     16.21 18.60 17.75 3.87\nOlist  22.05")
    open(SPD + "results_final_part1.md", "w").write("\n".join(O))
