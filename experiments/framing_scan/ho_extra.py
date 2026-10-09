import numpy as np, pandas as pd
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
R = pd.read_pickle(SP + "ho_R.pkl"); DAY = pd.Timedelta(days=1); O = []
def P(s=""): print(s); O.append(str(s))
v = R[R.split == "val"]; rows = []
for w, g in v.groupby("win"):
    rows.append(dict(win=w, n=len(g), ours_MAE=np.abs(g.raw - g.dur).mean(), ours_corr_MAE=np.abs(g.cor - g.dur).mean(), pratik_MAE=np.abs(g.p_win - g.dur).mean(), ours_bias=(g.raw - g.dur).mean(), corr_bias=(g.cor - g.dur).mean(), pratik_bias=(g.p_win - g.dur).mean()))
P("## Validation per window (monthly refit; Jan..May1-25)\n```\n" + pd.DataFrame(rows).round(2).to_string(index=False) + "\n```")
te = R[R.split == "test"].reset_index(drop=True); rng = np.random.default_rng(0)
ae = {c: np.abs(te[c] - te.dur).values for c in ["raw", "cor", "p_his"]}
b = [(ae["p_his"][i].mean() - ae["raw"][i].mean()) for i in (rng.integers(0, len(te), len(te)) for _ in range(300))]
P(f"Test MAE gap Pratik(his) - ours raw: {np.mean(b):.2f}, 95% bootstrap [{np.percentile(b,2.5):.2f}, {np.percentile(b,97.5):.2f}]")
def sc(col): return ((te.tq + pd.to_timedelta(te[col], unit="D")) - (te.order_estimated_delivery_date.dt.normalize() + DAY)).dt.total_seconds().values / 86400
te["m"] = te.tq.dt.to_period("M"); y = te.late.values.astype(bool)
def rec(s, ix):
    top = np.zeros(len(ix), bool); mm = te.m.values[ix]
    for m in np.unique(mm):
        j = np.where(mm == m)[0]; k = int(np.ceil(.1 * len(j))); top[j[np.argsort(-s[ix][j], kind="stable")[:k]]] = True
    return (top & y[ix]).sum() / y[ix].sum()
sa, sp = sc("raw"), sc("p_his"); d = []
for _ in range(300):
    ix = rng.integers(0, len(te), len(te)); d.append(rec(sa, ix) - rec(sp, ix))
P(f"Top-10% recall gap ours raw - Pratik: {np.mean(d):.3f}, 95% bootstrap [{np.percentile(d,2.5):.3f}, {np.percentile(d,97.5):.3f}]")
open(SP + "results_handover_part2.md", "w").write("\n".join(O))
