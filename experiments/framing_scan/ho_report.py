import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score as APS
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
sel, best, F = pickle.load(open(SP + "ho_sel.pkl", "rb")); PR = pd.read_pickle(SP + "ho_pratik.pkl").set_index("order_id")
DAY = pd.Timedelta(days=1)
def met(y, p):
    e = p - y; return dict(n=len(y), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(), R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())
rows = []
for wi, (idx, raw, cor) in sel.items():
    w = F.iloc[idx][["order_id", "tq", "dur", "th", "deliv", "late", "order_estimated_delivery_date"]].copy(); w["win"] = wi; w["split"] = "val" if wi < 5 else "test"
    w["raw"], w["cor"] = raw, cor; rows.append(w)
R = pd.concat(rows, ignore_index=True)
R["p_win"] = PR.p_win.reindex(R.order_id).values; R["p_his"] = PR.p_his.reindex(R.order_id).values
R["in_his"] = R.order_id.isin(PR.index); R["his_test"] = R.order_id.map(PR.split).eq("test")
R.to_pickle(SP + "ho_R.pkl")
O = []
def P(s=""): print(s, flush=True); O.append(str(s))
hist = PR[PR.split.eq("test")]
te = R[R.split == "test"]
P(f"Our valid-handover test rows: {len(te)}; Pratik test cohort: {len(hist)}; overlap: {te.order_id.isin(hist.index).sum()}; ours not in his: {(~te.order_id.isin(hist.index)).sum()}; his not in ours: {(~hist.index.isin(te.order_id)).sum()}")
cm = te[te.order_id.isin(hist.index)]
d_ = np.abs(cm.dur.values - PR.delivery_days.reindex(cm.order_id).values); P(f"target agreement on common test rows: max |dur_ours - delivery_days_his| = {d_.max():.4f} days")
# ---------- table
def tab(df, col, label):
    return dict(model=label, **met(df.dur.values, df[col].values))
def block(split, common_only):
    d = R[R.split == split]
    d = d[d.in_his & (d.his_test if split == "test" else True)] if common_only else d
    return d
def pooled_and_mean(split, common_only, col, label):
    d = block(split, common_only)
    if split == "val":
        ms = [met(g.dur.values, g[col].values) for _, g in d.groupby("win")]
        return dict(model=label, scope=f"{split} mean of 5 windows", n=sum(m["n"] for m in ms), **{k: np.mean([m[k] for m in ms]) for k in ["MAE", "RMSE", "bias", "R2"]})
    return dict(model=label, scope=split, **met(d.dur.values, d[col].values))
out = []
for common in (False, True):
    for split in ("val", "test"):
        for col, lab in [("raw", "Ours M2 handover (raw)"), ("cor", "Ours M2 handover (bias-corrected)"), ("p_win", "Pratik linear (refit monthly, same windows)")] + ([("p_his", "Pratik linear (his protocol: fit once on his train)")] if split == "test" else []):
            if split == "val" and col == "p_his": continue
            r = pooled_and_mean(split, common, col, lab)
            if not np.isnan(r["MAE"]): out.append(dict(pop="common-with-Pratik" if common else "ours-all-valid", **r))
T1 = pd.DataFrame(out); P("## Handover-stage ETA accuracy (days; bias = mean(pred - actual), + = over-predict)\n```\n" + T1.round(3).to_string(index=False) + "\n```")
P("Pratik reported (his protocol, his 19,230 test rows): MAE 3.888, RMSE 5.369, bias +2.763, R2 0.176. Reproduced here: " + str(met(hist.delivery_days.values, PR.p_his[hist.index].values)))
# ---------- by month (test, common rows)
mon = []
d = block("test", True).copy(); d["m"] = d.tq.dt.to_period("M").astype(str)
for m, g in d.groupby("m"):
    row = {"month": m, "n": len(g)}
    for col, lab in [("raw", "ours_raw"), ("cor", "ours_corr"), ("p_his", "pratik")]:
        mm = met(g.dur.values, g[col].values); row[lab + "_MAE"] = mm["MAE"]; row[lab + "_bias"] = mm["bias"]
    mon.append(row)
P("## Test by purchase month (common rows)\n```\n" + pd.DataFrame(mon).round(2).to_string(index=False) + "\n```")
# ---------- early warning
EW = []
def ew(df, col, label, frac=.10):
    d = df.copy(); d["eta"] = d.tq + pd.to_timedelta(d[col], unit="D"); d["prom_end"] = d.order_estimated_delivery_date.dt.normalize() + DAY
    d["score"] = (d.eta - d.prom_end) / DAY; d["m"] = d.tq.dt.to_period("M")
    top = np.zeros(len(d), bool)
    for m, ix in d.groupby("m").indices.items():
        k = int(np.ceil(frac * len(ix))); top[ix[np.argsort(-d.score.values[ix], kind="stable")[:k]]] = True
    y = d.late.values.astype(bool); flag = (d.score > 0).values
    caught = top & y
    warn_prom = ((d.prom_end - d.th) / DAY).values[caught]; warn_del = ((d.deliv - d.th) / DAY).values[caught]
    return dict(model=label, n=len(d), late=int(y.sum()), top10_recall=caught.sum() / y.sum(), top10_prec=caught.sum() / top.sum(), PR_AUC=APS(y, d.score),
                rule_flags=int(flag.sum()), rule_recall=(flag & y).sum() / y.sum(), rule_prec=(flag & y).sum() / max(flag.sum(), 1),
                med_warn_to_promise_d=np.median(warn_prom), med_warn_to_delivery_d=np.median(warn_del))
d = block("test", False)
EW += [ew(d, "raw", "ours raw ETA (all valid test)"), ew(d, "cor", "ours bias-corrected ETA")]
dc = block("test", True); EW += [ew(dc, "raw", "ours raw (common rows)"), ew(dc, "cor", "ours corrected (common rows)"), ew(dc, "p_his", "Pratik linear (common rows)")]
P("## Late-order early warning at handover (promise = Olist estimated date; top-10% per purchase month by ETA - promise)\n```\n" + pd.DataFrame(EW).round(3).to_string(index=False) + "\n```")
open(SP + "results_handover_part1.md", "w").write("\n".join(O))
