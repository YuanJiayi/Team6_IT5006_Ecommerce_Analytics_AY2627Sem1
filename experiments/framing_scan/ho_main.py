"""Handover-stage ETA from our M2 carrier-leg model (features as-of handover) vs Pratik's handover linear model; early warning."""
import os; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, warnings, pickle, itertools, multiprocessing as mp
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
warnings.filterwarnings("ignore")
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
T = pd.Timestamp; DAY = pd.Timedelta(days=1)
F = pd.read_pickle(SP + "feat.pkl"); H = pd.read_pickle(SP + "ho_feat_th.pkl"); PR = pd.read_pickle(SP + "ho_pratik.pkl")
valid = (F.th.notna() & (F.th >= F.tq) & (F.th < F.deliv)).values
F["leg"] = F.leg.where(valid); F["handover"] = F.handover.where(valid)
for c in H.columns: F[c] = H[c].values                      # dynamic features as-of handover
F["h_dow"] = F.th.dt.dayofweek; F["h_hour"] = F.th.dt.hour
F = F[valid].reset_index(drop=True)
VAL = [(T(f"2018-0{m}-01"), T(f"2018-0{m+1}-01") if m < 5 else T("2018-05-26")) for m in range(1, 6)]
TST = [(T("2018-05-26"), T("2018-06-01")), (T("2018-06-01"), T("2018-07-01")), (T("2018-07-01"), T("2018-08-01")), (T("2018-08-01"), T("2018-09-01"))]
WINS = [("val", i, w) for i, w in enumerate(VAL)] + [("test", i, w) for i, w in enumerate(TST)]
CATS = ["seller_state_c", "customer_state_c", "cat_c"]
GEO = ["dist", "capital", "z3_dur", "z5_dur", "z3_dur30", "z3_dur90", "z3_leg", "z5_leg", "z3_leg30", "z5_n", "city_late"]
FLB = GEO + ["rd90", "rt_spread", "rd_std", "rd_n", "rl90", "route_vol7", "weight", "freight", "price", "n_items", "dow", "same_state", "backlog"] + CATS
FLH = FLB + ["handover", "h_dow", "h_hour"]
X = F[sorted(set(FLH))].astype(float)
tq, th, deliv = F.tq.values, F.th.values, F.deliv.values
dur, leg, hand = F.dur.values, F.leg.values, F.handover.values
thday = F.th.dt.normalize().values
CFGS = [(s, l, f) for s in ("all", "180") for l in ("sq", "abs") for f in ("base", "+h")]
def model(cols, loss):
    ci = [cols.index(c) for c in CATS]
    return HGR(max_iter=120, learning_rate=0.08, random_state=0, categorical_features=ci, loss="squared_error" if loss == "sq" else "absolute_error")
def fitpred(tr, tgt, cols, loss):
    return model(cols, loss).fit(X.iloc[tr][cols].values, leg[tr])
def run_window(args):
    wi, cfg = args; split, i, (W, E) = WINS[wi]; scope, loss, fs = cfg; cols = FLB if fs == "base" else FLH
    tr = np.where(deliv < W.to_datetime64())[0]
    if scope == "180": tr = tr[tq[tr] >= (W - 180 * DAY).to_datetime64()]
    tr = tr[np.argsort(tq[tr])]
    rng = np.where((tq >= (W - 150 * DAY).to_datetime64()) & (tq < E.to_datetime64()))[0]
    pl = np.zeros(len(F)); pl[rng] = fitpred(tr, leg, cols, loss).predict(X.iloc[rng][cols].values)
    for f in np.array_split(np.arange(len(tr)), 3):       # out-of-fold for rows the final model was trained on
        tgt = tr[f][np.isin(tr[f], rng)]
        if len(tgt): pl[tgt] = fitpred(tr[np.setdiff1d(np.arange(len(tr)), f)], leg, cols, loss).predict(X.iloc[tgt][cols].values)
    raw = hand[rng] + np.clip(pl[rng], 0, None)
    win = np.where((tq[rng] >= W.to_datetime64()) & (tq[rng] < E.to_datetime64()))[0]
    # bias-corrected: per handover day D, median residual (actual - raw) of orders handed over in [D-90, D-45) and already delivered before D
    res = dur[rng] - raw; hd = thday[rng]; dl = deliv[rng]; corr = np.zeros(len(win))
    for D in np.unique(hd[win]):
        m = (hd >= D - np.timedelta64(90, "D")) & (hd < D - np.timedelta64(45, "D")) & (dl < D)
        off = np.median(res[m]) if m.sum() >= 30 else 0.0
        corr[hd[win] == D] = off
    return wi, cfg, rng[win], raw[win], raw[win] + corr
def met(y, p):
    e = p - y; return dict(n=len(y), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(), R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())
O = []
def P(s=""): print(s, flush=True); O.append(str(s))
if __name__ == "__main__":
    ctx = mp.get_context("fork")
    # ---- phase A: validation only, all configs
    with ctx.Pool(8) as pool: A = pool.map(run_window, [(wi, c) for wi in range(5) for c in CFGS])
    vres = {}
    for wi, cfg, idx, raw, cor in A: vres.setdefault(cfg, {})[wi] = (idx, raw, cor)
    rows = []
    for cfg, w in vres.items():
        m = [met(dur[w[i][0]], w[i][1]) for i in range(5)]; mc = [met(dur[w[i][0]], w[i][2]) for i in range(5)]
        rows.append(dict(cfg=str(cfg), val_MAE=np.mean([x["MAE"] for x in m]), val_R2=np.mean([x["R2"] for x in m]), val_bias=np.mean([x["bias"] for x in m]), val_MAE_corr=np.mean([x["MAE"] for x in mc]), val_bias_corr=np.mean([x["bias"] for x in mc])))
    VT = pd.DataFrame(rows).sort_values("val_MAE"); P("## Validation config search (mean of 5 monthly windows; selection = lowest raw val MAE)\n```\n" + VT.round(3).to_string(index=False) + "\n```")
    best = eval(VT.iloc[0].cfg); P(f"Selected config (scope, loss, features): {best}")
    # ---- phase B: test once, selected config
    with ctx.Pool(4) as pool: B = pool.map(run_window, [(wi, best) for wi in range(5, 9)])
    sel = {wi: (idx, raw, cor) for wi, cfg, idx, raw, cor in B}
    for wi in range(5): sel[wi] = vres[best][wi]
    pickle.dump((sel, best, F[["order_id", "tq", "th", "deliv", "dur", "handover", "leg", "late", "promise", "order_estimated_delivery_date", "route"]]), open(SP + "ho_sel.pkl", "wb"))
