import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, pandas as pd, warnings, pickle, multiprocessing as mp
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
d = pd.read_pickle(SP + "feat.pkl"); T = pd.Timestamp; DAY = pd.Timedelta(days=1)
VAL = [(T(f"2018-0{m}-01"), T(f"2018-0{m+1}-01") if m < 5 else T("2018-05-26")) for m in range(1, 6)]
TST = [(T("2018-05-26"), T("2018-06-01")), (T("2018-06-01"), T("2018-07-01")), (T("2018-07-01"), T("2018-08-01")), (T("2018-08-01"), T("2018-09-01"))]
WINS = [("val", i, w) for i, w in enumerate(VAL)] + [("test", i, w) for i, w in enumerate(TST)]
TG = np.array([.5, .6, .7, .8, .85, .88, .9, .92, .93, .95, .96, .97, .98, .99, .995]); REP = [.90, .95, .97]
need = d.need.values.astype(float); pday = d.pday.values; deliv = d.deliv.values; tq = d.tq.values
OUT = []
def P(s=""): print(s, flush=True); OUT.append(s)
CAL = (90, 45)
spread = d.rt_spread.fillna(d.rt_spread.median()).clip(lower=1).values
def scale_of(kind, mu): return {"1": np.ones_like(mu), "mu": np.clip(mu, 3, None), "sp": spread}[kind]
def promises(mu, sk, rng, win):
    lo, hi = CAL; sc = scale_of(sk, mu); out = np.full((len(win), len(TG)), np.nan); wd = pday[win]
    for day in np.unique(wd):
        cm = rng[(pday[rng] >= day - np.timedelta64(lo, "D")) & (pday[rng] < day - np.timedelta64(hi, "D")) & (deliv[rng] < day)]
        off = np.quantile((need[cm] - mu[cm]) / sc[cm], TG); m = wd == day
        out[m] = np.maximum(np.ceil(mu[win[m]][:, None] + off[None, :] * sc[win[m]][:, None]), 1)
    return out
widx, wrng = {}, {}
for split, i, (W, E) in WINS:
    key = (split, i)
    widx[key] = np.where((tq >= W.to_datetime64()) & (tq < E.to_datetime64()))[0]
    wrng[key] = np.where((tq >= (W - 100 * DAY).to_datetime64()) & (tq < E.to_datetime64()))[0]
vk = [("val", i) for i in range(5)]; tk = [("test", i) for i in range(4)]
res = {}; mus = {}   # res[(method, cfg)][key] = promise array ; mus[(method,cfg)][key]=mu on window rows
def run_rule(name, cfg, mu):
    for sk in ["1", "mu", "sp"]:
        for key in widx:
            res.setdefault((name, cfg + (sk,)), {})[key] = promises(mu, sk, wrng[key], widx[key])
            mus.setdefault((name, cfg + (sk,)), {})[key] = mu[widx[key]]
def at(ot, mp, x):
    o_ = np.argsort(ot); return float(np.interp(x, ot[o_], mp[o_], left=np.nan, right=np.nan))
def front(A, w): return (A >= need[w][:, None]).mean(0), A.mean(0)
def olist(keys):
    w = np.concatenate([widx[k] for k in keys]); return d.promise.values[w].mean(), 1 - d.late.values[w].mean()
def val_score(k):   # mean over 5 val windows of d@95
    v = []
    for key in vk:
        ot, mp = front(res[k][key], widx[key]); v.append(at(ot, mp, .95))
    return np.nanmean(v), sum(np.isnan(v))
def pick(method):
    rows = [(k, *val_score(k)) for k in res if k[0] == method]
    rows = [r for r in rows if r[2] == 0] or rows
    return min(rows, key=lambda r: r[1])[0]
# ---------- rules
run_rule("R0", (), d.mu_route.values)
for w in [180, None]:
    for k in [10, 20, 40]: run_rule("R1", (w, k), d[f"r1_{w}_{k}"].values)
P("# Geo experiment results\n\n## Rule variants on validation (mean over 5 monthly windows of promise days at 95% on-time)")
for m in ["R0", "R1"]:
    for k in sorted([k for k in res if k[0] == m], key=str): P(f"  {k}: {val_score(k)[0]:.2f}")
BR1 = pick("R1"); BR0 = pick("R0"); P(f"R0 chosen {BR0}; R1 chosen {BR1}")
w1, k1 = BR1[1][0], BR1[1][1]; d["mu_r1"] = d[f"r1_{w1}_{k1}"]; mu_r1 = d.mu_r1.values
# ---------- models
CATS = ["seller_state_c", "customer_state_c", "cat_c"]
GEO = ["dist", "capital", "z3_dur", "z5_dur", "z3_dur30", "z3_dur90", "z3_leg", "z5_leg", "z3_leg30", "z5_n", "city_late"]
ROUTE = ["rd90", "rt_spread", "rd_std", "rd_n", "mu_route", "route_vol7"]
OPS = ["sell_hand", "sell_hand30", "sell_n", "backlog"]
ORD = ["weight", "freight", "price", "n_items", "dow", "same_state", "n_sellers"]
FALL = GEO + ROUTE + OPS + ORD + CATS
FH = OPS + ["route_vol7", "weight", "freight", "price", "n_items", "dow", "same_state", "dist", "n_sellers"] + CATS
FL = GEO + ["rd90", "rt_spread", "rd_std", "rd_n", "rl90", "route_vol7", "weight", "freight", "price", "n_items", "dow", "same_state", "backlog"] + CATS
X = d[sorted(set(FALL + FH + FL))].astype(float)
def hgb(cols, loss="sq", q=.9):
    ci = [cols.index(c) for c in CATS if c in cols]
    kw = dict(loss="quantile", quantile=q) if loss == "q90" else {}
    return HGR(max_iter=120, learning_rate=0.08, random_state=0, categorical_features=ci or None, **kw)
class M1:
    def __init__(s, loss): s.loss = loss; s.cols = FALL
    def fit(s, tr): s.m = hgb(s.cols, s.loss).fit(X.iloc[tr][s.cols].values, (need - mu_r1)[tr]); return s
    def mu(s, Xf, base): return base + s.m.predict(Xf[s.cols].values)
class M3:
    cols = [c for c in FALL if c not in CATS]
    def fit(s, tr):
        s.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(X.iloc[tr][s.cols].values, (need - mu_r1)[tr]); return s
    def mu(s, Xf, base): return base + s.m.predict(Xf[s.cols].values)
class M2:
    cols = sorted(set(FH + FL))
    def fit(s, tr):
        tr = tr[~np.isnan(d.handover.values[tr]) & ~np.isnan(d.leg.values[tr])]
        s.h = hgb(FH).fit(X.iloc[tr][FH].values, d.handover.values[tr]); s.l = hgb(FL).fit(X.iloc[tr][FL].values, d.leg.values[tr]); return s
    def mu(s, Xf, base): return s.h.predict(Xf[FH].values) + s.l.predict(Xf[FL].values)
MODELS = {"M1_sq": lambda: M1("sq"), "M1_q90": lambda: M1("q90"), "M2": M2, "M3": M3}
def do_window(wi):
    split, i, (W, E) = WINS[wi]
    key = (split, i); rng = wrng[key]; win = widx[key]; out = {}
    for scope in ["all", "180"]:
        tr = np.where((deliv < W.to_datetime64()) & ((tq >= (W - 180 * DAY).to_datetime64()) if scope == "180" else True))[0]
        tr = tr[np.argsort(tq[tr])]
        for name, mkm in MODELS.items():
            mu = np.zeros(len(d)); Xr = X.iloc[rng]; mu[rng] = mkm().fit(tr).mu(Xr, mu_r1[rng])
            for f in np.array_split(np.arange(len(tr)), 3):
                tgt = tr[f][np.isin(tr[f], rng)]
                if len(tgt): mu[tgt] = mkm().fit(tr[np.setdiff1d(np.arange(len(tr)), f)]).mu(X.iloc[tgt], mu_r1[tgt])
            for sk in ["1", "mu", "sp"]:
                out[(name, (scope, sk))] = (promises(mu, sk, rng, win), mu[win])
    print("window", key, flush=True)
    return key, out
if __name__ == "__main__":
    with mp.get_context("fork").Pool(8) as pool: rs = pool.map(do_window, range(len(WINS)), chunksize=1)
    for key, out in rs:
        for k, (p_, m_) in out.items(): res.setdefault(k, {})[key] = p_; mus.setdefault(k, {})[key] = m_
    pickle.dump((res, mus, widx), open(SP + "geo_res.pkl", "wb"))
