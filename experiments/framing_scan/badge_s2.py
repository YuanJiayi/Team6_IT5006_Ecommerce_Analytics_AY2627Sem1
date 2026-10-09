"""Stage 2: rules, R, cutoff modes, selection on validation, frozen test eval, importance. Writes results_badge.md"""
import os; os.environ['OMP_NUM_THREADS']='1'
exec(open(SP0 := "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/badge_s1.py").read().split("def cols(name)")[0])
from sklearn.metrics import average_precision_score as APS, roc_auc_score as AUC
rs = pickle.load(open(SP + "badge_scores.pkl", "rb")); RP, MU, TG = pickle.load(open(SP + "final_promise.pkl", "rb"))
pday = d.pday.values
def asof2(ev_k, ev_t, ev_v, q_k, q_t, win):
    ev = pd.DataFrame({"k": ev_k.values, "t": ev_t.values, "v": ev_v.values}).dropna().sort_values("t")
    q = pd.DataFrame({"k": q_k.values, "t": q_t.values, "i": np.arange(len(q_k))})
    m = np.full(len(q_k), np.nan); n = np.zeros(len(q_k))
    for k, qg in q.groupby("k"):
        g = ev[ev.k == k]
        if not len(g): continue
        t = g.t.values; cs = np.r_[0, np.cumsum(g.v.values.astype(float))]
        hi = np.searchsorted(t, qg.t.values, "left"); lo = np.searchsorted(t, qg.t.values - np.timedelta64(win, "D"), "left")
        c = hi - lo; i = qg.i.values; n[i] = c
        with np.errstate(invalid="ignore", divide="ignore"): m[i] = np.where(c >= 5, (cs[hi] - cs[lo]) / c, np.nan)
    return m
O = []
def P(s=""): print(s, flush=True); O.append(s)
vk = [("val", i) for i in range(5)]; tk = [("test", i) for i in range(4)]
widx, wrng, S = {}, {}, {}   # S[(method, D)][key] = full-length score array (nan outside rng)
for key, D, rng, win, out in rs:
    widx[key] = win; wrng[key] = rng
    for name, s in out.items():
        a = np.full(len(d), np.nan, dtype=float); a[rng] = s; S.setdefault((name, D), {})[key] = a
rule = {}
for D in (7, 10):
    sh = asof2(d.route, d.deliv, (d.need <= D).astype(float), d.route, d.tq, 90)
    rule[("B1", D)] = np.nan_to_num(sh, nan=-1.0)
    rule[("B2", D)] = np.where((d.same_state.values == 1) & ~np.isnan(d.rd90.values), -d.rd90.values, -np.inf)
    for m in ("B1", "B2"): S[(m, D)] = {k: rule[(m, D)] for k in widx}
MINN = 50
def cut_from(sc, yy, target):
    ok = np.isfinite(sc)
    if ok.sum() == 0: return np.inf
    sc, yy = sc[ok], yy[ok]; o = np.argsort(-sc, kind="stable"); sc, yy = sc[o], yy[o]
    cp = np.cumsum(yy) / np.arange(1, len(yy) + 1); k = np.where((cp >= target) & (np.arange(1, len(yy) + 1) >= MINN))[0]
    return sc[k.max()] if len(k) else np.inf
def badge_static(m, D, key, target, src):
    sc = np.concatenate([S[(m, D)][k][widx[k]] for k in src]); yy = np.concatenate([(need[widx[k]] <= D) for k in src]).astype(float)
    c = cut_from(sc, yy, target); return S[(m, D)][key][widx[key]] >= c
def badge_daily(m, D, key, target):
    s = S[(m, D)][key]; rng = wrng[key]; win = widx[key]; y = (need <= D).astype(float); wd = pday[win]; b = np.zeros(len(win), bool)
    for day in np.unique(wd):
        cm = rng[(pday[rng] >= day - np.timedelta64(90, "D")) & (pday[rng] < day - np.timedelta64(45, "D")) & (deliv[rng] < day)]
        c = cut_from(s[cm], y[cm], target); mm = wd == day; b[mm] = s[win[mm]] >= c
    return b
SRC = {("val", 1): [("val", 0)], ("val", 2): [("val", 0), ("val", 1)], ("val", 3): [("val", 1), ("val", 2)], ("val", 4): [("val", 2), ("val", 3)]}
FROZEN = [("val", 3), ("val", 4)]
def get_badge(m, D, mode, tgt, key):
    if mode == "daily": return badge_daily(m, D, key, tgt)
    if mode == "static_all": return badge_static(m, D, key, tgt, vk if key[0] == "test" else [("val", i) for i in range(key[1])])
    return badge_static(m, D, key, tgt, FROZEN if key[0] == "test" else SRC[key])
def stats(b, key, D):
    yy = need[widx[key]] <= D; n = int(b.sum()); br = int((b & ~yy).sum())
    return dict(n=n, broken=br, prec=(1 - br / n) if n else np.nan, cov=n / len(b), N=len(b))
def summ(bl, keys, D):
    st = [stats(bl[k], k, D) for k in keys]
    return dict(prec=(1 - sum(s["broken"] for s in st) / sum(s["n"] for s in st)) if any(s["n"] for s in st) else np.nan, cov=np.mean([s["cov"] for s in st]),
                n=sum(s["n"] for s in st), broken=sum(s["broken"] for s in st), pprec=1 - sum(s["broken"] for s in st) / max(1, sum(s["n"] for s in st)), pcov=sum(s["n"] for s in st) / sum(s["N"] for s in st))
VK = vk[1:]; DEL = [0, .01, .02, .03, .04]
# R candidates: promise <= D at conformal level j
def badge_R(D, j, key): return RP["M2"][key][:, j] <= D
FAM = {"B1": ["B1"], "B2": ["B2"], "C1": ["C1_C0.1", "C1_C1"], "C2": ["C2_a", "C2_b", "C2_c"]}
def select(fam, D, tgt):
    best = None; allc = []
    if fam == "R":
        for j in range(len(TG)):
            bl = {k: badge_R(D, j, k) for k in vk}; v = summ(bl, VK, D); allc.append(((j,), v))
    else:
        for m in FAM[fam]:
            for mode in ("static", "static_all", "daily"):
                for dl in DEL:
                    bl = {k: get_badge(m, D, mode, tgt + dl, k) for k in VK}; allc.append(((m, mode, dl), summ(bl, VK, D)))
    ok = [c for c in allc if c[1]["n"] > 0 and c[1]["prec"] >= tgt]
    return max(ok, key=lambda c: c[1]["cov"]) if ok else max(allc, key=lambda c: np.nan_to_num(c[1]["prec"])), allc
def eval_choice(fam, D, ch, keys):
    if fam == "R": return {k: badge_R(D, ch[0], k) for k in keys}
    m, mode, dl = ch; return {k: get_badge(m, D, mode, tgt_ + dl, k) for k in keys}
CH = {}; RES = {}
P("# Fast-delivery badge classifier on Olist\n")
P("Population: delivered orders only (conditional on delivery; late/undelivered orders absent, and recent purchases are right-censored toward fast ones, so test base rates are inflated).")
P("fast = delivered calendar date - purchase calendar date <= D. Val = Jan..May25 2018 monthly windows (cutoffs from prior windows need history, so val stats/selection use windows 2-5 (Feb-May); val precision is POOLED over those windows, coverage is the mean of window coverages). Cutoff modes: static (last 2 prior val windows), static_all (all prior val windows), daily (purchases 90-45d earlier, resolved), each with a target offset delta in 0..4pt. Test = 2018-05-26..2018-08-31, evaluated once with frozen rule.\n")
P("## Base rates")
rows = []
for D in (7, 10):
    rows.append([D] + [f"{np.mean(need[widx[k]] <= D):.1%}" for k in vk] + [f"{np.mean(need[widx[k]] <= D):.1%}" for k in tk])
P(pd.DataFrame(rows, columns=["D"] + [f"val{i+1}" for i in range(5)] + ["tMay26-31", "tJun", "tJul", "tAug"]).to_string(index=False))
FAMS = ["B1", "B2", "R", "C1", "C2"]
table = []
for D in (7, 10):
    for tgt_ in (.95, .90):
        for fam in FAMS:
            ch, allc = select(fam, D, tgt_); CH[(D, tgt_, fam)] = ch[0]
            bv = eval_choice(fam, D, ch[0], VK); bt = eval_choice(fam, D, ch[0], tk)
            v = summ(bv, VK, D); t = summ(bt, tk, D); RES[(D, tgt_, fam)] = (bv, bt, v, t)
            table.append(dict(D=D, tgt=tgt_, fam=fam, choice=str(ch[0]), val_prec=v["prec"], val_cov=v["cov"], val_n=v["n"], val_broken=v["broken"],
                              te_prec=t["pprec"], te_cov=t["pcov"], te_n=t["n"], te_broken=t["broken"]))
T_ = pd.DataFrame(table)
P("\n## Main table (val = mean over windows 2-5; test pooled)")
P(T_.round(4).to_string(index=False))
# primary D: classifier val coverage at 95
cv = {D: max(T_[(T_.D == D) & (T_.tgt == .95) & T_.fam.isin(["C1", "C2"])].val_cov) for D in (7, 10)}
P(f"\nBest classifier val coverage at precision 95%: {cv}")
PD = 10
P(f"Primary D = {PD}: at D=7 every method has <3% val coverage at 95% precision (not meaningful); D=10 reported in detail, D=7 in main table")
# monthly stability for primary D at 95
P("\n## Test by month (D=%d): precision / coverage / n / broken" % PD)
for tgt_ in (.95, .90):
    rows = []
    for fam in FAMS:
        bt = RES[(PD, tgt_, fam)][1]; r = [f"{fam}@{int(tgt_*100)}"]
        for k in tk:
            s = stats(bt[k], k, PD); r.append(f"{s['prec']:.1%}/{s['cov']:.1%}/{s['n']}/{s['broken']}")
        rows.append(r)
    P(pd.DataFrame(rows, columns=["cfg", "May26-31", "Jun", "Jul", "Aug"]).to_string(index=False))
P("\n## Val by window (D=%d), precision/coverage" % PD)
for tgt_ in (.95, .90):
    rows = []
    for fam in FAMS:
        bv = RES[(PD, tgt_, fam)][0]; r = [f"{fam}@{int(tgt_*100)}"]
        for k in VK: s = stats(bv[k], k, PD); r.append(f"{s['prec']:.1%}/{s['cov']:.1%}")
        rows.append(r)
    P(pd.DataFrame(rows, columns=["cfg", "Feb", "Mar", "Apr", "May"]).to_string(index=False))
# ranking quality
def rank_score(fam, D, ch, key):
    if fam == "R": return -RP["M2"][key].mean(1)
    return S[(ch[0], D)][key][widx[key]]
P("\n## Ranking quality (D=%d): PR-AUC / ROC-AUC, val mean over windows 2-5 | test pooled" % PD)
rows = []
for fam in FAMS:
    ch = CH[(PD, .95, fam)]; r = [fam]
    sv = []
    for k in VK:
        yy = need[widx[k]] <= PD; sc = rank_score(fam, PD, ch, k); sc = np.where(np.isfinite(sc), sc, -1e9); sv.append((APS(yy, sc), AUC(yy, sc)))
    yt = np.concatenate([need[widx[k]] <= PD for k in tk]); st = np.concatenate([rank_score(fam, PD, ch, k) for k in tk]); st = np.where(np.isfinite(st), st, -1e9)
    rows.append([fam, np.mean([a for a, _ in sv]), np.mean([b for _, b in sv]), APS(yt, st), AUC(yt, st), yt.mean()])
P(pd.DataFrame(rows, columns=["fam", "val PR", "val ROC", "test PR", "test ROC", "test base"]).round(3).to_string(index=False))
# promised days
P("\n## Avg promised days (M2 conformal 0.9425 level promise / Olist estimate), test, D=%d, precision 95%% badge" % PD)
j95 = int(np.argmin(np.abs(TG - 0.9425))); rows = []
for fam in FAMS:
    bt = RES[(PD, .95, fam)][1]; b = np.concatenate([bt[k] for k in tk]); w = np.concatenate([widx[k] for k in tk])
    pm = np.concatenate([RP["M2"][k][:, j95] for k in tk]); ol = d.promise.values[w]
    rows.append([fam, b.sum(), pm[b].mean() if b.any() else np.nan, pm[~b].mean(), ol[b].mean() if b.any() else np.nan, ol[~b].mean(), need[w][b].mean() if b.any() else np.nan])
P(pd.DataFrame(rows, columns=["fam", "n_badged", "M2 promise badged", "M2 promise other", "Olist est. badged", "Olist est. other", "actual days badged"]).round(1).to_string(index=False))
# overlap of badges
P("\n## Overlap on test (D=%d, 95%%): fraction of badges of row-family also badged by column-family" % PD)
bb = {f: np.concatenate([RES[(PD, .95, f)][1][k] for k in tk]) for f in FAMS}
P(pd.DataFrame([[a] + [(bb[a] & bb[b]).sum() / max(1, bb[a].sum()) for b in FAMS] for a in FAMS], columns=["fam"] + FAMS).round(2).to_string(index=False))
# permutation importance on best classifier (by val coverage), fit on data < val5 start, score val5
bestc = max(["C1", "C2"], key=lambda f: RES[(PD, .95, f)][2]["cov"]); bm = CH[(PD, .95, bestc)][0]
src = open(SP0).read(); ns = {}; exec(src[:src.index("def job")], ns)
W5 = VAL[4][0]; tr = np.where(deliv < W5.to_datetime64())[0]; win = widx[("val", 4)]
cl = ns["NC"] if bm.startswith("C1") else ns["FALL"]; Xc = ns["X"][cl].values; y = (need <= PD).astype(int)
mdl = ns["CFG"][bm]().fit(Xc[tr], y[tr]); base = APS(y[win], mdl.predict_proba(Xc[win])[:, 1]); rg = np.random.RandomState(0); imp = {}
for ci, c in enumerate(cl):
    dr = []
    for _ in range(3):
        Xp = Xc[win].copy(); Xp[:, ci] = rg.permutation(Xp[:, ci]); dr.append(base - APS(y[win], mdl.predict_proba(Xp)[:, 1]))
    imp[c] = np.mean(dr)
P(f"\n## Permutation importance, best classifier {bm} (D={PD}), model fit < May 2018 val window, PR-AUC drop on val window 5 (base PR-AUC {base:.3f})")
P(pd.Series(imp).sort_values(ascending=False).head(8).round(4).to_string())
open(SP + "results_badge.md", "w").write("\n".join(O))
