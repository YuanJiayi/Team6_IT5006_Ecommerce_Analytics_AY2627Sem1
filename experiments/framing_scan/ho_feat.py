"""Recompute the dynamic (history) features of feat.pkl as-of an arbitrary query-time column. Writes ho_feat_th.pkl (as-of handover) and checks the tq version against feat.pkl."""
exec(open("/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/geo_base.py").read())
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
F = pd.read_pickle(SP + "feat.pkl")
def asof2(ev_k, ev_t, ev_v, q_k, q_t, win=None):
    ev = pd.DataFrame({"k": ev_k.values, "t": ev_t.values, "v": ev_v.values}).dropna().sort_values("t")
    q = pd.DataFrame({"k": q_k.values, "t": q_t.values, "i": np.arange(len(q_k))}).dropna(subset=["k"])
    m = np.full(len(q_k), np.nan); n = np.zeros(len(q_k)); egs = {k: g for k, g in ev.groupby("k")}
    for k, qg in q.groupby("k"):
        if k not in egs: continue
        t = egs[k].t.values; v = egs[k].v.values.astype(float); cs = np.r_[0, np.cumsum(v)]
        hi = np.searchsorted(t, qg.t.values, "left")
        lo = np.searchsorted(t, qg.t.values - np.timedelta64(win, "D"), "left") if win else np.zeros_like(hi)
        cnt = hi - lo; i = qg.i.values; n[i] = cnt
        with np.errstate(invalid="ignore", divide="ignore"): m[i] = np.where(cnt > 0, (cs[hi] - cs[lo]) / cnt, np.nan)
    return m, n
def shrink(m, n, parent, k): return (np.nan_to_num(m) * n + k * parent) / (n + k)
def dyn(d, qt):
    o_ = pd.DataFrame(index=d.index); r = d.route; c = d.customer_state; dd = d.dur.mean()
    o_["rd90"], o_["rd_std"], o_["rd_n"], p50 = asof(r, d.deliv, d.dur, r, qt, 90, 0.5)
    _, _, _, p90 = asof(r, d.deliv, d.dur, r, qt, 90, 0.9); o_["rt_spread"] = p90 - p50
    cd90 = asof(c, d.deliv, d.dur, c, qt, 90)[0]
    base0 = o_.rd90.fillna(pd.Series(cd90, index=d.index)).fillna(dd).values
    rl, _ = asof2(r, d.deliv, d.leg, r, qt, 90); o_["rl90"] = pd.Series(rl, index=d.index).fillna(d.leg.mean())
    RAW = {}
    for lv, key in [("z3", d.cz3), ("z5", d.cz5)]: RAW[(lv, None)] = asof2(key, d.deliv, d.dur, key, qt, None)
    for w in [30, 90]: RAW[("z3", w)] = asof2(d.cz3, d.deliv, d.dur, d.cz3, qt, w)
    o_["z3_dur"] = shrink(*RAW[("z3", None)], base0, 20); o_["z5_dur"] = shrink(*RAW[("z5", None)], o_["z3_dur"].values, 20)
    o_["z3_dur30"] = shrink(*RAW[("z3", 30)], base0, 20); o_["z3_dur90"] = shrink(*RAW[("z3", 90)], base0, 20); o_["z5_n"] = RAW[("z5", None)][1]
    bl = o_.rl90.values
    m_, n_ = asof2(d.cz3, d.deliv, d.leg, d.cz3, qt, None); o_["z3_leg"] = shrink(m_, n_, bl, 20)
    m_, n_ = asof2(d.cz5, d.deliv, d.leg, d.cz5, qt, None); o_["z5_leg"] = shrink(m_, n_, o_["z3_leg"].values, 20)
    m_, n_ = asof2(d.cz3, d.deliv, d.leg, d.cz3, qt, 30); o_["z3_leg30"] = shrink(m_, n_, bl, 20)
    m_, n_ = asof2(d.customer_city, d.deliv, d.late, d.customer_city, qt, 180); o_["city_late"] = shrink(m_, n_, d.late.mean(), 20)
    al = df[["seller_id", "order_purchase_timestamp", "order_delivered_carrier_date"]].dropna(subset=["seller_id"])
    bk = np.zeros(len(d)); dq = pd.DataFrame({"seller_id": d.seller_id.values, "t": qt.values}); dq["i"] = np.arange(len(d))
    for s_, g in dq.groupby("seller_id"):
        a_ = al[al.seller_id == s_]; tp = a_.order_purchase_timestamp.values; th2 = a_.order_delivered_carrier_date.values
        th2 = np.where(np.isnat(th2), np.datetime64("2100-01-01"), th2); t = g.t.values[:, None]
        bk[g.i.values] = ((tp[None, :] < t) & (tp[None, :] >= t - np.timedelta64(30, "D")) & (th2[None, :] > t)).sum(1)
    o_["backlog"] = bk
    rv = np.zeros(len(d)); alr = df[["seller_state", "customer_state", "order_purchase_timestamp"]].dropna(); alr["route"] = alr.seller_state + ">" + alr.customer_state
    ev = {k: np.sort(g.order_purchase_timestamp.values) for k, g in alr.groupby("route")}
    qv = qt.values
    for k, ix in d.groupby("route").indices.items():
        t = ev[k]; q = qv[ix]; rv[ix] = np.searchsorted(t, q, "left") - np.searchsorted(t, q - np.timedelta64(7, "D"), "left")
    o_["route_vol7"] = rv
    return o_
if __name__ == "__main__":
    assert (F.order_id.values == d.order_id.values).all()
    valid = F.th.notna() & (F.th >= F.tq) & (F.th < F.deliv)
    F["leg"] = F.leg.where(valid)   # invalid handovers must not pollute leg-history
    chk = dyn(F, F.tq)
    for c in chk.columns:
        a, b = chk[c].values, d[c].values if False else pd.read_pickle(SP + "feat.pkl")[c].values
        print(c, "max abs diff vs feat.pkl", np.nanmax(np.abs(a - b)))
    H = dyn(F, F.th.fillna(F.tq)); H.to_pickle(SP + "ho_feat_th.pkl"); print("saved")
