"""Task 4: add payment type / approval delay to the seller-handover sub-model of M2; effect on checkout promise days at 95% on-time (validation)."""
SPD = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
src = open(SPD + "geo_run.py").read()
exec(src[:src.index("# ---------- rules")])
TG = np.round(np.arange(0.80, 0.9951, 0.0025), 4)
a = src.index("CATS = ["); b = src.index("def do_window")
w1, k1 = 180, 40; d["mu_r1"] = d[f"r1_{w1}_{k1}"]; mu_r1 = d.mu_r1.values
exec(src[a:b])
print("M2 handover features FH:", FH); print("M2 carrier-leg features FL:", FL)
print("payment/approval in FH or FL:", [c for c in FH + FL if "pay" in c or "approv" in c or "install" in c])
D = "/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"
oo = pd.read_csv(f"{D}/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_approved_at"])[["order_id", "order_approved_at", "order_purchase_timestamp"]]
pay = pd.read_csv(f"{D}/olist_order_payments_dataset.csv").sort_values("payment_value", ascending=False).drop_duplicates("order_id")[["order_id", "payment_type"]]
e = d[["order_id"]].merge(oo, on="order_id", how="left").merge(pay, on="order_id", how="left")
assert (e.order_id.values == d.order_id.values).all()
X["approval_delay"] = ((e.order_approved_at - e.order_purchase_timestamp) / DAY).values
X["pay_c"] = e.payment_type.astype("category").cat.codes.astype(float).replace(-1, np.nan).values
def m2_window(wi):
    split, i, (W, E) = WINS[wi]; key = (split, i); rng = wrng[key]; win = widx[key]
    tr = np.where(deliv < W.to_datetime64())[0]; tr = tr[np.argsort(tq[tr])]
    mu = np.zeros(len(d)); mu[rng] = M2().fit(tr).mu(X.iloc[rng], mu_r1[rng])
    for f in np.array_split(np.arange(len(tr)), 3):
        tgt = tr[f][np.isin(tr[f], rng)]
        if len(tgt): mu[tgt] = M2().fit(tr[np.setdiff1d(np.arange(len(tr)), f)]).mu(X.iloc[tgt], mu_r1[tgt])
    return key, promises(mu, "mu", rng, win), mu[win]
FH0 = list(FH); VARS = {"base": [], "+pay_type": ["pay_c"], "+approval_delay": ["approval_delay"], "+both": ["pay_c", "approval_delay"]}
def variant(win_i):
    name, wi = win_i; global FH
    FH = FH0 + VARS[name]
    return name, wi, m2_window(wi)
CATS_ORIG = list(CATS)
def at(ot, mp, x): o_ = np.argsort(ot); return float(np.interp(x, ot[o_], mp[o_], left=np.nan, right=np.nan))
if __name__ == "__main__":
    with mp.get_context("fork").Pool(8) as pool: rs = pool.map(variant, [(n, wi) for n in VARS for wi in range(5)])
    O = []
    def P(s=""): print(s, flush=True); O.append(str(s))
    tab = {}
    for name, wi, (key, prom, mu) in rs:
        w = widx[key]; A = prom; ot = (A >= need[w][:, None]).mean(0); mpm = A.mean(0)
        tab.setdefault(name, {})[wi] = (at(ot, mpm, .95), np.abs(mu - need[w]).mean())
    df_ = pd.DataFrame({n: [tab[n][i][0] for i in range(5)] for n in VARS}, index=[f"val{i}" for i in range(5)]); df_.loc["mean"] = df_.mean()
    mae = pd.DataFrame({n: [tab[n][i][1] for i in range(5)] for n in VARS}, index=[f"val{i}" for i in range(5)]); mae.loc["mean"] = mae.mean()
    P("## Checkout promise days at 95% on-time (validation windows), M2 handover sub-model variants\n```\n" + df_.round(3).to_string() + "\n```")
    P("## M2 point-estimate MAE vs needed days (validation)\n```\n" + mae.round(3).to_string() + "\n```")
    chg = (df_.drop(columns="base").sub(df_.base, axis=0)); P("## Change vs base (days; negative = shorter promise)\n```\n" + chg.round(3).to_string() + "\n```")
    open(SPD + "results_handover_part_pay.md", "w").write("\n".join(O))
