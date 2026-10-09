import pandas as pd, numpy as np, re
d = pd.read_pickle("rv_base.pkl")
x = d[d.delivered & d.score.notna()].copy()
ot = x[x.late == 0]
print("on-time delivered reviewed:", len(ot), "bad:", int(ot.bad.sum()), "rate", ot.bad.mean().round(3))
print("late:", len(x[x.late==1]), x[x.late==1].bad.mean().round(3))
ob = ot[ot.bad == 1]; og = ot[ot.bad == 0]
print("share of on-time bad with text:", (ob.text.str.len()>0).mean().round(3), " good:", (og.text.str.len()>0).mean().round(3))
pats = {
 "not received (não recebi/nao chegou)": r"n[aã]o recebi|n[aã]o chegou|n[aã]o foi entregue|nunca chegou|aguardando|ainda n[aã]o",
 "partial (recebi apenas/só/faltou/falta/um dos)": r"recebi (apenas|s[oó]|somente)|faltou|falta|apenas (um|1|uma)|s[oó] (um|chegou|recebi)|somente (um|1)|uma parte|parte do|um dos|outro produto n[aã]o|ainda falta|restante",
 "wrong/different (errado/diferente/trocado)": r"errad|diferente|trocad|n[aã]o corresponde|n[aã]o [eé] o que|outro (produto|modelo|cor)",
 "defect/broken (defeito/quebrado/danific)": r"defeit|quebrad|danific|estragad|n[aã]o funciona|parou de|rachad|amassad|avaria",
 "quality (qualidade/ruim/péssim/fraco/falsific)": r"qualidade|ruim|p[eé]ssim|fraco|falsific|r[aá]pido.*(estrag|quebr)|decepcion|n[aã]o recomendo|horr[ií]vel",
 "delay words (atraso/demor/prazo)": r"atras|demor|prazo|demorou",
 "delivery word (entrega)": r"entreg",
 "return/refund (devol/reembolso/estorno/cancel)": r"devol|reembols|estorn|cancel|reclam|troca",
 "seller contact (vendedor/loja/contato/resposta)": r"vendedor|loja|contato|resposta|atendimento|sem resposta",
}
rows=[]
for k,p in pats.items():
    f = lambda df: df.text.str.contains(p, regex=True).mean()
    rows.append((k, f(ob), f(og), f(x[(x.late==1)&(x.bad==1)])))
print(pd.DataFrame(rows, columns=["pattern","ontime_bad","ontime_good","late_bad"]).set_index("pattern").round(3).to_string())
# partial/not-received among multi-seller vs single
for g, nm in [(ot.n_sellers>=2,"multi-seller"),(ot.n_items>=2,"multi-item"),(ot.n_items==1,"single item")]:
    s = ob[g.loc[ob.index]]
    print(nm, "on-time bad n", len(s), "partial%", s.text.str.contains(pats[list(pats)[1]]).mean().round(3), "notrecv%", s.text.str.contains(pats[list(pats)[0]]).mean().round(3))
# cause classification (mutually exclusive, priority order) on on-time bad reviews
def cause(r):
    t = r.text
    if re.search(pats[list(pats)[1]], t) or (re.search(r"n[aã]o (recebi|chegou)|chegou apenas|chegou s[oó]", t) and r.n_items>=2): return "partial / missing item"
    if re.search(pats[list(pats)[0]], t): return "not received (per customer)"
    if re.search(pats[list(pats)[2]], t): return "wrong / not as described"
    if re.search(pats[list(pats)[3]], t): return "defective / damaged"
    if re.search(pats[list(pats)[4]], t): return "poor quality"
    if re.search(pats[list(pats)[5]], t): return "delivery time"
    if len(t)==0: return "no comment"
    return "other comment"
ob = ob.assign(cause=ob.apply(cause, axis=1))
print("\ncause shares of on-time bad reviews\n", ob.cause.value_counts(normalize=True).round(3).to_string())
# structural breakdowns (on-time)
def br(col, bins=None, q=None):
    s = ot[col]
    if q: s = pd.qcut(s, q, duplicates="drop")
    if bins: s = pd.cut(s, bins)
    t = ot.groupby(s, observed=True).bad.agg(["size","mean"]).round(3); print(f"\n{col}\n", t.to_string())
ot = ot.assign(ns=ot.n_sellers.clip(upper=3), ni=ot.n_items.clip(upper=4), nprod=ot.n_products.clip(upper=3))
for c in ["ns","ni","nprod"]: br(c)
br("price", q=5); br("freight_share", q=5); br("photos_min", bins=[-1,1,2,4,30]); br("desc_mean", q=5)
br("sel_bad_max", q=5); br("sel_n_max", bins=[-1,0,4,19,99,9999]); br("cat_bad_max", q=5); br("promise_days", q=5); br("handover_d", q=5)
cc = ot.groupby("cat").bad.agg(["size","mean"]); cc=cc[cc["size"]>=400].sort_values("mean",ascending=False); print(cc.head(8).round(3).to_string())
# multi-seller: delivered timestamp one package?
m = ot[ot.n_sellers>=2]
print("\nmulti-seller on-time n", len(m), "bad", m.bad.mean().round(3), " bad-with-text share", (m[m.bad==1].text.str.len()>0).mean().round(3))
print("multi-seller on-time vs late rates; late share among multi-seller delivered:", x[x.n_sellers>=2].late.mean().round(3), "vs single", x[x.n_sellers==1].late.mean().round(3))
for t in m[(m.bad==1)&(m.text.str.len()>20)].text.sample(8, random_state=1): print("  -", t[:140])
# decomposition: how much of on-time bad is explained by multi
print("\nshare of on-time bad that is multi-item(>=2):", (ob.n_items>=2).mean().round(3), " multi-seller:", (ob.n_sellers>=2).mean().round(3), "; of on-time orders:", (ot.n_items>=2).mean().round(3), (ot.n_sellers>=2).mean().round(3))
# the residual: single item single seller on-time bad
r1 = ob[(ob.n_items==1)]
print("single-item on-time bad share:", len(r1)/len(ob)); print(r1.apply(cause,axis=1).value_counts(normalize=True).round(3).to_string())
