"""Run Pratik's handover linear model (c66c7be, extracted in ./pratik) -> preds on his handover cohort. Three protocols:
 'his': his exact protocol (fit on primary train, score primary test); 'win': refit before each of our windows (deliv < W)."""
import sys, os, warnings; warnings.filterwarnings("ignore")
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
R = SP + "pratik"; sys.path.insert(0, R); sys.path.insert(0, R + "/experiments"); os.chdir(R)
import numpy as np, pandas as pd
import phase2_eta as E
table, spec, cov = E.load_stage_table(R); print(cov)
hs = E.stage_spec(spec, "handover"); T = pd.Timestamp
tr = table[table.split.eq("train")]; te = table[table.split.eq("test")]
out = table[["order_id", "order_purchase_timestamp", "split", "delivery_days"]].copy(); out["p_his"] = np.nan; out["p_win"] = np.nan
out.loc[te.index, "p_his"] = E.fit_predict("linear", {}, hs, tr, te)
print("his test MAE", np.abs(out.loc[te.index, "p_his"] - te.delivery_days).mean(), len(te))
W = [(T(f"2018-0{m}-01"), T(f"2018-0{m+1}-01") if m < 5 else T("2018-05-26")) for m in range(1, 6)] + \
    [(T("2018-05-26"), T("2018-06-01")), (T("2018-06-01"), T("2018-07-01")), (T("2018-07-01"), T("2018-08-01")), (T("2018-08-01"), T("2018-09-01"))]
for a, b in W:
    f = table[table.outcome_available_at < a]; s = table[(table.order_purchase_timestamp >= a) & (table.order_purchase_timestamp < b)]
    out.loc[s.index, "p_win"] = E.fit_predict("linear", {}, hs, f, s)
out["handover_days"] = table.handover_days; out["approval_days"] = table.approval_days
out.to_pickle(SP + "ho_pratik.pkl")
