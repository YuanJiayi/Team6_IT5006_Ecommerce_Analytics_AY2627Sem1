"""Stage 1: score all rows (window rows + 100d lookback via 3-fold OOF) for B1,B2 features and C1/C2 configs, per window, D in {7,10}."""
import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, pandas as pd, pickle, warnings, multiprocessing as mp
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
SP = "/private/tmp/claude-501/-Users-joshualum-Documents-VSCode-IT5006-Grp-6/69c8a3a0-dc52-4a45-be15-6c61b93aef54/scratchpad/"
d = pd.read_pickle(SP + "feat.pkl"); T = pd.Timestamp; DAY = pd.Timedelta(days=1)
VAL = [(T(f"2018-0{m}-01"), T(f"2018-0{m+1}-01") if m < 5 else T("2018-05-26")) for m in range(1, 6)]
TST = [(T("2018-05-26"), T("2018-06-01")), (T("2018-06-01"), T("2018-07-01")), (T("2018-07-01"), T("2018-08-01")), (T("2018-08-01"), T("2018-09-01"))]
WINS = [("val", i, w) for i, w in enumerate(VAL)] + [("test", i, w) for i, w in enumerate(TST)]
need = d.need.values.astype(float); deliv = d.deliv.values; tq = d.tq.values
CATS = ["seller_state_c", "customer_state_c", "cat_c"]
GEO = ["dist", "capital", "z3_dur", "z5_dur", "z3_dur30", "z3_dur90", "z3_leg", "z5_leg", "z3_leg30", "z5_n", "city_late"]
ROUTE = ["rd90", "rt_spread", "rd_std", "rd_n", "mu_route", "route_vol7", "rl90"]
OPS = ["sell_hand", "sell_hand30", "sell_n", "backlog"]
ORD = ["weight", "freight", "price", "n_items", "dow", "same_state", "n_sellers"]
FALL = GEO + ROUTE + OPS + ORD + CATS   # no month
X = d[FALL].astype(float)
NC = [c for c in FALL if c not in CATS]
CFG = {
 "C1_C0.1": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=300)),
 "C1_C1": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=300)),
}
def hg(**kw): return lambda: HGC(random_state=0, categorical_features=[FALL.index(c) for c in CATS], **kw)
CFG.update({"C2_a": hg(max_iter=150, learning_rate=0.08, max_leaf_nodes=31),
            "C2_b": hg(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0),
            "C2_c": hg(max_iter=100, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100)})
def cols(name): return NC if name.startswith("C1") else FALL
def job(args):
    wi, D = args
    split, i, (W, E) = WINS[wi]
    win = np.where((tq >= W.to_datetime64()) & (tq < E.to_datetime64()))[0]
    rng = np.where((tq >= (W - 100 * DAY).to_datetime64()) & (tq < E.to_datetime64()))[0]
    tr = np.where(deliv < W.to_datetime64())[0]; tr = tr[np.argsort(tq[tr])]
    y = (need <= D).astype(int); out = {}
    for name, mk in CFG.items():
        c = cols(name); Xc = X[c].values
        s = np.full(len(d), np.nan); s[rng] = mk().fit(Xc[tr], y[tr]).predict_proba(Xc[rng])[:, 1]
        for f in np.array_split(np.arange(len(tr)), 3):
            tgt = tr[f][np.isin(tr[f], rng)]
            if len(tgt): s[tgt] = mk().fit(Xc[tr[np.setdiff1d(np.arange(len(tr)), f)]], y[tr[np.setdiff1d(np.arange(len(tr)), f)]]).predict_proba(Xc[tgt])[:, 1]
        out[name] = s[rng].astype(np.float32)
    print("done", wi, D, flush=True)
    return (split, i), D, rng, win, out
if __name__ == "__main__":
    with mp.get_context("fork").Pool(8) as pool: rs = pool.map(job, [(w, D) for D in (7, 10) for w in range(len(WINS))], chunksize=1)
    pickle.dump(rs, open(SP + "badge_scores.pkl", "wb"))
