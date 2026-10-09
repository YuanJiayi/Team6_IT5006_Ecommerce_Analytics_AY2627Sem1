"""Seller-level snapshot tests: (A) 90d dormancy, (B) quality deterioration (next-90d bad-review rate)."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, mean_absolute_error as MAE
D="data"; DAY=pd.Timedelta(days=1)
o=pd.read_csv(f"{D}/olist_orders_dataset.csv",parse_dates=["order_purchase_timestamp","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"])
it=pd.read_csv(f"{D}/olist_order_items_dataset.csv",parse_dates=["shipping_limit_date"])
rv=pd.read_csv(f"{D}/olist_order_reviews_dataset.csv",parse_dates=["review_creation_date"])
se=pd.read_csv(f"{D}/olist_sellers_dataset.csv"); pr=pd.read_csv(f"{D}/olist_products_dataset.csv")
r=rv.sort_values("review_creation_date").groupby("order_id").agg(score=("review_score","min"),rcre=("review_creation_date","min"))
so=it.merge(pr[["product_id","product_category_name"]],on="product_id",how="left").groupby(["seller_id","order_id"]).agg(gmv=("price","sum"),fr=("freight_value","sum"),n=("price","size"),
     lim=("shipping_limit_date","max"),cat=("product_category_name","first"),nprod=("product_id","nunique")).reset_index()
so=so.merge(o[["order_id","order_status","order_purchase_timestamp","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"]],on="order_id").merge(r,on="order_id",how="left").merge(se[["seller_id","seller_state"]],on="seller_id")
so["t"]=so.order_purchase_timestamp
so["bad"]=(so.score<=2).astype(float).where(so.score.notna())
so["canc"]=so.order_status.isin(["canceled","unavailable"]).astype(float)
so["late"]=(so.order_delivered_customer_date>so.order_estimated_delivery_date).astype(float).where(so.order_delivered_customer_date.notna())
so["ho"]=(so.order_delivered_carrier_date-so.t)/DAY
so["miss"]=(so.order_delivered_carrier_date>so.lim).astype(float).where(so.order_delivered_carrier_date.notna())
first=so.groupby("seller_id").t.min(); states=se.set_index("seller_id").seller_state.astype("category").cat.codes
cats=so.cat.astype("category").cat.codes; so["catc"]=cats
BASE_BAD=(so.bad.dropna()).mean()

def feats(s):
    """as-of features at snapshot s; only info with timestamps < s"""
    H=so[so.t<s]; rows={}
    for w in (30,90,180):
        a=H[H.t>=s-w*DAY].groupby("seller_id").agg(**{f"o{w}":("order_id","size"),f"g{w}":("gmv","sum")}); rows[w]=a
    f=rows[180].join(rows[90]).join(rows[30]).fillna(0)
    a=H[H.t>=s-180*DAY]
    # event-known-by-snapshot outcomes
    rvd=a[a.rcre<s]; f["bad_n"]=rvd.groupby("seller_id").bad.size(); f["bad180"]=rvd.groupby("seller_id").bad.mean(); f["score180"]=rvd.groupby("seller_id").score.mean()
    dl=a[a.order_delivered_customer_date<s]; f["late180"]=dl.groupby("seller_id").late.mean()
    sh=a[a.order_delivered_carrier_date<s]; f["ho180"]=sh.groupby("seller_id").ho.mean(); f["miss180"]=sh.groupby("seller_id").miss.mean()
    f["canc180"]=a.groupby("seller_id").canc.mean()
    f["avg_gmv"]=f.g180/f.o180; f["fr_ratio"]=a.groupby("seller_id").fr.sum()/a.groupby("seller_id").gmv.sum()
    f["ncat"]=a.groupby("seller_id").cat.nunique(); f["items"]=a.groupby("seller_id").n.mean()
    last=H.groupby("seller_id").t.max(); f["rec"]=(s-last).dt.days.reindex(f.index)
    f["tenure"]=(s-first).dt.days.reindex(f.index)
    f["trend"]=(f.o30*3+1)/(f.o90+1); f["trend2"]=(f.o90*2+1)/(f.o180+1)
    f["state"]=states.reindex(f.index); f["topcat"]=a.groupby("seller_id").catc.agg(lambda x:x.mode().iat[0]); 
    plat=H[H.t>=s-30*DAY].shape[0]/max(1,H[(H.t>=s-60*DAY)&(H.t<s-30*DAY)].shape[0]); f["plat_trend"]=plat
    f["dow_gap"]=0
    return f[f.o180>0]

def outcome(s,f):
    W=so[(so.t>=s)&(so.t<s+90*DAY)]
    f["y_dorm"]=(~f.index.isin(W.seller_id.unique())).astype(int)
    g=W.groupby("seller_id").bad.agg(["mean","count"]); f["nb"]=g["count"].reindex(f.index); f["y_bad"]=g["mean"].reindex(f.index)
    return f

def snaps(a,b,step=7): return pd.date_range(a,b,freq=f"{step}D")
def build(ss):
    out=[]
    for s in ss:
        f=outcome(s,feats(s)); f["snap"]=s; out.append(f)
    return pd.concat(out)
TRAIN_V=snaps("2017-06-03","2017-08-31"); VAL=snaps("2017-12-02","2018-02-24"); TRAIN_F=snaps("2017-06-03","2018-02-24")
TEST=[pd.Timestamp("2018-05-26")]
dv_tr=build(TRAIN_V); dv_va=build(VAL); dv_trf=build(TRAIN_F); dv_te=build(TEST)
dv_trf.to_pickle("/dev/null") if False else None
X=[c for c in dv_te.columns if c not in("y_dorm","nb","y_bad","snap")]
print("rows",len(dv_tr),len(dv_va),len(dv_trf),len(dv_te))

def topk(y,sc,p=.10):
    k=max(1,int(len(y)*p)); i=np.argsort(-sc)[:k]; y=np.asarray(y); return y[i].mean(), y[i].sum()/y.sum()
def rep(name,y,sc):
    pa=AP(y,sc); au=AUC(y,sc); p,r=topk(y,sc); return f"{name:34s} PR-AUC {pa:.3f}  ROC {au:.3f}  prec@10% {p:.3f}  recall@10% {r:.3f}"

# ======= A: dormancy
print("\n=== A. Seller dormancy (active in last 180d, zero orders next 90d) ===")
for nm,d in (("val",dv_va),("test",dv_te)): print(nm,"n",len(d),"prevalence",d.y_dorm.mean().round(3),"trailing GMV share of dormant",(d.g180[d.y_dorm==1].sum()/d.g180.sum()).round(3))
def rules(d):
    return {"rule: days since last order":d.rec.values,"rule: -orders in last 90d":-d.o90.values,"rule: -orders in last 30d":-d.o30.values+d.rec.values*1e-6,
            "rule: -(orders180) tie-> recency":-d.o180.values*1000+d.rec.values}
best={}
for nm,d in (("val",dv_va),("test",dv_te)):
    for k,v in rules(d).items(): print(rep(f"[{nm}] {k}",d.y_dorm.values,v))
# tune on val
cand=[dict(max_iter=100,learning_rate=.05,max_depth=3,min_samples_leaf=50),dict(max_iter=200,learning_rate=.05,max_depth=4,min_samples_leaf=100),dict(max_iter=300,learning_rate=.03,max_depth=None,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=1.0)]
bs=None
for i,pr_ in enumerate(cand):
    m=HC(random_state=0,**pr_).fit(dv_tr[X],dv_tr.y_dorm); sc=m.predict_proba(dv_va[X])[:,1]; print(rep(f"[val] HGB cand{i}",dv_va.y_dorm.values,sc))
    if bs is None or AP(dv_va.y_dorm,sc)>bs[0]: bs=(AP(dv_va.y_dorm,sc),i)
pr_=cand[bs[1]]; print("chosen",bs[1])
m=HC(random_state=0,**pr_).fit(dv_trf[X],dv_trf.y_dorm); sc=m.predict_proba(dv_te[X])[:,1]; print(rep("[test] HGB (trained to 2018-02-24)",dv_te.y_dorm.values,sc))
# GMV at stake in top 10%
k=int(len(dv_te)*.1); i=np.argsort(-sc)[:k]; t=dv_te.iloc[i]
print(f"test top10% flagged {k} sellers: {t.y_dorm.mean():.1%} dormant; their trailing-180d GMV {t.g180.sum():,.0f} = {t.g180.sum()/dv_te.g180.sum():.1%} of total; mean orders180 flagged {t.o180.mean():.1f} vs all {dv_te.o180.mean():.1f}")
# high-value subset: sellers with >=10 orders180 (worth a retention call)
hv=dv_te[dv_te.o180>=10]; sch=m.predict_proba(hv[X])[:,1]
print("worth-saving sellers (>=10 orders/180d): n",len(hv),"prevalence",hv.y_dorm.mean().round(3)); print(rep("  [test, >=10 orders] HGB",hv.y_dorm.values,sch)); print(rep("  [test, >=10 orders] rule recency",hv.y_dorm.values,hv.rec.values))
# new-seller subset (tenure<=120d)
nw=dv_te[dv_te.tenure<=120]; scn=m.predict_proba(nw[X])[:,1]
print("new sellers (tenure<=120d): n",len(nw),"prevalence",nw.y_dorm.mean().round(3)); print(rep("  [test, new] HGB",nw.y_dorm.values,scn)); print(rep("  [test, new] rule recency",nw.y_dorm.values,nw.rec.values))
imp=pd.Series(0.0,index=X)
from sklearn.inspection import permutation_importance
pi=permutation_importance(m,dv_te[X],dv_te.y_dorm,scoring="average_precision",n_repeats=3,random_state=0); print("perm importance top:",pd.Series(pi.importances_mean,index=X).sort_values(ascending=False).head(6).round(3).to_dict())

# ======= B: quality deterioration
print("\n=== B. Seller quality deterioration: next-90d bad-review rate ===")
def sub(d): return d[(d.o180>=10)&(d.bad_n>=8)&(d.nb>=8)].copy()
bt,bv,btf,bte=sub(dv_tr),sub(dv_va),sub(dv_trf),sub(dv_te)
print("n val/test",len(bv),len(bte),"mean next bad rate val/test",bv.y_bad.mean().round(3),bte.y_bad.mean().round(3))
THR=0.25
for nm,d in (("val",bv),("test",bte)):
    print(nm,f"share with next bad>={THR}:",(d.y_bad>=THR).mean().round(3),"; of those, trailing bad180<",THR,":",((d.bad180<THR)&(d.y_bad>=THR)).sum()/(d.y_bad>=THR).sum())
    print(f"  [{nm}] regression MAE: platform mean {MAE(d.y_bad,np.full(len(d),btf.y_bad.mean())):.4f}  trailing180 rate {MAE(d.y_bad,d.bad180):.4f}  shrunk(k=20) {MAE(d.y_bad,(d.bad180*d.bad_n+btf.y_bad.mean()*20)/(d.bad_n+20)):.4f}")
yb=lambda d:(d.y_bad>=THR).astype(int)
for nm,d in (("val",bv),("test",bte)):
    print(rep(f"[{nm}] rule trailing bad rate",yb(d).values,d.bad180.values))
    print(rep(f"[{nm}] rule shrunk bad rate",yb(d).values,((d.bad180*d.bad_n+0.14*20)/(d.bad_n+20)).values))
    print(rep(f"[{nm}] rule late rate",yb(d).values,d.late180.fillna(0).values))
Xb=[c for c in X]
cb=[dict(max_iter=100,learning_rate=.05,max_depth=3,min_samples_leaf=30),dict(max_iter=200,learning_rate=.03,max_depth=3,min_samples_leaf=50,l2_regularization=2.0)]
bs=None
for i,p_ in enumerate(cb):
    mm=HC(random_state=0,**p_).fit(bt[Xb],yb(bt)); sc=mm.predict_proba(bv[Xb])[:,1]; print(rep(f"[val] HGB cand{i}",yb(bv).values,sc))
    if bs is None or AP(yb(bv),sc)>bs[0]: bs=(AP(yb(bv),sc),i)
mm=HC(random_state=0,**cb[bs[1]]).fit(btf[Xb],yb(btf)); sc=mm.predict_proba(bte[Xb])[:,1]; print(rep("[test] HGB cls",yb(bte).values,sc))
rg=HR(random_state=0,max_iter=150,learning_rate=.04,max_depth=3,min_samples_leaf=40).fit(btf[Xb],btf.y_bad); print("[test] HGB regression MAE",round(MAE(bte.y_bad,rg.predict(bte[Xb])),4))
rgv=HR(random_state=0,max_iter=150,learning_rate=.04,max_depth=3,min_samples_leaf=40).fit(bt[Xb],bt.y_bad); print("[val] HGB regression MAE",round(MAE(bv.y_bad,rgv.predict(bv[Xb])),4))
k=int(len(bte)*.1); i=np.argsort(-sc)[:k]; print("test top10% flagged:",k,"sellers; their next-90d orders",bte.nb.iloc[i].sum(),"mean next bad rate",bte.y_bad.iloc[i].mean().round(3),"vs all",bte.y_bad.mean().round(3))
