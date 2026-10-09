import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.ensemble import HistGradientBoostingRegressor as H
o,it,pr,se,cu=load()
it=it.merge(pr[["product_id","cat","product_weight_g","product_length_cm","product_height_cm","product_width_cm","product_photos_qty"]],on="product_id",how="left")
it["vol"]=it.product_length_cm*it.product_height_cm*it.product_width_cm
a=it.groupby("order_id").agg(n_items=("order_item_id","size"),price=("price","sum"),freight=("freight_value","sum"),w=("product_weight_g","sum"),vol=("vol","sum"),
  n_sell=("seller_id","nunique"),seller_id=("seller_id","first"),cat=("cat","first"),photos=("product_photos_qty","first"))
d=o.merge(a,on="order_id").merge(se[["seller_id","seller_state"]],on="seller_id").merge(cu[["customer_id","customer_state"]],on="customer_id")
d=d[d.order_approved_at.notna()&(d.order_status!="canceled")|d.order_delivered_carrier_date.notna()]
d=d[d.order_approved_at.notna()].copy()
d["ho"]=(d.order_delivered_carrier_date-d.order_approved_at)/DAY
print("handed over share",d.ho.notna().mean(),"neg ho",(d.ho<0).mean())
d=d.sort_values("order_approved_at").reset_index(drop=True)
t=d.order_approved_at.values
# seller features
H_mean=np.full(len(d),np.nan);H_med=H_mean.copy();H_last5=H_mean.copy();n_hist=np.zeros(len(d));back=np.zeros(len(d));back_all=back.copy();vol30=back.copy()
for s,g in d.groupby("seller_id"):
    idx=g.index.values; A=g.order_approved_at.values; C=g.order_delivered_carrier_date.values; ho=g.ho.values
    ok=~np.isnan(ho)&(ho>=0); order=np.argsort(C[ok]); Cs=C[ok][order]; hs=ho[ok][order]
    cs=np.concatenate([[0],np.cumsum(hs)])
    for j,i in enumerate(idx):
        k=np.searchsorted(Cs,A[j],side="left")   # handed over strictly before t
        n_hist[i]=k
        if k>=3:
            H_mean[i]=cs[k]/k; H_med[i]=np.median(hs[max(0,k-20):k]); H_last5[i]=hs[max(0,k-5):k].mean()
        m=(A<A[j])&(A>=A[j]-np.timedelta64(30,"D"))
        Cm=np.where(np.isnat(C[m]),np.datetime64("2100-01-01"),C[m])
        back[i]=(Cm>A[j]).sum(); vol30[i]=m.sum()
        m2=(A<A[j]); Cm2=np.where(np.isnat(C[m2]),np.datetime64("2100-01-01"),C[m2]); back_all[i]=(Cm2>A[j]).sum()
d["h_mean"]=H_mean;d["h_med"]=H_med;d["h_last5"]=H_last5;d["n_hist"]=n_hist;d["backlog30"]=back;d["vol30"]=vol30;d["backlog_all"]=back_all
d["wd"]=d.order_approved_at.dt.weekday;d["hr"]=d.order_approved_at.dt.hour;d["doy"]=d.order_approved_at.dt.dayofyear
d["same_state"]=(d.seller_state==d.customer_state).astype(int)
d["pay_lag"]=(d.order_approved_at-d.order_purchase_timestamp)/DAY
for c in ["cat","seller_state","customer_state"]: d[c+"_c"]=d[c].astype("category").cat.codes
d["seller_c"]=d.seller_id.astype("category").cat.codes
F=["n_items","price","freight","w","vol","n_sell","photos","h_mean","h_med","h_last5","n_hist","backlog30","vol30","backlog_all","wd","hr","same_state","pay_lag","cat_c","seller_state_c","customer_state_c"]
lab=d[d.ho.notna()&(d.ho>=0)&(d.ho<60)].copy()
# only orders whose handover is observable (approved well before data end)
lab=lab[lab.order_approved_at<"2018-08-15"]
print("n",len(lab),"ho mean",lab.ho.mean(),"median",lab.ho.median(),"p90",lab.ho.quantile(.9))
tr=lab[lab.order_approved_at<"2018-03-01"];va=lab[(lab.order_approved_at>="2018-03-01")&(lab.order_approved_at<BOUND)];tr2=lab[lab.order_approved_at<BOUND];te=lab[lab.order_approved_at>=BOUND]
glob_med=tr.ho.median()
def base(x,col):
    return x[col].fillna(x.ho.pipe(lambda _:glob_med))
def rep(name,x,p):
    y=x.ho.values; r=dict(MAE=np.abs(y-p).mean(),RMSE=np.sqrt(((y-p)**2).mean()))
    # SLA: predict handover within 3d? classify error in 'promise' use: coverage of 90% quantile later
    return r
res={}
for nm,x in [("val",va),("test",te)]:
    for b in ["h_mean","h_med","h_last5"]: res[(nm,b)]=rep(b,x,base(x,b).values)
    res[(nm,"global_median")]=rep("g",x,np.full(len(x),glob_med))
    res[(nm,"state_median")]=rep("s",x,x.seller_state.map(tr.groupby("seller_state").ho.median()).fillna(glob_med).values)
cfgs=[dict(learning_rate=.05,max_iter=300,max_depth=4,min_samples_leaf=50),dict(learning_rate=.05,max_iter=200,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=1)]
best=None
for loss in ["absolute_error","squared_error"]:
  for c in cfgs:
    m=H(loss=loss,random_state=0,**c).fit(tr[F],tr.ho); p=m.predict(va[F]); mae=np.abs(va.ho-p).mean(); print(loss,c,"val MAE",round(mae,3),"RMSE",np.sqrt(((va.ho-p)**2).mean()))
    if best is None or mae<best[0]: best=(mae,loss,c)
_,loss,c=best; print("chosen",loss,c)
m=H(loss=loss,random_state=0,**c).fit(tr[F],tr.ho);res[("val","GBM")]=rep("m",va,m.predict(va[F]))
m2=H(loss=loss,random_state=0,**c).fit(tr2[F],tr2.ho);pt=m2.predict(te[F]);res[("test","GBM")]=rep("m",te,pt)
# no-history/no-backlog ablation
for nm,drop in [("GBM_no_backlog",["backlog30","vol30","backlog_all"]),("GBM_no_hist",["h_mean","h_med","h_last5","n_hist"])]:
    f=[x for x in F if x not in drop]; mm=H(loss=loss,random_state=0,**c).fit(tr2[f],tr2.ho); res[("test",nm)]=rep(nm,te,mm.predict(te[f]))
print(pd.DataFrame(res).T.round(3))
# SLA business view: set per-order SLA = predicted P80 handover; check coverage/ width. quantile GBM
q=H(loss="quantile",quantile=.8,random_state=0,**c).fit(tr2[F],tr2.ho); qp=q.predict(te[F])
sm=te.h_med.fillna(glob_med)+ (tr2.ho-tr2.h_med.fillna(glob_med)).quantile(.8)  # baseline: seller median + global 80th resid
print("P80 SLA test: GBM cover",(te.ho<=qp).mean(),"mean SLA days",qp.mean(),"| baseline cover",(te.ho<=sm).mean(),"mean SLA",sm.mean())
# late-handover tail: orders with ho>5d, AUC of predicted ho vs baseline
from sklearn.metrics import roc_auc_score as AUC
yb=(te.ho>5).astype(int); print("share ho>5d",yb.mean(),"AUC GBM",AUC(yb,pt),"AUC seller_med",AUC(yb,te.h_med.fillna(glob_med)))
print("backlog corr with ho test",te[["backlog30","ho"]].corr(method="spearman").iloc[0,1])
# stronger baselines: calendar-aware
tr2=tr2.copy();tr2["hb"]=tr2.hr//6;te=te.copy();te["hb"]=te.hr//6;va=va.copy();va["hb"]=va.hr//6;tr=tr.copy();tr["hb"]=tr.hr//6
for nm,fit,x in [("val",tr,va),("test",tr2,te)]:
    cal=fit.groupby(["wd","hb"]).ho.median(); calx=x.set_index(["wd","hb"]).index.map(lambda k:cal.get(k,glob_med)).values.astype(float)
    off=fit.ho.groupby([fit.wd,fit.hb]).median(); 
    p1=calx; p2=np.where(x.h_med.notna(),x.h_med.values+(calx-glob_med),calx)
    # seller median of calendar-adjusted: multiply version
    print(nm,"calendar median MAE",np.abs(x.ho-p1).mean(),"seller_med+cal offset MAE",np.abs(x.ho-p2).mean())
