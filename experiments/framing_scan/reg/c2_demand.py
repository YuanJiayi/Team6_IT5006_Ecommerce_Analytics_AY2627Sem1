import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.linear_model import Ridge
o,it,pr,se,cu=load()
o=o[o.order_status!="canceled"]
it=it.merge(pr[["product_id","cat"]],on="product_id",how="left").merge(o[["order_id","order_purchase_timestamp","customer_id"]],on="order_id").merge(cu[["customer_id","customer_state"]],on="customer_id")
it["wk"]=it.order_purchase_timestamp.dt.to_period("W-SUN").dt.start_time
W=pd.date_range("2017-01-02","2018-08-20",freq="W-MON")
topS=it.groupby("customer_state").order_id.nunique().nlargest(10).index;topC=it.groupby("cat").order_id.nunique().nlargest(10).index
def series(key,vals,metric):
    out={}
    for v in vals:
        x=it[it[key]==v]; s=x.groupby("wk").order_id.nunique() if metric=="orders" else x.groupby("wk").price.sum()
        out[f"{key}:{v}"]=s.reindex(W).fillna(0)
    return pd.DataFrame(out)
LAGS=8;HOR=[1,2,3,4]
BF=pd.Timestamp("2017-11-20")  # week of Black Friday 2017-11-24
def run(Y,name):
    n=len(W);rows=[]
    for j,c in enumerate(Y.columns):
        y=Y[c].values
        for T in range(LAGS+4,n):
            lag=y[T-LAGS+1:T+1][::-1]; m4=lag[:4].mean();m8=lag.mean();m12=y[max(0,T-11):T+1].mean()
            for h in HOR:
                if T+h>=n: continue
                tw=W[T+h]; 
                sn=y[T+h-52] if T+h-52>=0 else np.nan
                rows.append(dict(s=c,sid=j,T=T,h=h,tw=tw,y=y[T+h],last=lag[0],m4=m4,m8=m8,m12=m12,sn=sn,**{f"l{k}":lag[k]/(m4+1) for k in range(LAGS)},
                  r8=m4/(m8+1),slope=(lag[:3].mean()-lag[3:6].mean())/(m8+1),bf=int(tw==BF),bfp=int(tw==BF+pd.Timedelta(weeks=1)),xmas=int(tw.month==12 and tw.day>=18),
                  newyr=int(tw.month==1 and tw.day<=7),wkofyr=tw.isocalendar()[1],lm4=np.log1p(m4),tr_bf=int(W[T]>=BF)))
    D_=pd.DataFrame(rows);D_["scale"]=D_.m4+1
    FE=[f"l{k}" for k in range(LAGS)]+["r8","slope","bf","bfp","xmas","newyr","h","lm4","sid"]
    D_["yn"]=D_.y/D_.scale
    # eval: origins from 2018-01 onward; model trained on rows with target week <= origin week (T+h known)
    orig=[T for T in range(n) if W[T]>=pd.Timestamp("2018-01-01")]
    preds={k:[] for k in ["ridge","gbm","gbm_log_resid"]};idx=[]
    cfgs=[dict(learning_rate=.05,max_iter=150,max_depth=3,min_samples_leaf=30),dict(learning_rate=.05,max_iter=100,max_leaf_nodes=8,min_samples_leaf=60,l2_regularization=2)]
    pr_=pd.DataFrame(index=D_.index,columns=[f"gbm{i}" for i in range(2)]+["ridge"],dtype=float)
    for T in orig:
        if T%2: continue  # refit every 2 weeks, use for T and T+1
        trn=D_[(D_["T"]+D_.h<=T)]
        if len(trn)<300: continue
        tst=D_[D_["T"].isin([T,T+1])&(D_["T"]+D_.h<n)]
        X=trn[FE].copy();Xt=tst[FE]
        for i,c in enumerate(cfgs):
            m=H(loss="absolute_error",random_state=0,**c).fit(X,trn.yn);pr_.loc[tst.index,f"gbm{i}"]=m.predict(Xt)*tst.scale
        FR=[f for f in FE if f!="sid"]
        mr=Ridge(alpha=3).fit(trn[FR],trn.yn);pr_.loc[tst.index,"ridge"]=np.clip(mr.predict(tst[FR]),0,None)*tst.scale
    E=D_.join(pr_);E=E[E["T"].isin([T for T in orig])&E.ridge.notna()].copy()
    E["sn_f"]=E.sn.fillna(E.m4)
    E["val"]=E.tw<BOUND
    res={}
    for nm,msk in [("val",E.val),("test",~E.val)]:
        x=E[msk]
        for b in ["last","m4","m8","m12"]: res[(nm,b)]=wape(x.y,x[b])
        for b in ["ridge","gbm0","gbm1"]: res[(nm,b)]=wape(x.y,x[b])
        res[(nm,"n_obs")]=len(x)
        for h in HOR:
            xh=x[x.h==h];res[(nm,f"h{h} m4")]=wape(xh.y,xh.m4);res[(nm,f"h{h} last")]=wape(xh.y,xh["last"]);res[(nm,f"h{h} bestmodel")]=min(wape(xh.y,xh.ridge),wape(xh.y,xh.gbm0),wape(xh.y,xh.gbm1))
    r=pd.Series(res).unstack(0);print("\n==",name);print(r.round(3))
    return E
if __name__=="__main__":
    allr={}
    for key,vals,metric in [("customer_state",topS,"orders"),("cat",topC,"orders"),("customer_state",topS,"gmv"),("cat",topC,"gmv")]:
        run(series(key,vals,metric),f"{key} {metric}")
    tot=pd.DataFrame({"total":it.groupby("wk").order_id.nunique().reindex(W).fillna(0)})
    print("total weekly orders tail",tot.total.tail(30).astype(int).values)
