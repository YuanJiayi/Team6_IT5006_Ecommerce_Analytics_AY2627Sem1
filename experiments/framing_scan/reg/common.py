import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
D="/Users/joshualum/Documents/VSCode/IT5006_Grp_6/data"
BOUND=pd.Timestamp("2018-05-26"); DAY=pd.Timedelta(days=1)
def load():
    o=pd.read_csv(f"{D}/olist_orders_dataset.csv",parse_dates=["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date"])
    it=pd.read_csv(f"{D}/olist_order_items_dataset.csv")
    pr=pd.read_csv(f"{D}/olist_products_dataset.csv")
    se=pd.read_csv(f"{D}/olist_sellers_dataset.csv"); cu=pd.read_csv(f"{D}/olist_customers_dataset.csv")
    tr=pd.read_csv(f"{D}/product_category_name_translation.csv")
    pr=pr.merge(tr,how="left",on="product_category_name"); pr["cat"]=pr.product_category_name_english.fillna("unk")
    return o,it,pr,se,cu
def wape(y,p): y=np.asarray(y);p=np.asarray(p); return np.abs(y-p).sum()/max(np.abs(y).sum(),1e-9)
