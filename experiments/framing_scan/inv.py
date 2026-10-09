import pandas as pd, glob, os
for f in sorted(glob.glob("data/olist_*.csv")+["data/product_category_name_translation.csv"]):
    d=pd.read_csv(f); print(f"\n## {os.path.basename(f)} rows={len(d)}")
    for c in d.columns:
        s=d[c]; r=f"{c} [{s.dtype}] miss={s.isna().mean():.1%} card={s.nunique()}"
        if pd.api.types.is_numeric_dtype(s): r+=f" range={s.min():.4g}..{s.max():.4g}"
        elif "timestamp" in c or "date" in c: r+=f" range={s.min()}..{s.max()}"
        print("  ",r)
