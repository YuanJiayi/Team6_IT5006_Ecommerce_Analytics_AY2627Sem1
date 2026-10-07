"""Linear-regression assumption diagnostics for ridge_100 on the primary training orders (validation only; test rows never used)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_classification import load_primary_training, temporal_folds  # noqa: E402
from phase2_regression import feature_columns, make_regression_pipeline, regression_metrics  # noqa: E402

OUT = Path(__file__).with_suffix("")
SEED = 42
RIDGE = ("ridge", {"alpha": 100.0})


def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def vif_table(Z, names):
    rows = []
    for j, name in enumerate(names):
        others = np.delete(Z, j, axis=1)
        if np.std(Z[:, j]) == 0:
            rows.append({"feature": name, "vif": np.nan, "note": "constant column"})
            continue
        r2 = LinearRegression().fit(others, Z[:, j]).score(others, Z[:, j])
        rows.append({"feature": name, "vif": float(1 / (1 - r2)) if r2 < 1 - 1e-12 else float("inf"), "note": ""})
    return pd.DataFrame(rows)


def breusch_pagan(X, resid):
    """Koenker (studentised) LM test: n*R^2 of squared residuals on the design matrix."""
    X1 = np.column_stack([np.ones(len(X)), X])
    u = resid ** 2
    beta = np.linalg.lstsq(X1, u, rcond=None)[0]
    r2 = 1 - np.sum((u - X1 @ beta) ** 2) / np.sum((u - u.mean()) ** 2)
    df = np.linalg.matrix_rank(X1) - 1
    lm = len(u) * r2
    return {"lm": float(lm), "df": int(df), "p_value": float(stats.chi2.sf(lm, df))}


def durbin_watson(resid_ordered):
    return float(np.sum(np.diff(resid_ordered) ** 2) / np.sum(resid_ordered ** 2))


def binned_mean(x, y, bins=20):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    return (np.array([x[idx == b].mean() for b in range(bins)]),
            np.array([y[idx == b].mean() for b in range(bins)]))


def residual_diagnostics(tag, fitted, resid, X, order_time, month, rng):
    z = (resid - resid.mean()) / resid.std(ddof=1)
    sub = rng.choice(len(resid), size=min(5000, len(resid)), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    bx, by = binned_mean(fitted, resid)
    ax[0].scatter(fitted[sub], resid[sub], s=4, alpha=.3, color="tab:blue")
    ax[0].plot(bx, by, color="tab:red", lw=2, label="binned mean (20 quantile bins)")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set(title=f"Residual vs fitted ({tag})", xlabel="fitted days", ylabel="residual (actual - fitted)")
    ax[0].legend()
    root_abs = np.sqrt(np.abs(z))
    bx, by = binned_mean(fitted, root_abs)
    ax[1].scatter(fitted[sub], root_abs[sub], s=4, alpha=.3, color="tab:blue")
    ax[1].plot(bx, by, color="tab:red", lw=2)
    ax[1].set(title=f"Scale-location ({tag})", xlabel="fitted days", ylabel="sqrt|standardized residual|")
    stats.probplot(z[sub], dist="norm", plot=ax[2])
    ax[2].set_title(f"QQ plot, standardized residuals ({tag}, 5,000 pts)")
    fig.tight_layout()
    fig.savefig(OUT / f"residuals_{tag}.png", dpi=120)
    plt.close(fig)

    ordered = resid[np.argsort(order_time, kind="stable")]
    by_month = pd.Series(resid).groupby(month).agg(["mean", "count"]).rename(columns={"mean": "mean_residual"})
    by_month.index.name = "purchase_month"
    by_month.to_csv(OUT / f"residual_by_month_{tag}.csv")
    fig, a = plt.subplots(figsize=(8, 4))
    a.bar(by_month.index.astype(str), by_month["mean_residual"])
    a.axhline(0, color="k", lw=.8)
    a.set(title=f"Mean residual by purchase month ({tag})", ylabel="mean residual (days)")
    plt.setp(a.get_xticklabels(), rotation=60)
    fig.tight_layout()
    fig.savefig(OUT / f"residual_by_month_{tag}.png", dpi=120)
    plt.close(fig)
    return {"n": len(resid), "mean_residual": float(resid.mean()),
            "skewness": float(stats.skew(resid)), "excess_kurtosis": float(stats.kurtosis(resid)),
            "breusch_pagan": breusch_pagan(X, resid), "durbin_watson": durbin_watson(ordered),
            "max_abs_month_mean_residual": float(by_month["mean_residual"].abs().max())}


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    train, spec = load_primary_training(ROOT)
    folds = temporal_folds(train, spec)
    columns = sum(feature_columns(spec), [])
    X, y = train[columns], train[spec["regression_target"]].to_numpy(dtype=float)
    n = len(train)
    assert n == 75099, n
    result = {"n_train": n}

    def fresh(family="ridge", params=RIDGE[1]):
        return make_regression_pipeline(family, params, spec)

    with threadpool_limits(limits=1):
        # 1. fit statistics
        ols, ridge = fresh("linear", {}), fresh()
        ols.fit(X, y)
        ridge.fit(X, y)
        Z = dense(ols[:-1].transform(X))
        names = list(ols[0].get_feature_names_out())
        p = Z.shape[1]
        rank = np.linalg.matrix_rank(np.column_stack([np.ones(n), Z]))
        r2_ols, r2_ridge = ols.score(X, y), ridge.score(X, y)
        F = (r2_ols / p) / ((1 - r2_ols) / (n - p - 1))
        result["fit"] = {
            "p_columns_after_preprocessing": p, "design_rank_with_intercept": int(rank),
            "ols": {"r2": r2_ols, "adj_r2": adj_r2(r2_ols, n, p), "F": F, "F_df": [p, n - p - 1],
                    "F_p_value": float(stats.f.sf(F, p, n - p - 1))},
            "ridge_100": {"r2": r2_ridge, "adj_r2": adj_r2(r2_ridge, n, p)},
        }
        folds_rows = []
        for w, (fit, valid) in enumerate(folds):
            m = fresh().fit(X.iloc[fit], y[fit])
            for part, idx in (("fit_in_sample", fit), ("validation", valid)):
                folds_rows.append({"window": w, "part": part, "orders": len(idx),
                                   **regression_metrics(y[idx], m.predict(X.iloc[idx]))})
        fold_df = pd.DataFrame(folds_rows)
        fold_df.to_csv(OUT / "train_vs_validation_by_window.csv", index=False)
        result["train_vs_validation_mean"] = fold_df.groupby("part")[["mae", "rmse", "r2"]].mean().to_dict("index")

        # 2. VIF on numeric columns after linear preprocessing (before one-hot)
        keep = [i for i, nm in enumerate(names) if not nm.startswith("category__")]
        vif = vif_table(Z[:, keep], [names[i] for i in keep]).sort_values("vif", ascending=False)
        vif.to_csv(OUT / "vif.csv", index=False)
        result["vif_top"] = vif.head(8).to_dict("records")

        # 3a. in-sample ridge residuals
        fitted = ridge.predict(X)
        month_train = train["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
        t_train = train["order_purchase_timestamp"].to_numpy()
        result["residuals_in_sample"] = residual_diagnostics("in_sample", fitted, y - fitted, Z,
                                                             t_train, month_train, rng)

        # 3b. pooled out-of-fold validation residuals (raw target) + 4. log-target sensitivity
        oof, cols = [], {}
        log_rows, raw_rows = [], []
        for w, (fit, valid) in enumerate(folds):
            raw = fresh().fit(X.iloc[fit], y[fit])
            pr = raw.predict(X.iloc[valid])
            raw_rows.append(regression_metrics(y[valid], pr))
            lg = fresh().fit(X.iloc[fit], np.log1p(y[fit]))
            pl = np.expm1(lg.predict(X.iloc[valid]))
            log_rows.append(regression_metrics(y[valid], pl))
            oof.append(pd.DataFrame({"idx": valid, "pred": pr, "pred_log": pl}))
        oof = pd.concat(oof, ignore_index=True)
        idx = oof["idx"].to_numpy()
        Zv = Z[idx]  # full-train design columns (one-hot widths differ per fold)
        month_v, t_v = month_train[idx], t_train[idx]
        for tag, col in (("validation_oof", "pred"), ("validation_oof_logtarget", "pred_log")):
            pred = oof[col].to_numpy()
            result[f"residuals_{tag}"] = residual_diagnostics(tag, pred, y[idx] - pred, Zv, t_v, month_v, rng)

    summary = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in ("mae", "rmse", "r2")}
    result["log_target_sensitivity"] = {
        "raw_ridge_100_mean_validation": summary(raw_rows),
        "log1p_ridge_100_mean_validation": summary(log_rows),
        "per_window_raw": raw_rows, "per_window_log": log_rows,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("log_target_sensitivity",)}, indent=1, default=float))
    print(json.dumps({k: v for k, v in result["log_target_sensitivity"].items() if "mean" in k}, indent=1))


if __name__ == "__main__":
    main()
