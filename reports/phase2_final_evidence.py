"""Evidence for the final Phase 2 report: tables, number macros and pgfplots figures.

Run from the repository root:
    python reports/phase2_final_evidence.py

Reads the saved model outputs in results/phase2/{promise,handover} and the raw Olist tables. It fits no model.
Writes results/phase2/final/*.csv, reports/generated/numbers.tex (one macro per number quoted in the report) and
reports/generated/fig_*.tex (pgfplots, so the report compiles in an online LaTeX editor).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_classification import load_primary_training, temporal_folds  # noqa: E402
from phase2_promise import load_promise_table  # noqa: E402

RES = ROOT / "results" / "phase2"
OUT = RES / "final"
GEN = ROOT / "reports" / "generated"
GAMMA = 0.05
BAD_REVIEW = 2  # review score at or below this counts as a bad review
MONTH_LABEL = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
               1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}

macros: dict[str, str] = {}


def put(name, value, digits=None, pct=False, suffix=""):
    """Register a LaTeX macro. `pct` multiplies by 100; `digits` rounds."""
    if isinstance(value, str):
        text = value
    else:
        v = float(value) * (100 if pct else 1)
        text = f"{v:,.{digits}f}".replace(",", "{,}") if digits is not None else str(v)
    macros[name] = text + suffix


def load_json(path):
    return json.loads(Path(path).read_text())


def day_to_date(day):
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(day, unit="D")


# ---------------------------------------------------------------- promise engine
def promise_tables(full, train, folds):
    """Validation comparison per window, test per month, monthly late share and buffer level."""
    adaptive_val = pd.read_csv(RES / "promise/adaptive_validation.csv")
    eng = adaptive_val[(adaptive_val.model == "engine") & np.isclose(adaptive_val.gamma, GAMMA)].set_index("window")
    promise_by_id = full.set_index("order_id")
    rows = []
    for window, (_, valid) in enumerate(folds):
        v = promise_by_id.loc[train.iloc[valid]["order_id"]]
        rows.append({
            "window": window,
            "start": v["order_purchase_timestamp"].min().date().isoformat(),
            "end": v["order_purchase_timestamp"].max().date().isoformat(),
            "orders": len(v),
            "late_share": float(v["is_late"].mean()),
            "olist_mean_promise": float(v["olist_promise"].mean()),
            "olist_on_time": float(1 - v["is_late"].mean()),
            "engine_mean_promise": float(eng.loc[window, "mean_promise"]),
            "engine_on_time": float(eng.loc[window, "on_time"]),
        })
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "validation_windows_comparison.csv", index=False)

    # monthly late share (all delivered-cohort orders) and mean buffer level on scoring days
    full = full.copy()
    full["month"] = full["order_purchase_timestamp"].dt.to_period("M")
    monthly = full.groupby("month").agg(orders=("is_late", "size"), late_share=("is_late", "mean")).reset_index()
    trace = pd.read_csv(RES / "promise/adaptive_level_trace.csv")
    trace = trace[trace.model == "engine"]
    scoring_days = set()
    for _, valid in folds:
        scoring_days |= set(train.iloc[valid]["order_purchase_timestamp"].dt.normalize().map(lambda d: (d - pd.Timestamp("1970-01-01")).days))
    test_days = set(full[full.split == "test"]["order_purchase_timestamp"].dt.normalize().map(lambda d: (d - pd.Timestamp("1970-01-01")).days))
    trace = trace[trace.day.isin(scoring_days | test_days)].copy()
    trace["month"] = trace.day.map(lambda d: day_to_date(d).to_period("M"))
    level = trace.groupby("month")["level"].mean().rename("mean_buffer_level").reset_index()
    monthly = monthly.merge(level, on="month", how="left")
    monthly["month"] = monthly["month"].astype(str)
    monthly.to_csv(OUT / "monthly_late_share_and_buffer.csv", index=False)
    return val, monthly


def promise_numbers(val, monthly):
    test = load_json(RES / "promise/adaptive_test_results.json")
    fixed = load_json(RES / "promise/test_results.json")
    sel = load_json(RES / "promise/selection.json")
    adap = load_json(RES / "promise/adaptive_selection.json")
    put("EngineTestPromise", test["engine"]["mean_promise"], 2)
    put("EngineTestMedian", test["engine"]["median_promise"], 0)
    put("EngineTestOnTime", test["engine"]["on_time"], 2, pct=True)
    put("OlistTestPromise", test["olist"]["mean_promise"], 2)
    put("OlistTestMedian", test["olist"]["median_promise"], 0)
    put("OlistTestOnTime", test["olist"]["on_time"], 2, pct=True)
    put("TestPromiseGap", test["olist"]["mean_promise"] - test["engine"]["mean_promise"], 1)
    put("FixedTestPromise", fixed["selected"]["mean_promise"], 1)
    put("FixedTestOnTime", fixed["selected"]["on_time"], 1, pct=True)
    put("FixedLevel", sel["fixed_level"], 4)
    put("FixedValOnTime", sel["mean_validation_on_time_at_fixed_level"], 1, pct=True)
    put("MatchedEngine", fixed["matched_reliability_mean_promise"]["selected"], 1)
    put("MatchedGap", fixed["gate_1"]["test_matched_gap_days"], 1)
    put("SelectedGamma", adap["selected_gamma"]["engine"], 2)
    mean_val = adap["mean_validation"][adap["selected"]][str(adap["selected_gamma"]["engine"])]
    put("EngineValPromise", mean_val["mean_promise"], 1)
    put("EngineValOnTime", mean_val["on_time"], 1, pct=True)
    put("OlistValPromise", val["olist_mean_promise"].mean(), 1)
    put("OlistValOnTime", val["olist_on_time"].mean(), 1, pct=True)
    for i, word in enumerate(["A", "B", "C", "D", "E"]):
        r = val.iloc[i]
        put(f"Win{word}Start", r["start"])
        put(f"Win{word}End", r["end"])
        put(f"Win{word}Late", r["late_share"], 1, pct=True)
        put(f"Win{word}Olist", r["olist_mean_promise"], 1)
        put(f"Win{word}Engine", r["engine_mean_promise"], 1)
        put(f"Win{word}OlistOnTime", r["olist_on_time"], 1, pct=True)
        put(f"Win{word}EngineOnTime", r["engine_on_time"], 1, pct=True)
    put("PeakPromise", val["engine_mean_promise"].max(), 0)
    # regression diagnostics for the selected model and the best-MAE model
    summ = pd.read_csv(RES / "promise/validation_summary.csv").set_index("candidate")
    chosen = sel["selected"]
    put("SelName", chosen.replace("_", r"\_"))
    for tag, name in [("Sel", chosen), ("Forest", "forest_leaf20"), ("Ridge", "ridge_100"), ("Mean", "mean_baseline"),
                      ("Route", "route_baseline"), ("Linear", "linear_baseline"), ("Tree", "tree_depth6")]:
        put(f"{tag}MAE", summ.loc[name, "mean_mae"], 2)
        put(f"{tag}RMSE", summ.loc[name, "mean_rmse"], 2)
        put(f"{tag}RSq", summ.loc[name, "mean_r2"], 2)
        put(f"{tag}PromiseNinetyFive", summ.loc[name, "mean_promise_at_95"], 1)
    pred = pd.read_csv(RES / "promise/adaptive_test_predictions.csv")
    for tag, col in [("Engine", "mu_engine"), ("RouteBase", "mu_route")]:
        err = pred["need"] - pred[col]
        put(f"Test{tag}MAE", err.abs().mean(), 2)
        put(f"Test{tag}RMSE", float(np.sqrt((err ** 2).mean())), 2)
        put(f"Test{tag}RSq", 1 - (err ** 2).sum() / ((pred["need"] - pred["need"].mean()) ** 2).sum(), 2)
    put("TestOrders", len(pred), 0)
    # one-standard-error rule on the adaptive engine: per-window mean promise at each candidate's chosen gamma
    asel = load_json(RES / "promise/adaptive_selection.json")
    av = pd.read_csv(RES / "promise/adaptive_validation.csv")
    rows = []
    for name, g in asel["gamma_by_candidate"].items():
        per = av[av.candidate == name].groupby("gamma")[["on_time", "mean_promise"]].mean()
        g_used = g if g is not None else per["on_time"].idxmax()
        w = av[(av.candidate == name) & np.isclose(av.gamma, g_used)].sort_values("window")
        rows.append({"candidate": name, "eligible": g is not None, "gamma": g_used,
                     "mean_on_time": w["on_time"].mean(), "mean_promise": w["mean_promise"].mean(),
                     "per_window": list(w["mean_promise"])})
    tab = pd.DataFrame(rows).set_index("candidate")
    best = tab[tab.eligible & ~tab.index.isin(["route_baseline"])]["mean_promise"].idxmin()
    for name in tab.index:
        gap = np.array(tab.loc[name, "per_window"]) - np.array(tab.loc[best, "per_window"])
        tab.loc[name, "gap_to_best"] = gap.mean()
        tab.loc[name, "se_of_gap"] = gap.std(ddof=1) / np.sqrt(len(gap))
    tab.drop(columns="per_window").to_csv(OUT / "promise_adaptive_selection.csv")
    put("BestName", best.replace("_", r"\_"))
    for tag, name in [("Sel", chosen), ("Forest", "forest_leaf20"), ("Linear", "linear_baseline"),
                      ("RidgeOne", "ridge_1"), ("RouteAd", "route_baseline")]:
        put(f"{tag}AdOnTime", tab.loc[name, "mean_on_time"], 1, pct=True)
        put(f"{tag}AdPromise", tab.loc[name, "mean_promise"], 1)
        put(f"{tag}AdGap", tab.loc[name, "gap_to_best"], 2)
        put(f"{tag}AdGapSE", tab.loc[name, "se_of_gap"], 2)
    lin = tab.loc[[n for n in ["linear_baseline", "ridge_1", "ridge_10", "ridge_100"]]]
    put("LinOnTimeLow", lin["mean_on_time"].min(), 1, pct=True)
    put("LinOnTimeHigh", lin["mean_on_time"].max(), 1, pct=True)
    put("LinPromiseLow", lin["mean_promise"].min(), 1)
    put("LinPromiseHigh", lin["mean_promise"].max(), 1)
    fixed_sel = sel.get("fixed_buffer_rule_choice", "")
    put("FixedRuleName", fixed_sel.replace("_", r"\_"))
    cls = pd.read_csv(RES / "handover/validation_windows.csv")
    cls = cls[cls.variant == "primary"].pivot(index="window", columns="candidate", values="pr_auc")
    gap = cls["forest_leaf20"] - cls["logistic"]
    put("ClsGap", gap.mean(), 3)
    put("ClsGapSE", gap.std(ddof=1) / np.sqrt(len(gap)), 3)
    put("ClsBestWins", int((gap > 0).sum()), 0)
    pm = pd.read_csv(RES / "promise/adaptive_test_per_month.csv")
    pm.to_csv(OUT / "promise_test_per_month.csv", index=False)
    for _, r in pm.iterrows():
        tag = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug"}[int(r["month"][-2:])]
        put(f"Test{tag}Engine", r["engine_mean_promise"], 1)
        put(f"Test{tag}Olist", r["olist_mean_promise"], 1)
        put(f"Test{tag}EngineOnTime", r["engine_on_time"], 1, pct=True)
        put(f"Test{tag}OlistOnTime", r["olist_on_time"], 1, pct=True)
    # headline late-share facts
    m = monthly.set_index("month")
    put("BlackFridayLate", val.iloc[1]["late_share"], 0, pct=True)
    put("NovLate", m.loc["2017-11", "late_share"], 1, pct=True)
    put("MarLate", m.loc["2018-03", "late_share"], 1, pct=True)
    put("FebMarLate", val.iloc[4]["late_share"], 0, pct=True)
    normal = m.loc[["2017-03", "2017-04", "2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10"]]
    put("NormalLate", (normal["late_share"] * normal["orders"]).sum() / normal["orders"].sum(), 0, pct=True)
    put("AllOrders", int(monthly["orders"].sum()), 0)


# ---------------------------------------------------------------- classifier
def classifier_numbers():
    sel = load_json(RES / "handover/selection.json")
    test = pd.read_csv(RES / "handover/test_metrics.csv").set_index("ranking")
    val = pd.read_csv(RES / "handover/validation_summary.csv")
    gains = pd.read_csv(RES / "handover/test_gains_curve.csv")
    imp = pd.read_csv(RES / "handover/primary_permutation_importance.csv")
    cov = sel["coverage"]
    put("CohortOrders", cov["handover_cohort"], 0)
    put("ExcludedOrders", cov["excluded_no_valid_handover"], 0)
    put("ClsTestPR", test.loc["model", "pr_auc"], 2)
    put("ClsTestROC", test.loc["model", "roc_auc"], 2)
    put("ClsTestLate", test.loc["model", "late_rate"], 1, pct=True)
    put("ClsTestPrec", test.loc["model", "precision"], 1, pct=True)
    put("ClsTestRec", test.loc["model", "recall"], 0, pct=True)
    put("ClsTestFone", test.loc["model", "f1"], 2)
    put("ClsTestMCC", test.loc["model", "mcc"], 2)
    put("SlackTestPR", test.loc["slack_rule", "pr_auc"], 2)
    put("SlackTestROC", test.loc["slack_rule", "roc_auc"], 2)
    put("ClsNetBenefit", test.loc["model", "net_benefit_per_order"], 3)
    put("ActShare", sel["variants"]["primary"]["k"], 0, pct=True)
    put("BenefitRatio", sel["benefit_ratio"], 0)
    primary = val[val.variant == "primary"].set_index("candidate")
    for tag, name in [("Forest", "forest_leaf20"), ("Logit", "logistic"), ("LogitBal", "logistic_balanced"), ("Tree", "tree_depth6")]:
        put(f"Val{tag}PR", primary.loc[name, "pr_auc"], 3)
        put(f"Val{tag}ROC", primary.loc[name, "roc_auc"], 3)
    put("ValForestRecallTen", primary.loc["forest_leaf20", "recall_top10"], 0, pct=True)
    checkout = val[val.variant == "checkout"].set_index("candidate")
    put("CheckoutPR", checkout.loc["forest_leaf20", "pr_auc"], 2)
    diff = sel["variants"]["primary"]["value_test"]["model_minus_slack_rule"]["pr_auc"]
    put("ValDiffPR", diff["difference"], 3)
    put("ValDiffLow", diff["ci_low"], 3)
    put("ValDiffHigh", diff["ci_high"], 3)
    # gains at fixed shares
    g = gains.set_index("share_acted")
    for share in (0.01, 0.05, 0.10, 0.20):
        tag = {0.01: "One", 0.05: "Five", 0.10: "Ten", 0.20: "Twenty"}[share]
        row = g.iloc[(g.index - share).to_series().abs().argmin()]
        put(f"Gain{tag}Model", row["model"], 0, pct=True)
        put(f"Gain{tag}Slack", row["slack_rule"], 0, pct=True)
        put(f"Prec{tag}Model", row["model"] * test.loc["model", "late_rate"] / share, 0, pct=True)
    gains[gains.variant == "primary"].to_csv(OUT / "classifier_gains_test.csv", index=False)
    top = imp.sort_values("mean_pr_auc_drop", ascending=False).head(6)
    top.to_csv(OUT / "classifier_importance_top.csv", index=False)
    for i, (_, r) in enumerate(top.iterrows()):
        put(f"Imp{'ABCDEF'[i]}Name", r["feature"].replace("_", r"\_"))
        put(f"Imp{'ABCDEF'[i]}Drop", r["mean_pr_auc_drop"], 3)
    monthly = pd.read_csv(RES / "handover/test_monthly_counts.csv")
    monthly.to_csv(OUT / "classifier_test_monthly.csv", index=False)
    for _, r in monthly.iterrows():
        tag = MONTH_LABEL[int(r["month"][-2:])]
        put(f"Test{tag}Orders", r["orders"], 0)
        put(f"Test{tag}LateCount", r["late"], 0)
        put(f"Test{tag}Caught", r["caught"], 0)
        put(f"Test{tag}Acted", r["acted"], 0)
    return gains


# ---------------------------------------------------------------- supporting facts
def supporting_facts(cohort_ids):
    orders = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                         parse_dates=["order_purchase_timestamp", "order_delivered_carrier_date",
                                      "order_delivered_customer_date", "order_estimated_delivery_date"])
    orders = orders[orders.order_id.isin(cohort_ids)].copy()
    ok = orders["order_delivered_carrier_date"].notna() & (orders["order_delivered_carrier_date"] > orders["order_purchase_timestamp"]) & (
        orders["order_delivered_carrier_date"] < orders["order_delivered_customer_date"])
    sub = orders[ok]
    secs = 86400.0
    total = (sub["order_delivered_customer_date"] - sub["order_purchase_timestamp"]).dt.total_seconds() / secs
    carrier = (sub["order_delivered_customer_date"] - sub["order_delivered_carrier_date"]).dt.total_seconds() / secs
    share = float(np.cov(carrier, total)[0, 1] / np.var(total, ddof=1))
    put("CarrierShare", share, 0, pct=True)
    put("CarrierMean", carrier.mean(), 1)
    put("TotalMean", total.mean(), 1)

    reviews = pd.read_csv(ROOT / "data/olist_order_reviews_dataset.csv", usecols=["order_id", "review_score"])
    score = reviews.groupby("order_id")["review_score"].min()
    orders["late"] = (orders["order_delivered_customer_date"].dt.normalize() > orders["order_estimated_delivery_date"].dt.normalize()).astype(int)
    orders["score"] = orders["order_id"].map(score)
    rated = orders.dropna(subset=["score"])
    bad = rated.groupby("late")["score"].apply(lambda s: float((s <= BAD_REVIEW).mean()))
    put("BadReviewOnTime", bad[0], 0, pct=True)
    put("BadReviewLate", bad[1], 0, pct=True)
    put("BadReviewRatio", bad[1] / bad[0], 1)
    put("RatedOrders", len(rated), 0)

    customers = pd.read_csv(ROOT / "data/olist_customers_dataset.csv", usecols=["customer_id", "customer_unique_id"])
    orders = orders.merge(customers, on="customer_id", how="left").sort_values("order_purchase_timestamp")
    per = orders.groupby("customer_unique_id")
    n_orders = per.size()
    put("RepeatShare", float((n_orders >= 2).mean()), 1, pct=True)
    put("UniqueCustomers", len(n_orders), 0)
    first = per.head(1).set_index("customer_unique_id")
    first["repeat"] = (n_orders.reindex(first.index) >= 2).astype(int)
    rep = first.groupby("late")["repeat"].mean()
    put("RepeatAfterOnTime", rep[0], 1, pct=True)
    put("RepeatAfterLate", rep[1], 1, pct=True)
    pd.DataFrame({"fact": ["carrier_share_of_variance", "bad_review_on_time", "bad_review_late",
                           "repeat_after_on_time_first", "repeat_after_late_first"],
                  "value": [share, bad[0], bad[1], rep[0], rep[1]]}).to_csv(OUT / "supporting_facts.csv", index=False)


# ---------------------------------------------------------------- figures
def coords(xs, ys):
    return " ".join(f"({x},{y:.4f})" for x, y in zip(xs, ys))


def fig_monthly(monthly):
    m = monthly[(monthly["month"] >= "2017-03") & (monthly["month"] <= "2018-08")].reset_index(drop=True)
    labels = ",".join(pd.Timestamp(x + "-01").strftime("%b%y") for x in m["month"])
    idx = list(range(len(m)))
    late = coords(idx, m["late_share"] * 100)
    lv = m.dropna(subset=["mean_buffer_level"])
    level = coords([i for i in idx if not np.isnan(m.loc[i, "mean_buffer_level"])], lv["mean_buffer_level"] * 100)
    text = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=\linewidth,height=6.2cm,ybar,bar width=7pt,ymin=0,ymax=26,ylabel={{Orders late (\%)}},
 symbolic x coords={{{labels}}},xtick=data,x tick label style={{rotate=60,anchor=east,font=\scriptsize}},
 enlarge x limits=0.04,legend style={{at={{(0.02,0.97)}},anchor=north west,font=\scriptsize}},
 axis y line*=left]
\addplot[fill=gray!55,draw=none] coordinates {{{" ".join(f"({pd.Timestamp(x + '-01').strftime('%b%y')},{y:.3f})" for x, y in zip(m['month'], m['late_share'] * 100))}}};
\addlegendentry{{Orders late (left axis)}}
\end{{axis}}
\begin{{axis}}[width=\linewidth,height=6.2cm,ymin=80,ymax=100,axis y line*=right,axis x line=none,ylabel={{Buffer level (\%)}},
 symbolic x coords={{{labels}}},xtick=data,legend style={{at={{(0.02,0.80)}},anchor=north west,font=\scriptsize}},enlarge x limits=0.04]
\addplot[thick,mark=*,mark size=1.4pt] coordinates {{{" ".join(f"({pd.Timestamp(m.loc[i, 'month'] + '-01').strftime('%b%y')},{m.loc[i, 'mean_buffer_level'] * 100:.3f})" for i in idx if not np.isnan(m.loc[i, 'mean_buffer_level']))}}};
\addlegendentry{{Engine buffer level (right axis)}}
\end{{axis}}
\end{{tikzpicture}}"""
    (GEN / "fig_monthly.tex").write_text(text)


def fig_windows(val):
    names = ",".join(f"W{i + 1}" for i in range(len(val)))
    olist = " ".join(f"(W{i + 1},{v:.2f})" for i, v in enumerate(val["olist_mean_promise"]))
    engine = " ".join(f"(W{i + 1},{v:.2f})" for i, v in enumerate(val["engine_mean_promise"]))
    text = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.92\linewidth,height=5.4cm,ybar,bar width=9pt,ymin=0,ylabel={{Mean promise (days)}},
 symbolic x coords={{{names}}},xtick=data,enlarge x limits=0.12,legend style={{at={{(0.02,0.97)}},anchor=north west,font=\scriptsize}}]
\addplot[fill=gray!55,draw=none] coordinates {{{olist}}};
\addplot[fill=blue!55!black,draw=none] coordinates {{{engine}}};
\legend{{Olist promise,Promise engine}}
\end{{axis}}
\end{{tikzpicture}}"""
    (GEN / "fig_windows.tex").write_text(text)


def fig_gains(gains):
    g = gains[gains.variant == "primary"]
    step = max(len(g) // 40, 1)
    g = g.iloc[::step]
    plots = [("model", "Late-warning model", "thick,blue!60!black"), ("slack_rule", "Remaining-slack rule", "thick,dashed,orange!80!black"),
             ("random", "Random list", "thin,gray")]
    body = "\n".join(rf"\addplot[{style},no marks] coordinates {{{coords(g['share_acted'] * 100, g[col] * 100)}}};" + f"\n\\addlegendentry{{{label}}}"
                     for col, label, style in plots)
    text = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.92\linewidth,height=5.6cm,xlabel={{Share of orders acted on (\%)}},ylabel={{Late orders caught (\%)}},
 xmin=0,xmax=100,ymin=0,ymax=100,legend style={{at={{(0.97,0.05)}},anchor=south east,font=\scriptsize}}]
{body}
\end{{axis}}
\end{{tikzpicture}}"""
    (GEN / "fig_gains.tex").write_text(text)


def write_macros():
    lines = ["% Generated by reports/phase2_final_evidence.py: do not edit by hand."]
    for name, text in sorted(macros.items()):
        lines.append(rf"\newcommand{{\{name}}}{{{text}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    train, spec = load_primary_training()
    folds = temporal_folds(train, spec)
    full, _, _ = load_promise_table()
    val, monthly = promise_tables(full, train, folds)
    promise_numbers(val, monthly)
    gains = classifier_numbers()
    supporting_facts(set(full["order_id"]))
    fig_monthly(monthly)
    fig_windows(val)
    fig_gains(gains)
    write_macros()
    print(f"{len(macros)} macros written to {GEN / 'numbers.tex'}")


if __name__ == "__main__":
    main()
