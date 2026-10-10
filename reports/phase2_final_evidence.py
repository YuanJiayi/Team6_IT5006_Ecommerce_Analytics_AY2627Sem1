"""Evidence for the final Phase 2 report: tables, number macros and pgfplots figures.

Run from the repository root:
    python reports/phase2_final_evidence.py

Reads the saved model outputs in results/phase2/{eta,handover} and the raw Olist tables. It fits no model.
Writes results/phase2/final/*.csv, reports/generated/numbers.tex (one macro per number quoted in the report) and
reports/generated/fig_*.tex (pgfplots and plain tables, so the report compiles in an online LaTeX editor).

Story (NEW DIRECTION, docs/phase2_handoff.md): the regression is Pratik's two-stage delivery-time estimate
(checkout estimate, updated at carrier handover); the classifier is the handover late-warning model. The promise
engine and the remaining-slack rule are not part of this report (stretch goal / dropped comparison).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_eta import CHECKOUT_STAGE, HANDOVER_STAGE, load_stage_table  # noqa: E402

RES = ROOT / "results" / "phase2"
OUT = RES / "final"
GEN = ROOT / "reports" / "generated"
BAD_REVIEW = 2  # review score at or below this counts as a bad review
MONTH_LABEL = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
               1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
STAGE_TAG = {CHECKOUT_STAGE: "Checkout", HANDOVER_STAGE: "Handover"}

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


def escape(name):
    return str(name).replace("_", r"\_")


# ---------------------------------------------------------------- regression (two-stage estimate)
def regression_numbers():
    sel = load_json(RES / "eta/selection.json")
    cv = pd.read_csv(RES / "eta/cv_summary.csv")
    ladder = pd.read_csv(RES / "eta/ladder_summary.csv").set_index("stage")
    test = pd.read_csv(RES / "eta/test_metrics.csv")
    rand = pd.read_csv(RES / "eta/random_benchmark.csv")

    cov = sel["coverage"]
    put("CohortOrders", cov["handover_cohort"], 0)
    put("ExcludedOrders", cov["excluded_no_valid_handover"], 0)
    put("EtaTrainOrders", cov["train_orders"], 0)
    put("EtaTestOrders", cov["test_orders"], 0)

    # Information ladder: one fixed model (ridge alpha 100), inputs added by prediction point.
    for stage, tag in [("checkout", "LadderCheckout"), ("checkout_transit", "LadderTransit"),
                       ("handover_elapsed", "LadderElapsed"), ("handover", "LadderHandover")]:
        row = ladder.loc[stage]
        put(f"{tag}MAE", row["mean_mae"], 2)
        put(f"{tag}RSq", row["mean_r2"], 3)
        put(f"{tag}MinRSq", row["min_r2"], 3)

    selected = {CHECKOUT_STAGE: sel["checkout_selected_candidate"], HANDOVER_STAGE: sel["selected_candidate"]}
    for stage, tag in STAGE_TAG.items():
        put(f"{tag}SelName", escape(selected[stage]))
        put(f"{tag}BestName", escape(sel["lowest_mae_candidate"][stage]))
        stage_cv = cv[cv["stage"] == stage].set_index("candidate")
        for name_tag, name in [("Sel", selected[stage]), ("Best", sel["lowest_mae_candidate"][stage]),
                               ("Mean", "mean_baseline")]:
            row = stage_cv.loc[name]
            put(f"{tag}{name_tag}MAE", row["mean_mae"], 2)
            put(f"{tag}{name_tag}RMSE", row["mean_rmse"], 2)
            put(f"{tag}{name_tag}RSq", row["mean_r2"], 3)
        if HANDOVER_STAGE == stage and "elapsed_plus_route_baseline" in stage_cv.index:
            row = stage_cv.loc["elapsed_plus_route_baseline"]
            put(f"{tag}RuleMAE", row["mean_mae"], 2)
            put(f"{tag}RuleRSq", row["mean_r2"], 3)
        # stage-2 one-SE line-up: gap and SE of the selected candidate to the best
        s2 = {r["candidate"]: r for r in sel["stage2"][stage]}
        if selected[stage] in s2:
            put(f"{tag}SelGap", s2[selected[stage]]["gap_to_best"], 3)
            put(f"{tag}SelGapSE", s2[selected[stage]]["se_of_gap"], 3)

    # Later period (post hoc), each stage's own selected model, overall.
    test_overall = test[test["scope"] == "overall"].set_index(["stage", "candidate"])
    for stage, tag in STAGE_TAG.items():
        row = test_overall.loc[(stage, selected[stage])]
        put(f"{tag}TestMAE", row["mae"], 2)
        put(f"{tag}TestRMSE", row["rmse"], 2)
        put(f"{tag}TestRSq", row["r2"], 3)
        put(f"{tag}TestBias", row["mean_signed_error"], 2)
    put("TestOrders", int(test_overall.loc[(HANDOVER_STAGE, selected[HANDOVER_STAGE]), "orders"]), 0)
    test[test["scope"] == "overall"].to_csv(OUT / "eta_test_overall.csv", index=False)
    monthly = test[test["scope"] != "overall"].rename(columns={"scope": "month"})
    monthly.to_csv(OUT / "eta_test_monthly.csv", index=False)

    # Same-period benchmark (random split; secondary), each stage's own selected model.
    rand_i = rand.set_index(["stage", "candidate"])
    for stage, tag in STAGE_TAG.items():
        row = rand_i.loc[(stage, selected[stage])]
        put(f"{tag}RandValRSq", row["cv_mean_r2"], 3)
        put(f"{tag}RandTestRSq", row["test_r2"], 3)
        put(f"{tag}RandTestMAE", row["test_mae"], 2)

    # Feature importance / coefficients of each stage's own selected model.
    for stage, tag, stem in [(CHECKOUT_STAGE, "Checkout", "checkout_selected"), (HANDOVER_STAGE, "Handover", "selected")]:
        coef_path, imp_path = RES / f"eta/{stem}_coefficients.csv", RES / f"eta/{stem}_permutation_importance.csv"
        if coef_path.exists():
            coef = pd.read_csv(coef_path)
            # One-hot state/category dummies dominate by magnitude but are not individually interpretable;
            # report the top numeric/engineered inputs instead (the categorical columns remain in the model).
            coef = coef[~coef["feature"].str.startswith("category__")]
            top = (coef.groupby("feature")["coefficient"].mean().abs().sort_values(ascending=False).head(3))
            kind = "coefficient (mean $|$weight$|$ over windows, numeric and engineered inputs; one-hot " \
                   "state/category dummies excluded from this ranking)"
        else:
            imp = pd.read_csv(imp_path)
            top = imp.groupby("feature")["importance_mean"].mean().sort_values(ascending=False).head(3)
            kind = "permutation importance (mean MAE rise over windows)"
        put(f"{tag}ImpKind", kind)
        for i, (feature, value) in enumerate(top.items()):
            put(f"{tag}Imp{'ABC'[i]}Name", escape(feature))
            put(f"{tag}Imp{'ABC'[i]}Val", value, 3)

    cv.to_csv(OUT / "eta_cv_summary.csv", index=False)
    return sel, selected


# ---------------------------------------------------------------- classifier
def classifier_numbers():
    sel = load_json(RES / "handover/selection.json")
    test = pd.read_csv(RES / "handover/test_metrics.csv")
    test = test[test["variant"] == "primary"].set_index("ranking")
    val = pd.read_csv(RES / "handover/validation_summary.csv")
    gains = pd.read_csv(RES / "handover/test_gains_curve.csv")
    imp = pd.read_csv(RES / "handover/primary_permutation_importance.csv")
    cov = sel["coverage"]
    primary = sel["variants"]["primary"]
    chosen, best = primary["selected"], primary["best_pr_auc_candidate"]

    put("CohortOrders", cov["handover_cohort"], 0)
    put("ExcludedOrders", cov["excluded_no_valid_handover"], 0)
    put("ClsSelName", escape(chosen))
    put("ClsBestName", escape(best))
    put("ClsTestPR", test.loc["model", "pr_auc"], 2)
    put("ClsTestROC", test.loc["model", "roc_auc"], 2)
    put("ClsTestLate", test.loc["model", "late_rate"], 1, pct=True)
    put("ClsTestPrec", test.loc["model", "precision"], 1, pct=True)
    put("ClsTestRec", test.loc["model", "recall"], 0, pct=True)
    put("ClsTestFone", test.loc["model", "f1"], 2)
    put("ClsTestMCC", test.loc["model", "mcc"], 2)
    put("ClsNetBenefit", test.loc["model", "net_benefit_per_order"], 3)
    put("ActShare", primary["k"], 0, pct=True)
    put("BenefitRatio", sel["benefit_ratio"], 0)

    primary_val = val[val["variant"] == "primary"].set_index("candidate")
    put("ValSelPR", primary_val.loc[chosen, "pr_auc"], 3)
    put("ValSelROC", primary_val.loc[chosen, "roc_auc"], 3)
    put("ValBestPR", primary_val.loc[best, "pr_auc"], 3)
    put("ValSelRecallTen", primary_val.loc[chosen, "recall_top10"], 0, pct=True)
    # stage-2 one-SE line-up: gap and SE of the selected candidate to the best
    s2 = {r["candidate"]: r for r in primary["stage2"]}
    put("ClsSelGap", s2[chosen]["gap_to_best"], 3)
    put("ClsSelGapSE", s2[chosen]["se_of_gap"], 3)
    # the group-plain baseline for the selected model's own group (the group's first lineup entry)
    group_of = {c: g["group"] for g in sel["groups"] for c in g["configs"]}
    plain = next(c for c in primary["lineup"] if group_of.get(c) == group_of.get(chosen))
    put("ClsPlainName", escape(plain))
    put("ClsPlainPR", primary_val.loc[plain, "pr_auc"], 3)

    checkout = val[val["variant"] == "checkout"].set_index("candidate")
    checkout_sel = sel["checkout_variant"]["selected"]
    put("ClsCheckoutSelName", escape(checkout_sel))
    put("CheckoutPR", checkout.loc[checkout_sel, "pr_auc"], 3)

    # gains at fixed shares (model only; the slack rule is not a report comparison)
    g = gains[gains["variant"] == "primary"].set_index("share_acted")
    for share in (0.01, 0.05, 0.10, 0.20):
        tag = {0.01: "One", 0.05: "Five", 0.10: "Ten", 0.20: "Twenty"}[share]
        row = g.iloc[(g.index - share).to_series().abs().argmin()]
        put(f"Gain{tag}Model", row["model"], 0, pct=True)
        put(f"Prec{tag}Model", row["model"] * test.loc["model", "late_rate"] / share, 0, pct=True)
    gains[gains["variant"] == "primary"][["share_acted", "model", "random"]].to_csv(
        OUT / "classifier_gains_test.csv", index=False)

    top = imp.sort_values("mean_pr_auc_drop", ascending=False).head(6)
    top.to_csv(OUT / "classifier_importance_top.csv", index=False)
    for i, (_, r) in enumerate(top.iterrows()):
        put(f"Imp{'ABCDEF'[i]}Name", escape(r["feature"]))
        put(f"Imp{'ABCDEF'[i]}Drop", r["mean_pr_auc_drop"], 3)

    monthly = pd.read_csv(RES / "handover/test_monthly_counts.csv")
    monthly = monthly[monthly["variant"] == "primary"] if "variant" in monthly.columns else monthly
    monthly.to_csv(OUT / "classifier_test_monthly.csv", index=False)
    for _, r in monthly.iterrows():
        tag = MONTH_LABEL[int(r["month"][-2:])]
        put(f"Test{tag}Orders", r["orders"], 0)
        put(f"Test{tag}LateCount", r["late"], 0)
        put(f"Test{tag}Caught", r["caught"], 0)
        put(f"Test{tag}Acted", r["acted"], 0)
    val.to_csv(OUT / "classifier_validation_summary.csv", index=False)
    return gains


# ---------------------------------------------------------------- supporting facts and monthly late share
def monthly_late_share(table):
    full = table.copy()
    full["month"] = full["order_purchase_timestamp"].dt.to_period("M")
    monthly = full.groupby("month").agg(orders=("is_late", "size"), late_share=("is_late", "mean")).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly.to_csv(OUT / "monthly_late_share.csv", index=False)
    m = monthly.set_index("month")
    put("BlackFridayLate", m.loc["2017-11", "late_share"], 0, pct=True)
    put("FebMarLate", float(np.average(m.loc[["2018-02", "2018-03"], "late_share"],
                                       weights=m.loc[["2018-02", "2018-03"], "orders"])), 0, pct=True)
    normal_months = ["2017-03", "2017-04", "2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10"]
    normal = m.loc[[x for x in normal_months if x in m.index]]
    put("NormalLate", float(np.average(normal["late_share"], weights=normal["orders"])), 0, pct=True)
    put("AllOrders", int(monthly["orders"].sum()), 0)
    return monthly


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


# ---------------------------------------------------------------- figures and tables
def coords(xs, ys):
    return " ".join(f"({x},{y:.4f})" for x, y in zip(xs, ys))


def fig_monthly(monthly):
    m = monthly[(monthly["month"] >= "2017-03") & (monthly["month"] <= "2018-08")].reset_index(drop=True)
    labels = ",".join(pd.Timestamp(x + "-01").strftime("%b%y") for x in m["month"])
    text = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.95\linewidth,height=5.6cm,ybar,bar width=7pt,ymin=0,ymax=26,ylabel={{Orders late (\%)}},
 symbolic x coords={{{labels}}},xtick=data,x tick label style={{rotate=60,anchor=east,font=\scriptsize}},
 enlarge x limits=0.04]
\addplot[fill=gray!55,draw=none] coordinates {{{" ".join(f"({pd.Timestamp(x + '-01').strftime('%b%y')},{y:.3f})" for x, y in zip(m['month'], m['late_share'] * 100))}}};
\end{{axis}}
\end{{tikzpicture}}"""
    (GEN / "fig_monthly.tex").write_text(text)


def fig_ladder(ladder_windows):
    """Mean MAE by validation window, checkout vs handover: the two-stage improvement."""
    piv = ladder_windows.pivot_table(index="window", columns="stage", values="mae", aggfunc="mean")
    names = ",".join(f"W{i + 1}" for i in piv.index)
    checkout = " ".join(f"(W{i + 1},{v:.2f})" for i, v in zip(piv.index, piv["checkout"]))
    handover = " ".join(f"(W{i + 1},{v:.2f})" for i, v in zip(piv.index, piv["handover"]))
    text = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.92\linewidth,height=5.4cm,ybar,bar width=9pt,ymin=0,ylabel={{Mean validation MAE (days)}},
 symbolic x coords={{{names}}},xtick=data,enlarge x limits=0.12,legend style={{at={{(0.02,0.03)}},anchor=south west,font=\scriptsize}}]
\addplot[fill=gray!55,draw=none] coordinates {{{checkout}}};
\addplot[fill=blue!55!black,draw=none] coordinates {{{handover}}};
\legend{{Checkout estimate,Handover update}}
\end{{axis}}
\end{{tikzpicture}}"""
    (GEN / "fig_ladder.tex").write_text(text)


def fig_gains(gains):
    g = gains[gains.variant == "primary"]
    step = max(len(g) // 40, 1)
    g = g.iloc[::step]
    plots = [("model", "Late-warning model", "thick,blue!60!black"), ("random", "Random list", "thin,gray")]
    body = "\n".join(rf"\addplot[{style},no marks] coordinates {{{coords(g['share_acted'] * 100, g[col] * 100)}}};" + f"\n\\addlegendentry{{{label}}}"
                     for col, label, style in plots)
    text = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.92\linewidth,height=5.6cm,xlabel={{Share of orders acted on (\%)}},ylabel={{Late orders caught (\%)}},
 xmin=0,xmax=100,ymin=0,ymax=100,legend style={{at={{(0.97,0.05)}},anchor=south east,font=\scriptsize}}]
{body}
\end{{axis}}
\end{{tikzpicture}}"""
    (GEN / "fig_gains.tex").write_text(text)


def latex_table(frame, columns, header, caption, label, floatfmt="{:.3f}"):
    numeric = {c: pd.api.types.is_numeric_dtype(frame[c]) for c in columns}
    cols = "".join("r" if numeric[c] else "l" for c in columns)
    body_rows = []
    for _, r in frame.iterrows():
        cells = [floatfmt.format(r[c]) if numeric[c] else str(r[c]).replace("_", r"\_") for c in columns]
        body_rows.append(" & ".join(cells) + r" \\")
    text = (r"\begin{table}[H]" "\n" r"\centering\small" "\n" rf"\caption{{{caption}}}" "\n"
            rf"\label{{{label}}}" "\n" rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}" "\n" r"\toprule" "\n"
            + " & ".join(header) + r" \\" "\n" r"\midrule" "\n" + "\n".join(body_rows) + "\n"
            r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}")
    return text


def fig_appendix_regression(cv):
    rows = []
    for stage, stage_label in [(CHECKOUT_STAGE, "Checkout"), (HANDOVER_STAGE, "Handover")]:
        s = cv[cv["stage"] == stage].copy()
        s.insert(0, "Stage", stage_label)
        rows.append(s[["Stage", "candidate", "algorithm", "mean_mae", "mean_rmse", "mean_r2"]])
    frame = pd.concat(rows, ignore_index=True)
    frame["candidate"] = frame["candidate"].astype(str)
    frame["algorithm"] = frame["algorithm"].astype(str)
    text = latex_table(frame, ["Stage", "candidate", "algorithm", "mean_mae", "mean_rmse", "mean_r2"],
                       ["Stage", "Candidate", "Family", "MAE", "RMSE", "$R^2$"],
                       "Every regression candidate, mean over five validation windows.", "tab:app-regression")
    (GEN / "fig_appendix_regression.tex").write_text(text)


def fig_appendix_classifier(val):
    s = val[val["variant"] == "primary"][["candidate", "pr_auc", "roc_auc", "recall_top10"]].copy()
    s["candidate"] = s["candidate"].astype(str)
    text = latex_table(s, ["candidate", "pr_auc", "roc_auc", "recall_top10"],
                       ["Candidate", "PR-AUC", "ROC-AUC", "Recall at top 10\\%"],
                       "Every classifier candidate, mean over five validation windows (handover stage).",
                       "tab:app-classifier")
    (GEN / "fig_appendix_classifier.tex").write_text(text)


def write_macros():
    lines = ["% Generated by reports/phase2_final_evidence.py: do not edit by hand."]
    for name, text in sorted(macros.items()):
        lines.append(rf"\newcommand{{\{name}}}{{{text}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    table, _, _ = load_stage_table()
    sel, selected = regression_numbers()
    gains = classifier_numbers()
    monthly = monthly_late_share(table)
    supporting_facts(set(table["order_id"]))
    fig_monthly(monthly)
    fig_ladder(pd.read_csv(RES / "eta/ladder_windows.csv"))
    fig_gains(gains)
    fig_appendix_regression(pd.read_csv(RES / "eta/cv_summary.csv"))
    fig_appendix_classifier(pd.read_csv(RES / "handover/validation_summary.csv"))
    write_macros()
    print(f"{len(macros)} macros written to {GEN / 'numbers.tex'}")


if __name__ == "__main__":
    main()
