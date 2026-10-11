"""Evidence for the final Phase 2 report: tables, number macros and pgfplots figures.

Run from the Phase2 folder:
    python reports/phase2_final_evidence.py

Reads the saved model outputs in results/phase2/{eta,handover} and the raw Olist tables. It fits no model.
Writes results/phase2/final/*.csv and reports/generated/*.tex, then refreshes only the
marked evidence blocks in reports/phase2_report_v4.tex so it also compiles standalone.

The regression estimates delivery time at checkout and updates it at carrier handover;
the classifier ranks late-order risk at handover.
"""

from __future__ import annotations

import json
import re
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
REPORT = ROOT / "reports" / "phase2_report_v4.tex"
BAD_REVIEW = 2  # review score at or below this counts as a bad review
MONTH_LABEL = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
               1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
STAGE_TAG = {CHECKOUT_STAGE: "Checkout", HANDOVER_STAGE: "Handover"}
REPORT_REVIEW_SHARE = 0.10  # Illustrative workload, not the model-selection objective.

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
def regression_numbers(table):
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
    for split, tag in [("train", "EtaTrain"), ("test", "EtaTest")]:
        delivery = table.loc[table["split"].eq(split), "delivery_days"]
        put(f"{tag}DeliveryMean", delivery.mean(), 1)
        put(f"{tag}DeliverySD", delivery.std(), 1)

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

    # Later period, each stage's own selected model, overall.
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
    from phase2_handover_classifier import monthly_flags
    from sklearn.metrics import matthews_corrcoef
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
    scores = pd.read_csv(RES / "handover/test_scores.csv")
    labels = scores["is_late"].to_numpy().astype(bool)
    flags = monthly_flags(scores["score"].to_numpy(), scores["month"].to_numpy(), REPORT_REVIEW_SHARE)
    caught, flagged = int((flags & labels).sum()), int(flags.sum())
    put("ClsTestPrec", caught / flagged, 1, pct=True)
    put("ClsTestRec", caught / int(labels.sum()), 0, pct=True)
    put("ClsTestFone", 2 * caught / (flagged + int(labels.sum())), 2)
    put("ClsTestMCC", matthews_corrcoef(labels, flags), 2)
    put("ActShare", REPORT_REVIEW_SHARE, 0, pct=True)

    primary_val = val[val["variant"] == "primary"].set_index("candidate")
    put("ValSelPR", primary_val.loc[chosen, "pr_auc"], 3)
    put("ValSelROC", primary_val.loc[chosen, "roc_auc"], 3)
    put("ValBestPR", primary_val.loc[best, "pr_auc"], 3)
    put("ValSelRecallTen", primary_val.loc[chosen, "recall_top10"], 0, pct=True)
    # stage-2 one-SE line-up: gap and SE of the selected candidate to the best
    s2 = {r["candidate"]: r for r in primary["stage2"]}
    put("ClsSelGap", s2[chosen]["gap_to_best"], 3)
    put("ClsSelGapSE", s2[chosen]["se_of_gap"], 3)
    # The unregularized baseline was evaluated on the same fixed windows after the original grid run.
    # The selected candidate was unchanged, so its saved test evaluation remains valid.
    audit = pd.read_csv(RES / "handover/unregularized_baseline_audit/validation_windows.csv")
    plain = audit[(audit["variant"] == "primary") & (audit["candidate"] == "logistic_unregularized")]
    if sorted(plain["window"].tolist()) != list(range(5)):
        raise ValueError("Expected five chronological windows for unregularized logistic baseline")
    put("ClsPlainName", escape("logistic_unregularized"))
    put("ClsPlainPR", plain["pr_auc"].mean(), 3)
    put("ClsDefaultPR", primary_val.loc["logistic_c1", "pr_auc"], 3)

    checkout = val[val["variant"] == "checkout"].set_index("candidate")
    checkout_sel = sel["checkout_variant"]["selected"]
    put("ClsCheckoutSelName", escape(checkout_sel))
    put("CheckoutPR", checkout.loc[checkout_sel, "pr_auc"], 3)

    # Remaining-slack rule: the operational benchmark the model must beat.
    op = pd.read_csv(RES / "handover/validation_operating_point.csv")
    op = op[op["variant"] == "primary"].set_index("ranking")
    put("RuleValPR", op.loc["slack_rule", "pr_auc"], 3)
    put("RuleTestPR", test.loc["slack_rule", "pr_auc"], 2)
    put("RuleTestROC", test.loc["slack_rule", "roc_auc"], 2)
    ablation_later = pd.read_csv(RES / "handover/slack_shape_ablation.csv")
    rule_later = ablation_later[(ablation_later.ranking == "slack_rule") & (ablation_later.period == "later")].iloc[0]
    rule_caught = round(rule_later.recall_top10 * int(labels.sum()))
    put("RuleTestPrec", rule_caught / flagged, 1, pct=True)
    put("RuleTestRec", rule_caught / int(labels.sum()), 0, pct=True)
    put("RuleTestFone", 2 * rule_caught / (flagged + int(labels.sum())), 2)
    gap = primary["value_test"]["model_minus_slack_rule"]["pr_auc"]
    put("RuleGapPR", gap["difference"], 3)
    put("RuleGapLow", gap["ci_low"], 3)
    put("RuleGapHigh", gap["ci_high"], 3)
    ablation = pd.read_csv(RES / "handover/slack_shape_ablation.csv")
    windows = ablation[ablation["period"] != "later"].pivot(index="period", columns="ranking", values="pr_auc")
    later = ablation[ablation["period"] == "later"].set_index("ranking")
    put("ShapeValWith", windows["with_slack_shape"].mean(), 3)
    put("ShapeValWithout", windows["without_slack_shape"].mean(), 3)
    backlog = pd.read_csv(ROOT / "experiments/backlog_feature/validation_results.csv")
    backlog_windows = backlog.pivot(index="window", columns="variant", values="pr_auc")
    if len(backlog_windows) != 5 or not np.allclose(
        backlog_windows["state_recent_vs_8wk"].to_numpy(),
        windows["with_slack_shape"].to_numpy(), atol=1e-8
    ):
        raise ValueError("Relative-backlog validation results do not match the selected model")
    put("BacklogBasePR", backlog_windows["baseline"].mean(), 3)
    put("BacklogWithPR", backlog_windows["state_recent_vs_8wk"].mean(), 3)
    put("BacklogWins", int((backlog_windows["state_recent_vs_8wk"] > backlog_windows["baseline"]).sum()), 0)
    three = pd.read_csv(RES / "handover/three_feature_validation_ablation.csv")
    three_windows = three.pivot(index="window", columns="variant", values="pr_auc")
    if len(three_windows) != 5 or not np.allclose(
        three_windows["full"].to_numpy(), windows["with_slack_shape"].to_numpy(), atol=1e-8
    ):
        raise ValueError("Three-feature ablation does not match the selected model")
    put("ShapeValNoThree", three_windows["without_all_three"].mean(), 3)
    put("ShapeThreeWins", int((three_windows["full"] > three_windows["without_all_three"]).sum()), 0)
    cadence = pd.read_csv(ROOT / "experiments/monthly_retraining/metrics.csv")
    cadence = cadence[cadence["cohort"] == "all"].set_index(["phase", "policy"])
    for tag, phase, policy in [("MonthlyValCaught", "validation", "monthly"),
                               ("QuarterlyValCaught", "validation", "quarterly"),
                               ("MonthlyRuleValCaught", "validation", "slack_rule"),
                               ("MonthlyTestCaught", "test", "monthly"),
                               ("FrozenTestCaught", "test", "frozen"),
                               ("MonthlyRuleTestCaught", "test", "slack_rule")]:
        put(tag, cadence.loc[(phase, policy), "caught"], 0)
    put("ShapeWinsWindows", int((windows["with_slack_shape"] > windows["without_slack_shape"]).sum()), 0)
    put("ShapeBeatsRuleWindows", int((windows["with_slack_shape"] > windows["slack_rule"]).sum()), 0)
    put("ShapeTestWith", later.loc["with_slack_shape", "pr_auc"], 2)
    put("ShapeTestWithout", later.loc["without_slack_shape", "pr_auc"], 2)
    put("ShapeRecallWith", later.loc["with_slack_shape", "recall_top10"], 0, pct=True)
    put("ShapeRecallWithout", later.loc["without_slack_shape", "recall_top10"], 0, pct=True)
    put("RuleRecallTen", later.loc["slack_rule", "recall_top10"], 0, pct=True)
    past = pd.read_csv(RES / "handover/past_promise_share.csv").set_index("period")
    put("PastPromiseVal", past.loc["validation", "late_already_past_promise"], 1, pct=True)
    put("PastPromiseTest", past.loc["later", "late_already_past_promise"], 1, pct=True)
    put("PastPromiseLateRate", past.loc["validation", "past_promise_late_rate"], 1, pct=True)

    # gains at fixed shares
    g = gains[gains["variant"] == "primary"].set_index("share_acted")
    for share in (0.01, 0.05, 0.10, 0.20):
        tag = {0.01: "One", 0.05: "Five", 0.10: "Ten", 0.20: "Twenty"}[share]
        row = g.iloc[(g.index - share).to_series().abs().argmin()]
        put(f"Gain{tag}Model", row["model"], 0, pct=True)
        put(f"Prec{tag}Model", row["model"] * test.loc["model", "late_rate"] / share, 0, pct=True)
    gains[gains["variant"] == "primary"][["share_acted", "model", "slack_rule", "random"]].to_csv(
        OUT / "classifier_gains_test.csv", index=False)

    top = imp.sort_values("mean_pr_auc_drop", ascending=False).head(6)
    top.to_csv(OUT / "classifier_importance_top.csv", index=False)
    readable = {"slack_shortfall": "slack shortfall", "handover_days": "elapsed time",
                "remaining_slack": "remaining slack",
                "state_backlog_recent_vs_8wk": "relative buyer-state backlog"}
    for i, (_, r) in enumerate(top.iterrows()):
        put(f"Imp{'ABCDEF'[i]}Name", escape(r["feature"]))
        put(f"Imp{'ABCDEF'[i]}Display", readable.get(r["feature"], escape(r["feature"])))
        put(f"Imp{'ABCDEF'[i]}Drop", r["mean_pr_auc_drop"], 3)

    monthly = scores.assign(flagged=flags, caught=flags & labels).groupby("month", sort=True).agg(
        orders=("is_late", "size"), late=("is_late", "sum"),
        acted=("flagged", "sum"), caught=("caught", "sum")).reset_index()
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
    monthly = full.groupby("month").agg(orders=("is_late", "size"), late_share=("is_late", "mean"),
                                        promised_days=("promised_days", "mean")).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly.to_csv(OUT / "monthly_late_share.csv", index=False)
    m = monthly.set_index("month")
    put("PromJun", m.loc["2018-06", "promised_days"], 1)
    put("PromAug", m.loc["2018-08", "promised_days"], 1)
    # Promise length over the training and validation span with real volume (March 2017 to May 2018).
    span = m.loc["2017-03":"2018-05", "promised_days"]
    put("PromSpanMin", span.min(), 0)
    put("PromSpanMax", span.max(), 0)
    put("PromJul", m.loc["2018-07", "promised_days"], 1)
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
    plots = [("model", "Late-warning model", "thick,blue!60!black"), ("slack_rule", "Remaining-slack rule", "thick,orange!80!black,dashed"),
             ("random", "Random list", "thin,gray")]
    body = "\n".join(rf"\addplot[{style},no marks] coordinates {{{coords(g['share_acted'] * 100, g[col] * 100)}}};" + f"\n\\addlegendentry{{{label}}}"
                     for col, label, style in plots)
    text = rf"""\begin{{tikzpicture}}
\begin{{axis}}[width=0.92\linewidth,height=5.6cm,xlabel={{Share selected for review (\%)}},ylabel={{Late orders identified (\%)}},
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
    if "logistic_unregularized" not in set(s["candidate"]):
        audit = pd.read_csv(RES / "handover/unregularized_baseline_audit/validation_windows.csv")
        plain = audit[(audit["variant"] == "primary") & (audit["candidate"] == "logistic_unregularized")]
        if sorted(plain["window"].tolist()) != list(range(5)):
            raise ValueError("Expected five chronological windows for unregularized logistic baseline")
        row = plain[["pr_auc", "roc_auc", "recall_top10"]].mean().to_dict()
        s = pd.concat([pd.DataFrame([{"candidate": "logistic_unregularized", **row}]), s], ignore_index=True)
    s["candidate"] = s["candidate"].astype(str)
    text = latex_table(s, ["candidate", "pr_auc", "roc_auc", "recall_top10"],
                       ["Candidate", "PR-AUC", "ROC-AUC", "Recall at top 10\\%"],
                       "Every classifier candidate, mean over five validation windows (handover stage).",
                       "tab:app-classifier")
    (GEN / "fig_appendix_classifier.tex").write_text(text)


def fig_classifier_windows():
    """Selected classifier by chronological window, including each no-skill PR baseline."""
    selected = load_json(RES / "handover/selection.json")["variants"]["primary"]["selected"]
    rows = pd.read_csv(RES / "handover/validation_windows.csv")
    rows = rows[(rows["variant"] == "primary") & (rows["candidate"] == selected)].sort_values("window")
    if len(rows) != 5:
        raise ValueError("Expected five selected handover-classifier validation windows")
    body = "\n".join(
        f'W{int(row.window) + 1} & {int(row.validation_orders):,} & '
        f'{row.late_rate * 100:.1f}\\% & {row.pr_auc:.3f} & '
        f'{row.roc_auc:.3f} & {row.precision_top10 * 100:.1f}\\% \\\\'
        for row in rows.itertuples()
    )
    text = (
        "\\begin{table}[H]\n\\centering\\small\n"
        "\\caption{The selected classifier in each validation window. The late rate of each window is the PR-AUC of a random list.}\n"
        "\\label{tab:classifier-windows}\n"
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Window & Orders & Late rate & PR-AUC & ROC-AUC & Top-10\\% precision " + r"\\" + "\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (GEN / "fig_classifier_windows.tex").write_text(text)


def fig_split_summary():
    """Show the selected models across apparent train, temporal validation and test."""
    train = pd.read_csv(OUT / "train_apparent_metrics.csv").set_index(["task", "stage"])
    test_reg = pd.read_csv(RES / "eta/test_metrics.csv")
    test_cls = pd.read_csv(RES / "handover/test_metrics.csv")
    sel_reg = load_json(RES / "eta/selection.json")
    sel_cls = load_json(RES / "handover/selection.json")["variants"]["primary"]["selected"]
    cv_reg = pd.read_csv(RES / "eta/cv_summary.csv")
    cv_cls = pd.read_csv(RES / "handover/validation_summary.csv")
    rows = []
    for stage, label in [("checkout", "Checkout MAE (days)"), ("handover", "Handover MAE (days)")]:
        candidate = sel_reg["checkout_selected_candidate" if stage == "checkout" else "selected_candidate"]
        val = cv_reg[(cv_reg.stage == stage) & (cv_reg.candidate == candidate)].iloc[0]
        tst = test_reg[(test_reg.stage == stage) & (test_reg.candidate == candidate) & (test_reg.scope == "overall")].iloc[0]
        rows.append(f"{label} & {train.loc[('regression', stage), 'mae']:.2f} & {val.mean_mae:.2f} & {tst.mae:.2f} " + r"\\")
    val_cls = cv_cls[(cv_cls.variant == "primary") & (cv_cls.candidate == sel_cls)].iloc[0]
    tst_cls = test_cls[(test_cls.variant == "primary") & (test_cls.ranking == "model")].iloc[0]
    rows.append(f"Late warning PR-AUC & {train.loc[('classification', 'handover'), 'pr_auc']:.3f} & {val_cls.pr_auc:.3f} & {tst_cls.pr_auc:.3f} " + r"\\")
    windows = pd.read_csv(RES / "handover/validation_windows.csv")
    windows = windows[(windows.variant == "primary") & (windows.candidate == sel_cls)]
    rows.append(f"Late prevalence & {train.loc[('classification', 'handover'), 'late_rate'] * 100:.1f}\\% & {windows.late_rate.mean() * 100:.1f}\\% & {tst_cls.late_rate * 100:.1f}\\% " + r"\\")
    text = (
        "\\begin{table}[H]\n\\centering\\small\n"
        "\\caption{Descriptive scores for the chosen models. Training is scored in sample; validation averages five separately fitted chronological windows; the test set uses a model fitted on pre-test data. Late rates and case mixes differ, so the PR-AUC values are not a like-for-like learning curve.}\n"
        "\\label{tab:split-summary}\n"
        "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
        "Metric & Training & Validation & Test set \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (GEN / "fig_split_summary.tex").write_text(text)


def fig_complete_daily_replay():
    """Summarise complete calendar-day batches scored by the unchanged selected model."""
    data = pd.read_csv(OUT / "complete_daily_replay_summary.csv")
    later = data.set_index("period").loc["later_opened"]
    for name, col in [
        ("FullDailyOrders", "orders"), ("FullDailyLate", "late"),
        ("FullDailyFlagged", "flagged"), ("FullDailyCaught", "caught"),
        ("FullDailyActiveDays", "days_with_handovers"),
        ("FullDailyMedianFlagged", "median_daily_flagged"),
        ("FullDailyPninetyFlagged", "p90_daily_flagged"),
        ("FullDailyCaughtOpen", "caught_still_open_next_09"),
        ("FullDailyBeforePromise", "caught_open_and_promise_not_passed_next_09"),
        ("FullDailyExpiredCaught", "caught_promise_passed_by_next_09"),
    ]:
        put(name, later[col], 0)
    put("FullDailyPrecision", later["precision"], 1, pct=True)
    put("FullDailyRecall", later["recall"], 1, pct=True)
    put("FullDailyCaughtPerDay", later["caught"] / later["days_with_handovers"], 0)
    put("FullDailyRecallRound", later["recall"], 0, pct=True)
    rows = []
    for row in data.itertuples():
        label = f"W{int(row.period.split('_')[1]) + 1}" if row.period.startswith("validation_") else "Test set"
        rows.append(
            f"{label} & {int(row.days_with_handovers)} & {int(row.orders):,} & "
            f"{row.late_rate * 100:.1f}\\% & {row.median_daily_flagged:.0f} & "
            f"{row.precision * 100:.1f}\\% & {row.recall * 100:.1f}\\% " + r"\\"
        )
    text = (
        "\\begin{table}[H]\n\\centering\\small\n"
        "\\caption{Day-by-day replay of the chosen model, flagging the top \\ActShare{}\\% of handovers each day. Each row covers every valid handover in its calendar span. Counts show which orders would be flagged, not the effect of acting on them.}\n"
        "\\label{tab:daily-replay}\n"
        "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
        "Period & Active days & Orders & Late rate & Median flagged/day & Precision & Recall \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (GEN / "fig_complete_daily_replay.tex").write_text(text)


def fig_roc_pr_confusion():
    """ROC/PR curves and confusion counts of the selected model on the later period, from its saved scores."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    sys.path.insert(0, str(ROOT))
    from phase2_handover_classifier import monthly_flags
    k = REPORT_REVIEW_SHARE
    scores = pd.read_csv(RES / "handover/test_scores.csv")
    y, s = scores["is_late"].to_numpy().astype(bool), scores["score"].to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    pick = lambda n, m: np.unique(np.linspace(0, n - 1, m).round().astype(int))
    roc_pts, pr_pts = pick(len(fpr), 70), pick(len(rec), 70)
    roc = " ".join(f"({fpr[i]:.4f},{tpr[i]:.4f})" for i in roc_pts)
    pr = " ".join(f"({rec[i]:.4f},{prec[i]:.4f})" for i in pr_pts[::-1])
    rate = y.mean()
    axis = r"width=\linewidth,height=5.1cm,xmin=0,xmax=1,ymin=0,ymax=1,title style={font=\small},tick label style={font=\scriptsize},label style={font=\scriptsize}"
    text = (rf"""\begin{{figure}}[H]
\centering
\begin{{minipage}}{{0.49\linewidth}}\centering
\begin{{tikzpicture}}
\begin{{axis}}[{axis},xlabel={{False positive rate}},ylabel={{True positive rate}},title={{ROC (AUC = {roc_auc_score(y, s):.3f})}}]
\addplot[thick,blue!65!black,no marks] coordinates {{{roc}}};
\addplot[thin,gray,dashed,no marks] coordinates {{(0,0) (1,1)}};
\end{{axis}}
\end{{tikzpicture}}
\end{{minipage}}\hfill
\begin{{minipage}}{{0.49\linewidth}}\centering
\begin{{tikzpicture}}
\begin{{axis}}[{axis},xlabel={{Recall}},ylabel={{Precision}},title={{PR (AP = {average_precision_score(y, s):.3f})}}]
\addplot[thick,blue!65!black,no marks] coordinates {{{pr}}};
\addplot[thin,gray,dashed,no marks] coordinates {{(0,{rate:.4f}) (1,{rate:.4f})}};
\end{{axis}}
\end{{tikzpicture}}
\end{{minipage}}
\caption{{Final handover logistic classifier on {len(y):,} test-set orders ({int(y.sum())} late). Curves are sampled points for display; AUC and average precision (AP) use all predictions. Dashed lines show random ranking. The top-\ActShare{{}}\% review list is drawn separately each month, so it is not a single point on these curves.}}
\label{{fig:roc-pr}}
\end{{figure}}""")
    (GEN / "fig_roc_pr.tex").write_text(text)
    flags = monthly_flags(s, scores["month"].to_numpy(), k)
    tp, fp = int((flags & y).sum()), int((flags & ~y).sum())
    fn, tn = int((~flags & y).sum()), int((~flags & ~y).sum())
    (GEN / "fig_confusion.tex").write_text(rf"""\begin{{table}}[H]
\centering\small
\caption{{Test-set confusion counts for the top-\ActShare{{}}\% monthly review list. Counts show who would be flagged, before any action is taken.}}
\label{{tab:app-confusion}}
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
& Actually late & Actually on time \\
\midrule
Selected for review & {tp:,} & {fp:,} \\
Not selected & {fn:,} & {tn:,} \\
\bottomrule
\end{{tabular}}
\end{{table}}""")


def fig_period_value():
    """Model against the remaining-slack rule at the illustrative review share."""
    k = REPORT_REVIEW_SHARE
    d = pd.read_csv(RES / "handover/slack_shape_ablation.csv")
    model = d[d.ranking == "with_slack_shape"].set_index("period").copy()
    rule = d[d.ranking == "slack_rule"].set_index("period").copy()
    model["caught_at_k"] = (model["recall_top10"] * model["late"]).round().astype(int)
    rule["caught_at_k"] = (rule["recall_top10"] * rule["late"]).round().astype(int)
    order = ["W1", "W2", "W3", "W4", "W5", "later"]
    rows = []
    for period in order:
        m, r = model.loc[period], rule.loc[period]
        label = "Test set" if period == "later" else period
        rows.append(f"{label} & {m.late / m.orders * 100:.1f}\\% & {m.late_past_promise / m.late * 100:.0f}\\% & "
                    f"{m.pr_auc:.3f} & {r.pr_auc:.3f} & {int(m.caught_at_k):,} & {int(r.caught_at_k):,} & "
                    f"{m.late * k:,.0f} " + r"\\")
    (GEN / "fig_period_value.tex").write_text(
        "\\begin{table}[ht]\n\\centering\\small\n"
        "\\caption{Model against the remaining-slack rule at the same workload: the top \\ActShare{}\\% of each "
        "validation window, or of each purchase month in the test set (26 May--August 2018). "
        "A random list would catch \\ActShare{}\\% of late orders.}\n"
        "\\label{tab:period-value}\n\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
        "& & Late already & \\multicolumn{2}{c}{PR-AUC} & \\multicolumn{3}{c}{Late orders caught} \\\\\n"
        "\\cmidrule(lr){4-5}\\cmidrule(l){6-8}\n"
        "Period & Late rate & past date & Model & Rule & Model & Rule & Random \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    oos = ["W3", "W4", "W5", "later"]
    put("OosModelCaught", model.loc[oos, "caught_at_k"].sum(), 0)
    put("OosRuleCaught", rule.loc[oos, "caught_at_k"].sum(), 0)
    put("OosGainPct", (model.loc[oos, "caught_at_k"].sum() / rule.loc[oos, "caught_at_k"].sum() - 1), 0, pct=True)
    put("OosWins", int((model.loc[oos, "caught_at_k"] > rule.loc[oos, "caught_at_k"]).sum()), 0)
    put("WFiveModelCaught", model.loc["W5", "caught_at_k"], 0)
    put("WFiveRuleCaught", rule.loc["W5", "caught_at_k"], 0)
    put("WTwoModelCaught", model.loc["W2", "caught_at_k"], 0)
    put("WTwoRuleCaught", rule.loc["W2", "caught_at_k"], 0)
    put("LaterModelCaught", model.loc["later", "caught_at_k"], 0)
    put("LaterRuleCaught", rule.loc["later", "caught_at_k"], 0)
    windows = order[:5]
    gains = model.loc[windows, "caught_at_k"] / rule.loc[windows, "caught_at_k"] - 1
    put("ValGainMin", gains.min(), 0, pct=True)
    put("ValGainMax", gains.max(), 0, pct=True)
    put("LaterLossPct", 1 - model.loc["later", "caught_at_k"] / rule.loc["later", "caught_at_k"], 0, pct=True)
    # Calm (under 7% late) versus busy validation windows, by the share of late orders.
    rate = model.loc[windows, "late"] / model.loc[windows, "orders"]
    calm, busy = gains[rate < 0.07], gains[rate >= 0.07]
    put("CalmWindows", " and ".join(calm.index), "")
    put("CalmGainMin", calm.min(), 0, pct=True)
    put("CalmGainMax", calm.max(), 0, pct=True)
    put("BusyGainMin", busy.min(), 0, pct=True)
    put("BusyGainMax", busy.max(), 0, pct=True)
    put("CalmRateMax", rate[rate < 0.07].max(), 0, pct=True)
    put("BusyRateMin", rate[rate >= 0.07].min(), 0, pct=True)
    put("BusyRateMax", rate[rate >= 0.07].max(), 0, pct=True)
    past_share = model.loc[windows, "late_past_promise"] / model.loc[windows, "late"]
    put("PastPromiseWinMin", past_share.min(), 0, pct=True)
    put("PastPromiseWinMax", past_share.max(), 0, pct=True)
    lm = pd.read_csv(RES / "handover/later_monthly_model_vs_rule.csv").set_index("month")
    if (lm["past_flagged_model"] != lm["past_orders"]).any():
        raise ValueError("Expected the model to flag every parcel already past its promised date")
    put("AugModelNotPast", lm.loc["2018-08", "caught_not_past_model"], 0)
    put("AugRuleNotPast", lm.loc["2018-08", "caught_not_past_rule"], 0)
    put("JunModelNotPast", lm.loc["2018-06", "caught_not_past_model"], 0)
    put("JunRuleNotPast", lm.loc["2018-06", "caught_not_past_rule"], 0)
    put("TrainPromised", lm["train_promised_days"].iloc[0], 1)
    sp = pd.read_csv(RES / "handover/short_promise_model_vs_rule.csv").set_index("period")
    put("ShortCut", sp["short_cut_days"].iloc[0], 1)
    put("ShortShareAug", sp.loc["later_aug", "short_share"], 0, pct=True)
    vw_short = sp.loc[["W1", "W2", "W3", "W4", "W5"]]
    put("ShortValLateMin", vw_short["late"].min(), 0)
    put("ShortValLateMax", vw_short["late"].max(), 0)
    put("ShortValModelWins", int((vw_short["pr_auc_model"] > vw_short["pr_auc_rule"]).sum()), 0)
    put("ShortTestPRModel", sp.loc["later", "pr_auc_model"], 3)
    put("ShortTestPRRule", sp.loc["later", "pr_auc_rule"], 3)
    late_all = model.loc[order, "late"]
    put("DisruptLateShare", late_all[["W2", "W4", "W5"]].sum() / late_all.sum(), 0, pct=True)
    # Lift over random selection at the same share.
    put("LaterLift", model.loc["later", "caught_at_k"] / (model.loc["later", "late"] * k), 1)
    # Walk-forward: choose on windows before t only, score window t.
    vw = pd.read_csv(RES / "handover/validation_windows.csv")
    piv = vw[vw.variant == "primary"].pivot(index="window", columns="candidate", values="pr_auc")
    picks, gaps = [], []
    for t in range(2, 5):
        pick = piv.loc[:t - 1].mean().idxmax()
        picks.append(pick)
        gaps.append(piv.loc[t].max() - piv.loc[t, pick])
    selected = load_json(RES / "handover/selection.json")["variants"]["primary"]["selected"]
    put("WalkSame", sum(p == selected for p in picks), 0)
    put("WalkMaxGap", max(gaps), 3)


def fig_family_summary():
    """Main-text comparison of linear and tree-based variants with readable names."""
    val = pd.read_csv(RES / "handover/validation_summary.csv")
    val = val[val.variant == "primary"].set_index("candidate")
    tree_best = val.loc[[c for c in val.index if c.startswith("tree")], "pr_auc"].idxmax()
    forest_best = val.loc[[c for c in val.index if c.startswith("forest")], "pr_auc"].idxmax()
    boost_best = val.loc[[c for c in val.index if c.startswith("boost")], "pr_auc"].idxmax()
    candidates = [("Linear", "Plain logistic regression", "logistic_unregularized"),
                  ("Linear", "Regularised logistic regression", "logistic_c0.01"),
                  ("Tree-based", "Single decision tree", "tree_plain"),
                  ("Tree-based", "Tuned decision tree", tree_best),
                  ("Tree-based", "Random forest", forest_best),
                  ("Tree-based", "Gradient boosting", boost_best)]
    rows = []
    for family, name, candidate in candidates:
        r = val.loc[candidate]
        rows.append(f"{family} & {name} & {r.pr_auc:.3f} & {r.roc_auc:.3f} " + r"\\")
    abl = pd.read_csv(RES / "handover/slack_shape_ablation.csv")
    rule = abl[(abl.ranking == "slack_rule") & (abl.period != "later")][["pr_auc", "roc_auc"]].mean()
    rows.append(f"Benchmark & Remaining-slack rule & {rule.pr_auc:.3f} & {rule.roc_auc:.3f} " + r"\\")
    (GEN / "fig_family_summary.tex").write_text(
        "\\begin{table}[ht]\n\\centering\\small\n"
        "\\caption{Classification comparison, mean over five chronological validation windows. "
        "Full configurations are in Appendix "
        "Table~\\ref{tab:app-classifier}.}\n\\label{tab:family-summary}\n"
        "\\begin{tabular}{@{}llrr@{}}\n\\toprule\n"
        "Family & Configuration & PR-AUC & ROC-AUC \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def fig_regression_family_summary():
    """Main-text comparison of regression family baselines and selected variants."""
    cv = pd.read_csv(RES / "eta/cv_summary.csv").set_index(["stage", "candidate"])
    candidates = [("Benchmark", "Training mean", "mean_baseline"),
                  ("Linear", "Plain linear regression", "linear_baseline"),
                  ("Linear", "Ridge regression (selected)", "ridge_100"),
                  ("Tree-based", "Single decision tree", "tree_baseline"),
                  ("Tree-based", "Tuned decision tree", "tree_depth6"),
                  ("Tree-based", "Random forest", "forest_leaf20"),
                  ("Tree-based", "Gradient boosting", "boost_baseline")]
    rows = []
    for family, name, candidate in candidates:
        checkout = cv.loc[(CHECKOUT_STAGE, candidate), "mean_mae"]
        handover = cv.loc[(HANDOVER_STAGE, candidate), "mean_mae"]
        rows.append(f"{family} & {name} & {checkout:.2f} & {handover:.2f} " + r"\\")
    handover_rule = cv.loc[(HANDOVER_STAGE, "elapsed_plus_route_baseline"), "mean_mae"]
    rows.append(f"Benchmark & Elapsed time + route rule & -- & {handover_rule:.2f} " + r"\\")
    (GEN / "fig_regression_family_summary.tex").write_text(
        "\\begin{table}[ht]\n\\centering\\small\n"
        "\\caption{Regression comparison: mean MAE in days over five chronological validation windows. "
        "Lower is better. Full configurations are in Appendix Table~\\ref{tab:app-regression}.}"
        "\n\\label{tab:regression-family-summary}\n"
        "\\begin{tabular}{@{}llrr@{}}\n\\toprule\n"
        "Family & Configuration & Checkout & Handover \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def write_macros():
    lines = ["% Generated by reports/phase2_final_evidence.py: do not edit by hand."]
    for name, text in sorted(macros.items()):
        lines.append(rf"\newcommand{{\{name}}}{{{text}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")


def sync_standalone_report():
    """Refresh generated evidence in the report without rewriting its prose."""
    original = REPORT.read_text()
    report = original
    # Pick up newly added \input{generated/name} lines without a script edit.
    names = set(re.findall(r"\\input\{generated/([A-Za-z0-9_-]+)\}", report))
    names.update(re.findall(r"^% BEGIN GENERATED ([A-Za-z0-9_-]+) -- refreshed by phase2_final_evidence\.py$",
                            report, flags=re.MULTILINE))
    if not names:
        raise ValueError(f"No generated evidence found in {REPORT}")
    for name in sorted(names):
        begin = f"% BEGIN GENERATED {name} -- refreshed by phase2_final_evidence.py"
        end = f"% END GENERATED {name}"
        legacy_input = rf"\input{{generated/{name}}}"
        content = (GEN / f"{name}.tex").read_text().rstrip("\n")
        block = f"{begin}\n{content}\n{end}"
        if report.count(begin) == 1 and report.count(end) == 1:
            start = report.index(begin)
            finish = report.index(end, start) + len(end)
            report = report[:start] + block + report[finish:]
        elif report.count(legacy_input) == 1 and begin not in report and end not in report:
            report = report.replace(legacy_input, block, 1)
        else:
            raise ValueError(f"Expected exactly one generated block or input for {name} in {REPORT}")
    if report != original:
        REPORT.write_text(report)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    table, _, _ = load_stage_table()
    sel, selected = regression_numbers(table)
    gains = classifier_numbers()
    monthly = monthly_late_share(table)
    supporting_facts(set(table["order_id"]))
    fig_monthly(monthly)
    fig_ladder(pd.read_csv(RES / "eta/ladder_windows.csv"))
    fig_gains(gains)
    fig_appendix_regression(pd.read_csv(RES / "eta/cv_summary.csv"))
    fig_appendix_classifier(pd.read_csv(RES / "handover/validation_summary.csv"))
    fig_classifier_windows()
    fig_split_summary()
    fig_complete_daily_replay()
    fig_roc_pr_confusion()
    fig_period_value()
    fig_family_summary()
    fig_regression_family_summary()
    write_macros()
    sync_standalone_report()
    print(f"{len(macros)} macros written to {GEN / 'numbers.tex'}")


if __name__ == "__main__":
    main()
