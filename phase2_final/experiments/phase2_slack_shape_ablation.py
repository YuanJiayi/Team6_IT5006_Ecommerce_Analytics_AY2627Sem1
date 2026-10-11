"""Slack-shape ablation for the selected handover classifier.

Compares, on the same five validation windows and the later period, the selected logistic configuration with and
without the two slack-shape inputs, against the remaining-slack rule, including late orders caught at the validation-selected review share k. Also records how many late orders had already
passed their promised date at handover in each period. Writes results/phase2/handover/slack_shape_ablation.csv and
results/phase2/handover/past_promise_share.csv, and the selected model's later-period scores
(test_scores.csv) for the report's ROC/PR curves and confusion counts, and a monthly model-versus-rule breakdown of the
later period (later_monthly_model_vs_rule.csv), and a short-promise comparison (short_promise_model_vs_rule.csv).

Run from the phase2_final folder:  python experiments/phase2_slack_shape_ablation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2_eta import chronological_folds  # noqa: E402
from phase2_handover_classifier import (CANDIDATES, LABEL, PROMISE, SLACK_SHAPE_INPUTS, TOP_SHARE,  # noqa: E402
                                        fit_scores, load_cohort, monthly_flags, top_flags, variant_spec)

OUT = ROOT / "results" / "phase2" / "handover"


def slack_rule(fit_rows, rows):
    median = fit_rows["route_transit_90d_at_handover"].median()
    return -(rows[PROMISE] - rows["handover_days"] - rows["route_transit_90d_at_handover"].fillna(median)).to_numpy()


def main():
    table, spec, _ = load_cohort(ROOT)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    selected = json.loads((OUT / "selection.json").read_text())["variants"]["primary"]["selected"]
    family, params = next((f, p) for n, f, p in CANDIDATES if n == selected)
    with_shape = variant_spec(spec)
    without = variant_spec(spec)
    without["numeric_features"] = [f for f in without["numeric_features"] if f not in SLACK_SHAPE_INPUTS]
    logf = spec["linear_log_candidates"]
    rankers = {
        "with_slack_shape": lambda a, b: fit_scores(family, params, with_shape, a, b, logf),
        "without_slack_shape": lambda a, b: fit_scores(family, params, without, a, b, logf),
        "slack_rule": slack_rule,
    }
    selection = json.loads((OUT / "selection.json").read_text())["variants"]["primary"]
    share_k = float(selection["k"])
    y_test = test[LABEL].to_numpy()
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    periods = [(f"W{w + 1}", train.iloc[fit], train.iloc[valid], None) for w, (fit, valid) in enumerate(folds)]
    periods.append(("later", train, test, months))
    rows = []
    for name, rank in rankers.items():
        for period, fit_rows, rows_, period_months in periods:
            y = rows_[LABEL].to_numpy().astype(bool)
            scores = rank(fit_rows, rows_)
            # Same workload for every ranking: top share of each window, or of each purchase month in the later period.
            flags = (monthly_flags(scores, period_months, share_k) if period_months is not None
                     else top_flags(scores, share_k))
            rows.append({"ranking": name, "period": period, "orders": len(y), "late": int(y.sum()),
                         "late_past_promise": int((y & (rows_["past_promise_at_handover"] == 1).to_numpy()).sum()),
                         "pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores),
                         "flagged": int(flags.sum()), "caught_at_k": int((flags & y).sum()),
                         "recall_top10": float((y & (monthly_flags(scores, period_months, TOP_SHARE) if period_months is not None
                                                     else top_flags(scores, TOP_SHARE))).sum() / y.sum())})
            if name == "with_slack_shape" and period == "later":
                pd.DataFrame({"order_id": test["order_id"], "month": months, "is_late": y_test, "score": scores}).to_csv(
                    OUT / "test_scores.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "slack_shape_ablation.csv", index=False)

    # Final period by purchase month: where the model and the rule differ, among parcels not yet past their date.
    model_scores = rankers["with_slack_shape"](train, test)
    rule_scores = rankers["slack_rule"](train, test)
    past_t = (test["past_promise_at_handover"] == 1).to_numpy()
    y_t = y_test.astype(bool)
    flags_m, flags_r = monthly_flags(model_scores, months, share_k), monthly_flags(rule_scores, months, share_k)
    monthly = []
    for month in np.unique(months):
        i = months == month
        monthly.append({"month": month, "orders": int(i.sum()), "promised_days": float(test.loc[i, PROMISE].mean()),
                        "late_past": int((y_t & i & past_t).sum()), "late_not_past": int((y_t & i & ~past_t).sum()),
                        "past_flagged_model": int((flags_m & i & past_t).sum()), "past_orders": int((i & past_t).sum()),
                        "caught_not_past_model": int((flags_m & y_t & i & ~past_t).sum()),
                        "caught_not_past_rule": int((flags_r & y_t & i & ~past_t).sum())})
    monthly = pd.DataFrame(monthly)
    monthly["train_promised_days"] = float(train[PROMISE].mean())
    monthly.to_csv(OUT / "later_monthly_model_vs_rule.csv", index=False)

    # Short promises (below the training 10th percentile): model against rule among parcels not yet past their date.
    short_cut = float(train[PROMISE].quantile(0.10))
    short_rows = []
    for period, fit_rows, rows_, _ in periods:
        y = rows_[LABEL].to_numpy().astype(bool)
        j = (rows_[PROMISE] < short_cut).to_numpy() & (rows_["past_promise_at_handover"] != 1).to_numpy()
        m_s, r_s = rankers["with_slack_shape"](fit_rows, rows_), rankers["slack_rule"](fit_rows, rows_)
        short_rows.append({"period": period, "short_cut_days": short_cut, "short_share": float((rows_[PROMISE] < short_cut).mean()),
                           "orders": int(j.sum()), "late": int(y[j].sum()),
                           "pr_auc_model": average_precision_score(y[j], m_s[j]) if y[j].any() else np.nan,
                           "pr_auc_rule": average_precision_score(y[j], r_s[j]) if y[j].any() else np.nan})
    aug = months == "2018-08"
    short_rows.append({"period": "later_aug", "short_cut_days": short_cut,
                       "short_share": float((test.loc[aug, PROMISE] < short_cut).mean())})
    pd.DataFrame(short_rows).to_csv(OUT / "short_promise_model_vs_rule.csv", index=False)

    valid_rows = train.iloc[np.concatenate([v for _, v in folds])]
    share = []
    for period, rows_ in (("validation", valid_rows), ("later", test)):
        late = rows_[rows_[LABEL] == 1]
        past = rows_["past_promise_at_handover"] == 1
        share.append({"period": period, "late_orders": len(late),
                      "late_already_past_promise": float(late["past_promise_at_handover"].mean()),
                      "past_promise_orders_share": float(past.mean()),
                      "past_promise_late_rate": float(rows_.loc[past, LABEL].mean())})
    pd.DataFrame(share).to_csv(OUT / "past_promise_share.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index="ranking", columns="period", values="pr_auc").round(3))
    print(pd.DataFrame(share).round(3))


if __name__ == "__main__":
    main()
