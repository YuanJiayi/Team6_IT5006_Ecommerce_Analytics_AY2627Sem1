"""Post hoc label audit. Run: it5006-proj/bin/python experiments/phase2_label_definition_audit.py.

Reads frozen scores; only the separately marked timestamp experiment fits models.
Does not alter the saved model, data, results, cutoff, or report. Seed 42.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phase2_staged_audit import load, fit_model, probability  # noqa: E402

OUT = ROOT / "experiments/phase2_label_definition_audit"
SOURCE = ROOT / "experiments/phase2_staged_audit"
CUTOFF = 0.07878485347563163
SEED = 42


def measure(frame):
    y = frame.label.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    n = len(y)
    top = np.argsort(-p, kind="stable")[:int(np.ceil(.1 * n))]
    flagged = p >= CUTOFF
    rate = y.mean()
    precision10 = y[top].mean()
    return dict(orders=n, late=int(y.sum()), late_rate=rate,
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                pr_auc_over_rate=average_precision_score(y, p) / rate,
                top10_precision=precision10, top10_lift=precision10 / rate,
                top10_recall=y[top].sum() / y.sum(),
                cutoff_flagged_share=flagged.mean(),
                cutoff_precision=y[flagged].mean() if flagged.any() else np.nan,
                cutoff_recall=y[flagged].sum() / y.sum())


def describe(group, name):
    h = group.handover_after_approval_days
    return dict(group=name, orders=len(group), promised_days_median=group.promised_days.median(),
                promised_days_q25=group.promised_days.quantile(.25),
                promised_days_q75=group.promised_days.quantile(.75),
                same_state_pct=100 * group.same_state.mean(),
                handover_delay_known=int(h.notna().sum()),
                handover_delay_median=h.median(),
                handover_delay_q25=h.quantile(.25), handover_delay_q75=h.quantile(.75),
                frozen_checkout_score_median=group.checkout_probability.median(),
                frozen_checkout_score_mean=group.checkout_probability.mean(),
                frozen_checkout_top10_capture=group.checkout_top10.mean())


def main():
    OUT.mkdir(exist_ok=True)
    frame, spec = load()
    assert spec["random_state"] == SEED and spec["primary_cutoff"] == "2018-05-26"
    raw = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                      usecols=["order_id", "order_estimated_delivery_date"],
                      parse_dates=["order_estimated_delivery_date"])
    times = raw.order_estimated_delivery_date.dropna().dt.strftime("%H:%M:%S").value_counts()
    times.rename_axis("estimate_time").reset_index(name="orders").to_csv(OUT / "estimate_times.csv", index=False)
    assert len(times) == 1 and times.index[0] == "00:00:00"

    test = frame.loc[frame.split.eq("test")].copy()
    assert len(test) == 19363 and test.order_id.is_unique
    test["calendar_label"] = (test.order_delivered_customer_date.dt.normalize()
                              > test.order_estimated_delivery_date.dt.normalize()).astype(int)
    test["timestamp_label"] = (test.order_delivered_customer_date
                               > test.order_estimated_delivery_date).astype(int)
    assert test.calendar_label.equals(test.is_late)
    test["flipped"] = test.timestamp_label.ne(test.calendar_label)
    assert (test.loc[test.flipped, "calendar_label"] == 0).all()
    assert (test.loc[test.flipped, "order_delivered_customer_date"].dt.normalize().to_numpy()
            == test.loc[test.flipped, "order_estimated_delivery_date"].dt.normalize().to_numpy()).all()
    assert test.timestamp_label.sum() == 1021 and test.flipped.sum() == 347
    test["month"] = test.order_purchase_timestamp.dt.strftime("%Y-%m")

    rows = []
    for stage in ("checkout", "handover"):
        saved = pd.read_csv(SOURCE / f"{stage}_test_predictions.csv",
                            usecols=["order_id", "probability", "is_late"])
        joined = test.merge(saved, on="order_id", how="inner", validate="one_to_one", suffixes=("", "_saved"))
        assert joined.is_late.eq(joined.is_late_saved).all()
        if stage == "checkout":
            assert len(joined) == len(test)
            test = test.merge(saved[["order_id", "probability"]].rename(
                columns={"probability": "checkout_probability"}), on="order_id", validate="one_to_one")
            top_ids = set(test.nlargest(int(np.ceil(.1 * len(test))), "checkout_probability").order_id)
            test["checkout_top10"] = test.order_id.isin(top_ids)
        for label_col in ("calendar_label", "timestamp_label"):
            current = joined.assign(label=joined[label_col])
            for period, part in [("all", current), *list(current.groupby("month"))]:
                if part.label.nunique() < 2:
                    continue
                rows.append(dict(stage=stage, score_source="frozen", label=label_col,
                                 period=period, **measure(part)))
    pd.DataFrame(rows).to_csv(OUT / "frozen_scores.csv", index=False)

    groups = [("flipped_same_day", test[test.flipped]),
              ("calendar_late", test[test.calendar_label.eq(1)]),
              ("unchanged_on_time", test[test.timestamp_label.eq(0)])]
    pd.DataFrame([describe(part, name) for name, part in groups]).to_csv(OUT / "flip_groups.csv", index=False)
    test.groupby(["month", "flipped"]).size().rename("orders").reset_index().to_csv(OUT / "flip_by_month.csv", index=False)
    for col in ("customer_state", "seller_state"):
        test.groupby([col, "flipped"]).size().rename("orders").reset_index().to_csv(OUT / f"flip_by_{col}.csv", index=False)
    negative = test.loc[test.timestamp_label.eq(0), "checkout_probability"]
    rank_rows = []
    for name, part in groups[:2]:
        positives = part.checkout_probability
        y = np.r_[np.ones(len(positives)), np.zeros(len(negative))]
        p = np.r_[positives, negative]
        rank_rows.append(dict(group=name, positives=len(positives), comparison="unchanged on-time",
                              roc_auc=roc_auc_score(y, p), top10_capture=part.checkout_top10.mean()))
    pd.DataFrame(rank_rows).to_csv(OUT / "flip_ranking.csv", index=False)

    # Exploratory experiment: identical training eligibility, pipeline and C; new target only.
    frame["timestamp_label"] = (frame.order_delivered_customer_date
                                > frame.order_estimated_delivery_date).astype(int)
    train = frame.loc[frame.split.eq("train")].copy()
    test = frame.loc[frame.split.eq("test")].copy()
    refits = []
    for stage in ("checkout", "handover"):
        eligible = f"eligible_{stage}"
        fit = train.loc[train[eligible]]
        evaluate = test.loc[test[eligible]].copy()
        assert fit.outcome_available_at.lt(pd.Timestamp("2018-05-26")).all()
        model, features = fit_model(fit, spec, stage, target="timestamp_label")
        evaluate["probability"] = probability(model, evaluate, features)
        evaluate["label"] = evaluate.timestamp_label
        refits.append(dict(stage=stage, score_source="exploratory timestamp refit",
                           label="timestamp_label", period="all", **measure(evaluate)))
    pd.DataFrame(refits).to_csv(OUT / "exploratory_refits.csv", index=False)
    print(f"Post hoc label audit saved to {OUT}")
    print(f"Test calendar late={int(test.is_late.sum())}; timestamp late={int(test.timestamp_label.sum())}; flipped={int((test.timestamp_label != test.is_late).sum())}")


if __name__ == "__main__":
    main()
