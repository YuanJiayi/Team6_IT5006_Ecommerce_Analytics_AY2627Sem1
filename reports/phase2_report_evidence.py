"""Regenerate the classification report's tables and figures without model selection.

This script refits the previously frozen logistic model once, then diagnoses its
validation and held-out predictions. All test-period diagnostics are post hoc.
Run from the repository root: it5006-proj/bin/python reports/phase2_report_evidence.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2_classification import feature_columns, load_primary_training, make_pipeline  # noqa: E402

RESULTS = ROOT / "results/phase2/classification"
OUTPUT = ROOT / "reports/phase2_evidence"
K_VALUES = (0.01, 0.05, 0.10, 0.20)
BOOTSTRAPS = 1000
SEED = 42


def ranking(y, score, shares=K_VALUES):
    """Select ceil(k*n) highest scores; input order breaks exact ties."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    total_late = int(y.sum())
    rows = []
    for share in shares:
        selected = max(1, math.ceil(share * len(y)))
        found = int(y[order[:selected]].sum())
        precision = found / selected
        rows.append({"share": share, "reviewed": selected, "late_found": found,
                     "precision": precision, "recall": found / total_late,
                     "lift": precision / y.mean()})
    return rows


def score_summary(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    return {"orders": len(y), "late": int(y.sum()), "late_rate": float(y.mean()),
            "mean_prediction": float(score.mean()),
            "pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "ap_over_late_rate": float(average_precision_score(y, score) / y.mean())}


def bootstrap(y, score, *, seed=SEED):
    """Order bootstrap, conditional on this scored period and fitted model."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    samples = {"pr_auc": [], "roc_auc": []}
    samples.update({f"precision_{share:g}": [] for share in K_VALUES})
    for _ in range(BOOTSTRAPS):
        drawn = rng.integers(0, len(y), size=len(y))
        sampled_y, sampled_score = y[drawn], score[drawn]
        if sampled_y.min() == sampled_y.max():
            continue
        samples["pr_auc"].append(average_precision_score(sampled_y, sampled_score))
        samples["roc_auc"].append(roc_auc_score(sampled_y, sampled_score))
        for row in ranking(sampled_y, sampled_score):
            samples[f"precision_{row['share']:g}"].append(row["precision"])
    return {name: [float(x) for x in np.quantile(values, [0.025, 0.975])]
            for name, values in samples.items()}


def calibration_bins(frame, bins=5):
    # Equal-count bins keep even the low-prevalence windows readable.
    sorted_frame = frame.sort_values("probability", kind="stable").reset_index(drop=True)
    group = np.floor(np.arange(len(sorted_frame)) * bins / len(sorted_frame)).astype(int)
    return sorted_frame.assign(bin=group).groupby("bin").agg(
        mean_prediction=("probability", "mean"), observed_rate=("is_late", "mean"),
        orders=("is_late", "size"))


def plot_gains(validation, test):
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    shares = np.linspace(0.01, 0.30, 30)
    for fold, group in validation.groupby("fold", sort=True):
        rates = ranking(group.is_late, group.probability, shares)
        ax.plot(shares, [r["recall"] for r in rates], color="#a8c3d5", lw=0.8,
                label="Individual validation windows" if fold == 0 else None)
    validation_mean = np.mean([[r["recall"] for r in ranking(g.is_late, g.probability, shares)]
                               for _, g in validation.groupby("fold", sort=True)], axis=0)
    ax.plot(shares, validation_mean, color="#145a86", lw=2.2, label="Validation mean")
    ax.plot(shares, [r["recall"] for r in ranking(test.is_late, test.probability, shares)],
            color="#c44e39", lw=2.2, label="Later test (exploratory policy)")
    ax.plot([0, 0.3], [0, 0.3], "--", color="#777777", lw=1, label="Random selection")
    ax.set(xlabel="Share of orders reviewed", ylabel="Share of late orders found",
           title="Risk ranking concentrates late orders in a shorter list", xlim=(0, .30), ylim=(0, .70))
    ax.grid(alpha=.15)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "cumulative_gains.pdf")
    plt.close(fig)


def plot_calibration(validation, test):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharex=True, sharey=True)
    for fold, group in validation.groupby("fold", sort=True):
        bins = calibration_bins(group)
        axes[0].plot(bins.mean_prediction, bins.observed_rate, "o-", lw=1.2,
                     markersize=3, label=f"Window {fold + 1}")
    bins = calibration_bins(test)
    axes[1].plot(bins.mean_prediction, bins.observed_rate, "o-", color="#c44e39", lw=1.7,
                 markersize=4, label="Later test")
    for ax, title in zip(axes, ["Five validation windows", "Later test (exploratory)"]):
        ax.plot([0, .3], [0, .3], "--", color="#777777", lw=1, label="Perfect calibration")
        ax.set(title=title, xlim=(0, .3), ylim=(0, .5), xlabel="Mean predicted late risk")
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("Observed late rate")
    fig.tight_layout()
    fig.savefig(OUTPUT / "calibration_by_period.pdf")
    plt.close(fig)


def inline_report_figures(validation, test):
    """Keep the standalone LaTeX source compilable in the built-in editor."""
    report = ROOT / "reports/phase2_classification_draft.tex"
    source = report.read_text()
    shares = np.linspace(.01, .30, 30)

    def path(points):
        return " -- ".join(f"({x:.4f},{y:.4f})" for x, y in points)

    gain = [r"\begin{tikzpicture}[x=23cm,y=5.7cm]",
            r"\draw[->] (0,0)--(.32,0) node[right] {\scriptsize Share reviewed};",
            r"\draw[->] (0,0)--(0,.69) node[above] {\scriptsize Share of late orders found};",
            r"\draw[dashed,gray] (0,0)--(.30,.30);",
            r"\foreach \x/\lab in {.1/10,.2/20,.3/30} {\draw (\x,0)--(\x,-.008) node[below] {\scriptsize \lab\%};}",
            r"\foreach \y/\lab in {.2/20,.4/40,.6/60} {\draw (0,\y)--(-.004,\y) node[left] {\scriptsize \lab\%};}"]
    folds = list(validation.groupby("fold", sort=True))
    for _, group in folds:
        points = [(0, 0)] + [(s, r["recall"]) for s, r in zip(
            shares, ranking(group.is_late, group.probability, shares))]
        gain.append(r"\draw[blue!25,thin] " + path(points) + ";")
    mean_recall = np.mean([[r["recall"] for r in ranking(g.is_late, g.probability, shares)]
                           for _, g in folds], axis=0)
    gain.append(r"\draw[blue!75!black,very thick] " +
                path([(0, 0)] + list(zip(shares, mean_recall))) + ";")
    gain.append(r"\draw[red!75!black,very thick] " + path(
        [(0, 0)] + [(s, r["recall"]) for s, r in zip(
            shares, ranking(test.is_late, test.probability, shares))]) + ";")
    gain.extend([r"\node[anchor=west,blue!75!black] at (.175,.60) {\scriptsize Validation mean};",
                 r"\node[anchor=west,red!75!black] at (.175,.54) {\scriptsize Later test};",
                 r"\node[anchor=west,gray] at (.175,.48) {\scriptsize Random list};",
                 r"\end{tikzpicture}"])

    calibration = [r"\begin{tikzpicture}[x=15cm,y=7.3cm]"]
    for xshift, title in [("0cm", "Validation windows"), ("6.4cm", "Later test (exploratory)")]:
        calibration.append(r"\begin{scope}[xshift=" + xshift + "]")
        calibration.extend([
            r"\draw[->] (0,0)--(.31,0) node[right] {\scriptsize Predicted};",
            r"\draw[->] (0,0)--(0,.51) node[above] {\scriptsize Observed};",
            r"\draw[dashed,gray] (0,0)--(.30,.30);",
            r"\foreach \v in {.1,.2,.3} {\draw (\v,0)--(\v,-.009) node[below] {\tiny \v};}",
            r"\foreach \v in {.1,.2,.3,.4,.5} {\draw (0,\v)--(-.006,\v) node[left] {\tiny \v};}",
            r"\node[anchor=west] at (0,.55) {\small " + title + "};",
        ])
        if xshift == "0cm":
            colors = ["blue!75!black", "orange!85!black", "green!55!black",
                      "red!70!black", "violet!80!black"]
            for (fold, group), color in zip(folds, colors):
                bins = calibration_bins(group)
                points = list(zip(bins.mean_prediction, bins.observed_rate))
                calibration.append(r"\draw[" + color + r",thick] " + path(points) + ";")
                for point in points:
                    calibration.append(r"\fill[" + color + "] " + path([point]) + " circle[radius=.003];")
            for fold, color in enumerate(colors):
                y = .47 - fold * .035
                calibration.append(r"\draw[" + color + r",thick] " +
                                   path([(.17, y), (.19, y)]) + ";")
                calibration.append(r"\node[anchor=west] at " + path([(.195, y)]) +
                                   r" {\tiny Window " + str(fold + 1) + "};")
        else:
            bins = calibration_bins(test)
            points = list(zip(bins.mean_prediction, bins.observed_rate))
            calibration.append(r"\draw[red!75!black,very thick] " + path(points) + ";")
            for point in points:
                calibration.append(r"\fill[red!75!black] " + path([point]) + " circle[radius=.003];")
        calibration.append(r"\end{scope}")
    calibration.append(r"\end{tikzpicture}")

    replacements = {"GAINS": "\n".join(gain), "CALIBRATION": "\n".join(calibration)}
    for name, drawing in replacements.items():
        pattern = rf"(% BEGIN GENERATED {name} FIGURE\n).*?(\n% END GENERATED {name} FIGURE)"
        source, count = re.subn(pattern, lambda match: match.group(1) + drawing + match.group(2),
                                source, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError(f"Missing {name} figure markers in the report")
    report.write_text(source)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((RESULTS / "selection.json").read_text())
    assert selection["selected_candidate"] == "logistic_c01"
    assert np.isclose(selection["selected_threshold"], 0.07878485347563163)
    train, spec = load_primary_training(ROOT)
    assert spec["random_state"] == SEED and spec["primary_cutoff"] == "2018-05-26"
    table = pd.read_csv(ROOT / "data/phase2_order_table.csv",
                        parse_dates=["order_purchase_timestamp"])
    test = table.loc[table[spec["primary_split"]].eq("test")].copy()
    assert len(test) == 19363 and int(test["is_late"].sum()) == 674
    numeric, categorical = feature_columns(spec)
    log_columns = selection["log_transform"]["columns"] if selection["log_transform"]["adopted"] else ()
    pipeline = make_pipeline("linear", {"C": .1, "class_weight": None}, spec,
                             log_features=log_columns)
    with threadpool_limits(limits=1):
        pipeline.fit(train[numeric + categorical], train["is_late"])
        test["probability"] = pipeline.predict_proba(test[numeric + categorical])[:, 1]
    # This checks the refit against the first held-out result already reported.
    assert abs(average_precision_score(test.is_late, test.probability) - .0651) < .0001
    validation = pd.read_csv(RESULTS / "selected_validation_predictions.csv", float_precision="round_trip")
    dates = train[["order_id", "order_purchase_timestamp"]]
    validation = validation.merge(dates, on="order_id", validate="one_to_one")
    assert len(validation) == 38005

    test_pred = test[["order_id", "order_purchase_timestamp", "is_late", "probability"]]
    test_pred.to_csv(OUTPUT / "test_predictions.csv", index=False)
    period_rows = []
    for fold, group in validation.groupby("fold", sort=True):
        period_rows.append({"period": f"Validation {fold + 1}", **score_summary(group.is_late, group.probability)})
    period_rows.append({"period": "Later test", **score_summary(test.is_late, test.probability)})
    period_table = pd.DataFrame(period_rows)
    period_table.to_csv(OUTPUT / "period_scores.csv", index=False)

    top_rows = []
    for fold, group in validation.groupby("fold", sort=True):
        top_rows.extend({"period": f"Validation {fold + 1}", **row}
                        for row in ranking(group.is_late, group.probability))
    for row in ranking(test.is_late, test.probability):
        top_rows.append({"period": "Later test", **row})
    top_table = pd.DataFrame(top_rows)
    top_table.to_csv(OUTPUT / "top_k.csv", index=False)
    top_table.loc[top_table.period.ne("Later test")].groupby("share", as_index=False)[
        ["precision", "recall", "lift"]].mean().to_csv(OUTPUT / "top_k_validation_mean.csv", index=False)
    test_intervals = bootstrap(test.is_late, test.probability)
    (OUTPUT / "test_intervals.json").write_text(json.dumps(test_intervals, indent=2) + "\n")

    calibration_rows = []
    for fold, group in validation.groupby("fold", sort=True):
        calibration_rows.append(calibration_bins(group).assign(period=f"Validation {fold + 1}"))
    calibration_rows.append(calibration_bins(test).assign(period="Later test"))
    pd.concat(calibration_rows).to_csv(OUTPUT / "calibration_bins.csv", index_label="bin")

    alert_rows = []
    for name, frame in [("Pooled validation", validation), ("Later test", test)]:
        flagged = frame.probability.ge(selection["selected_threshold"])
        late = frame.is_late.eq(1)
        alert_rows.append({"period": name, "flagged": int(flagged.sum()),
                           "flagged_share": float(flagged.mean()),
                           "true_alerts": int((flagged & late).sum()),
                           "false_alerts": int((flagged & ~late).sum()),
                           "missed_late": int((~flagged & late).sum())})
    pd.DataFrame(alert_rows).to_csv(OUTPUT / "fixed_cutoff_alerts.csv", index=False)
    assert alert_rows[0]["flagged"] == 3600 and alert_rows[0]["true_alerts"] == 1145
    assert alert_rows[1]["flagged"] == 8316 and alert_rows[1]["true_alerts"] == 434

    months = test.assign(month=test.order_purchase_timestamp.dt.to_period("M").astype(str))
    monthly_rows = []
    for month, group in months.groupby("month", sort=True):
        if month not in ("2018-06", "2018-07", "2018-08"):
            continue
        ci = bootstrap(group.is_late, group.probability)
        top10 = ranking(group.is_late, group.probability, (.10,))[0]
        monthly_rows.append({"month": month, **score_summary(group.is_late, group.probability),
                             "top10_precision": top10["precision"],
                             "top10_recall": top10["recall"],
                             "pr_auc_low": ci["pr_auc"][0], "pr_auc_high": ci["pr_auc"][1],
                             "roc_auc_low": ci["roc_auc"][0], "roc_auc_high": ci["roc_auc"][1],
                             "top10_precision_low": ci["precision_0.1"][0],
                             "top10_precision_high": ci["precision_0.1"][1]})
    pd.DataFrame(monthly_rows).to_csv(OUTPUT / "monthly_test.csv", index=False)

    raw = pd.read_csv(ROOT / "data/olist_orders_dataset.csv",
                      parse_dates=["order_purchase_timestamp"])
    raw = raw.loc[raw.order_purchase_timestamp.between("2018-05-26", "2018-08-31 23:59:59")]
    in_progress = raw.loc[raw.order_status.isin(["shipped", "invoiced", "processing", "approved", "created"])
                          & raw.order_delivered_customer_date.isna()].copy()
    in_progress["month"] = in_progress.order_purchase_timestamp.dt.to_period("M").astype(str)
    missing = in_progress.groupby("month").size().reindex(
        ["2018-05", "2018-06", "2018-07", "2018-08"], fill_value=0)
    assert int(missing.sum()) == 193
    missing.rename("in_progress_without_delivery").to_csv(OUTPUT / "missing_outcomes.csv")

    plot_gains(validation, test)
    plot_calibration(validation, test)
    inline_report_figures(validation, test)
    print(period_table.to_string(index=False))
    print("\nTop-k:\n", top_table.to_string(index=False))
    print("\nTest bootstrap intervals:", test_intervals)
    print("\nMonthly:\n", pd.DataFrame(monthly_rows).to_string(index=False))
    print("\nStill in progress by month:\n", missing.to_string())


if __name__ == "__main__":
    main()
