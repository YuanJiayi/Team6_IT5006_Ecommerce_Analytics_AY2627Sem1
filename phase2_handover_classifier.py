"""Late warning at carrier handover, following docs/phase2_final_spec.md (classification half).

Run from the repository root:
    python phase2_handover_classifier.py

Order of work: validation of four fixed candidates over the five chronological windows, selection by mean PR-AUC,
a rank-based operating point (share k of each period's handovers) chosen on validation, baselines, the value test
against the slack rule (gate 2), the pre-declared history extension only if that gate fails, then one test scoring
after every choice is fixed. The checkout-stage variant is an appendix. The counterfactual simulation against the
engine's promise is not implemented yet: `run_variant` and the metric helpers take the label and promise as columns,
so an alternative promise column can be passed in later.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))

from phase2_classification import feature_columns, make_pipeline  # noqa: E402
from phase2_eta import (CANDIDATES as ETA_CANDIDATES, chronological_folds, fit_predict,  # noqa: E402
                        load_stage_table, stage_spec)
from phase2_recent_history_audit import asof_mean  # noqa: E402
from phase2_selection import two_stage_select  # noqa: E402

RESULT_DIR = ROOT / "results" / "phase2" / "handover"
LABEL = "is_late"
PROMISE = "promised_days"
# Two families (Linear; Tree-based: single tree, forest, gradient boosting). Each group's first entry is its plain
# baseline where the group has one; the rest is its tuning grid. Groups are listed from simplest to most complex.
LOGISTIC = [(f"logistic_c{c:g}{'_bal' if w else ''}", "linear", {"C": c, "class_weight": w})
            for c in (1.0, 0.01, 0.1, 10.0) for w in (None, "balanced")]  # first: sklearn default (C=1, unweighted)
TREE = [("tree_plain", "tree", {"max_depth": None, "min_samples_leaf": 1})] + [
    (f"tree_d{d}_l{l}", "tree", {"max_depth": d, "min_samples_leaf": l}) for d in (4, 6, 8, 12) for l in (20, 100)]
FOREST = [(f"forest_l{l}_{'sqrt' if m == 'sqrt' else 'half'}", "forest", {"max_depth": None, "min_samples_leaf": l, "max_features": m})
          for l in (5, 20, 50) for m in ("sqrt", 0.5)]
BOOST = [(f"boost_lr{lr:g}_n{n}_l{l}", "boost", {"learning_rate": lr, "max_iter": n, "min_samples_leaf": l})
         for lr in (0.03, 0.1) for n in (200, 500) for l in (20, 100)]
CANDIDATES = LOGISTIC + TREE + FOREST + BOOST
GROUPS = [("linear", "logistic_c1", [n for n, _, _ in LOGISTIC]),
          ("tree", "tree_plain", [n for n, _, _ in TREE]),
          ("forest", None, [n for n, _, _ in FOREST]),
          ("boost", None, [n for n, _, _ in BOOST])]
SMOKE_GROUPS = [("linear", "logistic_c1", ["logistic_c1"]), ("tree", None, ["tree_d6_l100"]),
                ("forest", None, ["forest_l20_sqrt"]), ("boost", None, ["boost_lr0.1_n200_l100"])]
ACTIVE = {"groups": GROUPS}  # switched to SMOKE_GROUPS by run(smoke=True)
K_GRID = [k / 100 for k in range(1, 31)]
BENEFIT = 5.0  # a caught late order is worth 5x the cost of acting on one order
BENEFIT_SENSITIVITY = (2.0, 10.0, 20.0)
TOP_SHARE = 0.10
BOOTSTRAPS, BOOTSTRAP_SEED = 1000, 42
HANDOVER_INPUTS = ["approval_days", "handover_days", "route_transit_90d_at_handover", "remaining_slack"]
EXTENSION_INPUTS = ["zip3_transit_90d_at_handover", "seller_handover_90d_at_handover"]
HISTORY_DAYS = 90
ZIP_PRIOR_STRENGTH, SELLER_PRIOR_STRENGTH = 20, 10
SELLER_HANDOVER_RANGE = (0, 60)  # same validity range as the frozen seller_ship_days feature


# ---------------------------------------------------------------------------------------------- features

def add_remaining_slack(table):
    """promised_days - handover_days - route_transit_90d_at_handover; NaN where transit history is unknown."""
    out = table.copy()
    out["remaining_slack"] = out[PROMISE] - out["handover_days"] - out["route_transit_90d_at_handover"]
    return out


def add_history_extension(table, customer_zip):
    """Leak-free as-of-handover history for the pre-declared extension.

    `customer_zip` maps order_id to customer_zip_code_prefix. Both values use only events completed strictly before
    the order's own handover: ZIP-3 carrier-leg transit of deliveries already made, and the seller's handover speed
    (purchase to handover) of orders already handed over. The order's own outcome is never in its history.
    """
    out = table.merge(customer_zip, on="order_id", how="left", validate="one_to_one")
    if out["customer_zip_code_prefix"].isna().any():
        raise ValueError("Every order needs a customer ZIP prefix")
    zip3 = out["customer_zip_code_prefix"].astype(int).astype(str).str.zfill(5).str[:3]
    handover = out["handover_time"].where(out["handover_valid"])
    out["zip3_transit_90d_at_handover"] = asof_mean(
        zip3, out["transit_days"], out["outcome_available_at"], zip3, handover,
        HISTORY_DAYS, ZIP_PRIOR_STRENGTH)
    speed = out["handover_days"].where(out["handover_days"].between(*SELLER_HANDOVER_RANGE))
    out["seller_handover_90d_at_handover"] = asof_mean(
        out["furthest_seller_id"], speed, handover, out["furthest_seller_id"], handover,
        HISTORY_DAYS, SELLER_PRIOR_STRENGTH)
    return out.drop(columns="customer_zip_code_prefix")


def load_cohort(root=ROOT):
    table, spec, coverage = load_stage_table(root)
    table = add_remaining_slack(table)
    orders = pd.read_csv(Path(root) / "data/olist_orders_dataset.csv", usecols=["order_id", "customer_id"])
    customers = pd.read_csv(Path(root) / "data/olist_customers_dataset.csv",
                            usecols=["customer_id", "customer_zip_code_prefix"])
    zips = orders.merge(customers, on="customer_id", validate="many_to_one")[["order_id", "customer_zip_code_prefix"]]
    return add_history_extension(table, zips), spec, coverage


def variant_spec(spec, extra=(), handover=True):
    """Feature contract for a variant: spec features, plus handover inputs and optional extras."""
    changed = stage_spec(spec, "handover") if handover else copy.deepcopy(spec)
    if handover:
        changed["numeric_features"] = changed["numeric_features"] + ["remaining_slack"]
    changed["numeric_features"] = changed["numeric_features"] + list(extra)
    return changed


def candidate_table():
    names = [n for _, _, configs in ACTIVE["groups"] for n in configs]
    lookup = {name: (family, params) for name, family, params in CANDIDATES}
    return [(name, *lookup[name]) for name in names]


# -------------------------------------------------------------------------------------------- model fits

def fit_scores(family, params, spec, fit_rows, score_rows, log_features):
    """Fit one candidate on `fit_rows` and return risk scores for `score_rows`. Preprocessing is inside the Pipeline."""
    numeric, categorical = feature_columns(spec)
    pipeline = make_pipeline(family, params, spec, log_features=log_features)
    if family == "forest":
        pipeline.set_params(model__n_jobs=-1)
    with threadpool_limits(limits=None if family == "forest" else 1):
        pipeline.fit(fit_rows[numeric + categorical], fit_rows[LABEL])
        return pipeline.predict_proba(score_rows[numeric + categorical])[:, 1]


# ---------------------------------------------------------------------------------------- ranking metrics

def top_count(n, share):
    return int(np.ceil(share * n - 1e-9))


def top_flags(scores, share):
    """Boolean flags for the highest `share` of scores (ties broken by position, stable)."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    flags = np.zeros(len(order), dtype=bool)
    flags[order[:top_count(len(order), share)]] = True
    return flags


def action_counts(y, flags):
    y, flags = np.asarray(y).astype(bool), np.asarray(flags).astype(bool)
    return {"orders": len(y), "acted": int(flags.sum()), "late": int(y.sum()), "caught": int((y & flags).sum())}


def net_benefit(counts, benefit):
    return benefit * counts["caught"] - counts["acted"]


def action_metrics(y, flags):
    """Precision, recall, F1 and MCC of acting on the flagged orders."""
    c = action_counts(y, flags)
    precision = c["caught"] / c["acted"] if c["acted"] else 0.0
    recall = c["caught"] / c["late"] if c["late"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mcc = float(matthews_corrcoef(np.asarray(y).astype(bool), np.asarray(flags).astype(bool))) if c["acted"] else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def random_action_metrics(late_rate, share):
    """Expected values of acting on a random `share` of orders: precision = late rate, recall = share, MCC = 0."""
    precision, recall = late_rate, share
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall),
            "mcc": 0.0}


def ranking_metrics(y, scores):
    return {"roc_auc": float(roc_auc_score(y, scores)), "pr_auc": float(average_precision_score(y, scores)),
            "late_rate": float(np.mean(y)),
            "recall_top10": action_metrics(y, top_flags(scores, TOP_SHARE))["recall"],
            "precision_top10": action_metrics(y, top_flags(scores, TOP_SHARE))["precision"]}


# --------------------------------------------------------------------------------------------- selection

def select_candidate(pr_auc):
    """Tune within each group on mean validation PR-AUC, then the one-standard-error rule across the line-up.

    `pr_auc` maps configuration -> per-window PR-AUC. Returns (chosen, best, reason, details).
    """
    chosen, details = two_stage_select(ACTIVE["groups"], pr_auc, higher_is_better=True)
    best = details["best"]
    reason = (f"best {best} (mean validation PR-AUC {np.mean(pr_auc[best]):.4f}); chosen {chosen} "
              f"({np.mean(pr_auc[chosen]):.4f}), the simplest line-up entry within one standard error of the best")
    return chosen, best, reason, details


def mean_net_benefit_per_order(windows, share, benefit):
    """Mean over validation windows of net benefit per handover order when acting on the top `share` of each window.

    `windows` is a list of (labels, scores). Only validation data can be passed in, which is how the operating
    point stays independent of the test period.
    """
    values = [net_benefit(action_counts(y, top_flags(s, share)), benefit) / len(y) for y, s in windows]
    return float(np.mean(values))


def choose_k(windows, benefit=BENEFIT):
    """Share k from K_GRID maximising mean validation net benefit; the smaller k wins ties."""
    curve = [mean_net_benefit_per_order(windows, k, benefit) for k in K_GRID]
    return K_GRID[int(np.argmax(curve))], curve


def monthly_flags(scores, months, share):
    """Top `share` by score within each month (the period over which the team acts)."""
    flags = np.zeros(len(scores), dtype=bool)
    scores, months = np.asarray(scores, dtype=float), np.asarray(months)
    for month in np.unique(months):
        index = np.flatnonzero(months == month)
        flags[index[top_flags(scores[index], share)]] = True
    return flags


# ---------------------------------------------------------------------------------------------- bootstrap

def _sorted_weighted(y, scores, weights):
    order = np.argsort(-scores, kind="stable")
    return y[order].astype(float), weights[order].astype(float)


def weighted_pr_auc(y, scores, weights):
    """Average precision with observation weights (a bootstrap draw is an integer weight per order)."""
    ys, ws = _sorted_weighted(y, scores, weights)
    caught, seen = np.cumsum(ws * ys), np.cumsum(ws)
    positives = caught[-1]
    precision = np.divide(caught, seen, out=np.zeros_like(caught), where=seen > 0)
    return float(np.sum(ws * ys * precision) / positives)


def weighted_recall_top(y, scores, weights, share):
    """Recall in the top `share` of the weighted sample (fractional cut at the boundary order)."""
    ys, ws = _sorted_weighted(y, scores, weights)
    cut = share * ws.sum()
    seen_before = np.cumsum(ws) - ws
    take = np.clip(cut - seen_before, 0, ws)
    return float(np.sum(take * ys) / np.sum(ws * ys))


def paired_bootstrap(windows, a_name, b_name, n_boot=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """95% interval for score `a` minus score `b` in PR-AUC and recall at the top 10%.

    Orders are resampled with replacement within each validation window (one set of draws shared by both rankings)
    and each metric is the mean over windows, matching how the model is selected.
    """
    rng = np.random.default_rng(seed)
    point = {}
    draws = {"pr_auc": [], "recall_top10": []}
    for name in ("pr_auc", "recall_top10"):
        point[name] = float(np.mean([_metric(name, w["y"], w[a_name], np.ones(len(w["y"])))
                                     - _metric(name, w["y"], w[b_name], np.ones(len(w["y"]))) for w in windows]))
    for _ in range(n_boot):
        diffs = {"pr_auc": [], "recall_top10": []}
        for w in windows:
            n = len(w["y"])
            weights = rng.multinomial(n, np.full(n, 1.0 / n))
            for name in diffs:
                diffs[name].append(_metric(name, w["y"], w[a_name], weights) - _metric(name, w["y"], w[b_name], weights))
        for name in draws:
            draws[name].append(np.mean(diffs[name]))
    return {name: {"difference": point[name], "ci_low": float(np.percentile(draws[name], 2.5)),
                   "ci_high": float(np.percentile(draws[name], 97.5))} for name in draws}


def _metric(name, y, scores, weights):
    return (weighted_pr_auc(y, scores, weights) if name == "pr_auc"
            else weighted_recall_top(y, scores, weights, TOP_SHARE))


def gate_verdict(interval):
    """Gate 2 passes when both 95% intervals for model minus slack rule lie above zero."""
    return bool(all(interval[m]["ci_low"] > 0 for m in ("pr_auc", "recall_top10")))


# ---------------------------------------------------------------------------------- validation machinery

def baseline_windows(train, folds, spec, regression_model):
    """Per window: labels and the two baseline rankings. Everything is fitted on the window's fitting rows only."""
    family, params = regression_model
    reg_spec = stage_spec(spec, "handover")
    out = []
    for fit, valid in folds:
        fit_rows, valid_rows = train.iloc[fit], train.iloc[valid]
        out.append({"y": valid_rows[LABEL].to_numpy(), **baseline_scores(fit_rows, valid_rows, reg_spec, family, params)})
    return out


def baseline_scores(fit_rows, score_rows, reg_spec, family, params):
    median = fit_rows["route_transit_90d_at_handover"].median()
    slack = (score_rows[PROMISE] - score_rows["handover_days"]
             - score_rows["route_transit_90d_at_handover"].fillna(median))
    predicted = fit_predict(family, params, reg_spec, fit_rows, score_rows)
    return {"slack_rule": -slack.to_numpy(), "regression_gap": predicted - score_rows[PROMISE].to_numpy()}


def validate(train, folds, spec_v, log_features, label):
    """Four candidates over the validation windows; returns per-window metrics, scores and the selection."""
    rows, scores = [], {}
    for name, family, params in candidate_table():
        started = time.monotonic()
        scores[name] = []
        for window, (fit, valid) in enumerate(folds):
            valid_rows = train.iloc[valid]
            s = fit_scores(family, params, spec_v, train.iloc[fit], valid_rows, log_features)
            scores[name].append(s)
            rows.append({"variant": label, "candidate": name, "window": window, "fit_orders": len(fit),
                         "validation_orders": len(valid), **ranking_metrics(valid_rows[LABEL].to_numpy(), s)})
        frame = pd.DataFrame(rows)
        part = frame[frame["candidate"] == name]
        print(f"[{label}] {name}: mean PR-AUC={part['pr_auc'].mean():.4f}, ROC-AUC={part['roc_auc'].mean():.4f}, "
              f"recall@10%={part['recall_top10'].mean():.4f}; {time.monotonic() - started:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("candidate", sort=False)[["pr_auc", "roc_auc", "recall_top10", "precision_top10"]].mean()
    chosen, best, reason, details = select_candidate(
        frame.pivot(index="window", columns="candidate", values="pr_auc").to_dict("list"))
    return {"label": label, "windows": frame, "summary": summary, "scores": scores,
            "selected": chosen, "best": best, "reason": reason, "details": details}


def operating_point(result, baselines, benefit=BENEFIT):
    """k and validation metrics at k for the selected model and both baselines."""
    chosen = result["scores"][result["selected"]]
    ys = [w["y"] for w in baselines]
    k, curve = choose_k(list(zip(ys, chosen)), benefit)
    rows = []
    rankings = {"model": chosen, "slack_rule": [w["slack_rule"] for w in baselines],
                "regression_gap": [w["regression_gap"] for w in baselines]}
    for name, per_window in rankings.items():
        window_metrics = [action_metrics(y, top_flags(s, k)) for y, s in zip(ys, per_window)]
        rows.append({"ranking": name, "k": k, **{m: float(np.mean([w[m] for w in window_metrics]))
                                                 for m in window_metrics[0]},
                     "net_benefit_per_order": mean_net_benefit_per_order(list(zip(ys, per_window)), k, benefit),
                     **_mean_ranking(ys, per_window)})
    rate = float(np.mean([y.mean() for y in ys]))
    rows.append({"ranking": "random", "k": k, **random_action_metrics(rate, k),
                 "net_benefit_per_order": float(np.mean([(benefit * y.mean() * k - k) for y in ys])),
                 "roc_auc": 0.5, "pr_auc": rate, "late_rate": rate})
    return k, curve, pd.DataFrame(rows)


def _mean_ranking(ys, per_window):
    metrics = [ranking_metrics(y, s) for y, s in zip(ys, per_window)]
    return {m: float(np.mean([x[m] for x in metrics])) for m in ("roc_auc", "pr_auc", "late_rate")}


def value_test(result, baselines):
    windows = [{"y": b["y"], "model": s, "slack_rule": b["slack_rule"]}
               for b, s in zip(baselines, result["scores"][result["selected"]])]
    interval = paired_bootstrap(windows, "model", "slack_rule")
    return {"variant": result["label"], "selected": result["selected"], "n_resamples": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED, "model_minus_slack_rule": interval, "gate_2_pass": gate_verdict(interval)}


def sensitivity(result, baselines):
    rows = []
    for benefit in (BENEFIT, *BENEFIT_SENSITIVITY):
        k, curve = choose_k(list(zip([w["y"] for w in baselines], result["scores"][result["selected"]])), benefit)
        rows.append({"benefit_ratio": benefit, "k": k, "validation_net_benefit_per_order": max(curve)})
    return pd.DataFrame(rows)


def importance(train, folds, spec_v, log_features, selected, seed):
    """Permutation importance (drop in PR-AUC) of the selected model on the validation windows, mean over windows."""
    family, params = next((f, p) for n, f, p in candidate_table() if n == selected)
    numeric, categorical = feature_columns(spec_v)
    columns = numeric + categorical
    rows = []
    for window, (fit, valid) in enumerate(folds):
        pipeline = make_pipeline(family, params, spec_v, log_features=log_features)
        if family == "forest":
            pipeline.set_params(model__n_jobs=-1)
        pipeline.fit(train.iloc[fit][columns], train.iloc[fit][LABEL])
        r = permutation_importance(pipeline, train.iloc[valid][columns], train.iloc[valid][LABEL],
                                   scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1)
        rows.extend({"window": window, "feature": f, "importance": float(m)}
                    for f, m in zip(columns, r.importances_mean))
    return (pd.DataFrame(rows).groupby("feature")["importance"].mean().sort_values(ascending=False)
            .rename("mean_pr_auc_drop").reset_index())


# ---------------------------------------------------------------------------------------------------- test

def gains_curve(y, scores, months, grid=np.linspace(0, 1, 101)):
    y = np.asarray(y).astype(bool)
    return [float((y & monthly_flags(scores, months, s)).sum() / y.sum()) for s in grid]


def score_test(table, spec, spec_v, log_features, result, k, regression_model, label, benefit_rows):
    """Single refit on all training handover rows, then test metrics at k (flags chosen within each purchase month)."""
    train = table[table[spec["primary_split"]].eq("train")]
    test = table[table[spec["primary_split"]].eq("test")].reset_index(drop=True)
    family, params = next((f, p) for n, f, p in candidate_table() if n == result["selected"])
    model = fit_scores(family, params, spec_v, train, test, log_features)
    family_r, params_r = regression_model
    base = baseline_scores(train, test, stage_spec(spec, "handover"), family_r, params_r)
    y = test[LABEL].to_numpy()
    months = test["order_purchase_timestamp"].dt.to_period("M").astype(str).to_numpy()
    rankings = {"model": model, "slack_rule": base["slack_rule"], "regression_gap": base["regression_gap"]}
    rows = []
    for name, s in rankings.items():
        flags = monthly_flags(s, months, k)
        counts = action_counts(y, flags)
        rows.append({"variant": label, "ranking": name, "k": k, **action_metrics(y, flags),
                     "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                     "late_rate": float(y.mean()), "net_benefit_per_order": net_benefit(counts, BENEFIT) / len(y)})
    rate = float(y.mean())
    rows.append({"variant": label, "ranking": "random", "k": k, **random_action_metrics(rate, k),
                 "roc_auc": 0.5, "pr_auc": rate, "late_rate": rate,
                 "net_benefit_per_order": BENEFIT * rate * k - k})
    sens = []
    for _, row in benefit_rows.iterrows():
        for name, s in rankings.items():
            counts = action_counts(y, monthly_flags(s, months, row["k"]))
            sens.append({"variant": label, "benefit_ratio": row["benefit_ratio"], "k": row["k"], "ranking": name,
                         "test_net_benefit_per_order": net_benefit(counts, row["benefit_ratio"]) / len(y)})
    grid = np.linspace(0, 1, 101)
    gains = pd.DataFrame({"variant": label, "share_acted": grid, "model": gains_curve(y, model, months, grid),
                          "slack_rule": gains_curve(y, base["slack_rule"], months, grid),
                          "regression_gap": gains_curve(y, base["regression_gap"], months, grid),
                          "random": grid})
    flags = monthly_flags(model, months, k)
    monthly = pd.DataFrame({"month": months, "late": y, "acted": flags, "caught": y & flags}).groupby("month").agg(
        orders=("late", "size"), late=("late", "sum"), acted=("acted", "sum"), caught=("caught", "sum"))
    monthly["variant"] = label
    return pd.DataFrame(rows), pd.DataFrame(sens), gains, monthly.reset_index()


# ------------------------------------------------------------------------------------------------- driver

def run(root=ROOT, out_dir=RESULT_DIR, smoke=False):
    started = time.monotonic()
    root, out_dir = Path(root), Path(out_dir)
    ACTIVE["groups"] = SMOKE_GROUPS if smoke else GROUPS
    out_dir.mkdir(parents=True, exist_ok=True)
    table, spec, coverage = load_cohort(root)
    train = table[table[spec["primary_split"]].eq("train")].reset_index(drop=True)
    folds = chronological_folds(train, spec)
    log_features = spec["linear_log_candidates"]
    selection = json.loads((root / "results/phase2/eta/selection.json").read_text())
    regression_model = next((f, p) for n, f, p in ETA_CANDIDATES if n == selection["selected_candidate"])
    baselines = baseline_windows(train, folds, spec, regression_model)

    variants = {"primary": variant_spec(spec)}
    validated = {}

    def validate_variant(label, spec_v):
        result = validate(train, folds, spec_v, log_features, label)
        k, curve, op = operating_point(result, baselines)
        result.update(k=k, curve=curve, op=op, value=value_test(result, baselines),
                      sens=sensitivity(result, baselines))
        validated[label] = result
        print(f"[{label}] selected {result['selected']}; k={k:.0%}; gate 2 pass="
              f"{result['value']['gate_2_pass']}; {result['value']['model_minus_slack_rule']}", flush=True)

    validate_variant("primary", variants["primary"])
    if not validated["primary"]["value"]["gate_2_pass"]:
        variants["extension"] = variant_spec(spec, EXTENSION_INPUTS)
        validate_variant("extension", variants["extension"])

    # Checkout-stage appendix: same candidates, spec inputs only, same handover cohort and windows.
    checkout = validate(train, folds, variant_spec(spec, handover=False), log_features, "checkout")

    # Everything is fixed; score the test period once per variant.
    test_rows, sens_rows, gains, monthly = [], [], [], []
    for label, result in validated.items():
        t, s, g, m = score_test(table, spec, variants[label], log_features, result, result["k"],
                                regression_model, label, result["sens"])
        test_rows.append(t), sens_rows.append(s), gains.append(g), monthly.append(m)

    pd.concat([r["windows"] for r in validated.values()] + [checkout["windows"]]).to_csv(
        out_dir / "validation_windows.csv", index=False)
    pd.concat([r["summary"].assign(variant=r["label"]).reset_index() for r in [*validated.values(), checkout]]
              ).to_csv(out_dir / "validation_summary.csv", index=False)
    pd.concat([r["op"].assign(variant=r["label"]) for r in validated.values()]).to_csv(
        out_dir / "validation_operating_point.csv", index=False)
    pd.concat([pd.DataFrame({"variant": r["label"], "k": K_GRID, "mean_net_benefit_per_order": r["curve"]})
               for r in validated.values()]).to_csv(out_dir / "validation_k_curve.csv", index=False)
    pd.concat([r["sens"].assign(variant=r["label"]) for r in validated.values()]).to_csv(
        out_dir / "sensitivity_validation.csv", index=False)
    pd.concat(test_rows).to_csv(out_dir / "test_metrics.csv", index=False)
    pd.concat(sens_rows).to_csv(out_dir / "sensitivity_test.csv", index=False)
    pd.concat(gains).to_csv(out_dir / "test_gains_curve.csv", index=False)
    pd.concat(monthly).to_csv(out_dir / "test_monthly_counts.csv", index=False)

    for label, result in validated.items():
        importance(train, folds, variants[label], log_features, result["selected"], spec["random_state"]).to_csv(
            out_dir / f"{label}_permutation_importance.csv", index=False)

    out = {"coverage": {**coverage, "train_orders": int(len(train))},
           "benefit_ratio": BENEFIT, "k_grid": [K_GRID[0], K_GRID[-1]],
           "smoke": smoke, "groups": [{"group": g, "baseline": b, "configs": c} for g, b, c in ACTIVE["groups"]],
           "variants": {label: {"selected": r["selected"], "best_pr_auc_candidate": r["best"], "reason": r["reason"],
                                "tuned": r["details"]["tuned"], "lineup": r["details"]["lineup"],
                                "stage2": r["details"]["stage2"],
                                "k": r["k"], "value_test": r["value"]} for label, r in validated.items()},
           "checkout_variant": {"selected": checkout["selected"], "reason": checkout["reason"],
                                "tuned": checkout["details"]["tuned"], "stage2": checkout["details"]["stage2"],
                                **checkout["summary"].loc[checkout["selected"]].to_dict()},
           "extension_run": "extension" in validated,
           "counterfactual_simulation": "not implemented in this run (depends on the promise engine output)",
           "runtime_seconds": round(time.monotonic() - started, 1)}
    (out_dir / "selection.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Done in {out['runtime_seconds']}s", flush=True)
    return validated, checkout, pd.concat(test_rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one configuration per group; use with --out")
    parser.add_argument("--out", default=str(RESULT_DIR), help="output directory")
    args = parser.parse_args()
    run(out_dir=args.out, smoke=args.smoke)
