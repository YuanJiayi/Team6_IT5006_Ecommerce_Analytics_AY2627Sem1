"""Two-stage model selection shared by the promise engine and the handover late classifier.

Stage 1 (tuning): within each model group, the configuration with the best mean validation score.
Stage 2 (selection): the line-up "plain baseline, tuned" for each group in order of complexity, and the paired
one-standard-error rule: the first (simplest) entry whose mean gap to the best entry is within one standard error of
that gap across the validation windows. A more complex model is chosen only if it beats every simpler one by more than
the window-to-window noise (James, Witten, Hastie and Tibshirani, An Introduction to Statistical Learning, Sec.
6.1.3, p. 214). Standard in statistical learning (it is glmnet's default `lambda.1se`), applied here across model
types rather than within one penalty path; the standard error comes from 5 non-independent validation windows.
"""

from __future__ import annotations

import numpy as np


def paired_gap(scores, best, higher_is_better):
    """Mean and standard error over windows of how much worse `scores` is than `best` (>= 0 means worse)."""
    a, b = np.asarray(scores, dtype=float), np.asarray(best, dtype=float)
    gap = b - a if higher_is_better else a - b
    se = float(gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1 else 0.0
    return float(gap.mean()), se


def one_se_choice(order, per_window, higher_is_better):
    """First entry of `order` within one standard error of the best entry. Returns (chosen, best, rows)."""
    mean = {n: float(np.mean(per_window[n])) for n in order}
    best = (max if higher_is_better else min)(order, key=mean.get)
    rows, chosen = [], None
    for n in order:
        gap, se = paired_gap(per_window[n], per_window[best], higher_is_better)
        rows.append({"candidate": n, "mean": mean[n], "gap_to_best": gap, "se_of_gap": se, "within_one_se": gap <= se})
        if chosen is None and gap <= se:
            chosen = n
    return chosen, best, rows


def two_stage_select(groups, per_window, higher_is_better):
    """Tune within each group, then choose across groups with the one-standard-error rule.

    `groups` is an ordered list of (group, baseline, configs): groups from simplest to most complex; `baseline` is the
    group's plain variant (or None when the group has no separate plain model); `configs` includes the baseline.
    `per_window` maps configuration -> per-window scores; configurations missing from it are ineligible.
    Returns (chosen, details) where details holds the tuned configuration per group and the stage-2 rows.
    """
    tuned, lineup = {}, []
    pick = max if higher_is_better else min
    for group, baseline, configs in groups:
        eligible = [c for c in configs if c in per_window]
        if not eligible:
            continue
        tuned[group] = pick(eligible, key=lambda c: float(np.mean(per_window[c])))
        for entry in (baseline, tuned[group]):
            if entry is not None and entry in per_window and entry not in lineup:
                lineup.append(entry)
    if not lineup:
        raise ValueError("No eligible configuration")
    chosen, best, rows = one_se_choice(lineup, per_window, higher_is_better)
    return chosen, {"tuned": tuned, "lineup": lineup, "best": best, "stage2": rows}
