"""As-of history features for Phase 2: statistics built only from outcomes known before each purchase.

Shared by data_prep.ipynb (which builds the order table), the stage models in eta.py and handover_classifier.py,
and the tests (which verify the result by brute force).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_of_counts_and_sums(history_keys, history_values, history_time, query_keys, query_time):
    """For each query row, count and sum the history values with the same key and time strictly before the query.

    A history row is usable only once its own time has passed (for example, the delivery time of an earlier order),
    so a row can never see its own outcome or any later one. Missing history values are ignored.
    """
    history = pd.DataFrame({"key": np.asarray(history_keys), "value": np.asarray(history_values, dtype=float),
                            "time": pd.to_datetime(np.asarray(history_time))}).dropna()
    queries = pd.DataFrame({"key": np.asarray(query_keys), "time": pd.to_datetime(np.asarray(query_time))})
    count = np.zeros(len(queries))
    total = np.zeros(len(queries))
    history = history.sort_values("time", kind="stable")
    by_key = {key: group for key, group in history.groupby("key", sort=False)}
    for key, group in queries.groupby("key", sort=False):
        known = by_key.get(key)
        if known is None:
            continue
        known_times = known["time"].to_numpy()
        query_times = group["time"].to_numpy()
        position = np.searchsorted(known_times, query_times, side="left")
        # Assert the point-in-time contract on the exact history rows used below.
        has_history = position > 0
        assert np.all(known_times[position[has_history] - 1] < query_times[has_history]), (
            "A history outcome was not known before its query purchase"
        )
        cumulative = np.r_[0.0, np.cumsum(known["value"].to_numpy())]
        rows = queries.index.get_indexer(group.index)
        count[rows] = position
        total[rows] = cumulative[position]
    return count, total


def as_of_shrunk_mean(history_keys, history_values, history_time, query_keys, query_time, strength):
    """Per-key mean of earlier history, pulled toward the overall mean of all earlier history.

    The overall mean is also computed as-of each query time, so no later information enters the prior.
    A query with no earlier history for its key gets the overall mean; a query with no earlier history at all
    is NaN and is left for the model pipeline's training-fold imputer.
    """
    n_key, sum_key = as_of_counts_and_sums(history_keys, history_values, history_time, query_keys, query_time)
    constant = np.zeros(len(history_values), dtype=int)
    n_all, sum_all = as_of_counts_and_sums(constant, history_values, history_time,
                                           np.zeros(len(query_time), dtype=int), query_time)
    overall = np.divide(sum_all, n_all, out=np.full(len(n_all), np.nan), where=n_all > 0)
    shrunk = (sum_key + overall * strength) / (n_key + strength)
    return np.where(n_key > 0, shrunk, overall)


def asof_mean(keys, values, available, query_keys, queries, days=None, strength=20):
    """Shrunk keyed mean of events available in [query-days, query)."""
    hist = pd.DataFrame({
        "key": np.asarray(keys, dtype=object),
        "value": np.asarray(values, dtype=float),
        "time": np.asarray(pd.to_datetime(available)),
    }).dropna(subset=["key", "value", "time"]).sort_values("time", kind="stable")
    query = pd.DataFrame({"key": np.asarray(query_keys, dtype=object),
                          "time": np.asarray(pd.to_datetime(queries))})
    result = np.full(len(query), np.nan)
    span = None if days is None else np.timedelta64(days, "D")

    def summarize(h, qt):
        times = h.time.to_numpy(dtype="datetime64[ns]")
        vals = h.value.to_numpy(dtype=float)
        end = np.searchsorted(times, qt, side="left")
        start = np.zeros(len(qt), dtype=int) if span is None else np.searchsorted(times, qt - span, side="left")
        sums = np.r_[0.0, np.cumsum(vals)]
        return end - start, sums[end] - sums[start]

    all_count, all_sum = summarize(hist, query.time.to_numpy(dtype="datetime64[ns]"))
    overall = np.divide(all_sum, all_count, out=np.full(len(query), np.nan), where=all_count > 0)
    groups = {key: group for key, group in hist.groupby("key", sort=False)}
    for key, part in query.groupby("key", sort=False):
        rows = part.index.to_numpy()
        if key not in groups:
            result[rows] = overall[rows]
            continue
        count, total = summarize(groups[key], part.time.to_numpy(dtype="datetime64[ns]"))
        prior = overall[rows]
        if strength == 0:
            result[rows] = np.divide(total, count, out=prior.copy(), where=count > 0)
        else:
            result[rows] = np.divide(total + strength * prior, count + strength,
                                     out=np.full(len(rows), np.nan), where=np.isfinite(prior))
    return result
