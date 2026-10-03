"""As-of history features for Phase 2: statistics built only from outcomes known before each purchase.

Shared by data_prep.ipynb (which builds the order table) and the tests (which verify the result by brute force).
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
        position = np.searchsorted(known["time"].to_numpy(), group["time"].to_numpy(), side="left")
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
