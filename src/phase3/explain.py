"""Exact per-input contributions for the linear late-risk champion.

The Phase 2 classifier is a logistic regression behind a ``ColumnTransformer``:
numeric inputs are median-imputed and standardised, skewed inputs are logged
before scaling, ``purchase_hour`` becomes a sine/cosine pair, and categorical
inputs become one-hot columns.  For a linear model on the logit scale the SHAP
values are exact and additive:

    contribution_j = coefficient_j * (transformed_j - background_mean_j)

where the background mean is the mean of each transformed column over the
training rows the champion was fitted on.  Contributions are aggregated back to
the original input names, so every numeric or categorical input gets one number
and ``base_value + sum(contributions) == logit`` for every scored row.

This module deliberately fails loudly if the champion is not linear: a tree
champion has no exact additive decomposition of this kind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_LABELS = {
    "promised_days": "Olist's promised days",
    "promise_slack": "Promise slack (days)",
    "route_typical_days": "Route's usual days",
    "seller_ship_days": "Seller dispatch time",
    "distance_km": "Distance",
    "same_state": "Same state",
    "purchase_hour": "Purchase hour",
    "n_items": "Items",
    "n_products": "Products",
    "n_sellers": "Sellers",
    "total_price": "Item value",
    "total_freight": "Freight",
    "total_weight_g": "Weight",
    "total_volume_cm3": "Volume",
    "max_installments": "Installments",
    "approval_days": "Approval days",
    "handover_days": "Days from purchase to handover",
    "route_transit_90d_at_handover": "Route's usual carrier time",
    "remaining_slack": "Spare time",
    "past_promise_at_handover": "Already past promise",
    "slack_shortfall": "Spare time below zero",
    "state_backlog_recent_vs_8wk": "Regional backlog",
    "customer_state": "Customer state",
    "seller_state": "Seller state",
    "product_category": "Product category",
    "payment_type": "Payment type",
    "purchase_dayofweek": "Purchase weekday",
}


def input_label(name):
    """A plain-language label for an original model input."""
    return INPUT_LABELS.get(name, str(name).replace("_", " "))


def _dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix, dtype=float)


def _column_mean(matrix):
    mean = matrix.mean(axis=0)
    return np.asarray(mean, dtype=float).ravel()


def _as_columns(columns):
    if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
        return list(columns)
    return [columns]


def original_inputs(prepare):
    """Map each transformed column to the original input it came from.

    ``ColumnTransformer.get_feature_names_out`` is prefixed (``numeric__...``),
    so the branches are walked directly and their own output names are used.
    """
    mapping = []
    for name, transformer, columns in prepare.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        columns = _as_columns(columns)
        if transformer == "passthrough":
            mapping.extend(columns)
            continue
        if name == "hour":
            output = list(transformer.get_feature_names_out(columns))
            if output != ["hour_sin", "hour_cos"]:
                raise ValueError("Unexpected purchase-hour expansion: {}".format(output))
            mapping.extend(["purchase_hour"] * len(output))
            continue
        output = list(transformer.get_feature_names_out(columns))
        if name == "category":
            for output_name in output:
                match = None
                for column in sorted(columns, key=len, reverse=True):
                    if output_name == column or output_name.startswith(column + "_"):
                        match = column
                        break
                if match is None:
                    raise ValueError(
                        "Cannot map categorical output {!r} back to {}".format(output_name, columns)
                    )
                mapping.append(match)
        else:
            if len(output) != len(columns):
                raise ValueError(
                    "Branch {!r} expanded {} inputs into {} outputs".format(name, len(columns), len(output))
                )
            mapping.extend(columns)
    return mapping


class LinearExplainer:
    """Exact additive contributions for a fitted linear Phase 2 pipeline."""

    def __init__(self, pipeline, columns, training_rows):
        try:
            prepare = pipeline.named_steps["prepare"]
            model = pipeline.named_steps["model"]
        except (AttributeError, KeyError) as error:
            raise ValueError("The champion is not a Phase 2 prepare/model pipeline") from error
        if not hasattr(model, "coef_"):
            raise ValueError(
                "Exact per-input contributions need a linear champion; "
                "this model has no coefficients"
            )
        self.pipeline = pipeline
        self.prepare = prepare
        self.model = model
        self.columns = list(columns)
        self.inputs = original_inputs(prepare)
        self.coef = np.asarray(model.coef_, dtype=float).reshape(-1)
        self.background = _column_mean(prepare.transform(training_rows[self.columns]))
        if len(self.coef) != len(self.background):
            raise ValueError(
                "The champion has {} coefficients but {} transformed columns".format(
                    len(self.coef), len(self.background)
                )
            )
        if len(self.inputs) != len(self.coef):
            raise ValueError(
                "The champion has {} coefficients but {} mapped inputs".format(
                    len(self.coef), len(self.inputs)
                )
            )
        intercept = getattr(model, "intercept_", 0.0)
        self.intercept = float(np.asarray(intercept).reshape(-1)[0])
        self.base_value = self.intercept + float(np.dot(self.coef, self.background))
        self.original_inputs = list(dict.fromkeys(self.inputs))

    def explain(self, rows):
        """Return (contributions by original input, logits) for ``rows``.

        ``contributions`` is a DataFrame indexed like ``rows`` with one column
        per original input.  ``base_value + contributions.sum(axis=1)`` equals
        the model logit for every row.
        """
        transformed = _dense(self.prepare.transform(rows[self.columns]))
        if transformed.shape[1] != len(self.coef):
            raise ValueError(
                "Prepared {} columns but the champion has {} coefficients".format(
                    transformed.shape[1], len(self.coef)
                )
            )
        values = (transformed - self.background) * self.coef
        contributions = pd.DataFrame(0.0, index=rows.index, columns=self.original_inputs)
        for position, name in enumerate(self.inputs):
            contributions[name] += values[:, position]
        logits = self.intercept + transformed @ self.coef
        return contributions, logits

    def top(self, contributions_row, count=3):
        """The ``count`` inputs that push this row's logit the most upward."""
        positive = contributions_row[contributions_row > 0]
        ordered = positive.sort_values(ascending=False, kind="stable").head(count)
        return [
            {"input": name, "label": input_label(name), "contribution": float(value)}
            for name, value in ordered.items()
        ]
