"""Read-only daily actions API for the Actions tab.

Two strategies are served for a chosen as-of date:

* ``GET /api/actions/queue?date=YYYY-MM-DD`` - the 09:00 rescue queue (Winnie's
  B_mix: late-risk and updated spare-time percentiles against the training
  orders, with the served champion as the model score), plus the exact
  linear-model contributions that push each parcel's late risk up.
* ``GET /api/actions/promises?date=YYYY-MM-DD`` - Jiayi's monthly promise
  policy (run B delivery-time estimate plus a regional 3.5% buffer).
* ``GET /api/actions/order?date=YYYY-MM-DD&order_id=...`` - whether one order
  is on that morning's list, or why it is not open then.

The queue and promises endpoints also return a live performance line
recomputed for the as-of date from outcomes known by then.  Model scores are
loaded once from the local MLflow store and reused; a missing store answers 503
rather than taking the whole dashboard down.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
for _entry in (str(REPO_ROOT / "src"), str(APP_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

PROMISE_RUN = "B"  # run B drops promised_days and promise_slack (18 inputs)
PROMISE_TARGET = 0.035
PERFORMANCE_WEEKS = 4

from build_overview_data import MIN_SAMPLE  # noqa: E402  shared minimum-orders rule

PROMISE_MIN_ORDERS = MIN_SAMPLE
PROMISE_DUE_SHARE = 0.99
PREDICTION_LOG_MAX_LINES = 10000
PREDICTION_LOG_LOCK = threading.Lock()
NO_CAPACITY_NOTE = (
    "No parcels to act on this morning: the daily list is 10% of yesterday's "
    "handovers, and there were none."
)
NOT_SCORED_MESSAGE = (
    "This order is not scored: it has no usable carrier handover, so it never appears "
    "on a morning list."
)


class ActionsUnavailable(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def promise_cohort(frame, as_of, policy_start):
    """The latest 4-week purchase cohort whose promises have essentially all passed.

    The window slides back from ``as_of`` until at least 99% of the orders
    bought in it have both promised dates before 09:00 on ``as_of``.  The
    returned ``due`` frame holds exactly those resolved orders, so ours and
    Olist's promises are compared on the same purchases.  Returns ``None``
    when no such window fits inside the policy's coverage.
    """
    import pandas as pd

    as_of = pd.Timestamp(as_of).normalize()
    cutoff = as_of + pd.Timedelta(hours=9)
    purchase = frame["order_purchase_timestamp"]
    purchase_day = purchase.dt.normalize()
    our_date = purchase_day + pd.to_timedelta(frame["our_promise_days"], unit="D")
    olist_date = purchase_day + pd.to_timedelta(frame["olist_promise_days"], unit="D")
    is_due = our_date.lt(cutoff) & olist_date.lt(cutoff)

    span = pd.Timedelta(weeks=PERFORMANCE_WEEKS)
    end = as_of
    while end - span >= policy_start:
        start = end - span
        in_window = purchase.ge(start) & purchase.lt(end)
        window_orders = int(in_window.sum())
        if window_orders and float(is_due[in_window].mean()) >= PROMISE_DUE_SHARE:
            return {
                "start": start,
                "end": end,
                "window_orders": window_orders,
                "due": frame[in_window & is_due],
            }
        end = end - pd.Timedelta(days=1)
    return None


def promise_metrics(due, cutoff):
    """Mean promise lengths and late rates for one already-resolved cohort.

    Lateness is on calendar days: actual delivery date after the promised date. For
    Olist's promise this must equal the raw ``is_late`` label; it is asserted here.
    """
    import numpy as np
    import pandas as pd

    delivered = due["outcome_available_at"].lt(cutoff)
    purchase_day = due["order_purchase_timestamp"].dt.normalize()
    actual_day = due["outcome_available_at"].dt.normalize()
    our_date = purchase_day + pd.to_timedelta(due["our_promise_days"], unit="D")
    olist_date = purchase_day + pd.to_timedelta(due["olist_promise_days"], unit="D")
    if not (actual_day.gt(olist_date).eq(due["is_late"].astype(bool))).all():
        raise ValueError("Calendar-day late definition does not equal the raw is_late label")
    our_late = (~delivered) | actual_day.gt(our_date)
    olist_late = (~delivered) | due["is_late"].astype(bool)
    return {
        "our_mean_promise_days": float(due["our_promise_days"].mean()),
        "our_late_rate": float(our_late.mean()),
        "olist_mean_promise_days": float(due["olist_promise_days"].mean()),
        "olist_late_rate": float(olist_late.mean()),
    }


class ActionsService:
    """The champion queue and promise policy, loaded once and reused."""

    def __init__(self, store=None, root=None, capacity=None, preload=True,
                 prediction_log=None):
        self.store = store
        self.root = root
        self.capacity = None if capacity is None else int(capacity)
        self.prediction_log = (Path(prediction_log) if prediction_log is not None
                               else APP_ROOT / "data" / "prediction_log.jsonl")
        self.error = None
        self.champions = None
        self.table = None
        self.spec = None
        self.context = None
        self.versions = None
        self._queue_cache = {}
        self._promise_cache = {}
        self._policy_table = None
        self._policy_spec = None
        self._eta_model = None
        self._eta_all = None
        self._month_frames = {}
        self._promise_tables = {}
        self._handover_by_order = None
        self._raw_order_ids = None
        self._month_states = {}
        if preload:
            try:
                self.load()
            except Exception as error:
                self.error = str(error)

    def load(self):
        from phase3 import queue as queue_module
        from phase3 import serving

        if self.root is None:
            root = queue_module.ROOT
        else:
            root = Path(self.root)
        champions = serving.load_champions(store=self.store, spec=None)
        table, spec, _ = queue_module.load_queue_table(root)
        table = table.reset_index(drop=True)
        context = serving.champion_context(champions, table, spec)
        self.champions = champions
        self.table = table
        self.spec = spec
        self.context = context
        self.versions = serving.model_versions(champions)
        self._queue_cache = {}
        self._promise_cache = {}
        self._policy_table = None
        self._policy_spec = None
        self._eta_model = None
        self._eta_all = None
        self._month_frames = {}
        self._promise_tables = {}
        self._handover_by_order = None
        self._raw_order_ids = None
        self._month_states = {}
        self.error = None
        return self

    # ------------------------------------------------------------------ queue

    def _date_state(self, as_of):
        """The late-risk model, context, version and explainer that serve ``as_of``.

        Late risk is the month's registered refit; the delivery-time models stay
        frozen.  States are cached per month, so one model load and one cohort
        scoring are reused for every date in that month.
        """
        import pandas as pd

        from phase3 import serving

        key = str(pd.Period(as_of, freq="M"))
        state = self._month_states.get(key)
        if state is None:
            model, version = serving.load_late_risk_for_date(as_of, store=self.store)
            context = serving.monthly_context(self.champions, model, self.table, self.spec)
            state = {"context": context, "model": model, "version": int(version.version),
                     "explainer": None, "refs": None}
            self._month_states[key] = state
        return state

    def _state_versions(self, state):
        return {**self.versions, "late_risk": state["version"]}

    def _month_explainer(self, state):
        if state["explainer"] is None:
            from phase3 import explain

            table = state["context"].table
            train = table[table[self.spec["primary_split"]].eq("train")]
            state["explainer"] = explain.LinearExplainer(
                state["model"], self.champions.classifier_columns, train)
        return state["explainer"]

    def _explanations(self, frame, as_of, state):
        """Top positive contributions for every row in the queue frame."""
        import pandas as pd

        explainer = self._month_explainer(state)
        if frame.empty:
            return {}, float(explainer.base_value)
        open_time = as_of + pd.Timedelta(hours=9)
        table = state["context"].table
        open_rows = table[
            table["handover_time"].lt(open_time)
            & (table["outcome_available_at"].isna()
               | table["outcome_available_at"].gt(open_time))
        ].set_index("order_id", drop=False)
        order_ids = [str(value) for value in frame["order_id"]]
        missing = [order_id for order_id in order_ids if order_id not in open_rows.index]
        if missing:
            raise RuntimeError(
                "The queue contains {} orders outside the scored open cohort".format(len(missing))
            )
        contributions, _ = explainer.explain(open_rows.loc[order_ids])
        return {
            order_id: explainer.top(contributions.loc[order_id])
            for order_id in order_ids
        }, explainer.base_value

    def _daily_capacity(self, as_of):
        """The day's Act block: an explicit override, otherwise Winnie's daily size."""
        from phase3 import queue as queue_module

        if self.capacity is not None:
            return self.capacity
        # Yesterday's handovers include parcels still with the carrier: they are
        # part of the volume that sets today's review size.
        return queue_module.daily_capacity(self.context.table, as_of)

    def queue(self, date):
        """Return the queue payload for day ``date`` (a date or ISO string)."""
        if self.context is None:
            raise ActionsUnavailable(self.error or "The champion queue is not loaded")
        from phase3 import queue as queue_module

        as_of = _parse_date(date)
        key = str(as_of.date())
        cached = self._queue_cache.get(key)
        if cached is not None:
            payload = dict(cached)
            payload["cached"] = True
            return payload

        started = time.perf_counter()
        state = self._date_state(as_of)
        versions = self._state_versions(state)
        capacity = self._daily_capacity(as_of)
        frame = queue_module.winnie_queue(as_of, context=state["context"], capacity=capacity)
        frame["tier"] = _tier(frame, capacity)
        import pandas as pd

        past = frame[frame["past_promise"]].sort_values(
            ["promise_date", "order_id"], kind="stable", na_position="last")
        ranked = frame[~frame["past_promise"]].sort_values(
            ["rank", "order_id"], kind="stable", na_position="last")
        frame = pd.concat([ranked, past], ignore_index=True)
        why, why_base_value = self._explanations(frame, as_of, state)
        performance = self._queue_performance(as_of, state)
        rows = [_row(row, versions, why[str(row["order_id"])])
                for _, row in frame.iterrows()]
        counts = {"act": int(frame["tier"].eq("act").sum()),
                  "watch": int(frame["tier"].eq("watch").sum()),
                  "none": int((frame["tier"].eq("none") & ~frame["past_promise"]).sum())}
        payload = {
            "ok": True,
            "date": str(as_of.date()),
            "as_of": (as_of + datetime.timedelta(hours=9)).isoformat(),
            "method": "blend",
            "capacity": capacity,
            "capacity_note": NO_CAPACITY_NOTE if capacity == 0 else None,
            "model_versions": versions,
            "why_base_value": float(why_base_value),
            "why_scale": "logit",
            "tiers": counts,
            "customer_update": int(frame["past_promise"].sum()),
            "performance": performance,
            "rows": rows,
            "elapsed_ms": _millis(started),
            "cached": False,
        }
        self._queue_cache[key] = payload
        return payload

    def _queue_performance(self, as_of, state):
        """Winnie's rescue metrics over the four weeks before ``as_of``.

        Both lines use her 10% daily size and selection window.  The blend is
        B_mix scored by the month's late-risk refit; the comparison is her B_rule,
        the remaining-slack ranking under the same daily process.  A parcel counts only when its delivery outcome was
        recorded before the selected date and it was not already past its
        promise at review.  The headline is late parcels flagged with at least
        three days to act; the raw late counts stay in the payload for the hover.
        """
        import pandas as pd

        from phase3 import queue as queue_module

        start = (as_of - pd.Timedelta(weeks=PERFORMANCE_WEEKS)).normalize()
        end = (as_of - pd.Timedelta(days=1)).normalize()
        cutoff = as_of + pd.Timedelta(hours=9)
        context = state["context"]
        process_start = queue_module._winnie_process_start(context) + pd.Timedelta(days=1)
        performance = {
            "weeks": PERFORMANCE_WEEKS,
            "start": str(start.date()),
            "end": str(end.date()),
            "daily_capacity": self._daily_capacity(as_of),
            "outcome_cutoff": cutoff.isoformat(),
        }

        blend_detail, _ = queue_module.winnie_daily_replay(
            context, end=end - pd.Timedelta(days=1))
        if len(blend_detail):
            blend_detail = blend_detail[
                blend_detail["morning"].ge(start) & blend_detail["morning"].le(end)]
        known = blend_detail[blend_detail["outcome_available_at"].lt(cutoff)]
        performance["blend"] = queue_module.rescue_metrics(known)

        slack_detail, _ = queue_module.winnie_daily_replay(
            context, end=end - pd.Timedelta(days=1), version="B_rule")
        if len(slack_detail):
            slack_detail = slack_detail[
                slack_detail["morning"].ge(start) & slack_detail["morning"].le(end)]
        known = slack_detail[slack_detail["outcome_available_at"].lt(cutoff)]
        performance["slack"] = queue_module.rescue_metrics(known)
        for metrics in (performance["blend"], performance["slack"]):
            if metrics["caught"] < MIN_SAMPLE:
                metrics["mean_days_left"] = None
        return performance

    # --------------------------------------------------------------- promises

    def _ensure_policy(self):
        if self._policy_table is not None:
            return
        import pandas as pd

        from phase3 import policy

        try:
            table, spec_full = policy.load_order_table()
            spec = policy.run_spec(spec_full, PROMISE_RUN)
            model = policy.fit_frozen_model(PROMISE_RUN, spec, table)
            eta_all = pd.Series(policy.predict_eta(model, spec, table), index=table.index)
        except Exception as error:
            raise ActionsUnavailable(
                "The promise policy could not be loaded: {}".format(error)
            ) from error
        self._policy_table = table
        self._policy_spec = spec
        self._eta_model = model
        self._eta_all = eta_all
        self._month_frames = {}
        self._promise_tables = {}

    def _check_promise_month(self, month):
        import pandas as pd

        first = self._policy_table.loc[
            self._policy_table["purchase_month"].eq(month), "order_purchase_timestamp"
        ].min()
        cutoff = pd.Timestamp(self._policy_spec["primary_cutoff"])
        if pd.notna(first) and first < cutoff:
            raise ValueError(
                "The promise policy scores months on or after the Phase 2 cutoff; "
                "move the selected date to June 2018 or later."
            )

    def _policy_start(self):
        """First day of the first month the live policy covers (June 2018)."""
        import pandas as pd

        cutoff = pd.Timestamp(self._policy_spec["primary_cutoff"])
        start = cutoff.to_period("M").to_timestamp()
        if start < cutoff:
            start = start + pd.offsets.MonthBegin(1)
        return start

    def _month_frame(self, month):
        import numpy as np

        from phase3 import policy

        key = str(month)
        if key not in self._month_frames:
            scored, buffer = policy.buffers_for(
                self._policy_table, month, self._eta_all, self._eta_all,
                PROMISE_TARGET, regional=True,
            )
            promise = np.ceil(self._eta_all.reindex(scored.index) + buffer)
            self._month_frames[key] = scored.assign(
                our_promise_days=promise,
                olist_promise_days=np.ceil(scored["promised_days"]),
                buffer_days=buffer,
            )
        return self._month_frames[key]

    def _promise_table(self, month, as_of):
        import pandas as pd

        from phase3 import policy

        as_of = pd.Timestamp(as_of).normalize()
        key = (str(month), str(as_of.date()))
        if key not in self._promise_tables:
            full = policy.promise_table(
                month.to_timestamp(), target=PROMISE_TARGET, run=PROMISE_RUN,
                model=self._eta_model, table=self._policy_table,
                spec=self._policy_spec, regional=True,
            )
            frame = self._month_frame(month)
            cutoff = as_of + pd.Timedelta(hours=9)
            known = frame[frame["order_purchase_timestamp"].lt(cutoff)]
            columns = ["region", "buffer_days", "olist_mean_promise_days",
                       "our_mean_promise_days", "n_orders"]
            rows = []
            for _, base in full.iterrows():
                selected = known[known["region"].eq(base["region"])]
                enough = len(selected) >= MIN_SAMPLE
                rows.append({
                    "region": base["region"],
                    "buffer_days": float(base["buffer_days"]),
                    "olist_mean_promise_days": (float(selected["olist_promise_days"].mean())
                                                if enough else None),
                    "our_mean_promise_days": (float(selected["our_promise_days"].mean())
                                              if enough else None),
                    "n_orders": int(len(selected)),
                })
            self._promise_tables[key] = pd.DataFrame(rows, columns=columns)
        return self._promise_tables[key]

    def _promise_performance(self, as_of):
        """Compare our promises with Olist's on the same resolved purchases.

        The cohort is the latest 4-week purchase window in which at least 99%
        of orders have both promised dates before 09:00 on ``as_of``.  Orders
        whose promises have not passed are dropped from both sides.  Lateness
        is known once the promised date passes: an order delivered after the
        promise is late, and an order still undelivered when its promised date
        has passed is late too.
        """
        import pandas as pd

        as_of = pd.Timestamp(as_of).normalize()
        cutoff = as_of + pd.Timedelta(hours=9)
        policy_start = self._policy_start()
        last_month = min(as_of.to_period("M"),
                         self._policy_table["purchase_month"].max())
        first_month = policy_start.to_period("M")
        frames = []
        if last_month >= first_month:
            for month in pd.period_range(first_month, last_month, freq="M"):
                frame = self._month_frame(month)
                if not frame.empty:
                    frames.append(frame)
        cohort = promise_cohort(pd.concat(frames), as_of, policy_start) if frames else None
        if cohort is None:
            return {
                "start": None,
                "end": None,
                "orders": 0,
                "not_yet_due": 0,
                "enough": False,
            }
        due = cohort["due"]
        orders = int(len(due))
        performance = {
            "start": str(cohort["start"].date()),
            "end": str(cohort["end"].date()),
            "orders": orders,
            "window_orders": int(cohort["window_orders"]),
            "not_yet_due": int(cohort["window_orders"] - orders),
            "enough": orders >= PROMISE_MIN_ORDERS,
        }
        if not orders:
            return performance
        performance.update(promise_metrics(due, cutoff))
        return performance

    def promises(self, date):
        """Return the monthly promise table and month-to-date performance."""
        if self.context is None:
            raise ActionsUnavailable(self.error or "The champion queue is not loaded")
        import pandas as pd

        as_of = _parse_date(date)
        key = str(as_of.date())
        cached = self._promise_cache.get(key)
        if cached is not None:
            payload = dict(cached)
            payload["cached"] = True
            return payload

        started = time.perf_counter()
        self._ensure_policy()
        month = as_of.to_period("M")
        self._check_promise_month(month)
        table = self._promise_table(month, as_of)
        regions = [
            {
                "region": str(row["region"]),
                "buffer_days": float(row["buffer_days"]),
                "our_mean_promise_days": (None if pd.isna(row["our_mean_promise_days"])
                                          else float(row["our_mean_promise_days"])),
                "olist_mean_promise_days": (None if pd.isna(row["olist_mean_promise_days"])
                                            else float(row["olist_mean_promise_days"])),
                "n_orders": int(row["n_orders"]),
            }
            for _, row in table.sort_values("n_orders", ascending=False).iterrows()
        ]
        payload = {
            "ok": True,
            "date": str(as_of.date()),
            "month": str(month),
            "run": PROMISE_RUN,
            "target": PROMISE_TARGET,
            "model_label": "Checkout delivery-time model without Olist's promise as an input",
            "regions": regions,
            "performance": self._promise_performance(as_of),
            "elapsed_ms": _millis(started),
            "cached": False,
        }
        self._promise_cache[key] = payload
        return payload

    def _raw_order_ids_set(self):
        """Every order ID in the raw Olist orders file, for not-scored lookups."""
        if self._raw_order_ids is None:
            import pandas as pd

            from phase3 import queue as queue_module

            root = Path(self.root) if self.root is not None else queue_module.ROOT
            path = root / "data" / "raw" / "olist_orders_dataset.csv"
            if not path.is_file():
                raise ActionsUnavailable(
                    "The raw order records are unavailable at {}".format(path))
            frame = pd.read_csv(path, usecols=["order_id"])
            self._raw_order_ids = set(frame["order_id"].astype(str))
        return self._raw_order_ids

    # ----------------------------------------------------------------- lookup

    def order_lookup(self, date, order_id):
        """Say whether one order is on the chosen morning's list, or why it is not."""
        if self.context is None:
            raise ActionsUnavailable(self.error or "The champion queue is not loaded")
        import pandas as pd

        from phase3 import queue as queue_module

        as_of = _parse_date(date)
        order_id = _parse_order_id(order_id)
        payload = self.queue(date)
        for row in payload["rows"]:
            if row["order_id"] != order_id:
                continue
            if row["past_promise"]:
                return {
                    "ok": True,
                    "order_id": order_id,
                    "found": True,
                    "state": "past_promise",
                    "message": "This order is still open but its promised date ({}) has passed, "
                    "so it is in the customer-update list.".format(_plain_date(row["promised_date"])),
                }
            return {
                "ok": True,
                "order_id": order_id,
                "found": True,
                "state": "open",
                "rank": row["rank"],
                "message": "This order is on today's list at rank {}.".format(row["rank"]),
            }

        table = self.context.table
        match = table[table["order_id"].astype(str).eq(order_id)]
        if match.empty:
            if order_id in self._raw_order_ids_set():
                return {
                    "ok": True,
                    "order_id": order_id,
                    "found": False,
                    "state": "not_scored",
                    "message": NOT_SCORED_MESSAGE,
                }
            return {
                "ok": True,
                "order_id": order_id,
                "found": False,
                "state": "unknown",
                "message": "No order with that ID is in the order records.",
            }
        record = match.iloc[0]
        purchased = record["order_purchase_timestamp"]
        handover = record["handover_time"]
        delivered = record["outcome_available_at"]
        open_time = as_of + datetime.timedelta(hours=9)
        morning = _plain_date(as_of)
        if purchased >= open_time:
            state, message = "not_bought", "This order had not been bought yet at 09:00 on {}.".format(morning)
        elif pd.isna(handover) or handover >= open_time:
            state, message = ("not_handed_over",
                              "This order had not been handed to the carrier yet at 09:00 on {}.".format(morning))
        elif pd.notna(delivered) and delivered <= open_time:
            state, message = ("delivered",
                              "This order had already been delivered by 09:00 on {}.".format(morning))
        elif handover < queue_module._winnie_process_start(self.context):
            state, message = ("before_review_period",
                              "This order was handed over before the daily review period, so it is not on the list.")
        else:
            state, message = ("reviewed_earlier",
                              "This order is still open but was picked on an earlier morning, so it is not "
                              "listed again.")
        return {"ok": True, "order_id": order_id, "found": False, "state": state, "message": message}

    # -------------------------------------------------------------- prediction

    def _prediction_reference(self, state):
        """The training distributions the live queue compares a parcel against."""
        from phase3 import queue as queue_module

        if state["refs"] is None:
            fit_rows = queue_module._winnie_fit_rows(state["context"])
            model_ref = fit_rows["risk"].to_numpy(dtype=float)
            rule_ref = queue_module.winnie_rule_scores(fit_rows, fit_rows)
            state["refs"] = (fit_rows, model_ref, rule_ref)
        return state["refs"]

    def _handover_times(self):
        if self._handover_by_order is None:
            table = self.context.table
            self._handover_by_order = dict(
                zip(table["order_id"].astype(str), table["handover_time"]))
        return self._handover_by_order

    def _would_rank(self, payload, blend, handover, order_id):
        """Where a parcel would sit in today's ranked pool, with stable tie-breaking."""
        import pandas as pd

        candidates = []
        for row in payload.get("rows", []):
            if (row.get("rank") is None or row.get("late_risk_percentile") is None
                    or row.get("spare_time_percentile") is None):
                continue
            candidates.append((
                (row["late_risk_percentile"] + row["spare_time_percentile"]) / 2.0,
                self._handover_times().get(row["order_id"], pd.Timestamp.max),
                row["order_id"],
            ))
        key = (handover, order_id)
        ahead = sum(
            1 for candidate_blend, candidate_handover, candidate_id in candidates
            if candidate_blend > blend
            or (candidate_blend == blend and (candidate_handover, candidate_id) < key)
        )
        return ahead + 1, len(candidates) + 1

    def predict(self, date, order_id):
        """Score one handed-over, still-open parcel exactly as today's queue would."""
        if self.context is None:
            raise ActionsUnavailable(self.error or "The champion queue is not loaded")
        import numpy as np
        import pandas as pd

        from phase3 import queue as queue_module

        started = time.perf_counter()
        as_of = _parse_date(date)
        order_id = _parse_order_id(order_id)
        month_state = self._date_state(as_of)
        versions = self._state_versions(month_state)
        queue_payload = self.queue(date)
        queue_row = next(
            (row for row in queue_payload.get("rows", []) if row["order_id"] == order_id), None)
        table = month_state["context"].table
        match = table[table["order_id"].astype(str).eq(order_id)]
        if match.empty:
            not_scored = order_id in self._raw_order_ids_set()
            payload = {
                "ok": True,
                "date": str(as_of.date()),
                "order_id": order_id,
                "found": False,
                "state": "not_scored" if not_scored else "unknown",
                "on_list": False,
                "customer_update": False,
                "model_versions": versions,
                "message": NOT_SCORED_MESSAGE if not_scored
                else "No order with that ID is in the order records.",
            }
            latency = _millis(started)
            payload["latency_ms"] = latency
            self._log_prediction(as_of, order_id, payload, None, latency)
            return payload

        record = match.iloc[0]
        target = match.iloc[[0]]
        purchased = record["order_purchase_timestamp"]
        handover = record["handover_time"]
        delivered = record["outcome_available_at"]
        open_time = as_of + pd.Timedelta(hours=9)
        morning = _plain_date(as_of)
        if purchased >= open_time:
            state, message = "not_bought", "This order had not been bought yet at 09:00 on {}.".format(morning)
        elif pd.isna(handover) or handover >= open_time:
            state, message = ("not_handed_over",
                              "This order had not been handed to the carrier yet at 09:00 on {}.".format(morning))
        elif pd.notna(delivered) and delivered <= open_time:
            state, message = ("delivered",
                              "This order had already been delivered by 09:00 on {}.".format(morning))
        else:
            state, message = None, None
        if state is not None:
            payload = {
                "ok": True,
                "date": str(as_of.date()),
                "order_id": order_id,
                "found": True,
                "state": state,
                "on_list": False,
                "customer_update": False,
                "model_versions": versions,
                "message": message,
            }
            latency = _millis(started)
            payload["latency_ms"] = latency
            self._log_prediction(as_of, order_id, payload, float(record["risk"]), latency)
            return payload

        late_risk_score = float(record["risk"])
        if queue_row is not None:
            if queue_row["past_promise"]:
                state = "past_promise"
                message = ("This order is still open but its promised date ({}) has passed, "
                           "so it is in the customer-update list.".format(
                               _plain_date(queue_row["promised_date"])))
                payload = {
                    "ok": True,
                    "date": str(as_of.date()),
                    "order_id": order_id,
                    "found": True,
                    "state": state,
                    "on_list": False,
                    "customer_update": True,
                    "rank": queue_row["rank"],
                    "open_orders": queue_row["open_orders"],
                    "late_risk_percentile": queue_row["late_risk_percentile"],
                    "spare_time": queue_row["spare_time"],
                    "spare_time_percentile": queue_row["spare_time_percentile"],
                    "predicted_delivery_date": queue_row["predicted_delivery_date"],
                    "promised_date": queue_row["promised_date"],
                    "days_left": queue_row["days_left"],
                    "tier": queue_row["tier"],
                    "why": queue_row["why"],
                    "model_versions": versions,
                    "message": message,
                }
            else:
                payload = {
                    "ok": True,
                    "date": str(as_of.date()),
                    "order_id": order_id,
                    "found": True,
                    "state": "open",
                    "on_list": True,
                    "customer_update": False,
                    "rank": queue_row["rank"],
                    "open_orders": queue_row["open_orders"],
                    "late_risk_percentile": queue_row["late_risk_percentile"],
                    "spare_time": queue_row["spare_time"],
                    "spare_time_percentile": queue_row["spare_time_percentile"],
                    "predicted_delivery_date": queue_row["predicted_delivery_date"],
                    "promised_date": queue_row["promised_date"],
                    "days_left": queue_row["days_left"],
                    "tier": queue_row["tier"],
                    "why": queue_row["why"],
                    "model_versions": versions,
                    "message": "This order is on today's list at rank {}.".format(queue_row["rank"]),
                }
            latency = _millis(started)
            payload["latency_ms"] = latency
            self._log_prediction(as_of, order_id, payload, late_risk_score, latency)
            return payload

        fit_rows, model_ref, rule_ref = self._prediction_reference(month_state)
        late_risk_percentile = float(
            queue_module.winnie_percentile(model_ref, np.array([late_risk_score], dtype=float))[0])
        rule_score = float(queue_module.winnie_rule_scores(fit_rows, target)[0])
        elapsed_days = (open_time - purchased).total_seconds() / 86400.0
        days_left_score = float(record[queue_module.PROMISE]) - elapsed_days
        updated_rule = -min(-rule_score, days_left_score)
        spare_time = -updated_rule
        spare_time_percentile = float(
            queue_module.winnie_percentile(rule_ref, np.array([updated_rule], dtype=float))[0])
        blend = (late_risk_percentile + spare_time_percentile) / 2.0
        rank, open_orders = self._would_rank(queue_payload, blend, handover, order_id)
        capacity = int(queue_payload.get("capacity") or 0)
        if handover < queue_module._winnie_process_start(month_state["context"]):
            state = "before_review_period"
            message = "This order was handed over before the daily review period, so it is not on the list."
        else:
            state = "reviewed_earlier"
            message = ("This order is still open but was picked on an earlier morning, so it is not "
                       "listed again.")
        explainer = self._month_explainer(month_state)
        contributions, _ = explainer.explain(target)
        why = explainer.top(contributions.iloc[0])
        payload = {
            "ok": True,
            "date": str(as_of.date()),
            "order_id": order_id,
            "found": True,
            "state": state,
            "on_list": False,
            "customer_update": False,
            "rank": rank,
            "open_orders": open_orders,
            "late_risk_percentile": late_risk_percentile,
            "spare_time": spare_time,
            "spare_time_percentile": spare_time_percentile,
            "predicted_delivery_date": _predicted_delivery_date(
                purchased, record["eta_days"]),
            "promised_date": str(record["promise_date"].date()),
            "days_left": int((record["promise_date"] - as_of).days),
            "tier": _tier_for_rank(rank, capacity),
            "why": why,
            "model_versions": versions,
            "message": message,
        }
        latency = _millis(started)
        payload["latency_ms"] = latency
        self._log_prediction(as_of, order_id, payload, late_risk_score, latency)
        return payload

    def _log_prediction(self, as_of, order_id, payload, late_risk_score, latency):
        """Append one JSON line per predict call, capped to the last 10,000 lines."""
        line = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"),
            "as_of_date": str(as_of.date()),
            "order_id": order_id,
            "state": payload.get("state"),
            "on_list": bool(payload.get("on_list")),
            "model_versions": dict(payload.get("model_versions") or {}),
            "late_risk_score": None if late_risk_score is None else float(late_risk_score),
            "late_risk_percentile": payload.get("late_risk_percentile"),
            "predicted_delivery_date": payload.get("predicted_delivery_date"),
            "spare_time": payload.get("spare_time"),
            "rank": payload.get("rank"),
            "latency_ms": latency,
        }
        path = Path(self.prediction_log)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with PREDICTION_LOG_LOCK:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line, sort_keys=True) + "\n")
                lines = path.read_text(encoding="utf-8").splitlines()
                if len(lines) > PREDICTION_LOG_MAX_LINES:
                    path.write_text(
                        "\n".join(lines[-PREDICTION_LOG_MAX_LINES:]) + "\n",
                        encoding="utf-8",
                    )
        except OSError as error:
            print("The prediction log could not be written: {}".format(error), file=sys.stderr)

    # --------------------------------------------------------------- responses

    def queue_response(self, date):
        try:
            return 200, self.queue(date)
        except ActionsUnavailable as error:
            return 503, {"ok": False, "error": {"code": "actions_unavailable",
                                                "message": error.message}}
        except ValueError as error:
            return 400, {"ok": False, "error": {"code": "bad_request", "message": str(error)}}
        except Exception:
            _log_error("queue")
            return 500, {"ok": False, "error": {"code": "server_error",
                                                "message": "The queue could not be built. Please try again."}}

    def promises_response(self, date):
        try:
            return 200, self.promises(date)
        except ActionsUnavailable as error:
            return 503, {"ok": False, "error": {"code": "actions_unavailable",
                                                "message": error.message}}
        except ValueError as error:
            return 400, {"ok": False, "error": {"code": "bad_request", "message": str(error)}}
        except Exception:
            _log_error("promises")
            return 500, {"ok": False, "error": {"code": "server_error",
                                                "message": "The promise table could not be built. Please try again."}}

    def order_response(self, date, order_id):
        try:
            return 200, self.order_lookup(date, order_id)
        except ActionsUnavailable as error:
            return 503, {"ok": False, "error": {"code": "actions_unavailable",
                                                "message": error.message}}
        except ValueError as error:
            return 400, {"ok": False, "error": {"code": "bad_request", "message": str(error)}}
        except Exception:
            _log_error("order")
            return 500, {"ok": False, "error": {"code": "server_error",
                                                "message": "The order could not be looked up. Please try again."}}

    def predict_response(self, date, order_id):
        try:
            return 200, self.predict(date, order_id)
        except ActionsUnavailable as error:
            return 503, {"ok": False, "error": {"code": "actions_unavailable",
                                                "message": error.message}}
        except ValueError as error:
            return 400, {"ok": False, "error": {"code": "bad_request", "message": str(error)}}
        except Exception:
            _log_error("predict")
            return 500, {"ok": False, "error": {"code": "server_error",
                                                "message": "The prediction could not be made. Please try again."}}


MONTH_NAMES = ("January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December")


def _plain_date(value):
    """A calendar date in words, e.g. ``9 July 2018``."""
    if value is None:
        return ""
    day = value.date() if hasattr(value, "date") else datetime.date.fromisoformat(str(value)[:10])
    return "{} {} {}".format(day.day, MONTH_NAMES[day.month - 1], day.year)


def _parse_order_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("The order ID is required.")
    order_id = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", order_id):
        raise ValueError("Order IDs are 32 characters long, using letters a-f and digits 0-9.")
    return order_id


def _log_error(where):
    """Keep the detail in the server log; the client only sees a plain message."""
    print("Actions {} failed:\n{}".format(where, traceback.format_exc()), file=sys.stderr)


def _millis(started):
    return round((time.perf_counter() - started) * 1000.0, 1)


def _parse_date(value):
    import pandas as pd

    if not isinstance(value, str) or not value.strip():
        raise ValueError("The date query parameter is required (YYYY-MM-DD).")
    try:
        day = datetime.date.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError("The date must be a valid YYYY-MM-DD calendar date.") from error
    return pd.Timestamp(day).normalize()


def _tier(frame, capacity):
    """act = the first daily capacity; watch = the next block; none = the rest."""
    ranks = frame["rank"].to_numpy(dtype=float)
    past = frame["past_promise"].to_numpy(dtype=bool)
    tier = [None] * len(frame)
    for position, rank in enumerate(ranks):
        if past[position] or rank != rank:
            tier[position] = "none"
        elif rank <= capacity:
            tier[position] = "act"
        elif rank <= 2 * capacity:
            tier[position] = "watch"
        else:
            tier[position] = "none"
    return tier


def _tier_for_rank(rank, capacity):
    """The block a rank would fall in on a normal morning."""
    if capacity <= 0 or rank is None:
        return "none"
    if rank <= capacity:
        return "act"
    if rank <= 2 * capacity:
        return "watch"
    return "none"


def _row(row, versions, why):
    promised = row["promise_date"]
    purchased = row["order_purchase_timestamp"]
    return {
        "order_id": str(row["order_id"]),
        "rank": None if _is_nan(row["rank"]) else int(row["rank"]),
        "tier": row["tier"],
        "eta_days": float(row["eta_days"]),
        "predicted_delivery_date": _predicted_delivery_date(purchased, row["eta_days"]),
        "promised_date": None if promised is None or promised != promised else str(promised.date()),
        "days_left": None if _is_nan(row["days_until_promise"]) else int(row["days_until_promise"]),
        "spare_time": float(row["spare_time_used"]),
        "late_risk_percentile": None if _is_nan(row["late_risk_percentile"])
        else float(row["late_risk_percentile"]),
        "spare_time_percentile": None if _is_nan(row["spare_time_percentile"])
        else float(row["spare_time_percentile"]),
        "model_rank": None if _is_nan(row["model_rank"]) else int(row["model_rank"]),
        "open_orders": int(row["open_orders"]),
        "model_versions": dict(versions),
        "past_promise": bool(row["past_promise"]),
        "customer_update": bool(row["past_promise"]),
        "suggested_action": str(row["suggested_action"]),
        "why": why,
    }


def _predicted_delivery_date(purchased, eta_days):
    if purchased is None or _is_nan(purchased) or _is_nan(eta_days):
        return None
    return str((purchased + datetime.timedelta(days=float(eta_days))).date())


def _is_nan(value):
    try:
        return value != value
    except TypeError:
        return False
