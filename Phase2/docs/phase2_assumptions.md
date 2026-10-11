# Phase 2 data and prediction assumptions

The final report states these boundaries in its Data Preparation and Limitations sections. This note collects them for anyone rerunning the code.

- The two regressions predict total purchase-to-delivery days. Checkout features come from the purchase, seller, customer, items and stored promise; the handover update may use elapsed time, approval time already observed, and route history available by handover.
- The classifier ranks handed-over orders for late delivery relative to Olist's estimated **calendar date**. Arrival on that date is on time. Delivery and review outcomes are never used as predictors for that order.
- The final modelling cohort contains eventual deliveries with a valid purchase-to-carrier-to-customer sequence. Cancelled, unfinished and invalid-sequence orders are outside this target population.
- Order-item and payment rows are aggregated to one row per order. The furthest seller represents multi-seller distance. ZIP-prefix coordinates are medians after implausible points are removed; distance is a straight-line proxy.
- The recorded promise, payment details and static ZIP map have no historical snapshot timestamps in the supplied data. Their exact checkout availability is assumed, not demonstrated.
- Training examples enter a fit only after their outcomes are recorded. Route and seller historical means use events completed strictly before the prediction time. Imputation, encoding and scaling are fitted within each chronological training fold.
- The five earlier chronological windows support model selection. The later test period describes performance under changed conditions. Monthly refitting and drift analyses were explored after that period was available and should not be presented as independent confirmation.
- Reviewing a flagged parcel has no observed carrier action, cost or measured causal effect in Olist's tables. The 10% queue is an illustrative workload setting.

The executable feature contract is [`data/phase2_feature_spec.json`](../data/phase2_feature_spec.json), and the construction logic and timing checks are in [`phase2_features.py`](../phase2_features.py), [`phase2_eta.py`](../phase2_eta.py) and [`tests/`](../tests/).
