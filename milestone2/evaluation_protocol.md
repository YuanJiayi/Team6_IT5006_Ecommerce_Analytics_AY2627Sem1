# Phase 2 evaluation protocol

Both tasks predict outcomes for an order at purchase time: whether it arrives after its estimated **calendar date**, and the number of days from purchase to delivery. They use the same delivered-order population and order-level features.

## Primary: future-order test

The primary test holds out orders purchased on or after **2018-05-26**, approximately the latest 20% of the eligible delivered orders by purchase date. Training uses earlier purchases **only if their delivery outcome was known before that date**. Orders bought earlier but delivered on or after the cutoff are excluded from this training snapshot. The later orders form the test set; neither task uses their outcomes for preprocessing, model selection, tuning, or threshold choice. This estimates performance on a later period of Olist orders, which matches the intended use of making predictions for future purchases.

Within the primary training set, use five expanding-time validation windows. For each window, fit on earlier purchases whose delivery outcomes were known before that window starts, then validate on purchases in that window. Fit imputers, encoders, scalers, feature selection, and any resampling within each fit only. Choose models and thresholds from these training-period windows, then evaluate once on the held-out later orders. Compare performance by purchase month within the held-out period to see whether the aggregate hides an unusual month. The test period's late rate and delivery-time distribution may differ from training; report both rather than forcing them to match.

## Secondary: same-history random benchmark

Retain the teammate's seeded, stratified random 80/20 split and five stratified folds as a **separate secondary experiment**. It estimates performance on unseen orders drawn from the same overall historical mix. A model trained for this experiment can include orders purchased after some of its test orders, so its score does not support a claim about deployment on later orders. Its test rows and training rows may overlap the primary experiment's partitions; never combine the two training sets or compare their scores as though they were measured on the same test population. Report the secondary result as context, not as the basis for choosing the final model or claiming future-order performance.

The notebook saves `split` and `cv_fold` for the primary experiment, plus `random_split` and `random_cv_fold` for the secondary one. `outcome_available_at` is split metadata, not a model feature. The primary folds require explicit expanding-window fit indices; passing `cv_fold` to `PredefinedSplit` would train on future folds and is invalid. See the notebook's loading example.

## Interpretation and limits

This is a retrospective test on the finite 2016–2018 Olist sample, not proof of performance after 2018. The delivered-only population omits orders with no recorded delivery, and later purchases can be affected by the dataset's end. Both the late rate and promised delivery window changed over time, so report the period shift and simple baselines alongside model metrics. The calendar date on which we run the analysis does not change the fixed historical cutoff.
