> **Superseded.** This review describes the first classification run on the original 18-feature set (with `purchase_month`). The feature contract has since changed (schema version 2, 20 features) and all results were regenerated. The numbers below are kept for history only; see [the notebook plan](phase2_classification_notebook_plan.md) and `results/phase2/classification/` for the current results.

# Chunk 2: classification review

The provisional choice is **logistic regression with C=0.1 and no class weighting**. This is validation evidence for the agreed delivered-only population, not final test performance. The complete experiment and figures are in [classification.ipynb](../notebooks/phase2/classification.ipynb); machine-readable evidence is in [results/phase2/classification](../results/phase2/classification/).

## Candidate comparison

All candidates use the same five temporal validation blocks (38,005 validation orders in total), with fold-specific imputation and encoding. Earlier training purchases must have been delivered before the validation block starts. PR-AUC here means average precision, averaged equally over the five folds. We tested 13 declared configurations, then three paired checks dropping payment features.

| Approach | Mean validation PR-AUC | Interpretation |
|---|---:|---|
| Training-prior baseline | 0.1067 | Constant probability; each fold's AP equals its late rate |
| Default logistic regression, C=1 | 0.1698 | Improves ranking over the baseline |
| Selected logistic regression, C=0.1 | **0.1808** | Best tested mean AP; stronger regularisation helps |
| Unrestricted decision tree | 0.1089 | Little improvement over the prior baseline |
| Best decision tree, depth 12 / leaf 20 | 0.1389 | Constraining the tree improves validation performance |
| Initial random forest | 0.1416 | Improves on the unrestricted single tree |
| Best random forest, depth 12 / leaf 20, without payments | 0.1693 | Better than the single tree, but below the selected linear model |

The decision tree and random forest belong to the same tree-based family. Retain the selected logistic and forest configurations for Chunk 4's explicit voting comparison. Extra complexity has not yet beaten the linear approach. These are results of a bounded search, not proof of an optimal model or statistically significant differences.

Fold prevalence ranges from 4.08% to 20.73%, and the selected logistic model's fold AP ranges from 0.0654 to 0.3660. Its mean fold ROC-AUC is 0.6384. The variation is material: the model has modest discrimination, and pooled scores alone conceal temporal differences.

## Payment sensitivity

Keeping the selected parameters fixed, removing `payment_type` and `max_installments` changes mean AP by -0.000054 for logistic regression, -0.002501 for the decision tree, and +0.000380 for the forest. The logistic difference is negligible for practical interpretation; this experiment provides little evidence that payment inputs materially improve that model. The strict highest-mean-AP rule retains them in the provisional logistic choice. This comparison does not verify their historical checkout availability.

## Alert threshold and operational meaning

The selected threshold is **0.0800304**, maximising MCC over the selected model's pooled validation predictions. The default 0.5 threshold generates no alerts. At the selected threshold:

| Measure | Validation result |
|---|---:|
| Precision: alerts that are truly late | 26.58% |
| Recall: late orders caught | 17.71% |
| F1 | 0.2126 |
| MCC | 0.1426 |
| Correct late alerts | 718 |
| False alerts | 1,983 |
| Missed late orders | 3,336 |
| Correct on-time classifications | 31,968 |

That means 2,701 alerts among 38,005 validation orders. It catches fewer than one in five late deliveries, while roughly three in four alerts are false. This is not yet a convincing operational alert system. MCC supplies a statistical threshold because no intervention cost or alert-capacity constraint was specified; another business priority could justify a different tradeoff.

The threshold was chosen and described on the same validation predictions, after model selection. These threshold metrics are therefore optimistic selection evidence, not an independent assessment. We must keep the chosen procedure fixed for the later holdout. Per-fold threshold results, the precision–recall curve, and confusion matrix are saved alongside the notebook.

## Verification and review gate

The full training workflow completed in a fresh notebook kernel. The review notebook also reruns from saved evidence with input and implementation hash checks. Focused tests cover training-fold boundaries, unavailable labels, preprocessing isolation, unseen categories, threshold optimisation with tied scores, and exact reproduction of saved metrics without holdout rows. Float round-trip parsing preserves scores at the threshold boundary.

The order table and feature contract were not changed. No primary holdout score, secondary random benchmark, or final model fit has been produced. The 60-day validation maturity gap, residual censoring, and assumed checkout availability remain documented limitations. Chunk 2 stops here for review; regression is the next chunk, with voting and final evaluation later.
