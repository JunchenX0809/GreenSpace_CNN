# Process Update — 2026-06-28

## Confusion Matrix Analysis — Current PyTorch Model

Model: `PyTorch_20260614_220926/best_mcmae`

Data: labeled test split, `n = 1,049`

Note: these confusion matrices are test-set analyses. The inference cache does not have true labels, so it cannot produce a true confusion matrix yet.

## Score Confusion Matrix

Rows are true score class. Columns are predicted score class. Each cell shows count and row percent.

| True \ Pred | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 74 (40.2%) | 68 (37.0%) | 32 (17.4%) | 4 (2.2%) | 6 (3.3%) |
| 2 | 40 (19.8%) | 82 (40.6%) | 65 (32.2%) | 14 (6.9%) | 1 (0.5%) |
| 3 | 6 (2.4%) | 46 (18.0%) | 123 (48.2%) | 71 (27.8%) | 9 (3.5%) |
| 4 | 0 (0.0%) | 8 (3.8%) | 49 (23.6%) | 95 (45.7%) | 56 (26.9%) |
| 5 | 0 (0.0%) | 0 (0.0%) | 4 (2.0%) | 36 (18.0%) | 160 (80.0%) |

Score summary:

| Metric | Value |
| --- | ---: |
| Exact class accuracy | 50.9% |
| Class MAE after rounding | 0.587 |

Important interpretation:

The score model predicts a continuous value (`score_ev`) during inference. The confusion matrix rounds that continuous value back into a hard class. This makes the confusion matrix stricter than MAE.

| True score | Predicted `score_ev` | Rounded class | Confusion matrix result | MAE error |
| ---: | ---: | ---: | --- | ---: |
| 4 | 3.49 | 3 | Wrong | 0.51 |
| 4 | 1.00 | 1 | Wrong | 3.00 |

The confusion matrix treats both examples as equally wrong, but MAE does not. MAE preserves how far the prediction is from the true score.

Observations:

- Score class `5` is the clearest class: 160 of 200 true class-5 examples were predicted as class 5.
- Most score mistakes are near-neighbor errors, especially `2 ↔ 3`, `3 ↔ 4`, and `4 ↔ 5`.
- Middle score classes remain less sharply separated than the upper tail.
- True score class `1` is often lifted to class `2` or `3`, so low-score separation is weaker than high-score separation.
- Extreme mistakes are uncommon; the model rarely predicts class `1` as `5` or class `5` as `1`.

## Veggie Confusion Matrix

Rows are true Veggie/vegetation class. Columns are predicted Veggie/vegetation class. Each cell shows count and row percent.

| True \ Pred | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 (36.4%) | 47 (53.4%) | 7 (8.0%) | 1 (1.1%) | 1 (1.1%) |
| 2 | 19 (6.0%) | 233 (73.7%) | 59 (18.7%) | 4 (1.3%) | 1 (0.3%) |
| 3 | 3 (1.1%) | 47 (16.7%) | 192 (68.3%) | 36 (12.8%) | 3 (1.1%) |
| 4 | 1 (0.5%) | 6 (2.7%) | 50 (22.5%) | 135 (60.8%) | 30 (13.5%) |
| 5 | 1 (0.7%) | 0 (0.0%) | 6 (4.2%) | 33 (23.2%) | 102 (71.8%) |

Veggie summary:

| Metric | Value |
| --- | ---: |
| Exact class accuracy | 66.2% |
| Class MAE after rounding | 0.378 |

Observations:

- Veggie has a stronger diagonal than score, meaning exact class agreement is better overall.
- Classes `2`, `3`, and `5` are the strongest Veggie classes.
- True Veggie class `1` is often predicted as class `2`, so the model tends to soften the lowest vegetation class upward.
- True Veggie class `4` is mostly correct, but still has meaningful spillover into classes `3` and `5`.
- Extreme mistakes are rare; most Veggie errors remain close to the true class.

## Binary Confusion Matrix Summary

Each binary variable uses its validation-tuned threshold from `thresholds_best_mcmae.csv`.

| Label | Positive support | Negative support | Threshold | FP | FN | Precision | Recall | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sports_field | 273 | 776 | 0.294 | 57 | 32 | 0.809 | 0.883 | 0.927 | 0.844 |
| multipurpose_open_area | 834 | 215 | 0.483 | 72 | 39 | 0.917 | 0.953 | 0.665 | 0.935 |
| children_s_playground | 125 | 924 | 0.370 | 81 | 33 | 0.532 | 0.736 | 0.912 | 0.617 |
| water_feature | 231 | 818 | 0.456 | 47 | 52 | 0.792 | 0.775 | 0.943 | 0.783 |
| walking_paths | 808 | 241 | 0.342 | 94 | 31 | 0.892 | 0.962 | 0.610 | 0.926 |
| built_structures | 430 | 619 | 0.480 | 80 | 71 | 0.818 | 0.835 | 0.871 | 0.826 |
| parking_lots | 320 | 729 | 0.432 | 72 | 40 | 0.795 | 0.875 | 0.901 | 0.833 |

Binary observations:

- `walking_paths` and `multipurpose_open_area` have very high recall, so the model catches most positives for these labels.
- `walking_paths` and `multipurpose_open_area` also have lower specificity, meaning they over-predict presence when the feature is absent.
- `children_s_playground` remains the weakest binary label: recall is moderate, but precision is low, so many predicted positives are false positives.
- `water_feature` is more balanced, with similar precision and recall and strong specificity.
- `sports_field`, `built_structures`, and `parking_lots` show generally usable behavior, with both false positives and false negatives in a manageable range.

## Regression-Style Evaluation Options

For score and Veggie, the model's native inference output is continuous (`score_ev`, `veg_ev`). Confusion matrices are useful diagnostics, but MAE-style metrics are better aligned with the regression training objective.

Current continuous test metrics:

| Target | MAE vs hard class | MAE vs rater mean |
| --- | ---: | ---: |
| Score | 0.627 | 0.576 |
| Veggie | 0.451 | 0.430 |

Additional useful regression-style checks:

| Check | What it tells us |
| --- | --- |
| MAE by true class | Which score/Veggie classes have the largest distance errors. |
| Median absolute error | Typical error size without being dominated by a few large misses. |
| Percent within 0.5 points | How often the continuous prediction is close enough to round to the same class. |
| Percent within 1.0 point | How often the model is directionally close even if exact class accuracy is low. |
| Bias by class | Whether the model systematically pushes low classes upward or high classes downward. |
| Prediction vs truth scatter/table | Whether predictions track the ordering of the true labels. |
| Error distribution | Whether most mistakes are small or whether there are many large misses. |
