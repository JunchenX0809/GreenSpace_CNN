# Process Update — 2026-05-09

## Scope (planned — no implementation yet)
---


trajectory so far with score & veg:
- **Iteration 1**: Hard labels (`score_class`, sparse CE) → low accuracy on val
- **Iteration 2**: Soft labels (`score_p_1..5`, categorical CE) → same generalization gap
- **Iteration 3 (this plan)**: Continuous target (`score_mean`, MAE) → ?

Each step has been a different way to **encode the same rater information**. `score_mean` is literally the expected value of the `score_p_1..5` distribution --> which we already train on in soft mode.
---

## Results — Regression + Oversampling (2026-05-11)

Regression heads for score/veg (`Dense(1)`, MAE loss, target = `score_mean`/`veg_mean`) + multi-label oversampling (children_s_playground 20%, water_feature 25%). Run: `20260511_200157/best_mcmae`.

### Overall metrics (test split, best_mcmae)

| Metric | Sparse (0418) | Soft (0503) | Regression+OS (0511) | Sparse→Reg Δ |
|---|---|---|---|---|
| **score_mae** | 0.691 | 0.686 | **0.636** | **-8.0%** |
| **score_mae_mean** | 0.649 | 0.641 | **0.588** | **-9.4%** |
| **veg_mae** | **0.453** | 0.458 | 0.474 | +4.6% |
| **veg_mae_mean** | **0.443** | 0.444 | 0.448 | +1.1% |
| score_acc | 0.493 | 0.488 | **0.510** | +3.4% |
| veg_acc | **0.636** | 0.623 | 0.631 | -0.8% |
| overall PR_AUC | 0.797 | 0.799 | **0.811** | **+1.8%** |
| overall F1@tuned | 0.743 | 0.734 | **0.757** | **+1.9%** |
| shade_acc_overall | 0.607 | 0.604 | **0.619** | +2.0% |

### Per-label binary head (test split, Sparse → Regression+OS)

| Label | Sparse F1 | Reg+OS F1 | Δ | Sparse PR_AUC | Reg+OS PR_AUC | Δ |
|---|---|---|---|---|---|---|
| water_feature | 0.490 | **0.581** | **+0.091** | 0.620 | **0.648** | **+0.028** |
| children_s_playground | 0.340 | **0.398** | **+0.058** | 0.440 | **0.464** | **+0.024** |
| sports_field | 0.770 | **0.810** | **+0.040** | 0.870 | **0.904** | **+0.034** |
| parking_lots | 0.730 | **0.767** | **+0.037** | 0.840 | 0.840 | 0.000 |
| Top 3 labels | ~0.89 avg | ~0.88 avg | flat | ~0.94 avg | ~0.94 avg | flat |

### Takeaway

- **Sparse → Soft was a wash** (all metrics within ±2%).
- **Regression + oversampling delivered clear gains**: score MAE down 8–9%, binary F1 up across all weak labels, shade improved as side effect.
- **One trade-off**: veg MAE slightly regressed (+4.6%) — veg was already well-calibrated under softmax.
- **children_s_playground** improved but remains the weakest label (0.398 F1, 0.464 PR_AUC).

---

## Next steps

**1. Regularization** — Add `Dropout(0.3)` before score/veg heads to narrow the ~3x train/val gap. Optionally try label smoothing on the binary head. Tune dropout rate based on val gap.

**2. Freeze strategy** — Partial backbone unfreeze (last ~30 layers instead of all ~240) to prevent the backbone from overfitting to ordinal signal while preserving general visual features.

---

## Links

- Prior ordinal / soft context: [0502_process_update.md](0502_process_update.md)
