# Process Update — 2026-04-19

## Run comparison — test split (`best_mcmae`)

| Metric | Baseline | New | rel_change_pct |
|--------|----------|-----|----------------|
| overall_PR_AUC | 0.8026 | 0.7966 | −0.76% |
| overall_F1@tuned | 0.7555 | 0.7432 | −1.63% |
| score_mae | 0.6937 | 0.6911 | −0.37% |
| veg_mae | 0.4584 | 0.4527 | −1.25% |
| shade_acc_overall | 0.6751 | 0.6074 | −10.03% |
| score_acc | 0.4462 | 0.4935 | +10.59% |
| veg_acc | 0.6098 | 0.6356 | +4.22% |

## Run comparison — test split (`best_prauc`)

Same comparison run as above, `compare_eval_runs.py` with `--variants best_prauc` (test split). `rel_change_pct` is `(new − baseline) / baseline` as a percentage.

| Metric | Baseline | New | rel_change_pct |
|--------|----------|-----|----------------|
| overall_PR_AUC | 0.8047 | 0.7940 | −1.33% |
| overall_F1@tuned | 0.7548 | 0.7433 | −1.52% |
| score_mae | 0.6972 | 0.6933 | −0.57% |
| veg_mae | 0.4564 | 0.4568 | +0.08% |
| shade_acc_overall | 0.6739 | 0.5846 | −13.25% |
| score_acc | 0.4439 | 0.4751 | +7.01% |
| veg_acc | 0.6121 | 0.6323 | +3.30% |

## Observation

### 1. Combo training + MAE

Adding **combo** training control, **score / veg MAE** improved slightly for both **best MCMAE** and **best PR-AUC** `.keras` runs vs baseline. **Caveat:** account for **training sample size** if preprocessing / n changed.

### 2. Macro binary vs accuracy

Some **macro binary** metrics dip while **accuracy** moves the other way — expected when **argmax** decisions and **probability** rankings don’t line up.

**Likely driver:** **`val_training_combo`** shifts **LR cuts** and **early stopping** vs PR-AUC-only, so the **training path** changes even though **`best_*.keras`** (PR-AUC) and **`best_mcmae_*.keras`** (MCMAE) are still chosen by those separate rules.

### 3. Label distributions (`04b_eval_visuals` → `score_veg_split_distributions.png`)

**Score:** **flat** 1–5 on train/val/test → MAE, based on the distribution plot, does show model learnt from images. For example: always class 3 on uniform {1,…,5} → will have a MAE of 1.2 for score; test score MAE **~0.69** is **well below** that. **Veg:** **skewed** but **same shape** on all splits → no obvious **split mismatch**. **Still:** shade / per-class / domain tolerance are separate checks.