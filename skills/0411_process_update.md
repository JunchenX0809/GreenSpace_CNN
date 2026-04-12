# Process Update — 2026-04-11

## Dropping `gardens` from the binary head

Rationale for excluding `gardens_p` from training and prediction:

1. Dedicates 10% of training stream to it (oversampling) — only 109 positives in train
2. Lets it dominate model selection via the highest inv-sqrt weight (`w=0.0958`)
3. Injects noise into the validation metric with only 31 positives
4. Dilutes the binary head's representational capacity for 7 learnable labels

Dropping it should produce a cleaner model for the remaining labels — removing both a noisy training signal and a noisy model-selection signal.

## Changes made

- `EXCLUDE_BINARY_LABELS` + filtered `binary_cols` in NB03 / NB04 / NB05
- `USE_GARDEN_OVERSAMPLING = False`
- NB02 + `src/` unchanged; `gardens_p` still in CSVs, not in model head
- Reversible: clear toggle list, restore oversampling flag
- Run-over-run diffs: `scripts/compare_eval_runs.py` → comparison CSVs + `rel_change_pct`

## Results — test split, post-`gardens` vs prior baseline

`rel_change_pct` = (new−baseline)/baseline. Side-by-side: both saved Keras checkpoints (NB04 compare CSVs have full baseline/new).

| Metric | best_prauc Δ% | best_mcmae Δ% |
|--------|---------------|----------------|
| overall_PR_AUC | +8.29% | +7.70% |
| overall_F1@tuned | +7.63% | +8.71% |
| overall_F1@0.5 | +12.54% | +9.63% |
| score_mae | +11.22% | +10.70% |
| veg_mae | +5.68% | +7.38% |
| shade_acc_overall | +0.68% | +4.61% |
| score_acc | −12.42% | −10.34% |
| veg_acc | −6.79% | −9.04% |

- Same sign on every row across both ckpts
- Binary-style / macro: ↑; ordinal MAE: ↑ (worse); score/veg acc: ↓
- after the gardens drop, that best keras weight still loses to the previous run’s  perf on all multi-head labels 

## Training history — cross-run comparison (what the val curves add)

Runs: **March** `20260321_214623` (with `gardens`) vs **April** `20260411_142755` (no `gardens`). Histories are warmup + fine-tune; epoch 1–5 frozen backbone, epoch 6+ full model.

| Pattern | March (gardens) | April (no gardens) |
|--------|------------------|---------------------|
| Late val PR-AUC | ~0.57 | ~0.73 |
| Late val score EV MAE (flat region) | ~0.70 | ~0.735 ( worse) |
| Late val veg EV MAE | ~0.427 | ~0.437 (similarly worse) |
| Alignment of “best” epochs | MCMAE best **after** PR-AUC best | MCMAE best **before** PR-AUC best |

Diagnostic read:

1. **getting rid of gardens** really does free the binary head to score higher on val (and train)—the April run’s val PR-AUC plateau is much higher than March’s.
2. For that same run, the **Multi-head surface** gets stuck **earlier** at a **slightly worse score MAE floor** than March’s late run, while training **continues to improve binary**. That lines up with the test table (worse score/veg MAE and acc on the new run).

## Next Step Brainstorming

1. Try higher weights for multi head lables again
we have loss_weights_v3 defined loss_weights_v3 = {
    'bin_head': 1,
    'shade_head': 1.0,
    'score_head': 1.2,
    'veg_head': 1.2,
} 
next training run can just uses the more focused weights to better account for score and veg.
2. Align “training control” a bit with multi head labels (score + veg)
Right now ReduceLROnPlateau and EarlyStopping watch binary PR-AUC only. We can monitor val MCMAE (or a small combo metric) for LR or for one of the early stopping rules.