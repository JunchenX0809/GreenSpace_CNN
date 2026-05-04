# Process Update — 2026-05-02

## Ordinal confusion heatmaps (`04c_mae_eval` → `presentation_visuals_only/eval_ordinal/`)

Test split, `**best_mcmae**` checkpoint: **5×5** argmax vs hard labels; heatmap cells are **row %** (each true-class row sums to **100%**).

## Observation

### 1. Veg vs score (matches NB04 story in [0419_process_update.md](0419_process_update.md))

Heatmaps line up with the **test** table there (`**best_mcmae`**): **lower veg MAE** and **higher veg accuracy** than score (**veg_mae ~0.45**, **score_mae ~0.69**; **veg_acc ~0.64**, **score_acc ~0.49**). **Veg** shows a **stronger diagonal**; **score** spreads more across **neighbor bins** — consistent with **flatter score support** vs **skewed veg** in `04b`, not a split artifact.

### 2. Conclusion

**Ordinal sanity holds:** mistakes are mostly **adjacent classes**, not extreme 1↔5 flips, so reported MAE is not masking a degenerate classifier. **NB04 MAE** still uses **expected value**, not argmax — report **both** when judging score/veg heads.

---

## Soft-trained run vs. sparse baseline (test split, `best_mcmae`)

Both checkpoints evaluated through the post-fix NB04 so the new soft-aligned columns line up. Sparse baseline = legacy `20260418_220126/best_mcmae_*.keras` re-evaluated today (numbers reproduce the 0419 process update within rounding — sanity check that NB02 splits regenerated identically). Soft run = `20260503_194454/best_mcmae_*.keras`.


| Metric                                | Direction     | Sparse baseline | Soft run | Δ%         |
| ------------------------------------- | ------------- | --------------- | -------- | ---------- |
| `score_mae` (vs hard class)           | lower better  | 0.691           | 0.686    | −0.72%     |
| `veg_mae` (vs hard class)             | lower better  | 0.453           | 0.458    | +1.10%     |
| `score_acc`                           | higher better | 0.493           | 0.488    | −1.01%     |
| `veg_acc`                             | higher better | 0.636           | 0.623    | −2.04%     |
| `overall_PR_AUC`                      | higher better | 0.797           | 0.799    | +0.25%     |
| `overall_F1@tuned`                    | higher better | 0.743           | 0.734    | −1.21%     |
| `shade_acc_overall`                   | higher better | 0.607           | 0.604    | −0.49%     |
| `score_ce` (vs soft true)             | lower better  | 1.593           | 1.671    | **+4.90%** |
| `veg_ce` (vs soft true)               | lower better  | 1.269           | 1.254    | −1.18%     |
| `score_mae_mean` (EV vs `score_mean`) | lower better  | 0.649           | 0.641    | −1.23%     |
| `veg_mae_mean` (EV vs `veg_mean`)     | lower better  | 0.443           | 0.444    | +0.23%     |


### Observations

1. `**best_mcmae` checkpointing selects on EV-MAE, not CE.** Soft training pushes the EV closer to `score_mean` (visible as the −1.23% on `score_mae_mean`), but the checkpoint isn't selected to minimise CE --> so `score_ce` worsening is expected.
2. **Score and veg respond differently.** Soft helped score's EV alignment but hurt its CE; veg got a small CE win and roughly flat EV. Likely related to the flatter score support vs. skewed veg distribution; soft training spreads probability mass on score, which helps EV but penalises CE.
3. **No legacy regression worth reverting.** All seven legacy columns stay inside ±2.1%; closest things to wins are `overall_PR_AUC` (+0.25%) and `score_mae` (−0.72%); biggest legacy loss is `veg_acc` (−2.04%).
4. **Next Step:** train a soft variant that monitors CE (or a CE-blended combo) for `BestMcMaeCheckpoint` instead of EV-MAE only. That would test whether the CE regression is a checkpoint-selection result vs. a true loss-landscape effect.

