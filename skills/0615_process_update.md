# Process Update — 2026-06-15

## Swin V2 + NAIP Comparison (RUN 1)

Run `PyTorch_20260609_220247` tested TorchGeo Swin V2 Base with NAIP RGB single-image Satlas weights at `512 x 512`. It is compared with the native-resolution TorchGeo ResNet-50 FMOW-GASSL run, `PyTorch_20260608_220356`.

| Test Metric | ResNet 512 | Swin V2 + NAIP | Delta |
| --- | ---: | ---: | ---: |
| Overall PR-AUC | `0.847` | `0.873` | `+0.026` |
| Overall ROC-AUC | `0.916` | `0.942` | `+0.026` |
| Overall tuned F1 | `0.797` | `0.823` | `+0.026` |
| Score MAE mean | `0.619` | `0.570` | `-0.049` |
| Veg MAE mean | `0.493` | `0.430` | `-0.064` |
| Shade accuracy conditional | `0.679` | `0.689` | `+0.010` |


## Per-Label Pattern

Priority-label gains:

- `children_s_playground`: PR-AUC `+0.050`, tuned F1 `+0.019`
- `water_feature`: PR-AUC `+0.022`, tuned F1 `+0.053`

Additional gains:

- `built_structures`: PR-AUC `+0.032`, tuned F1 `+0.028`
- `parking_lots`: PR-AUC `+0.030`, tuned F1 `+0.036`
- `multipurpose_open_area`: PR-AUC `+0.025`, tuned F1 `+0.002`
- `sports_field`: PR-AUC `+0.012`, tuned F1 `+0.023`
- `walking_paths`: PR-AUC `+0.011`, tuned F1 `+0.020`

## High-Level Observation

Swin V2 + NAIP improved every binary label and both continuous regression outcomes. The run was manually interrupted at epoch 22 due to runtime, but evaluation used the valid best-MC-MAE checkpoint saved at epoch 15. Because the architecture and pretrained dataset changed together, the gains cannot be attributed to NAIP weights alone.

## Swin V2 + NAIP With Updated Survey Data (RUN 2)

Run `PyTorch_20260614_220926` used the same Swin V2 + NAIP setup with expanded, regenerated splits (`3145/1048/1049` versus `3067/1022/1022`). Results are preliminary because existing images were reshuffled.

| Test Metric | Previous Run | Expanded-Data Run | Delta |
| --- | ---: | ---: | ---: |
| Overall PR-AUC | `0.873` | `0.888` | `+0.015` |
| Overall tuned F1 | `0.823` | `0.824` | `+0.001` |
| Overall ROC-AUC | `0.942` | `0.939` | `-0.003` |
| Score MAE mean | `0.570` | `0.576` | `+0.006` |
| Veg MAE mean | `0.430` | `0.430` | `0.000` |

### Per-Label Pattern

Most useful gains:

- `children_s_playground`: PR-AUC `+0.124`, tuned F1 `+0.074`
- `water_feature`: PR-AUC `+0.022`, tuned F1 remained flat

Small regressions appeared for `sports_field`, `built_structures`, `walking_paths`, and `parking_lots`.

### Preliminary Observation

The expanded-data run improved overall PR-AUC and priority-label detection, especially playgrounds, while overall tuned F1 and regression MAE remained broadly stable. Train-test gaps increased for some outcomes, indicating possible additional overfitting. The 20-epoch limit was sufficient for the best-MC-MAE checkpoint at epoch 17.
