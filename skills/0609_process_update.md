# Process Update — 2026-06-09

## Native-Resolution ResNet Comparison

Run `PyTorch_20260608_220356` retained the original `512 x 512` images, while `PyTorch_20260531_211840` resized them to `224 x 224`. Both used the same TorchGeo ResNet-50 FMOW-GASSL backbone, training schedule, test split, and evaluation flow.

| Test Metric | ResNet 224 | ResNet 512 | Delta |
| --- | ---: | ---: | ---: |
| Overall PR-AUC | `0.797` | `0.847` | `+0.049` |
| Overall ROC-AUC | `0.894` | `0.916` | `+0.022` |
| Overall tuned F1 | `0.744` | `0.797` | `+0.052` |
| Shade accuracy overall | `0.593` | `0.680` | `+0.087` |
| Score MAE mean | `0.631` | `0.619` | `-0.012` |
| Veg MAE mean | `0.509` | `0.493` | `-0.016` |

Lower MAE is better.

## Per-Label Pattern

Most useful gains:

- `water_feature`: PR-AUC `+0.135`, tuned F1 `+0.128`
- `children_s_playground`: PR-AUC `+0.079`, tuned F1 `+0.081`
- `parking_lots`: PR-AUC `+0.074`, tuned F1 `+0.050`
- `sports_field`: PR-AUC `+0.071`, tuned F1 `+0.098`

Smaller changes:

- `built_structures`: PR-AUC `+0.006`, tuned F1 `+0.012`
- `walking_paths`: PR-AUC `-0.003`, tuned F1 `-0.007`
- `multipurpose_open_area`: PR-AUC `-0.015`, tuned F1 `+0.005`

## High-Level Observation

Preserving `512 x 512` spatial detail improved every overall test metric and produced the largest gains for `water_feature` and `children_s_playground`. Native resolution should remain the preferred ResNet configuration, with repeated-seed testing reserved for confirming how much of the gain is attributable to resolution alone.
