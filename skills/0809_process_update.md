# Process Update — 2026-08-10

## External-Reviewer Handoff

The [README prediction-only quickstart](../README.md#prediction-only-clean-clone-trial-50-images)
now documents the clean-clone setup, external model/sample download, validation,
five-image smoke test, and full 50-image inference run. The new
[`models/README.md`](../models/README.md) records the required three-file bundle
layout and reference checksums.

No inference code changed.

## Superseding 50-Image Training Demo

The external handoff now begins with the downloadable 67-row/50-image training
trial before the separate unseen-image prediction trial. The refreshed
[`CORE_pipeline_v1.ipynb`](../notebooks/CORE_pipeline_v1.ipynb) runs the current
preprocessing, augmented/oversampled Swin V2 B + Satlas 1+1 trainer, evaluator,
production-bundle validator, and 50-image predictor. The main README records the
three Drive archives, exact manual extraction layout, and automatically created
output directories.
