"""Offline regression check for portable PyTorch run-bundle loading.

Proves a saved checkpoint reloads and predicts on a fresh machine WITHOUT any
pretrained-weight download and WITHOUT depending on a live split schema. It
creates a tiny checkpoint, its model config, and matching thresholds in a
temporary directory, then reloads them as a validated run bundle and runs
inference on two synthetic images. Run from the project root:

    python scripts/check_offline_checkpoint_load.py

Exits non-zero on failure.
"""

import os
import sys
import tempfile
from pathlib import Path

# Block weight downloads BEFORE importing torch: redirect caches to empty dirs
# and set offline flags. If any code path tried to fetch weights it would fail
# here rather than silently succeed over the network.
_OFFLINE_CACHE = Path(tempfile.mkdtemp(prefix="offline_cache_"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = str(_OFFLINE_CACHE / "torch")
os.environ["HF_HOME"] = str(_OFFLINE_CACHE / "hf")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from src_torch.artifacts import save_checkpoint
from src_torch.inference import list_inference_image_paths, predict_image_paths
from src_torch.models import build_torchgeo_model
from src_torch.run_bundle import load_run_bundle


def main() -> int:
    binary_cols = ["sports_field_p", "water_feature_p", "gardens_p"]
    run_tag = "PyTorch_offlinecheck"
    variant = "best_mcmae"
    device = torch.device("cpu")

    work = Path(tempfile.mkdtemp(prefix="offline_bundle_"))
    run_dir = work / "models" / "runs" / run_tag
    run_dir.mkdir(parents=True)
    monitoring_root = work / "monitoring_output"
    thr_dir = monitoring_root / "runs" / run_tag
    thr_dir.mkdir(parents=True)
    images_dir = work / "images"
    images_dir.mkdir()

    # Tiny ResNet-50 built WITHOUT pretrained weights, so construction needs no
    # network. ResNet-50 keeps the check fast; the load path is architecture
    # agnostic.
    model = build_torchgeo_model(
        model_name="resnet50",
        weight_name="ResNet50_Weights.FMOW_RGB_GASSL",
        load_pretrained_weights=False,
        num_binary=len(binary_cols),
    )

    model_config = {
        "run_tag": run_tag,
        "img_size": [512, 512],
        "binary_cols": binary_cols,
        "torch_data_config": {"img_size": [512, 512]},
        "torch_model_config": {
            "torchgeo_model_name": "resnet50",
            "torchgeo_weight": "ResNet50_Weights.FMOW_RGB_GASSL",
            "preserve_input_resolution": False,
            "num_binary": len(binary_cols),
            "num_shade": 2,
            "score_output_range": [1.0, 5.0],
            "veg_output_range": [1.0, 5.0],
        },
    }
    ckpt_path = run_dir / f"{variant}_{run_tag}.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=None,
        run_tag=run_tag,
        phase="offline_check",
        epoch=0,
        metrics={},
        model_config=model_config,
    )

    # Matching thresholds, keyed by the stripped label names.
    thr_csv = thr_dir / f"thresholds_{variant}.csv"
    with open(thr_csv, "w") as handle:
        handle.write("label,best_threshold\n")
        for name in (col[:-2] for col in binary_cols):
            handle.write(f"{name},0.5\n")

    # Two synthetic RGB inputs.
    rng = np.random.default_rng(0)
    for idx in range(2):
        arr = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images_dir / f"img_{idx}.jpg")

    # Reload as a validated bundle and predict — all offline.
    bundle = load_run_bundle(ckpt_path, device=device, monitoring_root=monitoring_root)
    assert bundle.bin_names == [col[:-2] for col in binary_cols], "bin_names mismatch"
    assert set(bundle.thresholds) == set(bundle.bin_names), "threshold labels mismatch"

    paths = list_inference_image_paths(images_dir)
    preds = predict_image_paths(bundle.model, paths, device=device, batch_size=2)
    assert preds["bin_head"].shape == (2, len(binary_cols)), "unexpected prediction shape"
    assert all(np.isfinite(value).all() for value in preds.values()), "non-finite predictions"

    print("PASS: offline run-bundle load + predict")
    print(f"  run_tag={bundle.run_tag} variant={bundle.variant}")
    print(f"  bin_names={bundle.bin_names}")
    print(f"  thresholds={bundle.thresholds}")
    print(f"  prediction rows={preds['bin_head'].shape[0]} cols={preds['bin_head'].shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
