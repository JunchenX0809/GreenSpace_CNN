"""Shared configuration for PyTorch smoke experiments.

The first PyTorch step reuses the current NB02 split outputs and the same
label/augmentation controls as the TensorFlow notebooks.
"""

import os
from pathlib import Path

from src.augmentation import AUG_PARAMS
from src.label_schema import EXPERIMENT_CONFIG, HEAD_PRESETS, resolve_label_cols

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the approved data root without embedding a host-specific path.

    The explicit argument wins, followed by ``GREENSPACE_DATA_ROOT``.  When
    neither is set, local development keeps using ``<project>/data``.
    """

    raw_root = data_root if data_root is not None else os.getenv("GREENSPACE_DATA_ROOT")
    return Path(raw_root).expanduser() if raw_root else PROJECT_ROOT / "data"


def resolve_split_dir(split_dir: str | Path | None = None) -> Path:
    """Resolve the directory that contains train/val/test split manifests."""

    return Path(split_dir).expanduser() if split_dir is not None else resolve_data_root() / "processed" / "splits"


def resolve_prediction_output_root(output_root: str | Path | None = None) -> Path:
    """Resolve where image-only prediction CSVs and plots are written."""

    raw_root = output_root if output_root is not None else os.getenv("GREENSPACE_PREDICTION_OUTPUT_ROOT")
    return Path(raw_root).expanduser() if raw_root else PROJECT_ROOT / "predictions"


# Backward-compatible import constant. Runtime loader functions call
# ``resolve_split_dir`` so command-line/environment configuration remains live.
SPLIT_DIR = resolve_split_dir()

TORCH_DATA_CONFIG = {
    "img_size": (512, 512),
    "batch_size": 4,
    "num_workers": 0,
    "pin_memory": False,
    "image_transform": "tf_parity",
    "backbone_preprocess": None,
    "use_oversampling": True,
    "use_augmentation": True,
    "oversampling_sanity_batches": 80,
}

TORCH_MODEL_CONFIG = {
    "backbone_priority": "torchgeo",
    "torchgeo_model_name": "swin_v2_b",
    "torchgeo_weight": "Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS",
    "load_pretrained_weights": True,
    "preserve_input_resolution": True,
    "fallback_backbone": "torchvision",
    "num_binary": 7,
    "num_shade": 2,
    "score_output_range": (1.0, 5.0),
    "veg_output_range": (1.0, 5.0),
}

TORCH_LOSS_WEIGHTS = {
    "bin_head": 1.0,
    "shade_head": 1.0,
    "score_head": 1.0,
    "veg_head": 1.0,
}

TORCH_TRAINING_SMOKE_CONFIG = {
    "seed": 37,
    "subset_size": 64,
    "debug_steps": 20,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "train_backbone": False,
}

TORCH_TRAINING_CONFIG = {
    "seed": 37,
    "test_run_mode": True,
    "test_warmup_epochs": 1,
    "test_finetune_epochs": 1,
    "warmup_epochs": 5,
    "finetune_epochs": 15,
    "warmup_learning_rate": 1e-3,
    "finetune_learning_rate": 1e-4,
    "fine_tune_backbone": True,
    "use_combo_training_control": True,
    "combo_w_bin": 0.5,
    "combo_w_ord": 0.5,
    "combo_mcmae_scale": 2.0,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-3,
    "restore_best_weights": True,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 2,
    "reduce_lr_min_delta": 1e-4,
    "mae_guardrail_delta": 0.05,
    "mae_guardrail_patience": 10,
    "device": "auto",
    "max_train_batches": None,
    "max_val_batches": None,
}

__all__ = [
    "AUG_PARAMS",
    "EXPERIMENT_CONFIG",
    "HEAD_PRESETS",
    "PROJECT_ROOT",
    "SPLIT_DIR",
    "TORCH_DATA_CONFIG",
    "TORCH_LOSS_WEIGHTS",
    "TORCH_MODEL_CONFIG",
    "TORCH_TRAINING_CONFIG",
    "TORCH_TRAINING_SMOKE_CONFIG",
    "resolve_data_root",
    "resolve_label_cols",
    "resolve_prediction_output_root",
    "resolve_split_dir",
]
