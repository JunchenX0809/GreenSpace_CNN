"""Shared configuration for PyTorch smoke experiments.

The first PyTorch step reuses the current NB02 split outputs and the same
label/augmentation controls as the TensorFlow notebooks.
"""

from pathlib import Path

from src.augmentation import AUG_PARAMS
from src.label_schema import EXPERIMENT_CONFIG, HEAD_PRESETS, resolve_label_cols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

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
    "torchgeo_model_name": "resnet50",
    "torchgeo_weight": "ResNet50_Weights.FMOW_RGB_GASSL",
    "load_pretrained_weights": True,
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
    "finetune_epochs": 100,
    "warmup_learning_rate": 1e-3,
    "finetune_learning_rate": 1e-4,
    "fine_tune_backbone": True,
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
    "resolve_label_cols",
]
