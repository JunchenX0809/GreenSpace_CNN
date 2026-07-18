"""Image-only PyTorch inference helpers for GreenSpace prediction exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src_torch.data import make_image_path_dataloader

SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def list_inference_image_paths(cache_dir: str | Path, limit: int | None = None) -> list[Path]:
    """Return supported cache images in deterministic filename order."""

    directory = Path(cache_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing inference image cache: {directory}")
    paths = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {directory}")
    names = [path.name.casefold() for path in paths]
    if len(names) != len(set(names)):
        raise ValueError(f"Inference cache has duplicate filenames ignoring case: {directory}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be a positive integer or None.")
        paths = paths[:limit]
    return paths


def load_thresholds(threshold_path: str | Path, binary_labels: Sequence[str]) -> dict[str, float]:
    """Load matching validation-tuned thresholds for the active binary schema."""

    path = Path(threshold_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing threshold CSV: {path}")
    table = pd.read_csv(path)
    required = {"label", "best_threshold"}
    if not required.issubset(table.columns):
        raise ValueError(f"Threshold CSV must contain {sorted(required)}: {path}")
    if table["label"].duplicated().any():
        raise ValueError(f"Threshold CSV contains duplicate labels: {path}")

    threshold_map = dict(zip(table["label"].astype(str), table["best_threshold"]))
    missing = [label for label in binary_labels if label not in threshold_map]
    if missing:
        raise ValueError(f"Threshold CSV is missing active labels: {missing}")
    thresholds = {label: float(threshold_map[label]) for label in binary_labels}
    invalid = {label: value for label, value in thresholds.items() if not np.isfinite(value) or not 0.0 <= value <= 1.0}
    if invalid:
        raise ValueError(f"Threshold CSV has invalid threshold values: {invalid}")
    return thresholds


def predict_image_paths(
    model: Any,
    image_paths: Sequence[str | Path],
    device: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Predict ordered unlabeled images and return post-activation head arrays."""

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for inference.") from exc

    loader = make_image_path_dataloader(image_paths, batch_size=batch_size, image_transform="rgb_255")
    chunks = {"bin_head": [], "shade_head": [], "score_head": [], "veg_head": []}
    model.eval()
    with torch.no_grad():
        for images in loader:
            outputs = model(images.to(device))
            chunks["bin_head"].append(torch.sigmoid(outputs["bin_head"]).detach().cpu().numpy())
            chunks["shade_head"].append(functional.softmax(outputs["shade_head"], dim=1).detach().cpu().numpy())
            chunks["score_head"].append(outputs["score_head"].detach().cpu().numpy())
            chunks["veg_head"].append(outputs["veg_head"].detach().cpu().numpy())

    predictions = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    count = len(image_paths)
    if any(values.shape[0] != count for values in predictions.values()):
        raise RuntimeError("Prediction row count does not match the requested image paths.")
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Inference produced non-finite prediction values.")
    return predictions


def build_prediction_dataframe(
    image_paths: Sequence[str | Path],
    predictions: dict[str, np.ndarray],
    binary_labels: Sequence[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Convert model predictions into the TensorFlow NB05-compatible CSV schema."""

    paths = [Path(path) for path in image_paths]
    count = len(paths)
    binary = np.asarray(predictions["bin_head"], dtype=np.float32)
    shade = np.asarray(predictions["shade_head"], dtype=np.float32)
    score = np.asarray(predictions["score_head"], dtype=np.float32).reshape(-1)
    veg = np.asarray(predictions["veg_head"], dtype=np.float32).reshape(-1)

    if binary.shape != (count, len(binary_labels)):
        raise ValueError(f"Unexpected binary prediction shape: {binary.shape}")
    if shade.shape != (count, 2):
        raise ValueError(f"Unexpected shade prediction shape: {shade.shape}")
    if score.shape[0] != count or veg.shape[0] != count:
        raise ValueError("Unexpected score or vegetation prediction shape.")
    missing = [label for label in binary_labels if label not in thresholds]
    if missing:
        raise ValueError(f"Missing thresholds for active labels: {missing}")

    rows: dict[str, Any] = {"image_filename": [path.name for path in paths]}
    for idx, label in enumerate(binary_labels):
        rows[f"{label}_prob"] = binary[:, idx]
        rows[f"{label}_pred"] = (binary[:, idx] >= thresholds[label]).astype(int)
    rows["shade_class"] = np.where(shade.argmax(axis=1) == 0, "minimal", "abundant")
    rows["shade_confidence"] = shade.max(axis=1)
    rows["score_ev"] = np.clip(score, 1.0, 5.0)
    rows["veg_ev"] = np.clip(veg, 1.0, 5.0)

    frame = pd.DataFrame(rows)
    if not frame["image_filename"].is_unique:
        raise ValueError("Inference image filenames must be unique in the prediction export.")
    return frame


def inference_output_tag(run_tag: str, limit: int | None) -> str:
    """Keep limited smoke artifacts separate from full model-run artifacts."""

    return run_tag if limit is None else f"{run_tag}_sample{limit}"
