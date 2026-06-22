"""Load a saved PyTorch run as one validated, transferable bundle.

A run bundle is the inseparable unit a collaborator needs to reproduce
evaluation or inference offline: a selected checkpoint, its saved model config,
and the matching validation-tuned thresholds. All three live in one
``models/runs/<tag>/`` directory. This loader validates those pieces together
so an incomplete or mismatched bundle fails clearly before any prediction.

Note: the input-transform contract (TF-parity ``[0, 1]`` vs TorchGeo
``[0, 255]``) is intentionally not carried here. The saved config's
``image_transform`` does not reliably reflect the transform used at inference,
and making that self-describing per run is a separate planned step. Inference
continues to apply the correct ``rgb_255`` transform via the inference helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src_torch.evaluation import (
    checkpoint_binary_cols,
    infer_run_tag_and_variant,
    load_torch_checkpoint_model,
)
from src_torch.inference import load_thresholds


@dataclass(frozen=True)
class RunBundle:
    """A saved model loaded with the artifacts needed to use it offline."""

    model: Any
    run_dir: Path
    checkpoint_path: Path
    threshold_path: Path
    run_tag: str
    variant: str
    binary_cols: list[str]  # saved order, '_p' suffixed; matches bin_head width
    bin_names: list[str]    # stripped names used for thresholds and output columns
    img_size: tuple[int, int]
    thresholds: dict[str, float]
    model_config: dict[str, Any]


def resolve_threshold_path(checkpoint_path: str | Path) -> Path:
    """Return the threshold CSV stored beside a selected checkpoint."""

    path = Path(checkpoint_path)
    _, variant = infer_run_tag_and_variant(path)
    return path.parent / f"thresholds_{variant}.csv"


def load_run_bundle(
    checkpoint_path: str | Path,
    device: Any | None = None,
) -> RunBundle:
    """Load and validate a checkpoint with its schema and matching thresholds."""

    path = Path(checkpoint_path)
    run_tag, variant = infer_run_tag_and_variant(path)
    model, model_config, _ = load_torch_checkpoint_model(path, device=device)

    binary_cols = checkpoint_binary_cols(model_config)
    # '_p' suffix convention, mirroring data.resolve_split_schema.
    bin_names = [col[:-2] for col in binary_cols]

    bin_head_width = int(model.bin_head.out_features)
    if bin_head_width != len(binary_cols):
        raise ValueError(
            f"Checkpoint bin_head width {bin_head_width} does not match saved "
            f"binary_cols count {len(binary_cols)}."
        )

    threshold_path = resolve_threshold_path(path)
    # load_thresholds raises if any active label is missing or out of [0, 1].
    thresholds = load_thresholds(threshold_path, bin_names)

    img_size = tuple(model_config.get("img_size", (512, 512)))

    return RunBundle(
        model=model,
        run_dir=path.parent,
        checkpoint_path=path,
        threshold_path=threshold_path,
        run_tag=run_tag,
        variant=variant,
        binary_cols=binary_cols,
        bin_names=bin_names,
        img_size=img_size,
        thresholds=thresholds,
        model_config=model_config,
    )
