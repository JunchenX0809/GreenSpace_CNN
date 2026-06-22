"""Load a saved PyTorch run as one validated, transferable bundle.

A run bundle is the inseparable unit a collaborator needs to reproduce
evaluation or inference offline: the checkpoint, its saved label schema and
input size, and the matching validation-tuned thresholds. Today the checkpoint
and config live in ``models/runs/<tag>/`` while thresholds live in
``monitoring_output/runs/<tag>/``. This loader resolves and validates those
pieces together so an incomplete or mismatched bundle fails clearly before any
prediction, instead of being stitched together by notebook cells.

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

from src_torch.config import PROJECT_ROOT
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
    run_tag: str
    variant: str
    binary_cols: list[str]  # saved order, '_p' suffixed; matches bin_head width
    bin_names: list[str]    # stripped names used for thresholds and output columns
    img_size: tuple[int, int]
    thresholds: dict[str, float]
    model_config: dict[str, Any]


def resolve_threshold_path(run_tag: str, variant: str, monitoring_root: Path | None = None) -> Path:
    """Map a checkpoint's run tag/variant to its tuned-threshold CSV."""

    root = monitoring_root or (PROJECT_ROOT / "monitoring_output")
    return root / "runs" / run_tag / f"thresholds_{variant}.csv"


def load_run_bundle(
    checkpoint_path: str | Path,
    device: Any | None = None,
    monitoring_root: Path | None = None,
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

    threshold_path = resolve_threshold_path(run_tag, variant, monitoring_root=monitoring_root)
    # load_thresholds raises if any active label is missing or out of [0, 1].
    thresholds = load_thresholds(threshold_path, bin_names)

    img_size = tuple(model_config.get("img_size", (512, 512)))

    return RunBundle(
        model=model,
        run_tag=run_tag,
        variant=variant,
        binary_cols=binary_cols,
        bin_names=bin_names,
        img_size=img_size,
        thresholds=thresholds,
        model_config=model_config,
    )
