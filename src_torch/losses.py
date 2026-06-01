"""One-batch loss and metric diagnostics for PyTorch smoke tests."""

from __future__ import annotations

from typing import Any

from src_torch.config import TORCH_LOSS_WEIGHTS


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install the repo requirements.txt before "
            "computing Torch losses."
        ) from exc
    return torch, functional


def shade_sample_weight(targets: dict[str, Any], binary_cols: list[str]) -> Any:
    """Return the NB03-equivalent shade sample weight from walking_paths_p."""

    torch, _ = _require_torch()
    if "walking_paths_p" not in binary_cols:
        return torch.ones_like(targets["shade_head"], dtype=torch.float32)
    idx = binary_cols.index("walking_paths_p")
    return targets["bin_head"][:, idx].to(dtype=torch.float32)


def compute_one_batch_diagnostics(
    outputs: dict[str, Any],
    targets: dict[str, Any],
    binary_cols: list[str],
    loss_weights: dict[str, float] = TORCH_LOSS_WEIGHTS,
) -> dict[str, float]:
    """Compute NB03-aligned one-batch losses and simple metrics.

    This is a smoke diagnostic, not a replacement for full validation metrics.
    Binary and shade heads are treated as logits. Score and vegetation heads are
    bounded scalar predictions in [1, 5].
    """

    torch, functional = _require_torch()

    def as_float(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        return float(value)

    y_bin = targets["bin_head"].to(dtype=torch.float32)
    y_shade = targets["shade_head"].to(dtype=torch.long)
    y_score = targets["score_head"].to(dtype=torch.float32)
    y_veg = targets["veg_head"].to(dtype=torch.float32)

    bin_loss = functional.binary_cross_entropy_with_logits(outputs["bin_head"], y_bin)

    shade_ce = functional.cross_entropy(outputs["shade_head"], y_shade, reduction="none")
    shade_sw = shade_sample_weight(targets, binary_cols).to(dtype=torch.float32)
    # Match the training intent: walking_paths_p reduces shade contribution.
    # Keep the denominator as batch size for Keras-style loss scaling.
    shade_loss = (shade_ce * shade_sw).mean()

    score_mae_per_sample = torch.abs(outputs["score_head"] - y_score).view(-1)
    veg_mae_per_sample = torch.abs(outputs["veg_head"] - y_veg).view(-1)
    score_loss = score_mae_per_sample.mean()
    veg_loss = veg_mae_per_sample.mean()

    weighted_total = (
        loss_weights["bin_head"] * bin_loss
        + loss_weights["shade_head"] * shade_loss
        + loss_weights["score_head"] * score_loss
        + loss_weights["veg_head"] * veg_loss
    )

    bin_pred = (outputs["bin_head"] >= 0).to(dtype=torch.float32)
    bin_acc = (bin_pred == y_bin).to(dtype=torch.float32).mean()
    shade_pred = outputs["shade_head"].argmax(dim=1)
    if float(shade_sw.sum()) > 0:
        shade_acc = ((shade_pred == y_shade).to(dtype=torch.float32) * shade_sw).sum() / shade_sw.sum()
    else:
        shade_acc = torch.tensor(float("nan"))

    score_veg_mae_mean = (score_loss + veg_loss) / 2.0

    return {
        "loss_total": as_float(weighted_total),
        "loss_bin_head": as_float(bin_loss),
        "loss_shade_head": as_float(shade_loss),
        "loss_score_head": as_float(score_loss),
        "loss_veg_head": as_float(veg_loss),
        "metric_bin_binary_accuracy": as_float(bin_acc),
        "metric_shade_sparse_categorical_accuracy_weighted": as_float(shade_acc),
        "metric_score_mae": as_float(score_loss),
        "metric_veg_mae": as_float(veg_loss),
        "metric_score_veg_mae_mean": as_float(score_veg_mae_mean),
        "shade_sample_weight_mean": as_float(shade_sw.mean()),
        "shade_sample_weight_sum": as_float(shade_sw.sum()),
    }
