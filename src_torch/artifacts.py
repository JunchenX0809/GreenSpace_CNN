"""Persistent run artifact helpers for PyTorch experiments."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src_torch.config import PROJECT_ROOT


def make_run_tag(prefix: str = "PyTorch") -> str:
    """Return a timestamped run tag compatible with existing run folders."""

    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def make_run_dir(run_tag: str, runs_root: Path | None = None) -> Path:
    """Create and return `models/runs/<run_tag>`."""

    root = runs_root or (PROJECT_ROOT / "models" / "runs")
    run_dir = root / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def json_ready(value: Any) -> Any:
    """Convert common scientific/PyTorch values into JSON-safe objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload atomically with stable formatting."""

    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with open(temporary, "w") as file:
            json.dump(json_ready(payload), file, indent=2)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    run_tag: str,
    phase: str,
    epoch: int,
    metrics: dict[str, Any],
    model_config: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> None:
    """Save a PyTorch checkpoint with enough context for later evaluation."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required to save checkpoints.") from exc

    payload = {
        "run_tag": run_tag,
        "phase": phase,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": json_ready(metrics),
        "model_config": json_ready(model_config),
    }
    if training_state is not None:
        payload["training_state"] = training_state

    temporary = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_training_curves(history: dict[str, list[Any]], run_dir: Path, warmup_epochs: int) -> Path | None:
    """Save standard and presentation-style curves when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    epochs = history.get("epoch", [])
    if not epochs:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_loss, ax_mae = axes

    for key, label in [
        ("loss_total", "Train Loss"),
        ("val_loss_total", "Val Loss"),
    ]:
        if key in history:
            ax_loss.plot(epochs, history[key], label=label)
    ax_loss.axvline(warmup_epochs + 0.5, color="gray", linestyle=":", alpha=0.7)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("PyTorch Total Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    for key, label in [
        ("metric_score_mae", "Train Score MAE"),
        ("val_metric_score_mae", "Val Score MAE"),
        ("metric_veg_mae", "Train Veg MAE"),
        ("val_metric_veg_mae", "Val Veg MAE"),
    ]:
        if key in history:
            ax_mae.plot(epochs, history[key], label=label)
    ax_mae.axvline(warmup_epochs + 0.5, color="gray", linestyle=":", alpha=0.7)
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("PyTorch Score/Veg MAE")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend()

    fig.tight_layout()
    out_path = run_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    save_training_metric_curves(
        history,
        run_dir / "training_metric_curves.png",
        warmup_epochs,
    )
    return out_path


def save_training_metric_curves(
    history: dict[str, list[Any]],
    output_path: Path,
    warmup_epochs: int,
) -> Path | None:
    """Save the PR-AUC and score/vegetation MAE epoch visual."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    epochs = history.get("epoch", [])
    if not epochs:
        return None

    def series(key: str) -> np.ndarray | None:
        values = history.get(key)
        if values is None:
            return None
        if len(values) != len(epochs):
            raise ValueError(
                f"history series {key!r} has {len(values)} values; "
                f"expected {len(epochs)}"
            )
        return np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=float,
        )

    def best_index(values: np.ndarray | None, mode: str) -> int | None:
        if values is None or not np.isfinite(values).any():
            return None
        finite_values = np.where(np.isfinite(values), values, np.nan)
        return int(
            np.nanargmax(finite_values)
            if mode == "max"
            else np.nanargmin(finite_values)
        )

    val_pr = series("val_metric_bin_head_weighted_pr_auc")
    train_pr = series("metric_bin_head_weighted_pr_auc")
    val_score = series("val_metric_score_mae")
    train_score = series("metric_score_mae")
    val_veg = series("val_metric_veg_mae")
    train_veg = series("metric_veg_mae")

    fig, (ax_pr, ax_mae) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if val_pr is not None:
        ax_pr.plot(epochs, val_pr, label="Val PR-AUC", color="tab:blue", linewidth=1.5)
        index = best_index(val_pr, "max")
        if index is not None:
            ax_pr.plot(
                epochs[index],
                val_pr[index],
                "*",
                color="tab:blue",
                markersize=14,
                label=f"Best val (epoch {epochs[index]})",
            )
    if train_pr is not None:
        ax_pr.plot(
            epochs,
            train_pr,
            label="Train PR-AUC",
            color="tab:blue",
            linestyle="--",
            alpha=0.6,
        )
    ax_pr.axvline(
        warmup_epochs + 0.5,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="Unfreeze backbone",
    )
    ax_pr.set_ylabel("Weighted PR-AUC")
    ax_pr.set_title("Binary Head — Weighted PR-AUC")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(loc="lower right", fontsize=9)

    for values, label, color, linestyle, alpha in [
        (val_score, "Val Score MAE", "tab:orange", "-", 1.0),
        (val_veg, "Val Veg MAE", "tab:green", "-", 1.0),
        (train_score, "Train Score MAE", "tab:orange", "--", 0.6),
        (train_veg, "Train Veg MAE", "tab:green", "--", 0.6),
    ]:
        if values is not None:
            ax_mae.plot(
                epochs,
                values,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=alpha,
            )

    if val_score is not None and val_veg is not None:
        mc_mae = (val_score + val_veg) / 2.0
        index = best_index(mc_mae, "min")
        if index is not None:
            ax_mae.plot(
                epochs[index],
                val_score[index],
                "*",
                color="tab:orange",
                markersize=14,
            )
            ax_mae.plot(
                epochs[index],
                val_veg[index],
                "*",
                color="tab:green",
                markersize=14,
                label=f"Best MCMAE (epoch {epochs[index]})",
            )

    converged_values: list[float] = []
    for values in (val_score, val_veg, train_score, train_veg):
        if values is not None:
            finite = values[len(values) // 2 :]
            converged_values.extend(finite[np.isfinite(finite)].tolist())
    if converged_values:
        ax_mae.set_ylim(0, max(converged_values) * 3)

    ax_mae.axvline(
        warmup_epochs + 0.5,
        color="gray",
        linestyle=":",
        alpha=0.7,
    )
    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("MAE")
    ax_mae.set_title("Score & Vegetation — MAE")
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
