"""PyTorch training helpers for smoke checks and first persistent runs."""

from __future__ import annotations

import hashlib
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src_torch.artifacts import make_run_dir, make_run_tag, save_checkpoint, save_json, save_training_curves
from src_torch.config import (
    EXPERIMENT_CONFIG,
    PROJECT_ROOT,
    TORCH_DATA_CONFIG,
    TORCH_LOSS_WEIGHTS,
    TORCH_MODEL_CONFIG,
    TORCH_TRAINING_CONFIG,
    TORCH_TRAINING_SMOKE_CONFIG,
)
from src_torch.data import load_split_df, make_eval_dataloader, make_train_dataloader, resolve_split_schema
from src_torch.losses import compute_one_batch_diagnostics
from src_torch.models import build_torchgeo_model


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install the repo requirements.txt before "
            "running Torch training smoke helpers."
        ) from exc
    return torch


def set_torch_seed(seed: int = TORCH_TRAINING_SMOKE_CONFIG["seed"]) -> None:
    """Set Python, NumPy, and PyTorch seeds for smoke reproducibility."""

    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


RESUME_SCHEMA_VERSION = 1


def split_frame_signature(frame: Any) -> dict[str, Any]:
    """Return a portable signature of split content, excluding host paths."""

    import pandas as pd

    columns = [column for column in frame.columns if column != "image_path"]
    stable = frame[columns].copy()
    row_hashes = pd.util.hash_pandas_object(
        stable,
        index=True,
        categorize=True,
    ).to_numpy()
    digest = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    return {"rows": int(len(frame)), "columns": columns, "sha256": digest}


def capture_training_rng_state(train_loader: Any | None = None) -> dict[str, Any]:
    """Capture epoch-boundary RNG and weighted-sampler state."""

    torch = _require_torch()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        state["torch_mps"] = torch.mps.get_rng_state()
    sampler = getattr(train_loader, "sampler", None)
    generator = getattr(sampler, "generator", None)
    if generator is not None:
        state["sampler_generator"] = generator.get_state()
    return state


def restore_training_rng_state(
    state: dict[str, Any],
    train_loader: Any | None = None,
) -> None:
    """Restore RNG state saved at the end of a completed epoch."""

    torch = _require_torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if (
        "torch_mps" in state
        and hasattr(torch, "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(state["torch_mps"])
    sampler = getattr(train_loader, "sampler", None)
    generator = getattr(sampler, "generator", None)
    if generator is not None and "sampler_generator" in state:
        generator.set_state(state["sampler_generator"])


def load_training_resume_checkpoint(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load and validate a checkpoint produced for epoch-boundary resume."""

    torch = _require_torch()
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("training_state")
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint is not resumable because training_state is missing: {path}"
        )
    if int(state.get("schema_version", -1)) != RESUME_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported resume checkpoint schema: "
            f"{state.get('schema_version')!r}"
        )
    required = {
        "phase",
        "phase_epoch",
        "global_epoch",
        "history",
        "best_mcmae",
        "best_prauc",
        "elapsed_seconds",
        "data_signature",
        "rng_state",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Resume checkpoint training_state is incomplete: {missing}")
    return payload


def _merge_resume_config(
    saved: dict[str, Any],
    requested: dict[str, Any] | None,
    *,
    mutable_keys: set[str],
    label: str,
) -> dict[str, Any]:
    """Use saved configuration and reject trajectory-changing overrides."""

    merged = dict(saved)
    for key, value in (requested or {}).items():
        if key in mutable_keys:
            merged[key] = value
            continue
        if key not in saved or saved[key] != value:
            raise ValueError(
                f"Cannot change {label} setting {key!r} when resuming: "
                f"saved={saved.get(key)!r}, requested={value!r}"
            )
    return merged


def set_backbone_trainable(model: Any, trainable: bool) -> None:
    """Set trainability for the TorchGeo backbone only."""

    for param in model.backbone.parameters():
        param.requires_grad = trainable


def trainable_parameter_summary(model: Any) -> dict[str, Any]:
    """Return parameter counts and trainable top-level groups."""

    total = 0
    trainable = 0
    groups: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        group = name.split(".", 1)[0]
        groups.setdefault(group, {"total": 0, "trainable": 0})
        groups[group]["total"] += count
        if param.requires_grad:
            groups[group]["trainable"] += count

    trainable_groups = [name for name, counts in groups.items() if counts["trainable"] > 0]
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_groups": trainable_groups,
        "groups": groups,
    }


def make_optimizer(model: Any, lr: float = TORCH_TRAINING_SMOKE_CONFIG["learning_rate"]) -> Any:
    """Create an Adam optimizer over trainable parameters only."""

    torch = _require_torch()
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters available for optimizer.")
    return torch.optim.Adam(params, lr=lr)


def _get_optimizer_lr(optimizer: Any) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _set_optimizer_lr(optimizer: Any, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def resolve_device(device: str = TORCH_TRAINING_CONFIG["device"]) -> Any:
    """Resolve configured train device."""

    torch = _require_torch()
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: tuple[Any, dict[str, Any]], device: Any) -> tuple[Any, dict[str, Any]]:
    """Move image and target tensors onto the selected device."""

    images, targets = batch
    return images.to(device), {name: value.to(device) for name, value in targets.items()}


def _total_loss_tensor(
    outputs: dict[str, Any],
    targets: dict[str, Any],
    binary_cols: list[str],
    loss_weights: dict[str, float] = TORCH_LOSS_WEIGHTS,
) -> Any:
    """Return differentiable total loss matching smoke diagnostics."""

    torch = _require_torch()
    import torch.nn.functional as functional

    y_bin = targets["bin_head"].to(dtype=torch.float32)
    y_shade = targets["shade_head"].to(dtype=torch.long)
    y_score = targets["score_head"].to(dtype=torch.float32)
    y_veg = targets["veg_head"].to(dtype=torch.float32)

    bin_loss = functional.binary_cross_entropy_with_logits(outputs["bin_head"], y_bin)
    shade_ce = functional.cross_entropy(outputs["shade_head"], y_shade, reduction="none")
    if "walking_paths_p" in binary_cols:
        shade_sw = targets["bin_head"][:, binary_cols.index("walking_paths_p")].to(dtype=torch.float32)
    else:
        shade_sw = torch.ones_like(y_shade, dtype=torch.float32)
    shade_loss = (shade_ce * shade_sw).mean()
    score_loss = torch.abs(outputs["score_head"] - y_score).view(-1).mean()
    veg_loss = torch.abs(outputs["veg_head"] - y_veg).view(-1).mean()
    return (
        loss_weights["bin_head"] * bin_loss
        + loss_weights["shade_head"] * shade_loss
        + loss_weights["score_head"] * score_loss
        + loss_weights["veg_head"] * veg_loss
    )


def _grad_summary(model: Any) -> dict[str, Any]:
    torch = _require_torch()
    trainable = [param for param in model.parameters() if param.requires_grad]
    grad_tensors = [param.grad for param in trainable if param.grad is not None]
    if not grad_tensors:
        return {"grad_param_tensors": 0, "grad_all_finite": False, "grad_norm": 0.0}

    all_finite = all(bool(torch.isfinite(grad).all()) for grad in grad_tensors)
    norm = torch.sqrt(sum(torch.sum(grad.detach() ** 2) for grad in grad_tensors))
    return {
        "grad_param_tensors": len(grad_tensors),
        "grad_all_finite": all_finite,
        "grad_norm": float(norm),
    }


def _average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No metric rows to average.")
    keys = rows[0].keys()
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = float(np.nanmean(values)) if values else float("nan")
    return out


def weighted_macro_pr_auc(
    y_pred: np.ndarray,
    train_df: Any,
    eval_df: Any,
    binary_cols: list[str],
    pos_threshold: float = 0.5,
) -> dict[str, Any]:
    """Mirror the TensorFlow weighted macro PR-AUC callback logic."""

    try:
        from sklearn.metrics import average_precision_score
    except ModuleNotFoundError:
        return {"value": float("nan"), "valid_labels": 0, "per_label": {}, "error": "sklearn missing"}

    bin_names = [col[:-2] for col in binary_cols]
    hard_train_names = [name for name in bin_names if name in train_df.columns]
    use_hard = len(hard_train_names) > 0
    label_names = hard_train_names if use_hard else bin_names

    if use_hard:
        y_train = train_df[label_names].fillna(0).astype(int).to_numpy()
    else:
        y_train = (
            train_df[binary_cols].fillna(0.0).astype(np.float32).to_numpy() >= pos_threshold
        ).astype(int)

    if use_hard and all(name in eval_df.columns for name in label_names):
        y_true = eval_df[label_names].fillna(0).astype(int).to_numpy()
    else:
        y_true = (
            eval_df[binary_cols].fillna(0.0).astype(np.float32).to_numpy() >= pos_threshold
        ).astype(int)

    prob_indices = np.array([bin_names.index(name) for name in label_names], dtype=np.int32)
    pred_aligned = y_pred[:, prob_indices]
    pos_counts = y_train.sum(axis=0).astype(np.float64)
    raw_weights = 1.0 / np.sqrt(np.maximum(pos_counts, 1.0))
    base_weights = raw_weights / np.sum(raw_weights)

    ap_vals = []
    valid_weights = []
    per_label = {}
    for idx, label in enumerate(label_names):
        truth = y_true[:, idx]
        if np.unique(truth).size < 2:
            per_label[label] = None
            continue
        ap = float(average_precision_score(truth, pred_aligned[:, idx]))
        per_label[label] = ap
        ap_vals.append(ap)
        valid_weights.append(float(base_weights[idx]))

    if not ap_vals:
        return {"value": float("nan"), "valid_labels": 0, "per_label": per_label}

    valid_weights = np.asarray(valid_weights, dtype=np.float64)
    valid_weights = valid_weights / np.sum(valid_weights)
    value = float(np.sum(valid_weights * np.asarray(ap_vals, dtype=np.float64)))
    return {"value": value, "valid_labels": len(ap_vals), "per_label": per_label}


def compute_training_combo(
    val_metrics: dict[str, Any],
    w_bin: float = 0.5,
    w_ord: float = 0.5,
    mae_scale: float = 2.0,
) -> dict[str, float]:
    """Mirror TensorFlow `ValComboTrainingMetric` for training control."""

    pr_auc = val_metrics.get("metric_bin_head_weighted_pr_auc")
    if pr_auc is None or not np.isfinite(float(pr_auc)):
        return {}

    score_mae = val_metrics.get("metric_score_mae")
    veg_mae = val_metrics.get("metric_veg_mae")
    if score_mae is None or veg_mae is None:
        return {"training_combo": float(pr_auc)}

    mc_mae = (float(score_mae) + float(veg_mae)) / 2.0
    ord_term = max(0.0, 1.0 - min(mc_mae / max(float(mae_scale), 1e-8), 1.0))
    combo = float(w_bin) * float(pr_auc) + float(w_ord) * ord_term
    return {
        "training_combo": combo,
        "training_combo_mcmae": mc_mae,
        "training_combo_ord_term": ord_term,
    }


class PlateauTrainingControl:
    """PyTorch equivalent of EarlyStopping + ReduceLROnPlateau on one max metric."""

    def __init__(
        self,
        monitor: str,
        early_patience: int = 10,
        early_min_delta: float = 0.0,
        reduce_lr_patience: int = 2,
        reduce_lr_factor: float = 0.5,
        reduce_lr_min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ) -> None:
        self.monitor = monitor
        self.early_patience = int(early_patience)
        self.early_min_delta = float(early_min_delta)
        self.reduce_lr_patience = int(reduce_lr_patience)
        self.reduce_lr_factor = float(reduce_lr_factor)
        self.reduce_lr_min_delta = float(reduce_lr_min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.best = float("-inf")
        self.wait = 0
        self.lr_wait = 0
        self.lr_best = float("-inf")
        self.best_state_dict = None
        self.best_epoch = None

    def update(self, model: Any, optimizer: Any, epoch: int, logs: dict[str, Any]) -> bool:
        torch = _require_torch()
        current = logs.get(self.monitor)
        if current is None or not np.isfinite(float(current)):
            print(f"[TrainingControl] Missing {self.monitor} at epoch {epoch}; skipping LR/early-stopping check.")
            return False

        current = float(current)
        if current > self.best + self.early_min_delta:
            self.best = current
            self.wait = 0
            self.lr_wait = 0
            self.best_epoch = int(epoch)
            if self.restore_best_weights:
                self.best_state_dict = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
        else:
            self.wait += 1

        if current > self.lr_best + self.reduce_lr_min_delta:
            self.lr_best = current
            self.lr_wait = 0
            return False

        self.lr_wait += 1
        if self.lr_wait >= self.reduce_lr_patience:
            old_lr = _get_optimizer_lr(optimizer)
            new_lr = old_lr * self.reduce_lr_factor
            _set_optimizer_lr(optimizer, new_lr)
            self.lr_wait = 0
            print(
                f"[ReduceLROnPlateau] Epoch {epoch}: reducing learning rate "
                f"from {old_lr:.6g} to {new_lr:.6g}; {self.monitor}={current:.4f}, best={self.best:.4f}"
            )

        if self.wait >= self.early_patience:
            print(
                f"[EarlyStopping] Stopping training at epoch {epoch}: "
                f"{self.monitor}={current:.4f}, best={self.best:.4f}, "
                f"min_delta={self.early_min_delta:.4f}, patience={self.early_patience}"
            )
            return True
        return False

    def restore(self, model: Any, device: Any) -> None:
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict({name: tensor.to(device) for name, tensor in self.best_state_dict.items()})
            print(f"[EarlyStopping] Restored best weights from epoch {self.best_epoch}.")

    def state_dict(self) -> dict[str, Any]:
        """Return all mutable state needed to continue control decisions."""

        return {
            "monitor": self.monitor,
            "best": self.best,
            "wait": self.wait,
            "lr_wait": self.lr_wait,
            "lr_best": self.lr_best,
            "best_state_dict": self.best_state_dict,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore mutable control state from a resume checkpoint."""

        if state.get("monitor") != self.monitor:
            raise ValueError(
                "Resume control monitor mismatch: "
                f"saved={state.get('monitor')!r}, current={self.monitor!r}"
            )
        self.best = float(state["best"])
        self.wait = int(state["wait"])
        self.lr_wait = int(state["lr_wait"])
        self.lr_best = float(state["lr_best"])
        self.best_state_dict = state.get("best_state_dict")
        self.best_epoch = state.get("best_epoch")


class MaeGuardrailControl:
    """Fine-tune-only MAE guardrail matching TensorFlow `MaeGuardrail`."""

    def __init__(self, mae_delta: float = 0.05, mae_patience: int = 10) -> None:
        self.mae_delta = float(mae_delta)
        self.mae_patience = int(mae_patience)
        self.best_mc_mae = float("inf")
        self.bad_epochs = 0

    def update(self, epoch: int, logs: dict[str, Any]) -> bool:
        score_mae = logs.get("metric_score_mae")
        veg_mae = logs.get("metric_veg_mae")
        if score_mae is None or veg_mae is None:
            print(f"[MaeGuardrail] Missing score/veg MAE at epoch {epoch}; skipping guardrail check.")
            return False

        mc_mae = (float(score_mae) + float(veg_mae)) / 2.0
        if mc_mae < self.best_mc_mae:
            self.best_mc_mae = mc_mae
            self.bad_epochs = 0
            return False

        if mc_mae > (self.best_mc_mae + self.mae_delta):
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0

        if self.bad_epochs >= self.mae_patience:
            pr_auc = logs.get("metric_bin_head_weighted_pr_auc", float("nan"))
            print(
                "[MaeGuardrail] Stopping training: "
                f"metric_bin_head_weighted_pr_auc={float(pr_auc):.4f}, mc_mae={mc_mae:.4f}, "
                f"best_mc_mae={self.best_mc_mae:.4f}, delta={self.mae_delta:.4f}, "
                f"patience={self.mae_patience}"
            )
            return True
        return False

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_mc_mae": self.best_mc_mae,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_mc_mae = float(state["best_mc_mae"])
        self.bad_epochs = int(state["bad_epochs"])


def train_one_epoch(
    model: Any,
    loader: Any,
    binary_cols: list[str],
    optimizer: Any,
    device: Any,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one train epoch and average batch diagnostics."""

    model.train()
    rows = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        images, targets = move_batch_to_device(batch, device)
        outputs = model(images)
        loss = _total_loss_tensor(outputs, targets, binary_cols)
        loss.backward()
        grad_summary = _grad_summary(model)
        optimizer.step()
        metrics = compute_one_batch_diagnostics(outputs, targets, binary_cols)
        metrics.update(grad_summary)
        rows.append(metrics)
    return _average_metric_rows(rows)


def evaluate_one_epoch(
    model: Any,
    loader: Any,
    binary_cols: list[str],
    device: Any,
    train_df: Any | None = None,
    eval_df: Any | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run one validation epoch and optionally compute weighted macro PR-AUC."""

    torch = _require_torch()
    model.eval()
    rows = []
    pred_chunks = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, targets = move_batch_to_device(batch, device)
            outputs = model(images)
            rows.append(compute_one_batch_diagnostics(outputs, targets, binary_cols))
            pred_chunks.append(torch.sigmoid(outputs["bin_head"]).detach().cpu().numpy())
    metrics = _average_metric_rows(rows)
    if train_df is not None and eval_df is not None and pred_chunks:
        y_pred = np.concatenate(pred_chunks, axis=0)
        eval_frame = eval_df.iloc[: len(y_pred)] if max_batches is not None else eval_df
        pr_auc = weighted_macro_pr_auc(y_pred, train_df, eval_frame, binary_cols)
        metrics["metric_bin_head_weighted_pr_auc"] = pr_auc["value"]
        metrics["metric_bin_head_weighted_pr_auc_valid_labels"] = pr_auc["valid_labels"]
        metrics["metric_bin_head_pr_auc_per_label"] = pr_auc["per_label"]
    return metrics


def _append_history(history: dict[str, list[Any]], phase: str, epoch: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any]) -> None:
    history.setdefault("phase", []).append(phase)
    history.setdefault("epoch", []).append(epoch)
    for key, value in train_metrics.items():
        history.setdefault(key, []).append(value)
    for key, value in val_metrics.items():
        history.setdefault(f"val_{key}", []).append(value)


def run_persistent_warmup_finetune(
    run_tag: str | None = None,
    training_config: dict[str, Any] | None = None,
    *,
    data_config: dict[str, Any] | None = None,
    split_dir: str | Path | None = None,
    image_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run or resume the established warm-up and fine-tuning orchestration.

    Resume checkpoints are written after completed epochs. If a process stops
    inside an epoch, resuming restarts from the previous completed epoch.
    """

    torch = _require_torch()
    resume_payload = (
        load_training_resume_checkpoint(resume_from)
        if resume_from is not None
        else None
    )
    resume_state = (
        dict(resume_payload["training_state"])
        if resume_payload is not None
        else None
    )

    if resume_payload is not None:
        saved_model_config = dict(resume_payload.get("model_config") or {})
        saved_training_config = dict(
            saved_model_config.get("torch_training_config") or {}
        )
        saved_data_config = dict(saved_model_config.get("torch_data_config") or {})
        if not saved_training_config or not saved_data_config:
            raise ValueError(
                "Resume checkpoint is missing saved training or data configuration."
            )
        cfg = _merge_resume_config(
            saved_training_config,
            training_config,
            mutable_keys={"device"},
            label="training",
        )
        data_cfg = _merge_resume_config(
            saved_data_config,
            data_config,
            mutable_keys={"num_workers", "pin_memory"},
            label="data",
        )
        saved_run_tag = str(resume_payload.get("run_tag") or "")
        if not saved_run_tag:
            raise ValueError("Resume checkpoint is missing run_tag.")
        if run_tag is not None and run_tag != saved_run_tag:
            raise ValueError(
                f"Resume run_tag mismatch: saved={saved_run_tag!r}, "
                f"requested={run_tag!r}"
            )
        run_tag = saved_run_tag
        run_dir = Path(resume_from).resolve().parent
        if bool(resume_state.get("run_complete", False)):
            raise ValueError(
                f"Training run {run_tag} is already complete; use its final or "
                "best checkpoint instead of resuming."
            )
    else:
        cfg = dict(TORCH_TRAINING_CONFIG)
        if training_config:
            cfg.update(training_config)
        data_cfg = dict(TORCH_DATA_CONFIG)
        if data_config:
            data_cfg.update(data_config)
        run_tag = run_tag or make_run_tag("PyTorch")
        selected_runs_root = (
            Path(runs_root)
            if runs_root is not None
            else PROJECT_ROOT / "models" / "runs"
        )
        run_dir = selected_runs_root / run_tag
        if run_dir.exists():
            raise FileExistsError(f"Training run directory already exists: {run_dir}")

    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg["num_workers"])
    pin_memory = bool(data_cfg["pin_memory"])
    if batch_size < 1:
        raise ValueError("Training batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("Training num_workers cannot be negative.")

    set_torch_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg.get("device", "auto")))

    train_df = load_split_df("train", split_dir=split_dir)
    val_df = load_split_df("val", split_dir=split_dir)
    schema = resolve_split_schema(train_df)
    data_signature = {
        "train": split_frame_signature(train_df),
        "val": split_frame_signature(val_df),
    }
    if resume_state is not None and resume_state["data_signature"] != data_signature:
        raise ValueError(
            "Train/validation split content differs from the resume checkpoint. "
            "Resume requires the same image filenames, labels, ordering, and rows."
        )

    train_loader, oversampling_plan = make_train_dataloader(
        batch_size=batch_size,
        split_dir=split_dir,
        image_transform="rgb_255",
        use_oversampling=bool(data_cfg["use_oversampling"]),
        use_augmentation=bool(data_cfg["use_augmentation"]),
        seed=int(cfg["seed"]),
        return_plan=True,
        image_root=image_root,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_eval_dataloader(
        "val",
        batch_size=batch_size,
        split_dir=split_dir,
        image_transform="rgb_255",
        image_root=image_root,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_run_mode = bool(cfg["test_run_mode"])
    warmup_epochs = int(
        cfg["test_warmup_epochs"] if test_run_mode else cfg["warmup_epochs"]
    )
    finetune_epochs = int(
        cfg["test_finetune_epochs"] if test_run_mode else cfg["finetune_epochs"]
    )
    if warmup_epochs < 1 or finetune_epochs < 1:
        raise ValueError("Warm-up and fine-tuning epoch targets must both be positive.")
    max_train_batches = cfg.get("max_train_batches")
    max_val_batches = cfg.get("max_val_batches")
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_val_batches = None if max_val_batches is None else int(max_val_batches)

    if resume_payload is not None:
        model_config = dict(resume_payload["model_config"])
        model_cfg = dict(model_config["torch_model_config"])
        saved_binary_cols = list(model_config.get("binary_cols") or [])
        if saved_binary_cols != schema.binary_cols:
            raise ValueError(
                "Active binary-label schema differs from the resume checkpoint: "
                f"saved={saved_binary_cols}, current={schema.binary_cols}"
            )
    else:
        model_cfg = dict(TORCH_MODEL_CONFIG)
        model_config = {}

    model = build_torchgeo_model(
        model_name=str(model_cfg["torchgeo_model_name"]),
        weight_name=str(model_cfg["torchgeo_weight"]),
        load_pretrained_weights=(
            False
            if resume_payload is not None
            else bool(model_cfg["load_pretrained_weights"])
        ),
        preserve_input_resolution=bool(model_cfg["preserve_input_resolution"]),
        input_size=tuple(data_cfg["img_size"]),
        num_binary=len(schema.binary_cols),
        num_shade=int(model_cfg["num_shade"]),
        score_output_range=tuple(model_cfg["score_output_range"]),
        veg_output_range=tuple(model_cfg["veg_output_range"]),
    )
    model.to(device)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
    else:
        run_dir = make_run_dir(run_tag, runs_root=selected_runs_root)

    print(f"RUN_TAG: {run_tag}")
    print(f"RUN_DIR: {run_dir}")
    print(f"started_at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"resume_from: {Path(resume_from).resolve() if resume_from else None}")
    print(f"device: {device}")
    print(
        "data:",
        {
            "split_dir": str(Path(split_dir).resolve()) if split_dir else "project default",
            "image_root": str(Path(image_root).resolve()) if image_root else "manifest/default",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
    )
    print(f"test_run_mode={test_run_mode} -> warmup={warmup_epochs}, finetune={finetune_epochs}")
    print(f"batch caps: train={max_train_batches}, val={max_val_batches}")
    print("model metadata:", model.metadata())
    print("oversampling plan:", oversampling_plan.summary() if oversampling_plan else None)
    monitor_name = "training_combo" if bool(cfg["use_combo_training_control"]) else "metric_bin_head_weighted_pr_auc"
    print(
        "training control:",
        {
            "monitor": f"val_{monitor_name}",
            "mode": "max",
            "early_stopping_patience": int(cfg["early_stopping_patience"]),
            "early_stopping_min_delta": float(cfg["early_stopping_min_delta"]),
            "restore_best_weights": bool(cfg["restore_best_weights"]),
            "reduce_lr_factor": float(cfg["reduce_lr_factor"]),
            "reduce_lr_patience": int(cfg["reduce_lr_patience"]),
            "reduce_lr_min_delta": float(cfg["reduce_lr_min_delta"]),
            "fine_tune_mae_guardrail": {
                "delta": float(cfg["mae_guardrail_delta"]),
                "patience": int(cfg["mae_guardrail_patience"]),
            },
        },
    )

    best_mcmae_path = run_dir / f"best_mcmae_{run_tag}.pt"
    best_prauc_path = run_dir / f"best_prauc_{run_tag}.pt"
    final_path = run_dir / f"final_{run_tag}.pt"
    last_path = run_dir / f"last_{run_tag}.pt"
    config_path = run_dir / f"model_config_{run_tag}.json"
    history_path = run_dir / f"training_history_{run_tag}.json"

    if resume_state is None:
        history: dict[str, list[Any]] = {}
        global_epoch = 0
        phase = "warmup"
        phase_epoch = 0
        best_mcmae = float("inf")
        best_prauc = float("-inf")
        elapsed_before = 0.0
        model_config = {
            "run_tag": run_tag,
            "framework": "pytorch",
            "model_family": f"TorchGeo {model_cfg['torchgeo_model_name']}",
            "model_metadata": model.metadata(),
            "img_size": data_cfg["img_size"],
            "binary_cols": schema.binary_cols,
            "score_head_mode": EXPERIMENT_CONFIG["score_head_mode"],
            "veg_head_mode": EXPERIMENT_CONFIG["veg_head_mode"],
            "experiment_config": dict(EXPERIMENT_CONFIG),
            "torch_data_config": dict(data_cfg),
            "torch_model_config": dict(model_cfg),
            "torch_training_config": dict(cfg),
            "warmup_epochs": warmup_epochs,
            "finetune_epochs": finetune_epochs,
            "test_run_mode": test_run_mode,
            "fine_tune_backbone": bool(cfg["fine_tune_backbone"]),
            "training_control_monitor": f"val_{monitor_name}",
            "use_oversampling": bool(data_cfg["use_oversampling"]),
            "use_augmentation": bool(data_cfg["use_augmentation"]),
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "data_signature": data_signature,
        }
    else:
        history = dict(resume_state["history"])
        global_epoch = int(resume_state["global_epoch"])
        phase = str(resume_state["phase"])
        phase_epoch = int(resume_state["phase_epoch"])
        best_mcmae = float(resume_state["best_mcmae"])
        best_prauc = float(resume_state["best_prauc"])
        elapsed_before = float(resume_state["elapsed_seconds"])
        if phase not in {"warmup", "finetune"}:
            raise ValueError(f"Unsupported resume phase: {phase!r}")

    session_started = time.monotonic()

    def make_control() -> PlateauTrainingControl:
        return PlateauTrainingControl(
            monitor=monitor_name,
            early_patience=int(cfg["early_stopping_patience"]),
            early_min_delta=float(cfg["early_stopping_min_delta"]),
            reduce_lr_patience=int(cfg["reduce_lr_patience"]),
            reduce_lr_factor=float(cfg["reduce_lr_factor"]),
            reduce_lr_min_delta=float(cfg["reduce_lr_min_delta"]),
            restore_best_weights=bool(cfg["restore_best_weights"]),
        )

    def elapsed_seconds() -> float:
        return elapsed_before + (time.monotonic() - session_started)

    def append_epoch_timing(epoch_seconds: float) -> None:
        history.setdefault("epoch_seconds", []).append(float(epoch_seconds))
        history.setdefault("elapsed_seconds", []).append(float(elapsed_seconds()))
        history.setdefault("completed_at", []).append(
            datetime.now().isoformat(timespec="seconds")
        )

    def maybe_save_best(
        active_phase: str,
        optimizer: Any,
        val_metrics: dict[str, Any],
    ) -> None:
        nonlocal best_mcmae, best_prauc
        val_mcmae = float(val_metrics["metric_score_veg_mae_mean"])
        if val_mcmae < best_mcmae:
            best_mcmae = val_mcmae
            save_checkpoint(
                best_mcmae_path,
                model=model,
                optimizer=optimizer,
                run_tag=run_tag,
                phase=active_phase,
                epoch=global_epoch,
                metrics=val_metrics,
                model_config=model_config,
            )
        val_prauc = float(
            val_metrics.get("metric_bin_head_weighted_pr_auc", float("nan"))
        )
        if np.isfinite(val_prauc) and val_prauc > best_prauc:
            best_prauc = val_prauc
            save_checkpoint(
                best_prauc_path,
                model=model,
                optimizer=optimizer,
                run_tag=run_tag,
                phase=active_phase,
                epoch=global_epoch,
                metrics=val_metrics,
                model_config=model_config,
            )

    def save_last_checkpoint(
        *,
        active_phase: str,
        active_phase_epoch: int,
        phase_complete: bool,
        optimizer: Any,
        control: PlateauTrainingControl,
        guardrail: MaeGuardrailControl | None,
        metrics: dict[str, Any],
        run_complete: bool = False,
    ) -> None:
        training_state = {
            "schema_version": RESUME_SCHEMA_VERSION,
            "phase": active_phase,
            "phase_epoch": int(active_phase_epoch),
            "global_epoch": int(global_epoch),
            "phase_complete": bool(phase_complete),
            "run_complete": bool(run_complete),
            "history": history,
            "best_mcmae": float(best_mcmae),
            "best_prauc": float(best_prauc),
            "elapsed_seconds": float(elapsed_seconds()),
            "data_signature": data_signature,
            "control_state": control.state_dict(),
            "guardrail_state": (
                guardrail.state_dict() if guardrail is not None else None
            ),
            "rng_state": capture_training_rng_state(train_loader),
        }
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            run_tag=run_tag,
            phase=active_phase,
            epoch=global_epoch,
            metrics=metrics,
            model_config=model_config,
            training_state=training_state,
        )

    rng_restored = False
    warmup_control = make_control()
    if phase == "warmup":
        set_backbone_trainable(model, False)
        print("Warm-up trainable summary:", trainable_parameter_summary(model))
        warmup_optimizer = make_optimizer(
            model,
            lr=float(cfg["warmup_learning_rate"]),
        )
        if resume_payload is not None:
            warmup_optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            warmup_control.load_state_dict(resume_state["control_state"])
            restore_training_rng_state(resume_state["rng_state"], train_loader)
            rng_restored = True
            print(
                f"Resuming warm-up after phase epoch {phase_epoch} "
                f"(global epoch {global_epoch})."
            )

        warmup_loop_start = (
            warmup_epochs
            if resume_state is not None
            and bool(resume_state.get("phase_complete", False))
            else phase_epoch
        )
        for current_phase_epoch in range(warmup_loop_start + 1, warmup_epochs + 1):
            epoch_started = time.monotonic()
            global_epoch += 1
            train_metrics = train_one_epoch(
                model,
                train_loader,
                schema.binary_cols,
                warmup_optimizer,
                device,
                max_batches=max_train_batches,
            )
            val_metrics = evaluate_one_epoch(
                model,
                val_loader,
                schema.binary_cols,
                device,
                train_df=train_df,
                eval_df=val_df,
                max_batches=max_val_batches,
            )
            if bool(cfg["use_combo_training_control"]):
                val_metrics.update(
                    compute_training_combo(
                        val_metrics,
                        w_bin=float(cfg["combo_w_bin"]),
                        w_ord=float(cfg["combo_w_ord"]),
                        mae_scale=float(cfg["combo_mcmae_scale"]),
                    )
                )
            _append_history(
                history,
                "warmup",
                global_epoch,
                train_metrics,
                val_metrics,
            )
            maybe_save_best("warmup", warmup_optimizer, val_metrics)
            stop_for_plateau = warmup_control.update(
                model,
                warmup_optimizer,
                global_epoch,
                val_metrics,
            )
            epoch_seconds = time.monotonic() - epoch_started
            append_epoch_timing(epoch_seconds)
            phase_complete = bool(
                stop_for_plateau or current_phase_epoch >= warmup_epochs
            )
            print(
                f"warmup epoch {global_epoch}: "
                f"train_loss={train_metrics['loss_total']:.4f} "
                f"val_loss={val_metrics['loss_total']:.4f} "
                f"val_{monitor_name}={float(val_metrics.get(monitor_name, float('nan'))):.4f} "
                f"lr={_get_optimizer_lr(warmup_optimizer):.6g} "
                f"seconds={epoch_seconds:.1f} elapsed={elapsed_seconds():.1f}"
            )
            save_last_checkpoint(
                active_phase="warmup",
                active_phase_epoch=current_phase_epoch,
                phase_complete=phase_complete,
                optimizer=warmup_optimizer,
                control=warmup_control,
                guardrail=None,
                metrics=val_metrics,
            )
            phase_epoch = current_phase_epoch
            if stop_for_plateau:
                break
        warmup_control.restore(model, device)
        phase = "finetune"
        phase_epoch = 0

    set_backbone_trainable(model, bool(cfg["fine_tune_backbone"]))
    print("Fine-tune trainable summary:", trainable_parameter_summary(model))
    finetune_optimizer = make_optimizer(
        model,
        lr=float(cfg["finetune_learning_rate"]),
    )
    finetune_control = make_control()
    mae_guardrail = MaeGuardrailControl(
        mae_delta=float(cfg["mae_guardrail_delta"]),
        mae_patience=int(cfg["mae_guardrail_patience"]),
    )
    if resume_payload is not None and phase == "finetune" and not rng_restored:
        finetune_optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        finetune_control.load_state_dict(resume_state["control_state"])
        guardrail_state = resume_state.get("guardrail_state")
        if not isinstance(guardrail_state, dict):
            raise ValueError("Fine-tune resume checkpoint is missing guardrail state.")
        mae_guardrail.load_state_dict(guardrail_state)
        restore_training_rng_state(resume_state["rng_state"], train_loader)
        print(
            f"Resuming fine-tuning after phase epoch {phase_epoch} "
            f"(global epoch {global_epoch})."
        )

    finetune_loop_start = (
        finetune_epochs
        if resume_state is not None
        and str(resume_state.get("phase")) == "finetune"
        and bool(resume_state.get("phase_complete", False))
        else phase_epoch
    )
    for current_phase_epoch in range(finetune_loop_start + 1, finetune_epochs + 1):
        epoch_started = time.monotonic()
        global_epoch += 1
        train_metrics = train_one_epoch(
            model,
            train_loader,
            schema.binary_cols,
            finetune_optimizer,
            device,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate_one_epoch(
            model,
            val_loader,
            schema.binary_cols,
            device,
            train_df=train_df,
            eval_df=val_df,
            max_batches=max_val_batches,
        )
        if bool(cfg["use_combo_training_control"]):
            val_metrics.update(
                compute_training_combo(
                    val_metrics,
                    w_bin=float(cfg["combo_w_bin"]),
                    w_ord=float(cfg["combo_w_ord"]),
                    mae_scale=float(cfg["combo_mcmae_scale"]),
                )
            )
        _append_history(
            history,
            "finetune",
            global_epoch,
            train_metrics,
            val_metrics,
        )
        maybe_save_best("finetune", finetune_optimizer, val_metrics)
        stop_for_plateau = finetune_control.update(
            model,
            finetune_optimizer,
            global_epoch,
            val_metrics,
        )
        stop_for_guardrail = mae_guardrail.update(global_epoch, val_metrics)
        epoch_seconds = time.monotonic() - epoch_started
        append_epoch_timing(epoch_seconds)
        phase_complete = bool(
            stop_for_plateau
            or stop_for_guardrail
            or current_phase_epoch >= finetune_epochs
        )
        print(
            f"finetune epoch {global_epoch}: "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"val_loss={val_metrics['loss_total']:.4f} "
            f"val_{monitor_name}={float(val_metrics.get(monitor_name, float('nan'))):.4f} "
            f"lr={_get_optimizer_lr(finetune_optimizer):.6g} "
            f"seconds={epoch_seconds:.1f} elapsed={elapsed_seconds():.1f}"
        )
        save_last_checkpoint(
            active_phase="finetune",
            active_phase_epoch=current_phase_epoch,
            phase_complete=phase_complete,
            optimizer=finetune_optimizer,
            control=finetune_control,
            guardrail=mae_guardrail,
            metrics=val_metrics,
        )
        phase_epoch = current_phase_epoch
        if stop_for_plateau or stop_for_guardrail:
            break
    finetune_control.restore(model, device)

    model_config.update(
        {
            "final_model_path": str(final_path),
            "best_prauc_path": (
                str(best_prauc_path) if best_prauc_path.exists() else None
            ),
            "best_mc_mae_path": (
                str(best_mcmae_path) if best_mcmae_path.exists() else None
            ),
            "last_checkpoint_path": str(last_path),
            "total_elapsed_seconds": float(elapsed_seconds()),
        }
    )
    final_metrics = {
        key: values[-1]
        for key, values in history.items()
        if isinstance(values, list) and values
    }
    save_checkpoint(
        final_path,
        model=model,
        optimizer=finetune_optimizer,
        run_tag=run_tag,
        phase="final",
        epoch=global_epoch,
        metrics=final_metrics,
        model_config=model_config,
    )

    save_json(config_path, model_config)
    save_json(history_path, history)
    curves_path = save_training_curves(history, run_dir, warmup_epochs=warmup_epochs)
    save_last_checkpoint(
        active_phase="finetune",
        active_phase_epoch=phase_epoch,
        phase_complete=True,
        optimizer=finetune_optimizer,
        control=finetune_control,
        guardrail=mae_guardrail,
        metrics=final_metrics,
        run_complete=True,
    )

    print("Saved final checkpoint:", final_path)
    print("Saved resumable checkpoint:", last_path)
    print("Saved best MC-MAE checkpoint:", best_mcmae_path.exists(), best_mcmae_path)
    print("Saved best PR-AUC checkpoint:", best_prauc_path.exists(), best_prauc_path)
    print("Saved config:", config_path)
    print("Saved history:", history_path)
    if curves_path:
        print("Saved curves:", curves_path)
    print(f"completed_at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"total_elapsed_seconds: {elapsed_seconds():.1f}")

    return {
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "final_model_path": str(final_path),
        "best_mc_mae_path": str(best_mcmae_path) if best_mcmae_path.exists() else None,
        "best_prauc_path": str(best_prauc_path) if best_prauc_path.exists() else None,
        "last_checkpoint_path": str(last_path),
        "model_config_path": str(config_path),
        "training_history_path": str(history_path),
        "training_curves_path": str(curves_path) if curves_path else None,
        "resumed": resume_from is not None,
        "total_elapsed_seconds": float(elapsed_seconds()),
        "history": history,
    }


def backward_smoke_step(
    model: Any,
    batch: tuple[Any, dict[str, Any]],
    binary_cols: list[str],
) -> dict[str, Any]:
    """Compute loss and gradients without updating weights."""

    model.train()
    model.zero_grad(set_to_none=True)
    images, targets = batch
    outputs = model(images)
    loss = _total_loss_tensor(outputs, targets, binary_cols)
    loss.backward()
    diagnostics = compute_one_batch_diagnostics(outputs, targets, binary_cols)
    diagnostics.update(_grad_summary(model))
    return diagnostics


def optimizer_smoke_step(
    model: Any,
    batch: tuple[Any, dict[str, Any]],
    binary_cols: list[str],
    optimizer: Any,
) -> dict[str, Any]:
    """Run one optimizer update."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    images, targets = batch
    outputs = model(images)
    loss = _total_loss_tensor(outputs, targets, binary_cols)
    loss.backward()
    grad_summary = _grad_summary(model)
    optimizer.step()
    diagnostics = compute_one_batch_diagnostics(outputs, targets, binary_cols)
    diagnostics.update(grad_summary)
    return diagnostics


def run_repeated_batch_debug(
    model: Any,
    batch: tuple[Any, dict[str, Any]],
    binary_cols: list[str],
    optimizer: Any,
    steps: int = TORCH_TRAINING_SMOKE_CONFIG["debug_steps"],
) -> list[dict[str, Any]]:
    """Run repeated optimizer steps on one fixed batch."""

    history = []
    for step in range(1, steps + 1):
        metrics = optimizer_smoke_step(model, batch, binary_cols, optimizer)
        history.append({"step": step, **metrics})
    return history


def evaluate_loader(
    model: Any,
    loader: Any,
    binary_cols: list[str],
    max_batches: int | None = None,
) -> dict[str, float]:
    """Average smoke diagnostics over a loader."""

    torch = _require_torch()
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            outputs = model(images)
            metrics = compute_one_batch_diagnostics(outputs, targets, binary_cols)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
    if count == 0:
        raise ValueError("No batches available for evaluation.")
    return {key: value / count for key, value in totals.items()}


def run_tiny_train_val_epoch(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    binary_cols: list[str],
    optimizer: Any,
) -> dict[str, dict[str, float]]:
    """Run one small train epoch plus one validation pass."""

    train_totals: dict[str, float] = {}
    train_count = 0
    for images, targets in train_loader:
        metrics = optimizer_smoke_step(model, (images, targets), binary_cols, optimizer)
        for key, value in metrics.items():
            train_totals[key] = train_totals.get(key, 0.0) + float(value)
        train_count += 1
    if train_count == 0:
        raise ValueError("No train batches available.")

    train_summary = {key: value / train_count for key, value in train_totals.items()}
    val_summary = evaluate_loader(model, val_loader, binary_cols)
    return {"train": train_summary, "val": val_summary}
