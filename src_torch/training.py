"""PyTorch training helpers for smoke checks and first persistent runs."""

from __future__ import annotations

import random
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
from src_torch.models import build_torchgeo_resnet50_forward_model


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
) -> dict[str, Any]:
    """Run a persistent PyTorch warm-up + fine-tune test and save artifacts."""

    torch = _require_torch()
    cfg = dict(TORCH_TRAINING_CONFIG)
    if training_config:
        cfg.update(training_config)

    set_torch_seed(int(cfg["seed"]))
    run_tag = run_tag or make_run_tag("PyTorch")
    run_dir = make_run_dir(run_tag)
    device = resolve_device(str(cfg.get("device", "auto")))

    train_df = load_split_df("train")
    val_df = load_split_df("val")
    schema = resolve_split_schema(train_df)

    batch_size = int(TORCH_DATA_CONFIG["batch_size"])
    train_loader, oversampling_plan = make_train_dataloader(
        batch_size=batch_size,
        image_transform="rgb_255",
        use_oversampling=bool(TORCH_DATA_CONFIG["use_oversampling"]),
        use_augmentation=bool(TORCH_DATA_CONFIG["use_augmentation"]),
        seed=int(cfg["seed"]),
        return_plan=True,
    )
    val_loader = make_eval_dataloader("val", batch_size=batch_size, image_transform="rgb_255")

    model = build_torchgeo_resnet50_forward_model(load_pretrained_weights=TORCH_MODEL_CONFIG["load_pretrained_weights"])
    model.to(device)

    test_run_mode = bool(cfg["test_run_mode"])
    warmup_epochs = int(cfg["test_warmup_epochs"] if test_run_mode else cfg["warmup_epochs"])
    finetune_epochs = int(cfg["test_finetune_epochs"] if test_run_mode else cfg["finetune_epochs"])
    max_train_batches = cfg.get("max_train_batches")
    max_val_batches = cfg.get("max_val_batches")
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_val_batches = None if max_val_batches is None else int(max_val_batches)

    print(f"RUN_TAG: {run_tag}")
    print(f"RUN_DIR: {run_dir}")
    print(f"device: {device}")
    print(f"test_run_mode={test_run_mode} -> warmup={warmup_epochs}, finetune={finetune_epochs}")
    print(f"batch caps: train={max_train_batches}, val={max_val_batches}")
    print("oversampling plan:", oversampling_plan.summary() if oversampling_plan else None)

    best_mcmae = float("inf")
    best_prauc = float("-inf")
    best_mcmae_path = run_dir / f"best_mcmae_{run_tag}.pt"
    best_prauc_path = run_dir / f"best_prauc_{run_tag}.pt"
    final_path = run_dir / f"final_{run_tag}.pt"
    history: dict[str, list[Any]] = {}

    model_config = {
        "run_tag": run_tag,
        "framework": "pytorch",
        "model_family": "TorchGeo ResNet-50",
        "model_metadata": model.metadata(),
        "img_size": TORCH_DATA_CONFIG["img_size"],
        "binary_cols": schema.binary_cols,
        "score_head_mode": EXPERIMENT_CONFIG["score_head_mode"],
        "veg_head_mode": EXPERIMENT_CONFIG["veg_head_mode"],
        "experiment_config": dict(EXPERIMENT_CONFIG),
        "torch_data_config": dict(TORCH_DATA_CONFIG),
        "torch_model_config": dict(TORCH_MODEL_CONFIG),
        "torch_training_config": dict(cfg),
        "warmup_epochs": warmup_epochs,
        "finetune_epochs": finetune_epochs,
        "test_run_mode": test_run_mode,
        "fine_tune_backbone": bool(cfg["fine_tune_backbone"]),
        "use_oversampling": bool(TORCH_DATA_CONFIG["use_oversampling"]),
        "use_augmentation": bool(TORCH_DATA_CONFIG["use_augmentation"]),
    }

    global_epoch = 0

    set_backbone_trainable(model, False)
    print("Warm-up trainable summary:", trainable_parameter_summary(model))
    warmup_optimizer = make_optimizer(model, lr=float(cfg["warmup_learning_rate"]))
    for _ in range(warmup_epochs):
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
        _append_history(history, "warmup", global_epoch, train_metrics, val_metrics)
        print(f"warmup epoch {global_epoch}: train_loss={train_metrics['loss_total']:.4f} val_loss={val_metrics['loss_total']:.4f}")

        val_mcmae = float(val_metrics["metric_score_veg_mae_mean"])
        if val_mcmae < best_mcmae:
            best_mcmae = val_mcmae
            save_checkpoint(
                best_mcmae_path,
                model=model,
                optimizer=warmup_optimizer,
                run_tag=run_tag,
                phase="warmup",
                epoch=global_epoch,
                metrics=val_metrics,
                model_config=model_config,
            )
        val_prauc = float(val_metrics.get("metric_bin_head_weighted_pr_auc", float("nan")))
        if np.isfinite(val_prauc) and val_prauc > best_prauc:
            best_prauc = val_prauc
            save_checkpoint(
                best_prauc_path,
                model=model,
                optimizer=warmup_optimizer,
                run_tag=run_tag,
                phase="warmup",
                epoch=global_epoch,
                metrics=val_metrics,
                model_config=model_config,
            )

    set_backbone_trainable(model, bool(cfg["fine_tune_backbone"]))
    print("Fine-tune trainable summary:", trainable_parameter_summary(model))
    finetune_optimizer = make_optimizer(model, lr=float(cfg["finetune_learning_rate"]))
    for _ in range(finetune_epochs):
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
        _append_history(history, "finetune", global_epoch, train_metrics, val_metrics)
        print(f"finetune epoch {global_epoch}: train_loss={train_metrics['loss_total']:.4f} val_loss={val_metrics['loss_total']:.4f}")

        val_mcmae = float(val_metrics["metric_score_veg_mae_mean"])
        if val_mcmae < best_mcmae:
            best_mcmae = val_mcmae
            save_checkpoint(
                best_mcmae_path,
                model=model,
                optimizer=finetune_optimizer,
                run_tag=run_tag,
                phase="finetune",
                epoch=global_epoch,
                metrics=val_metrics,
                model_config=model_config,
            )
        val_prauc = float(val_metrics.get("metric_bin_head_weighted_pr_auc", float("nan")))
        if np.isfinite(val_prauc) and val_prauc > best_prauc:
            best_prauc = val_prauc
            save_checkpoint(
                best_prauc_path,
                model=model,
                optimizer=finetune_optimizer,
                run_tag=run_tag,
                phase="finetune",
                epoch=global_epoch,
                metrics=val_metrics,
                model_config=model_config,
            )

    save_checkpoint(
        final_path,
        model=model,
        optimizer=finetune_optimizer,
        run_tag=run_tag,
        phase="final",
        epoch=global_epoch,
        metrics={key: values[-1] for key, values in history.items() if values},
        model_config=model_config,
    )

    model_config.update(
        {
            "final_model_path": str(final_path),
            "best_prauc_path": str(best_prauc_path) if best_prauc_path.exists() else None,
            "best_mc_mae_path": str(best_mcmae_path) if best_mcmae_path.exists() else None,
        }
    )
    config_path = run_dir / f"model_config_{run_tag}.json"
    history_path = run_dir / f"training_history_{run_tag}.json"
    save_json(config_path, model_config)
    save_json(history_path, history)
    curves_path = save_training_curves(history, run_dir, warmup_epochs=warmup_epochs)

    print("Saved final checkpoint:", final_path)
    print("Saved best MC-MAE checkpoint:", best_mcmae_path.exists(), best_mcmae_path)
    print("Saved best PR-AUC checkpoint:", best_prauc_path.exists(), best_prauc_path)
    print("Saved config:", config_path)
    print("Saved history:", history_path)
    if curves_path:
        print("Saved curves:", curves_path)

    return {
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "final_model_path": str(final_path),
        "best_mc_mae_path": str(best_mcmae_path) if best_mcmae_path.exists() else None,
        "best_prauc_path": str(best_prauc_path) if best_prauc_path.exists() else None,
        "model_config_path": str(config_path),
        "training_history_path": str(history_path),
        "training_curves_path": str(curves_path) if curves_path else None,
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
