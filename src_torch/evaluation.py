"""PyTorch evaluation helpers aligned to TensorFlow NB04 report outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src_torch.config import PROJECT_ROOT, TORCH_DATA_CONFIG, TORCH_LOSS_WEIGHTS
from src_torch.data import load_split_df, make_dataloader, resolve_split_schema
from src_torch.models import build_torchgeo_resnet50_forward_model
from src_torch.training import resolve_device


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for evaluation.") from exc
    return torch, functional


def infer_run_tag_and_variant(model_path: str | Path) -> tuple[str, str]:
    """Infer run tag and variant from a PyTorch checkpoint path."""

    path = Path(model_path)
    run_tag = path.parent.name if path.parent.name.startswith("PyTorch_") else None
    stem = path.stem
    if stem.startswith("best_mcmae_"):
        variant = "best_mcmae"
    elif stem.startswith("best_prauc_"):
        variant = "best_prauc"
    elif stem.startswith("final_"):
        variant = "final"
    else:
        variant = "unknown_model"
    if run_tag is None:
        for idx, part in enumerate(path.parts):
            if part == "runs" and idx + 1 < len(path.parts):
                run_tag = path.parts[idx + 1]
                break
    if run_tag is None:
        raise ValueError(f"Could not infer run tag from {path}")
    return run_tag, variant


def find_latest_pytorch_checkpoint(
    runs_root: Path | None = None,
    preferred_variant: str = "best_mcmae",
) -> Path:
    """Return a checkpoint from the newest `models/runs/PyTorch_*` folder."""

    root = runs_root or (PROJECT_ROOT / "models" / "runs")
    run_dirs = sorted(root.glob("PyTorch_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No PyTorch run folders found under {root}")
    patterns = {
        "best_mcmae": ["best_mcmae_*.pt", "best_prauc_*.pt", "final_*.pt"],
        "best_prauc": ["best_prauc_*.pt", "best_mcmae_*.pt", "final_*.pt"],
        "final": ["final_*.pt", "best_mcmae_*.pt", "best_prauc_*.pt"],
    }.get(preferred_variant, ["best_mcmae_*.pt", "best_prauc_*.pt", "final_*.pt"])
    for run_dir in run_dirs:
        for pattern in patterns:
            candidates = sorted(run_dir.glob(pattern))
            if candidates:
                return candidates[-1]
    raise FileNotFoundError(f"No PyTorch checkpoints found under {root}")


def load_torch_checkpoint_model(model_path: str | Path, device: Any | None = None) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load a GreenSpace TorchGeo model checkpoint for evaluation."""

    torch, _ = _require_torch()
    path = Path(model_path)
    run_tag, _ = infer_run_tag_and_variant(path)
    config_path = path.parent / f"model_config_{run_tag}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")

    import json

    model_config = json.loads(config_path.read_text())
    device = device or resolve_device("auto")
    torch_model_config = model_config.get("torch_model_config", {})
    torch_data_config = model_config.get("torch_data_config", {})
    saved_img_size = tuple(torch_data_config.get("img_size", model_config.get("img_size", (512, 512))))
    model = build_torchgeo_resnet50_forward_model(
        load_pretrained_weights=bool(torch_model_config.get("load_pretrained_weights", True)),
        preserve_input_resolution=bool(torch_model_config.get("preserve_input_resolution", False)),
        input_size=saved_img_size,
    )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_config, checkpoint


def make_plain_eval_loader(
    split: str,
    batch_size: int = TORCH_DATA_CONFIG["batch_size"],
):
    """Evaluation loader with no augmentation and no oversampling."""

    return make_dataloader(
        split,
        batch_size=batch_size,
        shuffle=False,
        image_transform="rgb_255",
        augment=False,
    )


def _move_targets(targets: dict[str, Any], device: Any) -> dict[str, Any]:
    return {name: value.to(device) for name, value in targets.items()}


def _eval_batch_losses(outputs: dict[str, Any], targets: dict[str, Any]) -> dict[str, float]:
    """Unweighted NB04-style per-head losses/metrics for one batch."""

    torch, functional = _require_torch()
    y_bin = targets["bin_head"].to(dtype=torch.float32)
    y_shade = targets["shade_head"].to(dtype=torch.long)
    y_score = targets["score_head"].to(dtype=torch.float32)
    y_veg = targets["veg_head"].to(dtype=torch.float32)

    bin_loss = functional.binary_cross_entropy_with_logits(outputs["bin_head"], y_bin)
    shade_loss = functional.cross_entropy(outputs["shade_head"], y_shade)
    score_loss = torch.abs(outputs["score_head"] - y_score).view(-1).mean()
    veg_loss = torch.abs(outputs["veg_head"] - y_veg).view(-1).mean()
    total = (
        TORCH_LOSS_WEIGHTS["bin_head"] * bin_loss
        + TORCH_LOSS_WEIGHTS["shade_head"] * shade_loss
        + TORCH_LOSS_WEIGHTS["score_head"] * score_loss
        + TORCH_LOSS_WEIGHTS["veg_head"] * veg_loss
    )
    bin_acc = ((outputs["bin_head"] >= 0).to(dtype=torch.float32) == y_bin).to(dtype=torch.float32).mean()
    shade_acc = (outputs["shade_head"].argmax(dim=1) == y_shade).to(dtype=torch.float32).mean()
    return {
        "loss": float(total.detach()),
        "bin_head_loss": float(bin_loss.detach()),
        "shade_head_loss": float(shade_loss.detach()),
        "score_head_loss": float(score_loss.detach()),
        "veg_head_loss": float(veg_loss.detach()),
        "bin_head_binary_accuracy": float(bin_acc.detach()),
        "shade_head_sparse_categorical_accuracy": float(shade_acc.detach()),
        "score_head_mae": float(score_loss.detach()),
        "veg_head_mae": float(veg_loss.detach()),
    }


def _average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No rows to average.")
    keys = rows[0].keys()
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}


def predict_split(
    model: Any,
    split: str,
    device: Any,
    batch_size: int = TORCH_DATA_CONFIG["batch_size"],
    max_batches: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    """Predict one split and return split df, predictions, and loss-monitor metrics."""

    torch, functional = _require_torch()
    df = load_split_df(split)
    loader = make_plain_eval_loader(split, batch_size=batch_size)
    rows = []
    pred_chunks = {"bin_head": [], "shade_head": [], "score_head": [], "veg_head": []}
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            targets = _move_targets(targets, device)
            outputs = model(images)
            rows.append(_eval_batch_losses(outputs, targets))
            pred_chunks["bin_head"].append(torch.sigmoid(outputs["bin_head"]).detach().cpu().numpy())
            pred_chunks["shade_head"].append(functional.softmax(outputs["shade_head"], dim=1).detach().cpu().numpy())
            pred_chunks["score_head"].append(outputs["score_head"].detach().cpu().numpy())
            pred_chunks["veg_head"].append(outputs["veg_head"].detach().cpu().numpy())

    preds = {name: np.concatenate(chunks, axis=0) for name, chunks in pred_chunks.items()}
    used_df = df.iloc[: len(preds["bin_head"])].reset_index(drop=True)
    return used_df, preds, _average_rows(rows)


def _binary_truth_and_probs(df: pd.DataFrame, preds: dict[str, np.ndarray], binary_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    bin_names = [col[:-2] for col in binary_cols]
    hard_names = [name for name in bin_names if name in df.columns]
    if hard_names:
        y_true = df[hard_names].fillna(0).astype(int).to_numpy()
        y_prob = np.stack([preds["bin_head"][:, bin_names.index(name)] for name in hard_names], axis=1)
        return y_true, y_prob, hard_names
    y_true = (df[binary_cols].fillna(0.0).astype(np.float32).to_numpy() >= 0.5).astype(int)
    return y_true, preds["bin_head"], bin_names


def _macro_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    vals = []
    for idx in range(y_true.shape[1]):
        if np.unique(y_true[:, idx]).size >= 2:
            vals.append(float(average_precision_score(y_true[:, idx], y_prob[:, idx])))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_loss_monitoring(predictions_by_split: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float]]], binary_cols: list[str]) -> pd.DataFrame:
    """Create TensorFlow-compatible loss monitoring table."""

    rows = []
    for split, (df, preds, loss_metrics) in predictions_by_split.items():
        y_true, y_prob, _ = _binary_truth_and_probs(df, preds, binary_cols)
        row = {"split": split, **loss_metrics}
        row["bin_head_pr_auc"] = _macro_average_precision(y_true, y_prob)
        rows.append(row)
    keep = [
        "split",
        "loss",
        "bin_head_loss",
        "shade_head_loss",
        "score_head_loss",
        "veg_head_loss",
        "bin_head_binary_accuracy",
        "bin_head_pr_auc",
        "shade_head_sparse_categorical_accuracy",
        "score_head_mae",
        "veg_head_mae",
    ]
    return pd.DataFrame(rows)[keep]


def tune_thresholds_f1(y_true_mat: np.ndarray, y_prob_mat: np.ndarray, label_names: list[str], min_pos: int = 1) -> pd.DataFrame:
    """Tune per-label thresholds on validation to maximize F1."""

    from sklearn.metrics import precision_recall_curve

    rows = []
    for idx, name in enumerate(label_names):
        y_true = np.asarray(y_true_mat[:, idx]).astype(int)
        y_prob = np.asarray(y_prob_mat[:, idx]).astype(float)
        n_pos = int(y_true.sum())
        n = int(len(y_true))
        pos_rate = float(y_true.mean()) if n else float("nan")
        if np.unique(y_true).size < 2 or n_pos < min_pos:
            rows.append(
                {
                    "label": name,
                    "best_threshold": np.nan,
                    "best_f1": np.nan,
                    "best_precision": np.nan,
                    "best_recall": np.nan,
                    "pos_rate": pos_rate,
                    "n_pos": n_pos,
                    "n": n,
                    "note": "single-class" if np.unique(y_true).size < 2 else "too-few-positives",
                }
            )
            continue
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        if thresholds.size == 0:
            rows.append(
                {
                    "label": name,
                    "best_threshold": np.nan,
                    "best_f1": np.nan,
                    "best_precision": np.nan,
                    "best_recall": np.nan,
                    "pos_rate": pos_rate,
                    "n_pos": n_pos,
                    "n": n,
                    "note": "no-thresholds",
                }
            )
            continue
        p = precision[:-1]
        r = recall[:-1]
        f1 = (2 * p * r) / (p + r + 1e-12)
        best_f1 = float(np.max(f1))
        best_idx = int(np.flatnonzero(f1 == best_f1)[0])
        rows.append(
            {
                "label": name,
                "best_threshold": float(thresholds[best_idx]),
                "best_f1": best_f1,
                "best_precision": float(p[best_idx]),
                "best_recall": float(r[best_idx]),
                "pos_rate": pos_rate,
                "n_pos": n_pos,
                "n": n,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def _pred_ordinal_class(pred_arr: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(pred_arr.squeeze(), 1, 5)).astype(np.int64) - 1


def _expected_score_veg(pred_arr: np.ndarray) -> np.ndarray:
    return np.clip(pred_arr.squeeze(), 1, 5).astype(np.float32)


def evaluate_split_metrics(
    split_name: str,
    df: pd.DataFrame,
    preds: dict[str, np.ndarray],
    binary_cols: list[str],
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Create TensorFlow-compatible overall and per-label rows for one split."""

    from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

    y_bin_true, pred_bin_aligned, label_names = _binary_truth_and_probs(df, preds, binary_cols)
    f1_05_list = []
    roc_list = []
    ap_list = []
    f1_tuned_list = []
    label_rows = []

    for idx, name in enumerate(label_names):
        y_true = y_bin_true[:, idx]
        y_prob = pred_bin_aligned[:, idx]
        y_hat_05 = (y_prob >= 0.5).astype(int)
        p05, r05, f105, _ = precision_recall_fscore_support(y_true, y_hat_05, average="binary", zero_division=0)
        f1_05_list.append(float(f105))
        roc = np.nan
        ap = np.nan
        if np.unique(y_true).size >= 2:
            roc = float(roc_auc_score(y_true, y_prob))
            ap = float(average_precision_score(y_true, y_prob))
            roc_list.append(roc)
            ap_list.append(ap)

        thr = thresholds.get(name, np.nan)
        pt = rt = f1t = np.nan
        if np.isfinite(thr):
            y_hat_t = (y_prob >= float(thr)).astype(int)
            pt, rt, f1t, _ = precision_recall_fscore_support(y_true, y_hat_t, average="binary", zero_division=0)
            pt, rt, f1t = float(pt), float(rt), float(f1t)
            f1_tuned_list.append(f1t)

        label_rows.append(
            {
                "split": split_name,
                "label": name,
                "support_pos": int(np.sum(y_true == 1)),
                "support_neg": int(np.sum(y_true == 0)),
                "P@0.5": float(p05),
                "R@0.5": float(r05),
                "F1@0.5": float(f105),
                "ROC_AUC": roc,
                "PR_AUC": ap,
                "thr_val": float(thr) if np.isfinite(thr) else np.nan,
                "P@thr": pt,
                "R@thr": rt,
                "F1@thr": f1t,
            }
        )

    y_shade_true = df["shade_class"].fillna(0).astype(int).to_numpy()
    y_score_true = df["score_class"].fillna(1).astype(int).to_numpy() - 1
    y_veg_true = df["veg_class"].fillna(1).astype(int).to_numpy() - 1

    shade_pred = preds["shade_head"].argmax(axis=1)
    shade_acc = float(accuracy_score(y_shade_true, shade_pred))
    if "walking_paths_p" in df.columns:
        w = df["walking_paths_p"].fillna(0.0).to_numpy(dtype=np.float32)
    elif "walking_paths" in df.columns:
        w = df["walking_paths"].fillna(0).to_numpy(dtype=np.float32)
    else:
        w = None
    shade_acc_conditional = np.nan
    shade_acc_paths_present = np.nan
    if w is not None:
        shade_correct = (shade_pred == y_shade_true).astype(np.float32)
        denom = float(np.sum(w))
        if denom > 1e-8:
            shade_acc_conditional = float(np.sum(w * shade_correct) / denom)
        mask = w >= 0.5
        if int(np.sum(mask)) > 0:
            shade_acc_paths_present = float(accuracy_score(y_shade_true[mask], shade_pred[mask]))

    score_expected = _expected_score_veg(preds["score_head"])
    veg_expected = _expected_score_veg(preds["veg_head"])
    score_true_1to5 = (y_score_true + 1).astype(np.float32)
    veg_true_1to5 = (y_veg_true + 1).astype(np.float32)

    score_mae_mean = float(np.mean(np.abs(score_expected - df["score_mean"].astype(np.float32).to_numpy()))) if "score_mean" in df.columns else np.nan
    veg_mae_mean = float(np.mean(np.abs(veg_expected - df["veg_mean"].astype(np.float32).to_numpy()))) if "veg_mean" in df.columns else np.nan

    overall_row = {
        "split": split_name,
        "n_samples": int(len(df)),
        "n_labels": int(len(label_names)),
        "overall_F1@0.5": float(np.mean(f1_05_list)) if f1_05_list else np.nan,
        "overall_ROC_AUC": float(np.mean(roc_list)) if roc_list else np.nan,
        "overall_PR_AUC": float(np.mean(ap_list)) if ap_list else np.nan,
        "overall_F1@tuned": float(np.mean(f1_tuned_list)) if f1_tuned_list else np.nan,
        "shade_acc_overall": shade_acc,
        "shade_acc_conditional": shade_acc_conditional,
        "shade_acc_paths_present": shade_acc_paths_present,
        "score_acc": float(accuracy_score(y_score_true, _pred_ordinal_class(preds["score_head"]))),
        "veg_acc": float(accuracy_score(y_veg_true, _pred_ordinal_class(preds["veg_head"]))),
        "score_mae": float(np.mean(np.abs(score_expected - score_true_1to5))),
        "veg_mae": float(np.mean(np.abs(veg_expected - veg_true_1to5))),
        "score_ce": np.nan,
        "veg_ce": np.nan,
        "score_mae_mean": score_mae_mean,
        "veg_mae_mean": veg_mae_mean,
    }
    return overall_row, pd.DataFrame(label_rows)


def evaluate_all_splits(
    predictions_by_split: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float]]],
    binary_cols: list[str],
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create overall and per-label split report tables."""

    overall_rows = []
    label_parts = []
    for split in ("train", "val", "test"):
        df, preds, _ = predictions_by_split[split]
        overall_row, label_df = evaluate_split_metrics(split, df, preds, binary_cols, thresholds)
        overall_rows.append(overall_row)
        label_parts.append(label_df)

    overall_df = pd.DataFrame(overall_rows)
    overall_df["split"] = pd.Categorical(overall_df["split"], categories=["train", "val", "test"], ordered=True)
    overall_df = overall_df.sort_values("split").reset_index(drop=True)

    per_label_df = pd.concat(label_parts, ignore_index=True)
    per_label_df["split"] = pd.Categorical(per_label_df["split"], categories=["train", "val", "test"], ordered=True)
    per_label_df = per_label_df.sort_values(["split", "F1@thr", "F1@0.5"], ascending=[True, False, False]).reset_index(drop=True)
    return overall_df, per_label_df


def save_evaluation_outputs(
    *,
    run_tag: str,
    variant: str,
    loss_monitor_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    per_label_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save PyTorch evaluation outputs using TensorFlow NB04 naming."""

    monitoring_dir = PROJECT_ROOT / "monitoring_output" / "runs" / run_tag
    report_dir = PROJECT_ROOT / "report_outputs" / "runs" / run_tag
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "loss_monitor": monitoring_dir / f"loss_monitor_{variant}.csv",
        "thresholds": monitoring_dir / f"thresholds_{variant}.csv",
        "overall": report_dir / f"overall_metrics_by_split_{variant}.csv",
        "per_label": report_dir / f"per_label_metrics_by_split_{variant}.csv",
    }
    loss_monitor_df.to_csv(paths["loss_monitor"], index=False)
    thresholds_df.to_csv(paths["thresholds"], index=False)
    overall_df.to_csv(paths["overall"], index=False)
    per_label_df.to_csv(paths["per_label"], index=False)
    return paths
