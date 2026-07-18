"""Build per-label binary confusion matrices for GreenSpace model outputs.

GreenSpace's binary head is multi-label, so this script writes one 2x2
confusion matrix per active binary variable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
_CACHE_ROOT = Path("/private/tmp/greenspace_cnn_confusion_cache")
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix

from src_torch.config import PROJECT_ROOT as CONFIG_PROJECT_ROOT
from src_torch.config import TORCH_DATA_CONFIG
from src_torch.evaluation import (
    checkpoint_binary_cols,
    find_latest_pytorch_checkpoint,
    infer_run_tag_and_variant,
    load_torch_checkpoint_model,
    predict_split,
)

BINARY_LABELS = [0, 1]


def _label_name(binary_col: str) -> str:
    return binary_col[:-2] if binary_col.endswith("_p") else binary_col


def _load_thresholds(path: Path, label_names: list[str], mode: str) -> dict[str, float]:
    if mode == "point5":
        return {label: 0.5 for label in label_names}
    if not path.is_file():
        raise FileNotFoundError(f"Missing threshold CSV: {path}")
    table = pd.read_csv(path)
    required = {"label", "best_threshold"}
    if not required.issubset(table.columns):
        raise ValueError(f"Threshold CSV must contain {sorted(required)}: {path}")
    if table["label"].duplicated().any():
        raise ValueError(f"Threshold CSV contains duplicate labels: {path}")
    values = dict(zip(table["label"].astype(str), table["best_threshold"]))
    missing = [label for label in label_names if label not in values]
    if missing:
        raise ValueError(f"Threshold CSV is missing active labels: {missing}")
    return {label: float(values[label]) for label in label_names}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def _row_percent(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=np.float64), where=totals != 0)


def _plot_binary_matrix(row_pct: np.ndarray, counts: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell = 170
    left = 150
    top = 112
    width = left + 2 * cell + 54
    height = top + 2 * cell + 94

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((left, 30), title, fill="black", font=font)
    draw.text((left + 82, height - 44), "Predicted", fill="black", font=font)
    draw.text((22, top + 132), "True", fill="black", font=font)

    headers = ["0 absent", "1 present"]
    for idx, header in enumerate(headers):
        x = left + idx * cell + cell // 2 - 30
        draw.text((x, top - 30), header, fill="black", font=font)
        draw.text((left - 86, top + idx * cell + cell // 2 - 4), header, fill="black", font=font)

    for row in range(2):
        for col in range(2):
            value = float(row_pct[row, col] * 100.0)
            intensity = int(245 - min(value, 100.0) * 1.8)
            fill = (255, intensity + 5 if intensity <= 250 else 255, intensity)
            x0 = left + col * cell
            y0 = top + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(120, 120, 120))

            count = int(counts[row, col])
            name = {0: {0: "TN", 1: "FP"}, 1: {0: "FN", 1: "TP"}}[row][col]
            text = f"{name}\n{value:.1f}%\n(n={count})"
            text_color = "white" if value >= 55 else "black"
            bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.multiline_text(
                (x0 + (cell - text_w) / 2, y0 + (cell - text_h) / 2),
                text,
                fill=text_color,
                font=font,
                spacing=4,
                align="center",
            )

    image.save(output_path)


def _save_one_matrix(
    *,
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    source_tag: str,
    output_dir: Path,
    save_csv: bool,
) -> dict[str, Any]:
    counts = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)
    tn, fp, fn, tp = counts.ravel()
    row_pct = _row_percent(counts)

    safe_label = label.replace("/", "_")
    counts_path = output_dir / f"{safe_label}_binary_confusion_{source_tag}_counts.csv"
    row_pct_path = output_dir / f"{safe_label}_binary_confusion_{source_tag}_row_pct.csv"
    png_path = output_dir / f"{safe_label}_binary_confusion_{source_tag}.png"
    if save_csv:
        counts_df = pd.DataFrame(
            counts,
            index=pd.Index(["true_0_absent", "true_1_present"], name=label),
            columns=pd.Index(["pred_0_absent", "pred_1_present"], name="prediction"),
        )
        row_pct_df = pd.DataFrame(row_pct, index=counts_df.index, columns=counts_df.columns)
        counts_df.to_csv(counts_path)
        row_pct_df.to_csv(row_pct_path)
    _plot_binary_matrix(
        row_pct,
        counts,
        title=f"{label} binary confusion ({source_tag})",
        output_path=png_path,
    )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else np.nan
    return {
        "label": label,
        "n": int(len(y_true)),
        "support_pos": int(tp + fn),
        "support_neg": int(tn + fp),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "f1": f1,
        "counts_csv": str(counts_path) if save_csv else "",
        "row_pct_csv": str(row_pct_path) if save_csv else "",
        "png": str(png_path),
    }


def _default_threshold_path(model_path: Path, run_tag: str, variant: str) -> Path:
    bundled = model_path.parent / f"thresholds_{variant}.csv"
    if bundled.is_file():
        return bundled
    return CONFIG_PROJECT_ROOT / "monitoring_output" / "runs" / run_tag / f"thresholds_{variant}.csv"


def _load_from_checkpoint(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str], dict[str, float], str]:
    model_path = Path(args.model_path).expanduser() if args.model_path else find_latest_pytorch_checkpoint()
    run_tag, variant = infer_run_tag_and_variant(model_path)
    model, model_config, _ = load_torch_checkpoint_model(model_path)
    binary_cols = checkpoint_binary_cols(model_config)
    label_names = [_label_name(col) for col in binary_cols]
    threshold_path = Path(args.threshold_csv).expanduser() if args.threshold_csv else _default_threshold_path(model_path, run_tag, variant)
    thresholds = _load_thresholds(threshold_path, label_names, args.threshold_mode)
    device = next(model.parameters()).device
    batch_size = int(args.batch_size or TORCH_DATA_CONFIG["batch_size"])
    df, preds, _ = predict_split(model, args.split, device, batch_size=batch_size)
    source_tag = f"{args.split}_{run_tag}_{variant}_{args.threshold_mode}"
    return df, preds, binary_cols, thresholds, source_tag


def _load_from_prediction_csv(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str], dict[str, float], str]:
    prediction_path = Path(args.prediction_csv).expanduser()
    label_path = Path(args.label_csv).expanduser()
    pred_df = pd.read_csv(prediction_path)
    label_df = pd.read_csv(label_path)
    if "image_filename" not in pred_df.columns or "image_filename" not in label_df.columns:
        raise ValueError("Both prediction and label CSVs must contain image_filename.")
    if not pred_df["image_filename"].is_unique:
        raise ValueError("Prediction CSV has duplicate image_filename values.")
    if not label_df["image_filename"].is_unique:
        raise ValueError("Label CSV has duplicate image_filename values.")

    label_names = sorted(
        col[:-5]
        for col in pred_df.columns
        if col.endswith("_prob") and f"{col[:-5]}_pred" in pred_df.columns and col[:-5] in label_df.columns
    )
    if not label_names:
        raise ValueError("Could not infer binary labels from *_prob/*_pred columns and label CSV hard-label columns.")

    merged = pred_df.merge(
        label_df[["image_filename", *label_names]],
        on="image_filename",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            "No prediction rows matched the label CSV by image_filename. "
            "A true confusion matrix requires ground-truth labels for the same images."
        )

    pred_arrays = {"bin_head": merged[[f"{label}_prob" for label in label_names]].to_numpy(dtype=np.float32)}
    thresholds = {label: 0.5 for label in label_names}
    if args.threshold_mode == "tuned":
        if args.threshold_csv:
            thresholds = _load_thresholds(Path(args.threshold_csv).expanduser(), label_names, args.threshold_mode)
        else:
            # Reuse existing hard predictions in exported inference CSVs when no
            # threshold artifact is supplied.
            thresholds = {label: np.nan for label in label_names}
    source_tag = args.source_tag or f"{prediction_path.stem}_{args.threshold_mode}"
    return merged, pred_arrays, [f"{label}_p" for label in label_names], thresholds, source_tag


def build_matrices(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prediction_csv or args.label_csv:
        if not args.prediction_csv or not args.label_csv:
            raise ValueError("--prediction-csv and --label-csv must be provided together.")
        df, preds, binary_cols, thresholds, source_tag = _load_from_prediction_csv(args)
    else:
        df, preds, binary_cols, thresholds, source_tag = _load_from_checkpoint(args)

    rows = []
    probs = np.asarray(preds["bin_head"], dtype=np.float32)
    if probs.shape[1] != len(binary_cols):
        raise ValueError(f"Binary prediction width {probs.shape[1]} does not match active label count {len(binary_cols)}.")

    for idx, binary_col in enumerate(binary_cols):
        label = _label_name(binary_col)
        true_col = label if label in df.columns else binary_col
        if true_col not in df.columns:
            raise ValueError(f"Missing truth column for {label!r}; expected {label!r} or {binary_col!r}.")
        y_true = df[true_col].fillna(0).astype(float).round().clip(0, 1).astype(int).to_numpy()
        if args.prediction_csv and args.threshold_mode == "tuned" and np.isnan(thresholds[label]):
            pred_col = f"{label}_pred"
            if pred_col not in df.columns:
                raise ValueError(f"Missing exported prediction column: {pred_col}")
            y_pred = df[pred_col].fillna(0).astype(int).clip(0, 1).to_numpy()
            threshold = np.nan
        else:
            threshold = thresholds[label]
            y_pred = (probs[:, idx] >= threshold).astype(int)
        rows.append(
            _save_one_matrix(
                label=label,
                y_true=y_true,
                y_pred=y_pred,
                threshold=float(threshold) if np.isfinite(threshold) else np.nan,
                source_tag=source_tag,
                output_dir=output_dir,
                save_csv=bool(args.save_csv),
            )
        )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"binary_confusion_summary_{source_tag}.csv"
    if args.save_csv:
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
    print(summary[["label", "support_pos", "support_neg", "threshold", "tn", "fp", "fn", "tp", "precision", "recall", "specificity", "f1"]].to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="PyTorch checkpoint path. Defaults to latest best_mcmae.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Labeled split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override checkpoint prediction batch size.")
    parser.add_argument("--threshold-csv", default=None, help="Threshold artifact. Defaults to the checkpoint run's thresholds CSV.")
    parser.add_argument("--threshold-mode", default="tuned", choices=["tuned", "point5"], help="Use validation-tuned thresholds or fixed 0.5.")
    parser.add_argument("--prediction-csv", default=None, help="Existing prediction CSV with *_prob and *_pred columns.")
    parser.add_argument("--label-csv", default=None, help="Ground-truth CSV with image_filename and hard binary label columns.")
    parser.add_argument("--source-tag", default=None, help="Output filename tag for prediction+label CSV mode.")
    parser.add_argument("--save-csv", action="store_true", help="Also save counts, row percentages, and summary CSVs.")
    parser.add_argument(
        "--output-dir",
        default=str(CONFIG_PROJECT_ROOT / "presentation_visuals_only" / "eval_binary"),
        help="Directory for confusion matrix PNG outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_matrices(parse_args())
