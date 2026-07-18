"""Build score/vegetation confusion matrices for GreenSpace model outputs.

Two workflows are supported:

1. Labeled split evaluation from a PyTorch checkpoint, e.g. the held-out test
   split used by NB04.
2. Existing prediction CSV plus an external label CSV. This is intended for
   inference-cache images after human labels are available.
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

from src_torch.config import TORCH_DATA_CONFIG
from src_torch.evaluation import (
    find_latest_pytorch_checkpoint,
    infer_run_tag_and_variant,
    load_torch_checkpoint_model,
    predict_split,
)

LABELS = [1, 2, 3, 4, 5]


def _rounded_class(values: Any) -> np.ndarray:
    """Convert regression-style EV predictions to hard 1-5 ordinal classes."""

    return np.rint(np.clip(np.asarray(values, dtype=np.float32).reshape(-1), 1, 5)).astype(np.int64)


def _row_percent(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals != 0,
    )


def _plot_matrix(row_pct: np.ndarray, counts: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell = 118
    left = 126
    top = 96
    right_pad = 36
    bottom_pad = 86
    width = left + cell * len(LABELS) + right_pad
    height = top + cell * len(LABELS) + bottom_pad

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((left, 28), title, fill="black", font=font)
    draw.text((left + 170, height - 42), "Predicted class", fill="black", font=font)
    draw.text((18, top + 220), "True class", fill="black", font=font)

    for idx, label in enumerate(LABELS):
        x = left + idx * cell + cell // 2
        y = top - 28
        draw.text((x - 4, y), str(label), fill="black", font=font)
        draw.text((left - 32, top + idx * cell + cell // 2 - 4), str(label), fill="black", font=font)

    for row in range(row_pct.shape[0]):
        for col in range(row_pct.shape[1]):
            value = float(row_pct[row, col] * 100.0)
            intensity = int(245 - min(value, 100.0) * 1.8)
            fill = (intensity, intensity + 5 if intensity <= 250 else 255, 255)
            x0 = left + col * cell
            y0 = top + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(120, 120, 120))

            count = int(counts[row, col])
            text = f"{value:.1f}%\n(n={count})"
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    head: str,
    source_tag: str,
    output_dir: Path,
    save_csv: bool,
) -> dict[str, Any]:
    counts = confusion_matrix(y_true, y_pred, labels=LABELS)
    row_pct = _row_percent(counts)
    accuracy = float(np.mean(y_true == y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    png_path = output_dir / f"{head}_confusion_{source_tag}.png"
    counts_path = output_dir / f"{head}_confusion_{source_tag}_counts.csv"
    row_pct_path = output_dir / f"{head}_confusion_{source_tag}_row_pct.csv"
    if save_csv:
        index = pd.Index(LABELS, name=f"true_{head}_class")
        columns = pd.Index(LABELS, name=f"pred_{head}_class")
        counts_df = pd.DataFrame(counts, index=index, columns=columns)
        row_pct_df = pd.DataFrame(row_pct, index=index, columns=columns)
        counts_df.to_csv(counts_path)
        row_pct_df.to_csv(row_pct_path)
    _plot_matrix(
        row_pct,
        counts,
        title=f"{head.title()} confusion matrix ({source_tag})",
        output_path=png_path,
    )
    return {
        "head": head,
        "n": int(len(y_true)),
        "accuracy": accuracy,
        "mae_class": mae,
        "counts_csv": str(counts_path) if save_csv else "",
        "row_pct_csv": str(row_pct_path) if save_csv else "",
        "png": str(png_path),
    }


def _load_from_checkpoint(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray], str]:
    model_path = Path(args.model_path).expanduser() if args.model_path else find_latest_pytorch_checkpoint()
    run_tag, variant = infer_run_tag_and_variant(model_path)
    model, model_config, _ = load_torch_checkpoint_model(model_path)
    device = next(model.parameters()).device
    batch_size = int(args.batch_size or TORCH_DATA_CONFIG["batch_size"])
    df, preds, _ = predict_split(model, args.split, device, batch_size=batch_size)
    source_tag = f"{args.split}_{run_tag}_{variant}"
    return df, preds, source_tag


def _load_from_prediction_csv(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray], str]:
    prediction_path = Path(args.prediction_csv).expanduser()
    label_path = Path(args.label_csv).expanduser()
    pred_df = pd.read_csv(prediction_path)
    label_df = pd.read_csv(label_path)
    required_pred = {"image_filename", "score_ev", "veg_ev"}
    required_labels = {"image_filename", "score_class", "veg_class"}
    missing_pred = required_pred - set(pred_df.columns)
    missing_labels = required_labels - set(label_df.columns)
    if missing_pred:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing_pred)}")
    if missing_labels:
        raise ValueError(f"Label CSV is missing columns: {sorted(missing_labels)}")

    if not pred_df["image_filename"].is_unique:
        raise ValueError("Prediction CSV has duplicate image_filename values.")
    if not label_df["image_filename"].is_unique:
        raise ValueError("Label CSV has duplicate image_filename values.")

    merged = pred_df.merge(
        label_df[["image_filename", "score_class", "veg_class"]],
        on="image_filename",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            "No prediction rows matched the label CSV by image_filename. "
            "A true confusion matrix requires ground-truth labels for the same inference-cache images."
        )

    preds = {
        "score_head": merged["score_ev"].to_numpy(dtype=np.float32),
        "veg_head": merged["veg_ev"].to_numpy(dtype=np.float32),
    }
    source_tag = args.source_tag or prediction_path.stem
    return merged, preds, source_tag


def build_matrices(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prediction_csv or args.label_csv:
        if not args.prediction_csv or not args.label_csv:
            raise ValueError("--prediction-csv and --label-csv must be provided together.")
        df, preds, source_tag = _load_from_prediction_csv(args)
    else:
        df, preds, source_tag = _load_from_checkpoint(args)

    rows = []
    for head, true_col, pred_key in (
        ("score", "score_class", "score_head"),
        ("veg", "veg_class", "veg_head"),
    ):
        y_true = df[true_col].fillna(1).astype(int).clip(1, 5).to_numpy()
        y_pred = _rounded_class(preds[pred_key])
        rows.append(
            _save_one_matrix(
                y_true=y_true,
                y_pred=y_pred,
                head=head,
                source_tag=source_tag,
                output_dir=output_dir,
                save_csv=bool(args.save_csv),
            )
        )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"confusion_summary_{source_tag}.csv"
    if args.save_csv:
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
    print(summary.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="PyTorch checkpoint path. Defaults to latest best_mcmae.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Labeled split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override checkpoint prediction batch size.")
    parser.add_argument("--prediction-csv", default=None, help="Existing inference prediction CSV with score_ev/veg_ev.")
    parser.add_argument("--label-csv", default=None, help="Ground-truth CSV with image_filename, score_class, veg_class.")
    parser.add_argument("--source-tag", default=None, help="Output filename tag for prediction+label CSV mode.")
    parser.add_argument("--save-csv", action="store_true", help="Also save counts, row percentages, and summary CSVs.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "presentation_visuals_only" / "eval_ordinal"),
        help="Directory for confusion matrix PNG outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_matrices(parse_args())
