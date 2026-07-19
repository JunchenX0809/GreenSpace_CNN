#!/usr/bin/env python3
"""Evaluate a trained PyTorch checkpoint and save the threshold + report bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_torch.data import load_split_df, resolve_split_schema  # noqa: E402
from src_torch.evaluation import (  # noqa: E402
    evaluate_all_splits,
    evaluate_loss_monitoring,
    find_latest_pytorch_checkpoint,
    infer_run_tag_and_variant,
    load_torch_checkpoint_model,
    predict_split,
    save_evaluation_outputs,
    tune_val_thresholds,
)
from src_torch.training import resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="Path to a specific <variant>_<run-tag>.pt")
    source.add_argument(
        "--preferred-variant",
        choices=("best_mcmae", "best_prauc", "final"),
        default="best_mcmae",
        help="Auto-select this variant from the newest run when --checkpoint is omitted",
    )
    parser.add_argument(
        "--runs-root",
        default=str(PROJECT_ROOT / "models" / "runs"),
        help="Root searched for PyTorch_* runs when --checkpoint is omitted",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int, help="Debug: cap batches per split")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1")

    if args.checkpoint:
        model_path = Path(args.checkpoint)
        if not model_path.is_file():
            parser.error(f"missing checkpoint: {model_path}")
    else:
        try:
            model_path = find_latest_pytorch_checkpoint(
                runs_root=Path(args.runs_root),
                preferred_variant=args.preferred_variant,
            )
        except FileNotFoundError as exc:
            parser.error(str(exc))

    run_tag, variant = infer_run_tag_and_variant(model_path)
    device = resolve_device(args.device)
    model, _model_config, _checkpoint = load_torch_checkpoint_model(model_path, device=device)

    schema = resolve_split_schema(load_split_df("train"))
    binary_cols = schema.binary_cols
    bin_names = schema.bin_names

    predict_kwargs: dict[str, object] = {"device": device, "max_batches": args.max_batches}
    if args.batch_size is not None:
        predict_kwargs["batch_size"] = args.batch_size

    predictions_by_split = {
        split: predict_split(model, split, **predict_kwargs)
        for split in ("train", "val", "test")
    }

    loss_monitor_df = evaluate_loss_monitoring(predictions_by_split, binary_cols)
    thresholds_df, best_thresholds = tune_val_thresholds(
        predictions_by_split["val"], bin_names, binary_cols
    )
    overall_df, per_label_df = evaluate_all_splits(
        predictions_by_split, binary_cols, best_thresholds
    )
    saved_paths = save_evaluation_outputs(
        run_tag=run_tag,
        variant=variant,
        loss_monitor_df=loss_monitor_df,
        thresholds_df=thresholds_df,
        overall_df=overall_df,
        per_label_df=per_label_df,
    )

    print(f"Evaluated checkpoint: {model_path}")
    print(f"run_tag: {run_tag}  variant: {variant}  device: {device}")
    print(f"Tuned thresholds: {len(best_thresholds)} / {len(bin_names)}")
    print("Saved artifacts:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
