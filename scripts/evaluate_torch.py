#!/usr/bin/env python3
"""Evaluate a packaged PyTorch checkpoint on train, validation, and test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_torch.data import load_split_df  # noqa: E402
from src_torch.evaluation import (  # noqa: E402
    checkpoint_binary_cols,
    evaluate_all_splits,
    evaluate_loss_monitoring,
    infer_run_tag_and_variant,
    load_torch_checkpoint_model,
    predict_split,
    save_evaluation_outputs,
    tune_validation_thresholds,
)
from src_torch.training import resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--checkpoint",
        "--model-path",
        dest="checkpoint",
        help="Checkpoint to evaluate; --model-path is kept as a compatibility alias",
    )
    selection.add_argument(
        "--run-dir",
        help="Explicit PyTorch run directory from which to select a checkpoint variant",
    )
    parser.add_argument(
        "--preferred-variant",
        choices=("best_mcmae", "best_prauc", "final"),
        default="best_mcmae",
        help="Checkpoint variant used with --run-dir (default: best_mcmae)",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data"),
        help="Root containing processed/splits and cache/images",
    )
    parser.add_argument("--split-dir", help="Override the train/val/test manifest directory")
    parser.add_argument("--image-root", help="Override the cached image directory")
    parser.add_argument("--monitoring-root", help="Override the monitoring_output/runs directory")
    parser.add_argument("--report-root", help="Override the report_outputs/runs directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _checkpoint_from_run_dir(
    parser: argparse.ArgumentParser,
    run_dir: Path,
    variant: str,
) -> Path:
    if not run_dir.is_dir():
        parser.error(f"missing run directory: {run_dir}")
    candidates = sorted(run_dir.glob(f"{variant}_*.pt"))
    if not candidates:
        parser.error(f"no {variant} checkpoint found in run directory: {run_dir}")
    if len(candidates) > 1:
        parser.error(
            f"multiple {variant} checkpoints found in {run_dir}; use --checkpoint explicitly"
        )
    return candidates[0]


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    data_root = Path(args.data_root)
    split_dir = Path(args.split_dir) if args.split_dir else data_root / "processed" / "splits"
    image_root = Path(args.image_root) if args.image_root else data_root / "cache" / "images"
    for split in ("train", "val", "test"):
        split_path = split_dir / f"{split}.csv"
        if not split_path.is_file():
            parser.error(f"missing {split} split manifest: {split_path}")
    if not image_root.is_dir():
        parser.error(f"missing image directory: {image_root}")

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else _checkpoint_from_run_dir(
            parser,
            Path(args.run_dir),
            args.preferred_variant,
        )
    )
    if not checkpoint_path.is_file():
        parser.error(f"missing checkpoint: {checkpoint_path}")
    return checkpoint_path, split_dir, image_root


def main(argv: list[str] | None = None) -> int:
    # Resolve explicit checkpoint and dataset locations.
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint_path, split_dir, image_root = _validate_args(parser, args)

    # Load the checkpoint and preserve its saved binary-label order.
    device = resolve_device(args.device)
    model, model_config, _ = load_torch_checkpoint_model(
        checkpoint_path,
        device=device,
    )
    run_tag, variant = infer_run_tag_and_variant(checkpoint_path)
    binary_cols = checkpoint_binary_cols(model_config)

    # Confirm every split is compatible with the checkpoint schema.
    for split in ("train", "val", "test"):
        split_columns = set(load_split_df(split, split_dir=split_dir).columns)
        missing = [column for column in binary_cols if column not in split_columns]
        if missing:
            parser.error(
                f"{split} split is incompatible with checkpoint binary labels: {missing}"
            )

    # Run inference once on train, validation, and test.
    predictions_by_split = {
        split: predict_split(
            model,
            split,
            device=device,
            batch_size=args.batch_size,
            split_dir=split_dir,
            image_root=image_root,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        for split in ("train", "val", "test")
    }
    # Tune binary thresholds on validation only.
    thresholds_df, threshold_map = tune_validation_thresholds(
        predictions_by_split["val"],
        binary_cols,
    )
    # Apply validation thresholds when building all evaluation tables.
    loss_monitor_df = evaluate_loss_monitoring(predictions_by_split, binary_cols)
    overall_df, per_label_df = evaluate_all_splits(
        predictions_by_split,
        binary_cols,
        threshold_map,
    )
    # Save portable thresholds, monitoring metrics, and reports.
    paths = save_evaluation_outputs(
        run_tag=run_tag,
        variant=variant,
        loss_monitor_df=loss_monitor_df,
        thresholds_df=thresholds_df,
        overall_df=overall_df,
        per_label_df=per_label_df,
        run_dir=checkpoint_path.parent,
        monitoring_root=args.monitoring_root,
        report_root=args.report_root,
    )

    print(f"Evaluation complete: {run_tag} ({variant})")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print("Artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
