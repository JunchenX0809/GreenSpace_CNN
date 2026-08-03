#!/usr/bin/env python3
"""Run a packaged PyTorch checkpoint on an unlabeled image directory."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_torch.config import resolve_prediction_output_root  # noqa: E402
from src_torch.inference import (  # noqa: E402
    build_prediction_dataframe,
    inference_output_tag,
    list_inference_image_paths,
    predict_image_paths,
    validate_prediction_dataframe,
)
from src_torch.run_bundle import load_run_bundle  # noqa: E402
from src_torch.training import resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", help="Explicit checkpoint that anchors the run bundle")
    selection.add_argument(
        "--run-dir",
        help="Explicit run directory containing one checkpoint for the preferred variant",
    )
    parser.add_argument(
        "--preferred-variant",
        choices=("best_mcmae", "best_prauc", "final"),
        default="best_mcmae",
        help="Checkpoint variant used with --run-dir (default: best_mcmae)",
    )
    parser.add_argument("--image-dir", required=True, help="Directory of unlabeled JPG/PNG images")
    parser.add_argument("--output", help="Exact output CSV path")
    parser.add_argument(
        "--output-dir",
        help="Output directory used when --output is omitted (default: predictions/)",
    )
    parser.add_argument(
        "--dataset-tag",
        help="Safe name used to keep image-folder outputs distinct (default: image directory name)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--limit", type=int, help="Predict only the first N sorted images")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV; otherwise existing files are protected",
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


def _safe_tag(parser: argparse.ArgumentParser, raw_tag: str) -> str:
    tag = raw_tag.strip()
    if not tag or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        parser.error(
            "--dataset-tag must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return tag


def _resolve_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path, str]:
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.output and args.output_dir:
        parser.error("--output and --output-dir cannot be combined")

    checkpoint = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else _checkpoint_from_run_dir(
            parser,
            Path(args.run_dir).expanduser(),
            args.preferred_variant,
        )
    )
    if not checkpoint.is_file():
        parser.error(f"missing checkpoint: {checkpoint}")

    image_dir = Path(args.image_dir).expanduser()
    if not image_dir.is_dir():
        parser.error(f"missing image directory: {image_dir}")
    if args.dataset_tag:
        dataset_tag = _safe_tag(parser, args.dataset_tag)
    else:
        dataset_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", image_dir.name).strip("._-")
        dataset_tag = dataset_tag or "images"
    return checkpoint, image_dir, dataset_tag


def _write_csv_atomic(frame, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite or choose another path."
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv.tmp",
            prefix=f".{output_path.name}.",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_output_target(
    parser: argparse.ArgumentParser,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists():
        if output_path.is_dir():
            parser.error(f"output path is a directory: {output_path}")
        if not overwrite:
            parser.error(
                f"output already exists: {output_path}. "
                "Use --overwrite or choose another path."
            )

    writable_parent = output_path.parent
    while not writable_parent.exists() and writable_parent != writable_parent.parent:
        writable_parent = writable_parent.parent
    if not writable_parent.is_dir():
        parser.error(f"no existing parent directory is available for: {output_path}")
    if not os.access(writable_parent, os.W_OK | os.X_OK):
        parser.error(f"output location is not writable: {writable_parent}")


def main(argv: list[str] | None = None) -> int:
    # Resolve one explicit run bundle and one image directory.
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint, image_dir, dataset_tag = _resolve_args(parser, args)

    # Load the checkpoint, saved label order, image size, and tuned thresholds together.
    device = resolve_device(args.device)
    bundle = load_run_bundle(checkpoint, device=device)
    image_paths = list_inference_image_paths(image_dir, limit=args.limit)

    # Resolve a collision-safe output path before expensive inference.
    if args.output:
        output_path = Path(args.output).expanduser()
        if output_path.suffix.lower() != ".csv":
            parser.error("--output must end in .csv")
    else:
        output_dir = resolve_prediction_output_root(args.output_dir)
        output_tag = inference_output_tag(
            bundle.run_tag,
            args.limit,
            dataset_tag=dataset_tag,
        )
        output_path = output_dir / f"predictions_{output_tag}.csv"
    _validate_output_target(parser, output_path, overwrite=args.overwrite)

    # Predict images in deterministic filename order.
    predictions = predict_image_paths(
        model=bundle.model,
        image_paths=image_paths,
        device=device,
        batch_size=args.batch_size,
        img_size=bundle.img_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    # Build and validate the stable public prediction schema.
    frame = build_prediction_dataframe(
        image_paths=image_paths,
        predictions=predictions,
        binary_labels=bundle.bin_names,
        thresholds=bundle.thresholds,
    )
    validate_prediction_dataframe(
        frame,
        binary_labels=bundle.bin_names,
        expected_rows=len(image_paths),
    )

    # Publish the completed CSV atomically; never replace it implicitly.
    try:
        _write_csv_atomic(frame, output_path, overwrite=args.overwrite)
    except (FileExistsError, OSError) as exc:
        parser.error(str(exc))

    print(f"Prediction complete: {bundle.run_tag} ({bundle.variant})")
    print(f"Checkpoint: {bundle.checkpoint_path.resolve()}")
    print(f"Thresholds: {bundle.threshold_path.resolve()}")
    print(f"Device: {device}")
    print(f"Images: {len(frame)}")
    print(f"Output: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
