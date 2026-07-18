#!/usr/bin/env python3
"""Run or resume the established PyTorch warm-up/fine-tuning pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_torch.training import run_persistent_warmup_finetune  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        required=True,
        help="smoke uses the existing 1+1 schedule; full uses the production schedule",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data"),
        help="Root containing processed/splits and cache/images",
    )
    parser.add_argument("--split-dir", help="Override the split manifest directory")
    parser.add_argument("--image-root", help="Override the cached image directory")
    parser.add_argument(
        "--runs-root",
        default=str(PROJECT_ROOT / "models" / "runs"),
    )
    parser.add_argument("--run-tag", help="Optional new-run folder name")
    parser.add_argument("--resume", help="Path to last_<run-tag>.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.run_tag:
        parser.error("--run-tag cannot be combined with --resume")
    for name in ("batch_size", "warmup_epochs", "finetune_epochs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    data_root = Path(args.data_root)
    split_dir = Path(args.split_dir) if args.split_dir else data_root / "processed" / "splits"
    image_root = Path(args.image_root) if args.image_root else data_root / "cache" / "images"
    for split in ("train", "val"):
        path = split_dir / f"{split}.csv"
        if not path.is_file():
            parser.error(f"missing {split} split manifest: {path}")
    if not image_root.is_dir():
        parser.error(f"missing image directory: {image_root}")
    if args.resume and not Path(args.resume).is_file():
        parser.error(f"missing resume checkpoint: {args.resume}")

    training_config: dict[str, object] = {
        "test_run_mode": args.mode == "smoke",
        "device": args.device,
    }
    if args.warmup_epochs is not None:
        key = "test_warmup_epochs" if args.mode == "smoke" else "warmup_epochs"
        training_config[key] = args.warmup_epochs
    if args.finetune_epochs is not None:
        key = "test_finetune_epochs" if args.mode == "smoke" else "finetune_epochs"
        training_config[key] = args.finetune_epochs

    data_config: dict[str, object] = {}
    if args.batch_size is not None:
        data_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_config["num_workers"] = args.num_workers
    if args.pin_memory is not None:
        data_config["pin_memory"] = args.pin_memory

    try:
        result = run_persistent_warmup_finetune(
            run_tag=args.run_tag,
            training_config=training_config,
            data_config=data_config,
            split_dir=split_dir,
            image_root=image_root,
            runs_root=args.runs_root,
            resume_from=args.resume,
        )
    except KeyboardInterrupt:
        print(
            "Training interrupted. Resume from the run's last_<run-tag>.pt; "
            "the active incomplete epoch will be repeated.",
            file=sys.stderr,
        )
        return 130

    print("Training artifacts:")
    for key in (
        "run_dir",
        "last_checkpoint_path",
        "best_mc_mae_path",
        "best_prauc_path",
        "final_model_path",
        "model_config_path",
        "training_history_path",
    ):
        if result.get(key):
            print(f"  {key}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
