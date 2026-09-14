#!/usr/bin/env python3
"""Run a saved GreenSpace model over the workstation's state/park image tree."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STATE_DIRECTORY_PATTERN = re.compile(r"^USA_([A-Za-z]{2})(?:_|$)")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RUN_SCHEMA_VERSION = 1
INVENTORY_COLUMNS = (
    "state_code",
    "park_code",
    "image_filename",
    "image_relative_path",
    "part_number",
)
FAILURE_COLUMNS = (
    "state_code",
    "park_code",
    "image_filename",
    "image_relative_path",
    "part_number",
    "error_type",
    "error_message",
)
TREE_METADATA_COLUMNS = ("state_code", "park_code", "image_relative_path")


@dataclass(frozen=True)
class ImageRecord:
    """One source image and its stable workstation-relative identity."""

    state_code: str
    park_code: str
    path: Path
    relative_path: str


@dataclass(frozen=True)
class StateInventory:
    """Deterministic image inventory for one state."""

    state_code: str
    source_directory: Path
    records: tuple[ImageRecord, ...]
    images_discovered: int
    jpg_directories: int
    ignored_files: int
    duplicate_basename_groups: int
    layout_warnings: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _safe_run_id(parser: argparse.ArgumentParser, raw_value: str) -> str:
    value = raw_value.strip()
    if not RUN_ID_PATTERN.fullmatch(value):
        parser.error(
            "--run-id must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return value


def _parse_states(parser: argparse.ArgumentParser, raw_value: str | None) -> tuple[str, ...] | None:
    if raw_value is None:
        return None
    values = [value.strip().upper() for value in raw_value.split(",") if value.strip()]
    if not values or any(not re.fullmatch(r"[A-Z]{2}", value) for value in values):
        parser.error("--states must be one or more comma-separated two-letter codes")
    if len(values) != len(set(values)):
        parser.error("--states contains a duplicate state code")
    return tuple(sorted(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    model_selection = parser.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--checkpoint",
        help="Explicit checkpoint that anchors the three-file run bundle",
    )
    model_selection.add_argument(
        "--run-dir",
        help="Directory containing one checkpoint for the preferred variant",
    )
    parser.add_argument(
        "--preferred-variant",
        choices=("best_mcmae", "best_prauc", "final"),
        default="best_mcmae",
    )

    parser.add_argument("--image-root", required=True, help="Root containing USA_XX state folders")
    parser.add_argument("--output-dir", required=True, help="Writable root for run outputs")
    parser.add_argument("--run-id", required=True, help="Stable name for this inventory or prediction run")

    input_selection = parser.add_mutually_exclusive_group()
    input_selection.add_argument(
        "--states",
        help="Optional comma-separated state codes; omission selects every discovered state",
    )
    input_selection.add_argument(
        "--image-path",
        action="append",
        help="Exact JPG path for a targeted smoke test; may be repeated",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Use only the first N sorted images; requires exactly one --states code",
    )
    parser.add_argument("--images-per-part", type=int, default=1000)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _checkpoint_from_run_dir(run_dir: Path, variant: str) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    candidates = sorted(run_dir.glob(f"{variant}_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No {variant} checkpoint found in: {run_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple {variant} checkpoints found in {run_dir}; use --checkpoint"
        )
    return candidates[0]


def _resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.inventory_only:
        return None
    if not args.checkpoint and not args.run_dir:
        raise ValueError("Prediction requires --checkpoint or --run-dir")
    checkpoint = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else _checkpoint_from_run_dir(
            Path(args.run_dir).expanduser(),
            args.preferred_variant,
        )
    )
    checkpoint = checkpoint.resolve(strict=False)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return checkpoint


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[tuple[str, ...] | None, str]:
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.images_per_part < 1:
        parser.error("--images-per-part must be at least 1")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")
    states = _parse_states(parser, args.states)
    if args.max_images is not None and (states is None or len(states) != 1):
        parser.error("--max-images requires exactly one state in --states")
    if args.resume and args.inventory_only:
        parser.error("--resume cannot be combined with --inventory-only")
    if args.resume and args.image_path:
        parser.error("--resume is not needed for an exact-image smoke test")
    if not args.inventory_only and not (args.checkpoint or args.run_dir):
        parser.error("prediction requires --checkpoint or --run-dir")
    return states, _safe_run_id(parser, args.run_id)


def discover_state_directories(
    image_root: Path,
    selected_states: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Return one top-level USA_XX directory per selected state."""

    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    selected = set(selected_states) if selected_states is not None else None
    candidates: dict[str, list[Path]] = {}
    for child in sorted(image_root.iterdir(), key=lambda path: path.name.casefold()):
        if not child.is_dir():
            continue
        match = STATE_DIRECTORY_PATTERN.match(child.name)
        if match is None:
            continue
        code = match.group(1).upper()
        if selected is not None and code not in selected:
            continue
        candidates.setdefault(code, []).append(child.resolve(strict=False))

    if selected is not None:
        missing = sorted(selected.difference(candidates))
        if missing:
            raise FileNotFoundError(f"State folders not found: {', '.join(missing)}")
    if not candidates:
        raise FileNotFoundError(f"No USA_XX state folders found in: {image_root}")

    duplicates = {code: paths for code, paths in candidates.items() if len(paths) > 1}
    if duplicates:
        detail = "; ".join(
            f"{code}: {', '.join(path.name for path in paths)}"
            for code, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Multiple top-level folders resolve to one state code: {detail}")
    return {code: paths[0] for code, paths in sorted(candidates.items())}


def discover_state_inventory(
    image_root: Path,
    state_code: str,
    state_directory: Path,
    max_images: int | None = None,
) -> StateInventory:
    """Discover direct JPG children of every nested directory named jpg."""

    image_root = image_root.resolve(strict=False)
    state_directory = state_directory.resolve(strict=False)
    records: list[ImageRecord] = []
    ignored_files = 0
    jpg_directories = 0
    layout_warnings: list[str] = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current, directories, filenames in os.walk(
        state_directory,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        directories.sort(key=str.casefold)
        filenames.sort(key=str.casefold)
        current_path = Path(current)
        if current_path.name.casefold() != "jpg":
            continue

        jpg_directories += 1
        directories.clear()
        relative_jpg = current_path.relative_to(state_directory)
        if len(relative_jpg.parts) != 3:
            layout_warnings.append(relative_jpg.as_posix())
        park_code = current_path.parent.name
        for filename in filenames:
            path = current_path / filename
            if path.suffix.casefold() != ".jpg":
                ignored_files += 1
                continue
            relative_path = path.relative_to(image_root).as_posix()
            records.append(
                ImageRecord(
                    state_code=state_code,
                    park_code=park_code,
                    path=path.resolve(strict=False),
                    relative_path=relative_path,
                )
            )

    records.sort(key=lambda item: (item.relative_path.casefold(), item.relative_path))
    normalized_paths = [record.relative_path.casefold() for record in records]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError(f"Duplicate relative image paths found for state {state_code}")

    duplicate_names = Counter(record.path.name.casefold() for record in records)
    duplicate_basename_groups = sum(count > 1 for count in duplicate_names.values())
    images_discovered = len(records)
    if max_images is not None:
        records = records[:max_images]

    return StateInventory(
        state_code=state_code,
        source_directory=state_directory,
        records=tuple(records),
        images_discovered=images_discovered,
        jpg_directories=jpg_directories,
        ignored_files=ignored_files,
        duplicate_basename_groups=duplicate_basename_groups,
        layout_warnings=tuple(layout_warnings),
    )


def _record_for_exact_path(image_root: Path, raw_path: str) -> ImageRecord:
    image_root = image_root.resolve(strict=False)
    path = Path(raw_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"Exact smoke-test image does not exist: {path}")
    if path.suffix.casefold() != ".jpg":
        raise ValueError(f"Exact smoke-test image must end in .jpg: {path}")
    if not _is_within(path, image_root):
        raise ValueError(f"Exact smoke-test image is outside --image-root: {path}")
    relative = path.relative_to(image_root)
    if len(relative.parts) < 4:
        raise ValueError(f"Exact smoke-test image does not match the state tree: {path}")
    state_match = STATE_DIRECTORY_PATTERN.match(relative.parts[0])
    if state_match is None:
        raise ValueError(f"Exact image is not under a USA_XX state folder: {path}")
    if path.parent.name.casefold() != "jpg":
        raise ValueError(f"Exact image is not directly inside a jpg directory: {path}")
    return ImageRecord(
        state_code=state_match.group(1).upper(),
        park_code=path.parent.parent.name,
        path=path,
        relative_path=relative.as_posix(),
    )


def discover_exact_inventories(
    image_root: Path,
    raw_paths: Sequence[str],
) -> dict[str, StateInventory]:
    """Build small state inventories from explicitly selected JPG paths."""

    records = [_record_for_exact_path(image_root, raw_path) for raw_path in raw_paths]
    path_keys = [record.relative_path.casefold() for record in records]
    if len(path_keys) != len(set(path_keys)):
        raise ValueError("--image-path contains a duplicate image")
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.state_code, []).append(record)
    inventories: dict[str, StateInventory] = {}
    for state_code, state_records in sorted(grouped.items()):
        ordered = sorted(
            state_records,
            key=lambda item: (item.relative_path.casefold(), item.relative_path),
        )
        inventories[state_code] = StateInventory(
            state_code=state_code,
            source_directory=image_root / Path(ordered[0].relative_path).parts[0],
            records=tuple(ordered),
            images_discovered=len(ordered),
            jpg_directories=len({record.path.parent for record in ordered}),
            ignored_files=0,
            duplicate_basename_groups=0,
            layout_warnings=(),
        )
    return inventories


def _partition_records(
    records: Sequence[ImageRecord],
    images_per_part: int,
) -> list[tuple[ImageRecord, ...]]:
    return [
        tuple(records[index : index + images_per_part])
        for index in range(0, len(records), images_per_part)
    ]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv_rows_atomic(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv.tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_dataframe_atomic(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv.tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _inventory_path(run_root: Path, state_code: str) -> Path:
    return run_root / "inventory" / f"USA_{state_code}_inventory.csv"


def _state_root(run_root: Path, state_code: str) -> Path:
    return run_root / "states" / f"USA_{state_code}"


def _part_paths(run_root: Path, state_code: str, part_number: int) -> tuple[Path, Path]:
    root = _state_root(run_root, state_code)
    suffix = f"USA_{state_code}_part_{part_number:05d}.csv"
    return root / f"predictions_{suffix}", root / f"failures_{suffix}"


def _write_inventory(
    run_root: Path,
    inventory: StateInventory,
    images_per_part: int,
) -> None:
    rows = []
    for index, record in enumerate(inventory.records):
        rows.append(
            {
                "state_code": record.state_code,
                "park_code": record.park_code,
                "image_filename": record.path.name,
                "image_relative_path": record.relative_path,
                "part_number": index // images_per_part + 1,
            }
        )
    _write_csv_rows_atomic(_inventory_path(run_root, inventory.state_code), INVENTORY_COLUMNS, rows)


def _load_inventory(
    image_root: Path,
    inventory_path: Path,
    state_code: str,
) -> StateInventory:
    with open(inventory_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INVENTORY_COLUMNS:
            raise ValueError(f"Inventory schema mismatch: {inventory_path}")
        rows = list(reader)
    records = []
    for row in rows:
        if row["state_code"] != state_code:
            raise ValueError(f"Inventory state mismatch: {inventory_path}")
        relative = row["image_relative_path"]
        records.append(
            ImageRecord(
                state_code=state_code,
                park_code=row["park_code"],
                path=image_root.joinpath(*relative.split("/")),
                relative_path=relative,
            )
        )
    source_directory = (
        image_root / Path(records[0].relative_path).parts[0]
        if records
        else image_root
    )
    return StateInventory(
        state_code=state_code,
        source_directory=source_directory,
        records=tuple(records),
        images_discovered=len(records),
        jpg_directories=0,
        ignored_files=0,
        duplicate_basename_groups=0,
        layout_warnings=(),
    )


def _configure_logging(run_root: Path, run_id: str) -> tuple[logging.Logger, Path]:
    logs_root = run_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"greenspace.state_tree.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s run_id={run_id} %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        logs_root / "inference.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, logs_root / "run_events.jsonl"


def _append_event(event_path: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp_utc": _utc_now(), "event": event, **fields}
    with open(event_path, "a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()


def _initial_state_summary(
    inventory: StateInventory,
    images_per_part: int,
    status: str = "pending",
) -> dict[str, Any]:
    total_parts = math.ceil(len(inventory.records) / images_per_part) if inventory.records else 0
    return {
        "state_code": inventory.state_code,
        "source_directory": str(inventory.source_directory),
        "images_discovered": inventory.images_discovered,
        "images_selected": len(inventory.records),
        "jpg_directories": inventory.jpg_directories,
        "ignored_files": inventory.ignored_files,
        "duplicate_basename_groups": inventory.duplicate_basename_groups,
        "layout_warnings": list(inventory.layout_warnings),
        "images_per_part": images_per_part,
        "parts_total": total_parts,
        "parts_completed": 0,
        "images_attempted": 0,
        "images_succeeded": 0,
        "images_failed": 0,
        "status": status,
        "last_update_utc": _utc_now(),
    }


def _aggregate_run_summary(
    config: dict[str, Any],
    state_summaries: dict[str, dict[str, Any]],
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    summaries = list(state_summaries.values())
    result = {
        "run_id": config["run_id"],
        "status": status,
        "created_at_utc": config["created_at_utc"],
        "last_update_utc": _utc_now(),
        "states_selected": len(summaries),
        "states_completed": sum(item["status"] == "complete" for item in summaries),
        "states_failed": sum(item["status"] == "failed" for item in summaries),
        "parts_total": sum(int(item["parts_total"]) for item in summaries),
        "parts_completed": sum(int(item["parts_completed"]) for item in summaries),
        "images_discovered": sum(int(item["images_discovered"]) for item in summaries),
        "images_selected": sum(int(item["images_selected"]) for item in summaries),
        "images_attempted": sum(int(item["images_attempted"]) for item in summaries),
        "images_succeeded": sum(int(item["images_succeeded"]) for item in summaries),
        "images_failed": sum(int(item["images_failed"]) for item in summaries),
        "state_codes": sorted(state_summaries),
        "checkpoint": config.get("checkpoint"),
        "requested_device": config.get("device"),
        **extra,
    }
    return result


def _validate_output_location(image_root: Path, output_root: Path) -> None:
    if output_root.resolve(strict=False) == image_root.resolve(strict=False):
        raise ValueError("--output-dir cannot be the same directory as --image-root")
    if not _is_within(output_root, image_root):
        return
    relative = output_root.resolve(strict=False).relative_to(image_root.resolve(strict=False))
    if relative.parts and STATE_DIRECTORY_PATTERN.match(relative.parts[0]):
        raise ValueError("--output-dir cannot be inside a USA_XX state directory")


def _new_run_root(output_root: Path, run_id: str, resume: bool) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / run_id
    if resume:
        if not run_root.is_dir():
            raise FileNotFoundError(f"Run to resume does not exist: {run_root}")
    else:
        if run_root.exists():
            raise FileExistsError(
                f"Run output already exists: {run_root}. Choose a new --run-id or use --resume."
            )
        run_root.mkdir()
    return run_root


def _run_config(
    args: argparse.Namespace,
    run_id: str,
    image_root: Path,
    output_root: Path,
    checkpoint: Path | None,
    requested_states: Sequence[str] | None,
    selected_states: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "selection_mode": "exact_images" if args.image_path else "states",
        "requested_states": list(requested_states) if requested_states is not None else None,
        "requested_image_paths": list(args.image_path or []),
        "selected_states": list(selected_states),
        "max_images": args.max_images,
        "images_per_part": args.images_per_part,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }


def _resume_identity(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "run_id",
        "image_root",
        "output_root",
        "checkpoint",
        "selection_mode",
        "requested_states",
        "requested_image_paths",
        "max_images",
        "images_per_part",
        "device",
        "batch_size",
        "num_workers",
        "pin_memory",
    )
    return {key: config.get(key) for key in keys}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_state_summary(run_root: Path, summary: dict[str, Any]) -> None:
    _write_json_atomic(
        _state_root(run_root, summary["state_code"]) / "state_summary.json",
        summary,
    )


def _load_or_create_inventories(
    args: argparse.Namespace,
    image_root: Path,
    run_root: Path,
    requested_states: Sequence[str] | None,
    logger: logging.Logger,
) -> dict[str, StateInventory]:
    if args.resume:
        config = _read_json(run_root / "run_config.json")
        inventories = {}
        for state_code in config.get("selected_states", []):
            path = _inventory_path(run_root, state_code)
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved inventory: {path}")
            inventories[state_code] = _load_inventory(image_root, path, state_code)
        if not inventories:
            raise ValueError("Saved run contains no state inventories")
        return inventories

    if args.image_path:
        inventories = discover_exact_inventories(image_root, args.image_path)
    else:
        state_directories = discover_state_directories(image_root, requested_states)
        inventories = {}
        for state_code, state_directory in state_directories.items():
            logger.info("Inventorying USA_%s under %s", state_code, state_directory)
            inventories[state_code] = discover_state_inventory(
                image_root,
                state_code,
                state_directory,
                max_images=args.max_images,
            )

    for inventory in inventories.values():
        _write_inventory(run_root, inventory, args.images_per_part)
    return inventories


def _empty_prediction_frame(binary_labels: Sequence[str]) -> Any:
    import pandas as pd

    from src_torch.inference import expected_prediction_columns

    return pd.DataFrame(
        columns=[*expected_prediction_columns(binary_labels), *TREE_METADATA_COLUMNS]
    )


def _build_tree_prediction_frame(
    successful_paths: Sequence[Path],
    predictions: dict[str, Any] | None,
    records_by_path: dict[str, ImageRecord],
    binary_labels: Sequence[str],
    thresholds: dict[str, float],
) -> Any:
    import pandas as pd

    from src_torch.inference import (
        build_prediction_values_dataframe,
        expected_prediction_value_columns,
        validate_prediction_values_dataframe,
    )

    if predictions is None:
        return _empty_prediction_frame(binary_labels)
    records = [records_by_path[_path_key(path)] for path in successful_paths]
    values = build_prediction_values_dataframe(
        predictions,
        binary_labels=binary_labels,
        thresholds=thresholds,
    )
    validate_prediction_values_dataframe(
        values[expected_prediction_value_columns(binary_labels)],
        binary_labels=binary_labels,
        expected_rows=len(records),
    )
    frame = pd.DataFrame({"image_filename": [record.path.name for record in records]})
    frame = pd.concat([frame, values.reset_index(drop=True)], axis=1)
    frame["state_code"] = [record.state_code for record in records]
    frame["park_code"] = [record.park_code for record in records]
    frame["image_relative_path"] = [record.relative_path for record in records]
    if not frame["image_relative_path"].is_unique:
        raise ValueError("Prediction output contains duplicate relative image paths")
    return frame


def _sanitize_error_message(message: str, record: ImageRecord) -> str:
    return message.replace(str(record.path), record.relative_path)


def _validate_completed_part(
    prediction_path: Path,
    failure_path: Path,
    records: Sequence[ImageRecord],
    binary_labels: Sequence[str],
) -> tuple[bool, int, int]:
    if not prediction_path.is_file() or not failure_path.is_file():
        return False, 0, 0

    import pandas as pd

    from src_torch.inference import (
        expected_prediction_columns,
        expected_prediction_value_columns,
        validate_prediction_values_dataframe,
    )

    try:
        prediction_frame = pd.read_csv(prediction_path)
        failure_frame = pd.read_csv(failure_path)
    except Exception:
        return False, 0, 0
    expected_prediction_columns_with_metadata = [
        *expected_prediction_columns(binary_labels),
        *TREE_METADATA_COLUMNS,
    ]
    if prediction_frame.columns.tolist() != expected_prediction_columns_with_metadata:
        return False, 0, 0
    if failure_frame.columns.tolist() != list(FAILURE_COLUMNS):
        return False, 0, 0
    if not prediction_frame.empty:
        try:
            validate_prediction_values_dataframe(
                prediction_frame[expected_prediction_value_columns(binary_labels)],
                binary_labels=binary_labels,
                expected_rows=len(prediction_frame),
            )
        except ValueError:
            return False, 0, 0

    output_paths = [
        *prediction_frame["image_relative_path"].astype(str).tolist(),
        *failure_frame["image_relative_path"].astype(str).tolist(),
    ]
    expected_paths = [record.relative_path for record in records]
    if len(output_paths) != len(set(path.casefold() for path in output_paths)):
        return False, 0, 0
    if {path.casefold() for path in output_paths} != {
        path.casefold() for path in expected_paths
    }:
        return False, 0, 0
    return True, len(prediction_frame), len(failure_frame)


def _validate_workstation_model(bundle: Any) -> None:
    model_config = bundle.model_config.get("torch_model_config", {})
    model_name = model_config.get("torchgeo_model_name")
    weight_name = model_config.get("torchgeo_weight")
    if model_name != "swin_v2_b":
        raise ValueError(
            f"Workstation inference requires the saved Swin V2 B model; found {model_name!r}"
        )
    if weight_name != "Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS":
        raise ValueError(
            "Workstation inference requires the saved Satlas Swin V2 B weight contract; "
            f"found {weight_name!r}"
        )


def _resolve_prediction_runtime(checkpoint: Path, requested_device: str) -> tuple[Any, Any]:
    import torch

    from src_torch.run_bundle import load_run_bundle
    from src_torch.training import resolve_device

    if requested_device.split(":", maxsplit=1)[0].lower() == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports that CUDA is unavailable")
    device = resolve_device(requested_device)
    bundle = load_run_bundle(checkpoint, device=device)
    _validate_workstation_model(bundle)
    return device, bundle


def _predict_part(
    records: Sequence[ImageRecord],
    bundle: Any,
    device: Any,
    args: argparse.Namespace,
    logger: logging.Logger,
    part_number: int,
) -> tuple[Any, list[dict[str, Any]]]:
    from src_torch.inference import predict_image_paths_resilient

    records_by_path = {_path_key(record.path): record for record in records}
    successful_paths, predictions, failures = predict_image_paths_resilient(
        model=bundle.model,
        image_paths=[record.path for record in records],
        device=device,
        batch_size=args.batch_size,
        img_size=bundle.img_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    prediction_frame = _build_tree_prediction_frame(
        successful_paths,
        predictions,
        records_by_path,
        bundle.bin_names,
        bundle.thresholds,
    )
    failure_rows = []
    for failure in failures:
        record = records_by_path[_path_key(failure.path)]
        message = _sanitize_error_message(failure.error_message, record)
        failure_rows.append(
            {
                "state_code": record.state_code,
                "park_code": record.park_code,
                "image_filename": record.path.name,
                "image_relative_path": record.relative_path,
                "part_number": part_number,
                "error_type": failure.error_type,
                "error_message": message,
            }
        )
        logger.warning(
            "Image failed state=USA_%s part=%d path=%s error=%s: %s",
            record.state_code,
            part_number,
            record.relative_path,
            failure.error_type,
            message,
        )
    if len(prediction_frame) + len(failure_rows) != len(records):
        raise RuntimeError("Part success/failure counts do not match its inventory")
    return prediction_frame, failure_rows


def _write_initial_artifacts(
    run_root: Path,
    config: dict[str, Any],
    inventories: dict[str, StateInventory],
    inventory_only: bool,
) -> dict[str, dict[str, Any]]:
    _write_json_atomic(run_root / "run_config.json", config)
    state_summaries = {}
    for state_code, inventory in inventories.items():
        status = "inventory_complete" if inventory_only else "pending"
        state_summaries[state_code] = _initial_state_summary(
            inventory,
            config["images_per_part"],
            status=status,
        )
        _write_state_summary(run_root, state_summaries[state_code])
    status = "inventory_complete" if inventory_only else "pending"
    _write_json_atomic(
        run_root / "run_summary.json",
        _aggregate_run_summary(config, state_summaries, status),
    )
    return state_summaries


def _load_state_summaries(
    run_root: Path,
    inventories: dict[str, StateInventory],
    images_per_part: int,
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for state_code, inventory in inventories.items():
        path = _state_root(run_root, state_code) / "state_summary.json"
        summaries[state_code] = (
            _read_json(path)
            if path.is_file()
            else _initial_state_summary(inventory, images_per_part)
        )
    return summaries


def _run_predictions(
    args: argparse.Namespace,
    run_root: Path,
    config: dict[str, Any],
    inventories: dict[str, StateInventory],
    logger: logging.Logger,
    event_path: Path,
    checkpoint: Path,
) -> int:
    device, bundle = _resolve_prediction_runtime(checkpoint, args.device)
    config["resolved_device"] = str(device)
    config["model_run_tag"] = bundle.run_tag
    config["model_variant"] = bundle.variant
    _write_json_atomic(run_root / "run_config.json", config)

    state_summaries = _load_state_summaries(
        run_root,
        inventories,
        args.images_per_part,
    )
    logger.info(
        "Preflight passed image_root=%s output=%s device=%s states=%s "
        "images_per_part=%d batch_size=%d workers=%d",
        config["image_root"],
        run_root,
        device,
        ",".join(sorted(inventories)),
        args.images_per_part,
        args.batch_size,
        args.num_workers,
    )
    _append_event(
        event_path,
        "run_started",
        run_id=config["run_id"],
        state_codes=sorted(inventories),
        device=str(device),
    )

    active_state: str | None = None
    try:
        for state_code, inventory in inventories.items():
            active_state = state_code
            parts = _partition_records(inventory.records, args.images_per_part)
            previous_summary = state_summaries.get(state_code, {})
            summary = _initial_state_summary(inventory, args.images_per_part, status="running")
            for field in (
                "images_discovered",
                "jpg_directories",
                "ignored_files",
                "duplicate_basename_groups",
                "layout_warnings",
            ):
                if field in previous_summary:
                    summary[field] = previous_summary[field]
            summary["started_at_utc"] = previous_summary.get("started_at_utc", _utc_now())
            summary["model_run_tag"] = bundle.run_tag
            summary["model_variant"] = bundle.variant
            summary["resolved_device"] = str(device)
            previous_elapsed = float(previous_summary.get("elapsed_seconds", 0.0))
            state_started = time.monotonic()
            logger.info(
                "Starting USA_%s discovered=%d selected=%d parts=%d",
                state_code,
                summary["images_discovered"],
                len(inventory.records),
                len(parts),
            )
            _append_event(
                event_path,
                "state_started",
                run_id=config["run_id"],
                state_code=state_code,
                images_discovered=summary["images_discovered"],
                images_selected=len(inventory.records),
                parts_total=len(parts),
            )

            for part_number, records in enumerate(parts, start=1):
                prediction_path, failure_path = _part_paths(
                    run_root,
                    state_code,
                    part_number,
                )
                valid, succeeded, failed = _validate_completed_part(
                    prediction_path,
                    failure_path,
                    records,
                    bundle.bin_names,
                )
                if args.resume and valid:
                    logger.info(
                        "Skipping completed USA_%s part %d/%d succeeded=%d failed=%d",
                        state_code,
                        part_number,
                        len(parts),
                        succeeded,
                        failed,
                    )
                    elapsed = 0.0
                else:
                    if args.resume and (prediction_path.exists() or failure_path.exists()):
                        logger.warning(
                            "Rebuilding incomplete USA_%s part %d/%d",
                            state_code,
                            part_number,
                            len(parts),
                        )
                    logger.info(
                        "Starting USA_%s part %d/%d images=%d",
                        state_code,
                        part_number,
                        len(parts),
                        len(records),
                    )
                    part_started = time.monotonic()
                    prediction_frame, failure_rows = _predict_part(
                        records,
                        bundle,
                        device,
                        args,
                        logger,
                        part_number,
                    )
                    _write_dataframe_atomic(prediction_frame, prediction_path)
                    _write_csv_rows_atomic(failure_path, FAILURE_COLUMNS, failure_rows)
                    valid, succeeded, failed = _validate_completed_part(
                        prediction_path,
                        failure_path,
                        records,
                        bundle.bin_names,
                    )
                    if not valid:
                        raise RuntimeError(
                            f"Published USA_{state_code} part {part_number} failed validation"
                        )
                    elapsed = time.monotonic() - part_started

                summary["parts_completed"] += 1
                summary["images_attempted"] += len(records)
                summary["images_succeeded"] += succeeded
                summary["images_failed"] += failed
                summary["elapsed_seconds"] = previous_elapsed + (
                    time.monotonic() - state_started
                )
                summary["last_update_utc"] = _utc_now()
                state_summaries[state_code] = summary
                _write_state_summary(run_root, summary)
                _write_json_atomic(
                    run_root / "run_summary.json",
                    _aggregate_run_summary(
                        config,
                        state_summaries,
                        "running",
                        resolved_device=str(device),
                    ),
                )
                rate = len(records) / elapsed if elapsed > 0 else None
                logger.info(
                    "USA_%s part %d/%d complete attempted=%d succeeded=%d failed=%d elapsed=%.1fs",
                    state_code,
                    part_number,
                    len(parts),
                    len(records),
                    succeeded,
                    failed,
                    elapsed,
                )
                _append_event(
                    event_path,
                    "part_completed",
                    run_id=config["run_id"],
                    state_code=state_code,
                    part_number=part_number,
                    parts_total=len(parts),
                    images_attempted=len(records),
                    images_succeeded=succeeded,
                    images_failed=failed,
                    elapsed_seconds=elapsed,
                    images_per_second=rate,
                    output_path=str(prediction_path),
                )

            summary["status"] = "complete"
            summary["completed_at_utc"] = _utc_now()
            summary["last_update_utc"] = summary["completed_at_utc"]
            summary["elapsed_seconds"] = previous_elapsed + (
                time.monotonic() - state_started
            )
            state_summaries[state_code] = summary
            _write_state_summary(run_root, summary)
            logger.info(
                "USA_%s complete parts=%d attempted=%d succeeded=%d failed=%d elapsed=%.1fs",
                state_code,
                summary["parts_completed"],
                summary["images_attempted"],
                summary["images_succeeded"],
                summary["images_failed"],
                summary["elapsed_seconds"],
            )
            _append_event(
                event_path,
                "state_completed",
                run_id=config["run_id"],
                state_code=state_code,
                parts_total=summary["parts_total"],
                images_attempted=summary["images_attempted"],
                images_succeeded=summary["images_succeeded"],
                images_failed=summary["images_failed"],
            )

        run_summary = _aggregate_run_summary(
            config,
            state_summaries,
            "complete",
            resolved_device=str(device),
            elapsed_seconds=sum(
                float(item.get("elapsed_seconds", 0.0))
                for item in state_summaries.values()
            ),
            completed_at_utc=_utc_now(),
        )
        _write_json_atomic(run_root / "run_summary.json", run_summary)
        _append_event(
            event_path,
            "run_completed",
            run_id=config["run_id"],
            images_attempted=run_summary["images_attempted"],
            images_succeeded=run_summary["images_succeeded"],
            images_failed=run_summary["images_failed"],
            parts_total=run_summary["parts_total"],
        )
        logger.info(
            "Run complete attempted=%d succeeded=%d failed=%d summary=%s",
            run_summary["images_attempted"],
            run_summary["images_succeeded"],
            run_summary["images_failed"],
            run_root / "run_summary.json",
        )
        return 0
    except BaseException as exc:
        if active_state is not None:
            summary = state_summaries[active_state]
            summary["status"] = "failed"
            summary["last_update_utc"] = _utc_now()
            summary["error_type"] = type(exc).__name__
            summary["error_message"] = str(exc)
            _write_state_summary(run_root, summary)
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        run_summary = _aggregate_run_summary(
            config,
            state_summaries,
            status,
            resolved_device=str(device),
            elapsed_seconds=sum(
                float(item.get("elapsed_seconds", 0.0))
                for item in state_summaries.values()
            ),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        _write_json_atomic(run_root / "run_summary.json", run_summary)
        _append_event(
            event_path,
            "run_interrupted" if status == "interrupted" else "run_failed",
            run_id=config["run_id"],
            state_code=active_state,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        if isinstance(exc, KeyboardInterrupt):
            logger.warning("Run interrupted; use --resume with the same arguments")
            return 130
        logger.exception("Run failed")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    requested_states, run_id = _validate_args(parser, args)
    image_root = Path(args.image_root).expanduser().resolve(strict=False)
    output_root = Path(args.output_dir).expanduser().resolve(strict=False)
    if not image_root.is_dir():
        parser.error(f"missing image root: {image_root}")
    try:
        _validate_output_location(image_root, output_root)
        checkpoint = _resolve_checkpoint(args)
        run_root = _new_run_root(output_root, run_id, args.resume)
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError, OSError) as exc:
        parser.error(str(exc))

    logger, event_path = _configure_logging(run_root, run_id)
    try:
        inventories = _load_or_create_inventories(
            args,
            image_root,
            run_root,
            requested_states,
            logger,
        )
        selected_states = tuple(sorted(inventories))
        config = _run_config(
            args,
            run_id,
            image_root,
            output_root,
            checkpoint,
            requested_states,
            selected_states,
        )
        if args.resume:
            saved_config = _read_json(run_root / "run_config.json")
            config["created_at_utc"] = saved_config["created_at_utc"]
            if _resume_identity(saved_config) != _resume_identity(config):
                raise ValueError(
                    "Resume arguments do not match run_config.json; use the original "
                    "arguments or choose a new --run-id"
                )
            config = saved_config
        else:
            _write_initial_artifacts(
                run_root,
                config,
                inventories,
                inventory_only=args.inventory_only,
            )

        free_bytes = shutil.disk_usage(output_root).free
        logger.info(
            "Inventory complete states=%d discovered=%d selected=%d output_free_gb=%.2f",
            len(inventories),
            sum(inventory.images_discovered for inventory in inventories.values()),
            sum(len(inventory.records) for inventory in inventories.values()),
            free_bytes / (1024 ** 3),
        )
        for inventory in inventories.values():
            logger.info(
                "USA_%s discovered=%d selected=%d jpg_directories=%d ignored_files=%d "
                "duplicate_basename_groups=%d layout_warnings=%d parts=%d",
                inventory.state_code,
                inventory.images_discovered,
                len(inventory.records),
                inventory.jpg_directories,
                inventory.ignored_files,
                inventory.duplicate_basename_groups,
                len(inventory.layout_warnings),
                math.ceil(len(inventory.records) / args.images_per_part)
                if inventory.records
                else 0,
            )
        _append_event(
            event_path,
            "inventory_completed",
            run_id=run_id,
            state_codes=selected_states,
            images_discovered=sum(item.images_discovered for item in inventories.values()),
            images_selected=sum(len(item.records) for item in inventories.values()),
        )
        if args.inventory_only:
            logger.info("Inventory-only run complete: %s", run_root / "run_summary.json")
            return 0
        if checkpoint is None:
            raise RuntimeError("Prediction checkpoint was not resolved")
        return _run_predictions(
            args,
            run_root,
            config,
            inventories,
            logger,
            event_path,
            checkpoint,
        )
    except KeyboardInterrupt:
        logger.warning("Inventory interrupted")
        _append_event(event_path, "inventory_interrupted", run_id=run_id)
        return 130
    except Exception as exc:
        logger.exception("Workstation inference setup failed")
        summary_path = run_root / "run_summary.json"
        if summary_path.is_file():
            try:
                summary = _read_json(summary_path)
                summary.update(
                    {
                        "status": "failed",
                        "last_update_utc": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                _write_json_atomic(summary_path, summary)
            except Exception:
                logger.exception("Could not update run_summary.json after setup failure")
        _append_event(
            event_path,
            "setup_failed",
            run_id=run_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
