#!/usr/bin/env python3
"""Append patch-center coordinates to state-tree prediction CSV parts."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath


PART_NAME = re.compile(r"^predictions_USA_([A-Z]{2})_part_\d{5}\.csv$")
REQUIRED_PREDICTION_COLUMNS = (
    "image_filename",
    "state_code",
    "park_code",
    "image_relative_path",
)
REQUIRED_POINT_COLUMNS = (
    "base_park_id",
    "full_park_id",
    "export_index",
    "state_code",
    "patch_id",
    "source_tfrecord",
    "center_x",
    "center_y",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root", required=True,
        help="The same root containing USA_XX state folders used for inference",
    )
    parser.add_argument(
        "--predictions-dir", required=True,
        help="Directory containing one state's predictions_USA_XX_part_00001.csv files",
    )
    parser.add_argument(
        "--output-dir",
        help="Destination for enriched CSVs (default: --predictions-dir)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace previously enriched outputs after the full join validates",
    )
    parser.add_argument(
        "--skip-bad-tables", action="store_true",
        help=(
            "Keep going when a park's patch-point table is missing, empty or malformed, "
            "or an image has no matching point: those rows get blank coordinates and are "
            "listed in skipped_coordinates_USA_XX.csv in the output directory"
        ),
    )
    return parser


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate CSV column in {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row in {path}")
    return columns, rows


def _require_columns(columns: list[str], required: tuple[str, ...], path: Path) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")


def _point_image_filename(row: dict[str, str], table: Path, line: int) -> str:
    source = row["source_tfrecord"]
    patch_id = row["patch_id"]
    full_id = row["full_park_id"]
    index = row["export_index"]
    if not source.endswith(".tfrecord.gz") or "/" in source or "\\" in source:
        raise ValueError(f"Invalid source_tfrecord at {table}:{line}")
    if not index.isdigit() or not patch_id.startswith(f"{full_id}_{index}_"):
        raise ValueError(f"Inconsistent patch ID or export index at {table}:{line}")
    suffix = patch_id.removeprefix(f"{full_id}_{index}_")
    if not re.fullmatch(r"p\d+", suffix):
        raise ValueError(f"Invalid patch suffix at {table}:{line}")
    return f"{full_id}_export{index}_{source.removesuffix('.tfrecord.gz')}_{suffix}.jpg"


def _load_points(table: Path, state: str, park: str) -> dict[str, tuple[str, str]]:
    if not table.is_file():
        raise FileNotFoundError(f"Missing patch-point table: {table}")
    columns, rows = _read_csv(table)
    _require_columns(columns, REQUIRED_POINT_COLUMNS, table)
    points: dict[str, tuple[str, str]] = {}
    for line, row in enumerate(rows, start=2):
        if row["base_park_id"] != park or row["state_code"] != state:
            raise ValueError(f"Park or state mismatch at {table}:{line}")
        filename = _point_image_filename(row, table, line)
        if filename in points:
            raise ValueError(f"Duplicate patch image {filename} in {table}")
        for field in ("center_x", "center_y"):
            try:
                finite = math.isfinite(float(row[field]))
            except ValueError:
                finite = False
            if not finite:
                raise ValueError(f"Invalid {field} at {table}:{line}")
        points[filename] = (row["center_x"], row["center_y"])
    return points


def _table_for_prediction(
    image_root: Path, row: dict[str, str], part: Path, line: int
) -> Path:
    relative_text = row["image_relative_path"]
    relative = PurePosixPath(relative_text)
    parts = relative.parts
    if (
        relative.is_absolute()
        or "\\" in relative_text
        or len(parts) < 4
        or any(component in ("", ".", "..") for component in relative_text.split("/"))
        or parts[-2] != "jpg"
        or parts[-3] != row["park_code"]
        or parts[-1] != row["image_filename"]
        or not parts[0].startswith(f"USA_{row['state_code']}_")
    ):
        raise ValueError(f"Invalid image_relative_path at {part}:{line}")
    return image_root.joinpath(*parts[:-2], "tables", f"{row['park_code']}_patch_points.csv")


def enrich_parts(
    image_root: Path,
    predictions_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    skip_bad_tables: bool = False,
    skipped: list[dict[str, str]] | None = None,
) -> tuple[int, int]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Missing predictions directory: {predictions_dir}")
    parts = sorted(
        path for path in predictions_dir.iterdir()
        if path.is_file() and PART_NAME.fullmatch(path.name)
    )
    if not parts:
        raise FileNotFoundError(f"No state-tree prediction parts in {predictions_dir}")
    states = {PART_NAME.fullmatch(path.name).group(1) for path in parts}
    if len(states) != 1:
        raise ValueError("--predictions-dir must contain parts for exactly one state")
    state = states.pop()
    targets = [output_dir / f"{part.stem}_with_coordinates.csv" for part in parts]
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing[0]}; use --overwrite")

    # Validate every row and every join before publishing any output.
    point_cache: dict[Path, dict[str, tuple[str, str]] | None] = {}
    table_errors: dict[Path, str] = {}
    if skipped is None:
        skipped = []
    seen_paths: set[str] = set()
    prepared: list[tuple[list[str], list[dict[str, str]]]] = []
    total_rows = 0
    for part in parts:
        columns, rows = _read_csv(part)
        _require_columns(columns, REQUIRED_PREDICTION_COLUMNS, part)
        if "center_x" in columns or "center_y" in columns:
            raise ValueError(f"Input already has coordinate columns: {part}")
        for line, row in enumerate(rows, start=2):
            if row["state_code"] != state:
                raise ValueError(f"State mismatch at {part}:{line}")
            relative = row["image_relative_path"]
            if relative.casefold() in seen_paths:
                raise ValueError(f"Duplicate prediction path at {part}:{line}: {relative}")
            seen_paths.add(relative.casefold())
            table = _table_for_prediction(image_root, row, part, line)
            if table not in point_cache:
                try:
                    point_cache[table] = _load_points(table, state, row["park_code"])
                except (OSError, ValueError) as exc:
                    if not skip_bad_tables:
                        raise
                    point_cache[table] = None
                    table_errors[table] = str(exc)
                    print(f"WARNING: skipping table: {exc}", file=sys.stderr)
            points = point_cache[table]
            coordinates = None if points is None else points.get(row["image_filename"])
            if coordinates is None:
                if points is None:
                    reason = f"bad patch-point table: {table_errors[table]}"
                else:
                    reason = f"no patch-point match in {table}"
                if not skip_bad_tables:
                    raise ValueError(
                        f"No patch-point match for {row['image_filename']} at {part}:{line} "
                        f"in {table}"
                    )
                skipped.append({
                    "part": part.name, "line": str(line),
                    "park_code": row["park_code"],
                    "image_relative_path": relative, "reason": reason,
                })
                coordinates = ("", "")
            row["center_x"], row["center_y"] = coordinates
        total_rows += len(rows)
        prepared.append((columns + ["center_x", "center_y"], rows))

    output_dir.mkdir(parents=True, exist_ok=True)
    for target, (columns, rows) in zip(targets, prepared):
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=output_dir,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    report = output_dir / f"skipped_coordinates_USA_{state}.csv"
    if skipped:
        with report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["part", "line", "park_code", "image_relative_path", "reason"],
            )
            writer.writeheader()
            writer.writerows(skipped)
    elif report.exists():
        report.unlink()
    return len(parts), total_rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else predictions_dir
    skipped: list[dict[str, str]] = []
    try:
        part_count, row_count = enrich_parts(
            Path(args.image_root).expanduser().resolve(), predictions_dir,
            output_dir, args.overwrite, args.skip_bad_tables, skipped,
        )
    except (OSError, ValueError) as exc:
        print(f"Coordinate join failed: {exc}", file=sys.stderr)
        return 1
    matched = row_count - len(skipped)
    print(f"Wrote {matched} matched predictions across {part_count} parts to {output_dir}")
    if skipped:
        parks = len({entry["park_code"] for entry in skipped})
        print(
            f"Skipped coordinates for {len(skipped)} predictions in {parks} parks; "
            f"see skipped_coordinates_USA_*.csv in {output_dir}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
