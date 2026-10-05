from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import append_patch_coordinates


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class AppendPatchCoordinatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.image_root = root / "images"
        self.predictions_dir = root / "predictions" / "USA_AL"
        self.predictions_dir.mkdir(parents=True)
        self.park = "04033-1402"
        self.state_folder = "USA_AL_2023_Full-archive"
        self.point_table = (
            self.image_root / self.state_folder / "USA_AL_2023_Full"
            / self.park / "tables" / f"{self.park}_patch_points.csv"
        )
        self.point_columns = [
            "base_park_id", "full_park_id", "export_index", "state_code",
            "patch_id", "source_tfrecord", "center_x", "center_y",
        ]
        self.prediction_columns = [
            "image_filename", "score_ev", "state_code", "park_code",
            "image_relative_path",
        ]
        self.parts: list[Path] = []
        point_rows = []
        for index in range(2):
            stem = f"USA_AL_{self.park}_1_{index}-00000"
            filename = f"{self.park}_1_export{index}_{stem}_p00000.jpg"
            point_rows.append({
                "base_park_id": self.park, "full_park_id": f"{self.park}_1",
                "export_index": str(index), "state_code": "AL",
                "patch_id": f"{self.park}_1_{index}_p00000",
                "source_tfrecord": f"{stem}.tfrecord.gz",
                "center_x": str(978336.0 + index),
                "center_y": str(1119477.0 + index),
            })
            part = self.predictions_dir / f"predictions_USA_AL_part_{index + 1:05d}.csv"
            self.parts.append(part)
            write_csv(part, self.prediction_columns, [{
                "image_filename": filename, "score_ev": str(index + 1),
                "state_code": "AL", "park_code": self.park,
                "image_relative_path": (
                    f"{self.state_folder}/USA_AL_2023_Full/{self.park}/jpg/{filename}"
                ),
            }])
        write_csv(self.point_table, self.point_columns, point_rows)

    def test_enriches_all_parts_and_preserves_prediction_values(self) -> None:
        parts, rows = append_patch_coordinates.enrich_parts(
            self.image_root, self.predictions_dir, self.predictions_dir
        )
        self.assertEqual((parts, rows), (2, 2))
        for index, part in enumerate(self.parts):
            original_columns, original_rows = append_patch_coordinates._read_csv(part)
            output = self.predictions_dir / f"{part.stem}_with_coordinates.csv"
            columns, enriched_rows = append_patch_coordinates._read_csv(output)
            self.assertEqual(columns, original_columns + ["center_x", "center_y"])
            self.assertEqual(
                [{key: row[key] for key in original_columns} for row in enriched_rows],
                original_rows,
            )
            self.assertEqual(enriched_rows[0]["center_x"], str(978336.0 + index))
            self.assertEqual(enriched_rows[0]["center_y"], str(1119477.0 + index))

    def test_missing_match_prevents_all_outputs(self) -> None:
        columns, rows = append_patch_coordinates._read_csv(self.parts[1])
        rows[0]["image_filename"] = rows[0]["image_filename"].replace("p00000", "p00001")
        rows[0]["image_relative_path"] = rows[0]["image_relative_path"].replace(
            "p00000", "p00001"
        )
        write_csv(self.parts[1], columns, rows)
        with self.assertRaisesRegex(ValueError, "No patch-point match"):
            append_patch_coordinates.enrich_parts(
                self.image_root, self.predictions_dir, self.predictions_dir
            )
        self.assertEqual(list(self.predictions_dir.glob("*_with_coordinates.csv")), [])

    def test_duplicate_point_fails(self) -> None:
        columns, rows = append_patch_coordinates._read_csv(self.point_table)
        write_csv(self.point_table, columns, [*rows, rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate patch image"):
            append_patch_coordinates.enrich_parts(
                self.image_root, self.predictions_dir, self.predictions_dir
            )

    def test_invalid_coordinate_fails_before_output(self) -> None:
        columns, rows = append_patch_coordinates._read_csv(self.point_table)
        rows[1]["center_y"] = "nan"
        write_csv(self.point_table, columns, rows)
        with self.assertRaisesRegex(ValueError, "Invalid center_y"):
            append_patch_coordinates.enrich_parts(
                self.image_root, self.predictions_dir, self.predictions_dir
            )
        self.assertEqual(list(self.predictions_dir.glob("*_with_coordinates.csv")), [])

    def test_existing_output_requires_overwrite(self) -> None:
        append_patch_coordinates.enrich_parts(
            self.image_root, self.predictions_dir, self.predictions_dir
        )
        with self.assertRaisesRegex(FileExistsError, "Output already exists"):
            append_patch_coordinates.enrich_parts(
                self.image_root, self.predictions_dir, self.predictions_dir
            )
        self.assertEqual(
            append_patch_coordinates.enrich_parts(
                self.image_root, self.predictions_dir, self.predictions_dir,
                overwrite=True,
            ),
            (2, 2),
        )


if __name__ == "__main__":
    unittest.main()
