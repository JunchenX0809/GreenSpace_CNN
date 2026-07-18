from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing import (
    aggregate_rater_labels,
    build_split_frames,
    clean_survey_dataframe,
    normalize_dataframe_columns,
)


def _raw_row(image_name: str, include_tile: str = "Yes") -> dict[str, object]:
    return {
        "Image Name": image_name,
        "Include Tile": include_tile,
        "Sports Field": "No",
        "Multipurpose Open Area": "Yes",
        "Children's Playground": "No",
        "Water Feature": "No",
        "Gardens": "No",
        "Walking Paths": "Yes",
        "Built Structures": "No",
        "Parking Lots": "No",
        "Shade Along Paths": "Minimal (< 10%)",
        "Structured Unstructured": "3",
        "Vegetation Cover Distribution": "Mixed",
    }


class CleaningTests(unittest.TestCase):
    def test_cleaning_normalizes_headers_and_filename_extension(self) -> None:
        cleaned = clean_survey_dataframe(
            pd.DataFrame([_raw_row("  Example.PnG  ")])
        )
        self.assertIn("children_s_playground", cleaned.columns)
        self.assertEqual(cleaned.loc[0, "image_filename"], "Example.png")

    def test_normalized_column_collisions_fail_by_default(self) -> None:
        frame = pd.DataFrame(columns=["Image Name", "Image-Name"])
        with self.assertRaisesRegex(ValueError, "collision"):
            normalize_dataframe_columns(frame)


class AggregationTests(unittest.TestCase):
    def test_aggregation_filters_rows_and_preserves_soft_hard_schema(self) -> None:
        first = _raw_row("a.JPG")
        second = _raw_row("a.JPG")
        second["Sports Field"] = "Yes"
        second["Shade Along Paths"] = "Abundant (> 10%)"
        second["Structured Unstructured"] = "5"
        second["Vegetation Cover Distribution"] = "Only trees"
        excluded = _raw_row("b.jpg", include_tile="No")

        cleaned = clean_survey_dataframe(
            pd.DataFrame([first, second, excluded])
        )
        soft, hard, summary = aggregate_rater_labels(cleaned)

        self.assertEqual(summary["before_rows"], 3)
        self.assertEqual(summary["included_rows"], 2)
        self.assertEqual(summary["included_unique_images"], 1)
        self.assertEqual(soft.loc[0, "image_filename"], "a.jpg")
        self.assertEqual(soft.loc[0, "n_ratings"], 2)
        self.assertEqual(soft.loc[0, "sports_field_p"], 0.5)
        self.assertEqual(soft.loc[0, "shade_p_minimal"], 0.5)
        self.assertEqual(soft.loc[0, "shade_p_abundant"], 0.5)
        self.assertEqual(soft.loc[0, "score_mean"], 4.0)
        self.assertEqual(soft.loc[0, "veg_mean"], 4.0)
        self.assertEqual(hard.loc[0, "sports_field"], 1)
        self.assertEqual(hard.loc[0, "shade_class"], 0)
        self.assertEqual(hard.loc[0, "score_class"], 3)
        self.assertEqual(hard.loc[0, "veg_class"], 3)


class SplitTests(unittest.TestCase):
    def test_split_matches_current_ratios_and_is_deterministic(self) -> None:
        raw = pd.DataFrame([_raw_row(f"image_{index}.jpg") for index in range(10)])
        cleaned = clean_survey_dataframe(raw)
        soft, hard, _ = aggregate_rater_labels(cleaned)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            for filename in soft["image_filename"]:
                (image_dir / filename).touch()

            first, first_summary = build_split_frames(soft, hard, image_dir)
            second, second_summary = build_split_frames(soft, hard, image_dir)

        self.assertEqual(first_summary, second_summary)
        self.assertEqual(
            {name: len(frame) for name, frame in first.items()},
            {"train": 6, "val": 2, "test": 2},
        )
        for split in ("train", "val", "test"):
            self.assertEqual(
                first[split]["image_filename"].tolist(),
                second[split]["image_filename"].tolist(),
            )
        memberships = [
            set(first[split]["image_filename"])
            for split in ("train", "val", "test")
        ]
        self.assertFalse(memberships[0] & memberships[1])
        self.assertFalse(memberships[0] & memberships[2])
        self.assertFalse(memberships[1] & memberships[2])


if __name__ == "__main__":
    unittest.main()
