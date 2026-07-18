from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.drive_download import (
    attach_drive_ids,
    build_drive_manifest,
    build_rated_download_manifest,
    download_images,
)


class ManifestTests(unittest.TestCase):
    def test_builds_existing_filelist_and_included_survey_contract(self) -> None:
        drive_manifest, drive_summary = build_drive_manifest(
            [
                {"file_id": "id-a", "name": "a.JPG", "mimeType": "image/jpeg"},
                {"file_id": "id-b", "name": "b.jpg", "mimeType": "image/jpeg"},
            ]
        )
        filelist = pd.DataFrame(
            {
                "FileId": [1, 2, 3],
                "ImageName": ["a.JPG", "b.jpg", "missing.png"],
                "ReviewFlag": ["", "", ""],
            }
        )
        filelist_with_ids, filelist_summary = attach_drive_ids(
            filelist,
            drive_manifest,
        )
        survey = pd.DataFrame(
            {
                "Image Name": ["a.JPG", "a.JPG", "b.jpg", "missing.png"],
                "Include Tile": ["Yes", "Yes", "No", "Yes"],
            }
        )
        rows, downloads, survey_summary = build_rated_download_manifest(
            survey,
            filelist_with_ids,
        )

        self.assertEqual(drive_summary["manifest_images"], 2)
        self.assertEqual(filelist_summary["filelist_matched"], 2)
        self.assertEqual(filelist_summary["filelist_missing"], 1)
        self.assertEqual(len(rows), 3)
        self.assertIn("ReviewFlag", rows.columns)
        self.assertEqual(downloads["image_filename"].tolist(), ["a.jpg"])
        self.assertEqual(survey_summary["included_unique_images"], 2)
        self.assertEqual(survey_summary["missing_unique_images"], 1)

    def test_duplicate_drive_names_preserve_notebook_first_match_policy(self) -> None:
        manifest, summary = build_drive_manifest(
            [
                {"file_id": "first", "name": "same.JPG"},
                {"file_id": "second", "name": "same.jpg"},
            ]
        )
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest.loc[0, "file_id"], "first")
        self.assertEqual(summary["duplicate_names"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_drive_manifest(
                [
                    {"file_id": "first", "name": "same.JPG"},
                    {"file_id": "second", "name": "same.jpg"},
                ],
                duplicate_policy="error",
            )


class DownloadTests(unittest.TestCase):
    def test_download_retries_atomically_and_skips_existing_files(self) -> None:
        manifest = pd.DataFrame(
            {
                "image_filename": ["a.jpg", "b.jpg"],
                "drive_file_id": ["id-a", "id-b"],
            }
        )
        attempts: dict[str, int] = {}

        def fetch(_drive: object, file_id: str) -> bytes:
            attempts[file_id] = attempts.get(file_id, 0) + 1
            if file_id == "id-a" and attempts[file_id] == 1:
                raise RuntimeError("temporary")
            return f"content-{file_id}".encode()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            (cache / "b.jpg").write_bytes(b"already cached")
            summary = download_images(
                object(),
                manifest,
                cache,
                fetch_bytes=fetch,
                sleep=lambda _seconds: None,
                show_progress=False,
            )

            self.assertEqual((cache / "a.jpg").read_bytes(), b"content-id-a")
            self.assertEqual((cache / "b.jpg").read_bytes(), b"already cached")
            self.assertFalse((cache / "a.jpg.part").exists())

        self.assertEqual(attempts, {"id-a": 2})
        self.assertEqual(summary["downloaded"], 1)
        self.assertEqual(summary["skipped_existing"], 1)
        self.assertEqual(summary["failed"], 0)

    def test_failed_or_unsafe_downloads_are_reported(self) -> None:
        manifest = pd.DataFrame(
            {
                "image_filename": ["failed.jpg", "../unsafe.jpg"],
                "drive_file_id": ["id-failed", "id-unsafe"],
            }
        )

        def fail(_drive: object, _file_id: str) -> bytes:
            raise RuntimeError("no access")

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = download_images(
                object(),
                manifest,
                temp_dir,
                max_retries=2,
                fetch_bytes=fail,
                sleep=lambda _seconds: None,
                show_progress=False,
            )
            self.assertFalse((Path(temp_dir) / "failed.jpg.part").exists())

        self.assertEqual(summary["requested"], 2)
        self.assertEqual(summary["failed"], 2)


if __name__ == "__main__":
    unittest.main()
