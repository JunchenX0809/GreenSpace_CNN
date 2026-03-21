"""Download all images from a nested Google Drive folder to a local cache.

Handles the two-level structure where a root folder contains subfolders
(one per parcel), each with one or more JPEG images inside.

Usage:
    python scripts/cache_drive_folder.py --folder-id <DRIVE_FOLDER_ID>
    python scripts/cache_drive_folder.py --folder-id <DRIVE_FOLDER_ID> --cache-dir data/cache/inference_images --limit 100
"""

import argparse
import time
from pathlib import Path

from tqdm import tqdm

# Ensure src/ is importable when running from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drive_utils import get_drive, download_file_bytes


def list_subfolders(drive, parent_id):
    """List all subfolders in a Drive folder."""
    items = drive.ListFile({
        'q': f"'{parent_id}' in parents and trashed = false and mimeType = 'application/vnd.google-apps.folder'",
        'supportsAllDrives': True,
        'includeItemsFromAllDrives': True,
    }).GetList()
    return [{'id': it['id'], 'name': it.get('title') or it.get('name')} for it in items]


def list_images_in_folder(drive, folder_id):
    """List all image files in a single Drive folder."""
    items = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed = false and mimeType contains 'image/'",
        'supportsAllDrives': True,
        'includeItemsFromAllDrives': True,
    }).GetList()
    return [{'id': it['id'], 'name': it.get('title') or it.get('name')} for it in items]


def main():
    parser = argparse.ArgumentParser(description='Cache Drive images locally.')
    parser.add_argument('--folder-id', required=True, help='Google Drive root folder ID')
    parser.add_argument('--cache-dir', default='data/cache/inference_images', help='Local cache directory')
    parser.add_argument('--limit', type=int, default=None, help='Max subfolders to process (for testing)')
    parser.add_argument('--max-retries', type=int, default=3)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    drive = get_drive(use_local_server=False)

    # Step 1: list subfolders
    print('Listing subfolders...')
    subfolders = list_subfolders(drive, args.folder_id)
    print(f'Found {len(subfolders)} subfolders')

    if args.limit:
        subfolders = subfolders[:args.limit]
        print(f'Limited to {len(subfolders)} subfolders')

    # Step 2: iterate subfolders, list images, download
    skipped = 0
    downloaded = 0
    failed = 0

    for sf in tqdm(subfolders, desc='Subfolders'):
        images = list_images_in_folder(drive, sf['id'])

        for img in images:
            out_path = cache_dir / img['name']

            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                continue

            ok = False
            for attempt in range(1, args.max_retries + 1):
                try:
                    raw = download_file_bytes(drive, img['id'])
                    out_path.write_bytes(raw)
                    downloaded += 1
                    ok = True
                    break
                except Exception as e:
                    if attempt == args.max_retries:
                        failed += 1
                        print(f'\nFAILED: {img["name"]} (id={img["id"]}) -> {e}')
                    else:
                        time.sleep(1.0 * attempt)

    print(f'\nDone. downloaded={downloaded} skipped={skipped} failed={failed}')
    print(f'Cache dir: {cache_dir.resolve()}')


if __name__ == '__main__':
    main()
