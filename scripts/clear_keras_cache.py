#!/usr/bin/env python3
"""
Delete cached application weights that can cause cross-environment shape mismatches.

This is intentionally conservative: it only targets EfficientNet-related cached files.

Run:
  python scripts/clear_keras_cache.py
"""

import os
from pathlib import Path


def main() -> None:
    keras_home = Path(os.getenv("KERAS_HOME") or (Path.home() / ".keras"))
    models_dir = keras_home / "models"

    print("KERAS_HOME:", keras_home)
    print("models_dir:", models_dir)

    if not models_dir.exists():
        print("Nothing to do (models directory does not exist).")
        return

    patterns = [
        "*efficientnetb0*",
        "*efficientnetb1*",
        "*efficientnetb2*",
        "*efficientnetb3*",
        "*efficientnetb4*",
        "*efficientnetb5*",
        "*efficientnetb6*",
        "*efficientnetb7*",
        "*efficientnet*",
    ]

    to_delete: list[Path] = []
    for pat in patterns:
        to_delete.extend(models_dir.glob(pat))

    # De-dup + only files
    uniq = []
    seen = set()
    for p in to_delete:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if p.is_file():
            uniq.append(p)

    if not uniq:
        print("No EfficientNet-related cached files found.")
        return

    print(f"Found {len(uniq)} file(s) to delete:")
    for p in sorted(uniq):
        print(" -", p)

    for p in uniq:
        try:
            p.unlink()
        except Exception as e:
            print("FAILED to delete:", p, "->", e)

    print("Done.")


if __name__ == "__main__":
    main()

