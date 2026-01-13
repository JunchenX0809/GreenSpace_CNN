#!/usr/bin/env python3
"""
Fail-fast check to standardize team Python versions.

Run:
  python scripts/check_python_version.py
"""

import sys


def main() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        raise SystemExit(
            f"Python {major}.{minor} detected. Please use Python 3.11+ for this repo.\n"
            "Tip (macOS/Homebrew): brew install python@3.11\n"
            "Then recreate your venv using that interpreter:\n"
            "  /opt/homebrew/bin/python3.11 -m venv .venv\n"
        )
    print(f"OK: Python {major}.{minor} (>= 3.11)")


if __name__ == "__main__":
    main()

