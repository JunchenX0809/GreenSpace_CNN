# Data Directory

This repo keeps the **`data/` folder structure** in Git, but ignores the **contents** (raw files, caches, derived CSVs) by default.

## Folder purpose
- **`data/raw/`**
  - Raw survey CSVs (may contain sensitive info) and any original inputs.
  - Not committed to Git.
- **`data/interim/`**
  - Intermediate manifests produced during preprocessing (e.g., filelists with Drive IDs).
  - Not committed to Git.
- **`data/cache/`**
  - Local caches for fast iteration (e.g., downloaded images).
  - Not committed to Git.
- **`data/processed/`**
  - Derived outputs used by training (labels, splits, thresholds, etc.).
  - Not committed to Git by default (to keep the repo lightweight).

## Notes for teammates
- If you are missing any files under `data/`, run `notebooks/02_data_preprocessing.ipynb` to regenerate them.
- If you want to share a particular processed artifact (e.g., `splits/*.csv`) across the team, we can whitelist it in `.gitignore` later.

