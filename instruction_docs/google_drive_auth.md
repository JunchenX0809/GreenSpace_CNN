# Google Drive authentication (PyDrive2 OAuth)

This project uses **PyDrive2 OAuth** to access images stored in a shared Google Drive folder.

## What you need
- A Google account that has access to the shared Drive folder
- The Drive folder ID (ask the project owner)
- An OAuth client secrets JSON file (ask the project owner)

## 1) Put the OAuth client secrets in the repo
Place the JSON file at:
- `secrets/client_secrets.json`

This path is the default used by `src/drive_utils.py` (and `secrets/` is gitignored).

## 2) Create a `.env` in the project root
Create `./.env` (not committed) and set at least:

```bash
# Required: Drive folder that contains images
GOOGLE_DRIVE_FOLDER_ID="YOUR_FOLDER_ID_HERE"
```

Optional knobs (defaults shown):

```bash
# If the folder is a Shared Drive, keep this true
GOOGLE_DRIVE_INCLUDE_SHARED=true

# Only list/download images (usually correct)
GOOGLE_DRIVE_MIME_PREFIX="image/"

# Defaults (normally don't change)
GOOGLE_OAUTH_CLIENT_SECRETS="secrets/client_secrets.json"
GOOGLE_OAUTH_CREDENTIALS_CACHE="secrets/credentials.json"
```

## 3) First-time auth flow (browser-based)
When you run the Drive steps in `notebooks/02_data_preprocessing.ipynb`, the code calls:
- `get_drive(use_local_server=True)`

On first run, this triggers `LocalWebserverAuth()`:
- Your browser opens to Google OAuth
- You log in and approve access
- Tokens are cached to `secrets/credentials.json` for future runs

After this, reruns usually **do not** prompt again (tokens refresh automatically).

## If browser auth doesn’t work
If the local webserver flow is blocked (remote machine / SSH / corporate policy), switch to:
- `get_drive(use_local_server=False)` (uses `CommandLineAuth()`)

You can do that in the notebook cell that creates `drive = get_drive(...)`.

