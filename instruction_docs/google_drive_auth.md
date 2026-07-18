# Google Drive authentication (PyDrive2 OAuth)

This project uses **PyDrive2 OAuth** to access images stored in a shared Google Drive folder.

## What you need
- A Google account that has access to the shared Drive folder
- The Drive folder ID: 1upr61AWU85uyOjVcVWqhJbXeJG296FN4
- An OAuth client secrets JSON file

## 1) Put the OAuth client secrets in the repo
### 1A) Download the OAuth client secrets JSON from Google Cloud Console
This is the file you likely “downloaded locally” before.

1. **Open Google Cloud Console → APIs & Services**
   - Select an existing project, or create a new one.

![After creating/selecting a project](images/1_after_creating_project.png)



2. **Configure the OAuth consent screen**
   - Go to **OAuth consent screen**
   ![APIs & Services page](images/2_in_API_servicing_page.png)
   - Fill out the required fields (app name + contact email)


![OAuth consent screen](images/3_in_auth_screen.png)

3. **Create an OAuth Client ID**
   - Go to **Clients → Create client → OAuth client ID**
   - Application type: **Desktop app** (recommended for this notebook workflow)
   - Create it

![Create OAuth client ID](images/4_in_client_creation_page.png)

4. **Download the JSON**
   - In **Credentials**, find the client you created and click **Download JSON**
   - The downloaded filename typically looks like:
     - `client_secret_XXXXXXXX.apps.googleusercontent.com.json`

### 1B) Put the JSON into this repo
Move/rename the downloaded JSON file to:
- `secrets/client_secrets.json`

This path is the default used by `src/drive_utils.py` (and `secrets/` is gitignored).

## 2) Create a `.env` in the project root
Create `./.env` (not committed) and set (copy & paste):

```bash
# Required: Drive folder that contains images
GOOGLE_DRIVE_FOLDER_ID=1upr61AWU85uyOjVcVWqhJbXeJG296FN4
```

Other knobs (defaults shown):

```bash
# If the folder is a Shared Drive, keep this true
GOOGLE_DRIVE_INCLUDE_SHARED=true

# Only list/download images (usually correct)
GOOGLE_DRIVE_MIME_PREFIX="image/"

# Defaults (normally don't change)
GOOGLE_OAUTH_CLIENT_SECRETS="secrets/client_secrets.json"
GOOGLE_OAUTH_CREDENTIALS_CACHE="secrets/credentials.json"
```

## 3) Download the rated survey images from a terminal

From the repository root, with the project virtual environment activated, run:

```bash
python scripts/download_drive_images.py \
  --survey-csv data/raw/0614_survey_response.csv \
  --filelist-csv data/filelist_0103.csv \
  --cache-dir data/cache/images
```

Replace the two CSV paths with the files for the current run. The folder ID is
read from `GOOGLE_DRIVE_FOLDER_ID` in `.env`; it can instead be supplied with
`--folder-id`. The script lists images that are direct children of that folder,
matching the established `02_data_preprocessing.ipynb` layout.

Useful safe-start options:

```bash
# Build and inspect the manifests without downloading image bytes
python scripts/download_drive_images.py \
  --survey-csv data/raw/0614_survey_response.csv \
  --filelist-csv data/filelist_0103.csv \
  --manifest-only

# Download only 50 unique rated images
python scripts/download_drive_images.py \
  --survey-csv data/raw/0614_survey_response.csv \
  --filelist-csv data/filelist_0103.csv \
  --limit 50
```

The command writes run-tagged Drive, filelist, and included-survey manifests plus
a JSON summary under `data/interim/`. Existing nonempty cached images are skipped.
Failed downloads are retried and leave no completed-looking partial file. The
command exits nonzero if a requested download still fails.

Use `--fail-on-missing` when every included survey image must match a Drive ID.
Use `python scripts/download_drive_images.py --help` for path overrides and the
full option list.

## 4) First-time auth flow (browser-based)

The download script calls:
- `get_drive(use_local_server=True)`

On first run, this triggers `LocalWebserverAuth()`:
- Your browser opens to Google OAuth
- You log in and approve access
- Tokens are cached to `secrets/credentials.json` for future runs

After this, reruns usually **do not** prompt again (tokens refresh automatically).

## If browser auth doesn’t work
If the local webserver flow is blocked (remote machine / SSH / corporate policy), run:

```bash
python scripts/download_drive_images.py \
  --survey-csv path/to/survey.csv \
  --filelist-csv path/to/filelist.csv \
  --command-line-auth
```

This uses PyDrive2 `CommandLineAuth()` instead of opening the local browser callback.
