from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def _find_project_root(start: Optional[Path] = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    for p in [candidate, *candidate.parents]:
        if (p / ".git").exists() or (p / "requirements.txt").exists():
            return p
    return candidate


def _load_env_defaults() -> Path:
    project_root = _find_project_root()
    # Load .env from project root first, then current directory (non-destructive)
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(".env", override=False)

    # Provide sensible defaults if unset
    os.environ.setdefault(
        "GOOGLE_OAUTH_CLIENT_SECRETS",
        str((project_root / "secrets/client_secrets.json").resolve()),
    )
    os.environ.setdefault(
        "GOOGLE_OAUTH_CREDENTIALS_CACHE",
        str((project_root / "secrets/credentials.json").resolve()),
    )
    return project_root


def get_drive(
    use_local_server: bool = True, client_secrets_path: Optional[str] = None
) -> GoogleDrive:
    """
    Create an authenticated GoogleDrive client using OAuth with refresh token caching.
    - Loads .env (root-aware) and supports explicit client_secrets_path override.
    - Persists tokens to GOOGLE_OAUTH_CREDENTIALS_CACHE.
    """
    _load_env_defaults()
    client_secrets = (
        client_secrets_path
        or os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS")
        or str((_find_project_root() / "secrets/client_secrets.json").resolve())
    )
    cred_cache = (
        os.environ.get("GOOGLE_OAUTH_CREDENTIALS_CACHE")
        or str((_find_project_root() / "secrets/credentials.json").resolve())
    )

    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(client_secrets)
    # Request offline access so Google issues a refresh_token
    gauth.settings["get_refresh_token"] = True

    # Try cached credentials first
    try:
        if Path(cred_cache).exists():
            gauth.LoadCredentialsFile(cred_cache)
    except Exception:
        pass

    # Obtain/refresh/authorize as needed
    if not getattr(gauth, "credentials", None) or not getattr(
        gauth.credentials, "refresh_token", None
    ):
        if use_local_server:
            gauth.LocalWebserverAuth()
        else:
            gauth.CommandLineAuth()
    elif getattr(gauth, "access_token_expired", False):
        try:
            gauth.Refresh()
        except Exception:
            if use_local_server:
                gauth.LocalWebserverAuth()
            else:
                gauth.CommandLineAuth()
    else:
        gauth.Authorize()

    # Persist credentials for reuse
    try:
        Path(cred_cache).parent.mkdir(parents=True, exist_ok=True)
        gauth.SaveCredentialsFile(cred_cache)
    except Exception:
        pass

    return GoogleDrive(gauth)


def list_files_in_folder(
    drive: GoogleDrive,
    folder_id: str,
    *,
    mime_prefix: Optional[str] = "image/",
    include_trashed: bool = False,
    include_shared_drives: bool = True,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    List files in a Drive folder; optionally filter by MIME prefix (e.g., 'image/').
    Returns dictionaries with: file_id, name, mimeType, size.
    """
    q = [f"'{folder_id}' in parents"]
    if not include_trashed:
        q.append("trashed = false")
    if mime_prefix:
        q.append(f"mimeType contains '{mime_prefix}'")
    params = {
        "q": " and ".join(q),
        "supportsAllDrives": include_shared_drives,
        "includeItemsFromAllDrives": include_shared_drives,
    }
    items = drive.ListFile(params).GetList()
    if limit is not None:
        items = items[: int(limit)]
    out: List[Dict] = []
    for it in items:
        out.append(
            {
                "file_id": it["id"],
                "name": it.get("title") or it.get("name"),
                "mimeType": it.get("mimeType"),
                "size": int(it.get("fileSize", 0)) if it.get("fileSize") else None,
            }
        )
    return out


def download_file_bytes(drive: GoogleDrive, file_id: str) -> bytes:
    """
    Download a Drive file to memory and return raw bytes.
    """
    f = drive.CreateFile({"id": file_id})
    f.FetchContent()  # loads into f.content (BytesIO)
    return f.content.getvalue()


