# `secrets/` (not committed)

This folder holds **local-only** credentials used to access Google Drive images.

## What goes here
- `client_secrets.json`: OAuth client secrets downloaded from Google Cloud Console
- `credentials.json`: OAuth token cache created automatically after first login

## Setup instructions
See `instruction_docs/google_drive_auth.md`.

## Safety
Do **not** commit any `.json` credential files. This repo is configured to only track this `README.md`.

