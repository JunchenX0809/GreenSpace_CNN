# Process Update — 2026-10-03

## Add Locations to Workstation Predictions

The new coordinate step is ready for the existing workstation layout. It reads
each completed state's prediction CSVs and the matching park `tables` files
under `Z:\GEE Derived`, then writes new CSVs with `center_x` and `center_y` beside
the originals. It does not rerun the model or need CUDA. Local Alabama
validation matched all 1,652 predictions across two parts.

### Run after inference completes

After these changes are pushed, pull the updated repository in PowerShell and
set the paths. Replace `<completed-run-id>` with the inference run folder name.

```powershell
cd "C:\Users\knobep01\Documents\Repositories\GreenSpace_CNN"
git pull
$Python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$ImageRoot = "Z:\GEE Derived"
$StatesRoot = "Z:\GEE Derived\inference_outputs\<completed-run-id>\states"
Test-Path scripts\append_patch_coordinates.py
Test-Path $Python
Test-Path $ImageRoot
Test-Path $StatesRoot
```

All four checks should return `True`. Then run these commands to process every
`USA_XX` state folder and all of its prediction parts:

```powershell
$StateDirs = @(Get-ChildItem $StatesRoot -Directory -Filter "USA_*")
if ($StateDirs.Count -eq 0) { throw "No state folders found" }
$StateDirs | ForEach-Object { & $Python scripts\append_patch_coordinates.py --image-root $ImageRoot --predictions-dir $_.FullName; if ($LASTEXITCODE -ne 0) { throw "Coordinate join failed for $($_.Name)" } }
```

The `&` tells PowerShell to run the Python executable saved in `$Python`; it
uses the project's `.venv` without activating it. Keep the script and all
options together on the same line, as in the earlier inference walkthrough.

### Confirm the result

Each state should report its matched row count and have one
`*_with_coordinates.csv` for each original prediction part. For Alabama, the
local reference is 1,000 rows in part 1 and 652 in part 2. The command stops
on a missing or ambiguous coordinate; inspect the reported park and image
before rerunning. Existing enriched files are protected unless `--overwrite`
is explicitly added.

The earlier [workstation inference walkthrough](0907_process_update.md)
contains the environment setup and PowerShell troubleshooting details.
