# Process Update — 2026-09-07

## Windows Workstation Inference Walkthrough — Part 1

### Purpose

Confirm whether the GreenSpace Python environment can use the workstation's
NVIDIA T1000 for CUDA inference. Do not start the full dataset yet.

Known workstation hardware:

- two Intel Xeon Gold 5315Y CPUs: 16 physical cores and 32 logical processors total;
- 511.66 GB system RAM;
- NVIDIA T1000 with 4 GB dedicated GPU memory.

The T1000 supports CUDA, but the driver, remote Windows session, and installed
PyTorch environment must also expose it.

## Step 1 — Open PowerShell

Connect to the workstation, open the Windows Start menu, search for
`PowerShell`, and open it.

## Step 2 — Check the NVIDIA driver

Run:

```powershell
nvidia-smi
```

Expected result: a table listing `NVIDIA T1000`, an NVIDIA driver version, GPU
memory, and a CUDA version.

- If the table appears, continue to Step 3.
- If PowerShell says the command is not recognized or returns an error, copy or
  screenshot the complete message and stop. Do not start full inference.

The CUDA version shown by `nvidia-smi` is the version supported by the driver.
It does not prove that the project’s PyTorch installation can use CUDA.

## Step 3 — Enter the project environment

Go to the cloned repository, replacing the example path with its real location:

```powershell
cd "C:\path\to\GreenSpace_CNN"
```

Activate the environment created from the README:

```powershell
.\.venv\Scripts\Activate.ps1
```

Check Python:

```powershell
python --version
```

The project expects Python 3.11 or 3.12; Python 3.11 is the tested version.

## Step 4 — Check CUDA inside PyTorch

Copy and run this one command:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

### Successful result

The important lines should resemble:

```text
CUDA available: True
GPU: NVIDIA T1000
```

This means the project environment can use CUDA.

### Unsuccessful result

- `No module named torch`: the repository dependencies are not installed in the
  active environment. Follow the README installation step, then repeat Step 4.
- `CUDA available: False`: do not guess or install additional CUDA software yet.
  Save the complete output from Steps 2 and 4 for troubleshooting.

## Step 5 — Record the result

Save or send:

1. the complete `nvidia-smi` output;
2. the complete PyTorch check output;
3. the Python version;
4. confirmation that the commands were run inside the repository's `.venv`.

## Initial inference settings

If Step 4 reports `CUDA available: True`, begin the later smoke test with:

```text
--device cuda --batch-size 1 --num-workers 2 --pin-memory
```

The T1000 has only 4 GB of GPU memory, so batch size 1 is the safe starting
point. Increase it only after a small inference test succeeds.

If CUDA is unavailable and CPU inference is required, begin with:

```text
--device cpu --batch-size 1 --num-workers 2 --no-pin-memory
```

`--batch-size` controls how many images the model processes together. It does
not control the number of rows written to each output CSV.

## Deferred until the server layout is known

The next walkthrough section will define:

- the full image-root path and state subdirectory structure;
- recursive state-by-state discovery;
- per-state or chunked CSV outputs;
- restart and completion behavior;
- the exact validation, smoke-test, and full inference commands.

No directory-iteration or inference-script changes are included in Part 1.
