# Model artifacts

Model checkpoints and run outputs are intentionally excluded from Git because
they are large generated artifacts. A clean clone therefore does not contain a
`models/runs/` payload.

For the external-review trial, download
`PyTorch_20260719_full_windows.zip` from the
[GreenSpace_CNN external-review files](https://drive.google.com/drive/folders/1tlsfN30WkAFBkwEmtTJmGt-uA6KbKXZA)
and extract it under `models/runs/` without renaming the run directory or its
files:

```text
models/
└── runs/
    └── PyTorch_20260719_full_windows/
        ├── best_mcmae_PyTorch_20260719_full_windows.pt
        ├── model_config_PyTorch_20260719_full_windows.json
        └── thresholds_best_mcmae.csv
```

All three files form one portable inference bundle. The loader derives the
model-configuration and threshold filenames from the run tag and checkpoint
variant, so moving or renaming individual files can make the bundle invalid.

Reference SHA-256 checksums:

```text
d9436d4a223f50f83114a207af3c5bc7a3bea213db135fcdbab612b0a3b0d4fa  best_mcmae_PyTorch_20260719_full_windows.pt
aac49f7498a7706063e6cb257d79eb568eace71ef056164ec40912f19451652a  model_config_PyTorch_20260719_full_windows.json
b7e8e8f6b63fab6fcee119493734aff5ae90e89b18f06888f2bdc430fce7313e  thresholds_best_mcmae.csv
```

The checkpoint is approximately 996 MiB. Only load the trusted,
checksum-verified project checkpoint; PyTorch checkpoints may execute unsafe
deserialization behavior when obtained from an untrusted source.
