# Pipeline Requirements (H5-first)

This document defines requirements for the 1pMCRI H5 production paths:

- Input: masknmf output H5 (`/motion_corrected`)
- Convert: `1pMCRI-production/h5_to_h5_fast.py`
- Run: `1pMCRI-production/run_1pMCRI_pipeline.m`
- Output: EXTRACT result MAT

TIFF support is optional/legacy and described at the end.

## 1. Required software

- MATLAB (toolboxes required by EXTRACT)
- EXTRACT repository on local disk
- Python (Anaconda/Miniconda recommended)
- Python packages:
  - `numpy`
  - `h5py`

Install:

```bash
pip install numpy h5py
```

## 2. Fast drive cache

Use a local NVMe path for conversion and EXTRACT I/O:

- Windows: `E:\EXTRACT-cache`
- Ubuntu: `/mnt/nvme/EXTRACT-cache`

## 3. Python executable

`run_1pMCRI_pipeline.m` accepts:

- Auto detect: `opts.python_exe = ''`
- Manual path (recommended):
  - Windows: `C:\Users\<user>\anaconda3\python.exe`
  - Ubuntu: `/home/<user>/miniconda3/bin/python`

## 4. Data format contract

Standard converted flow:
- Input H5 dataset name is fixed: `/motion_corrected`
- Output optimized dataset name is fixed: `/mov`
- EXTRACT reads `/mov`

Direct preoptimized flow:
- Input H5 is already EXTRACT-ready and contains `/mov`
- `opts.input_h5_preoptimized=true` is required to skip conversion

Common constraints:
- Input movie must be 3D
- Pixel dtype is typically `uint16` (recommended)

## 5. Standard execution flow

1. `run_1pMCRI_pipeline('', opts)` with `opts.input_h5`
2. Read input H5 from its original location
3. Convert `/motion_corrected` -> optimized `/mov`
4. Run EXTRACT from optimized H5
5. Save MAT with `output`, `config_used`, `meta`

## 5b. Direct preoptimized execution flow

1. `run_1pMCRI_pipeline('', opts)` with:
- `opts.input_h5 = <extract-ready.h5>`
- `opts.input_h5_preoptimized = true`
2. Pipeline validates `<input_h5>:/mov` as 3D
3. Pipeline skips `/motion_corrected -> /mov` conversion
4. EXTRACT runs directly from input H5
5. Save MAT with `output`, `config_used`, `meta`

## 6. H5 options

- `opts.input_h5`: masknmf H5 path (required for H5 flow)
- `opts.h5_skip_if_exists` (default `true`)
- `opts.h5_chunk_t` (default `256`)
- `opts.h5_chunk_x` (default `256`)
- `opts.h5_chunk_y` (default `256`)
- `opts.h5_compression` (default `0`)
- `opts.orientation_fix` (default `transpose_xy`)
  - `none`
  - `transpose_xy`
- `opts.input_h5_preoptimized` (default `false`)

## 7. Troubleshooting

- `python not found`:
  - Set `opts.python_exe` explicitly.
- `Input dataset not found: /motion_corrected`:
  - Check the masknmf H5 dataset name and path.
- `input_h5_preoptimized=true` but `/mov` not found:
  - Use conversion mode (`input_h5_preoptimized=false`) or export direct `/mov` from masknmf.
- Orientation mismatch:
  - Default is `transpose_xy`. If orientation is already correct, set `opts.orientation_fix = 'none'`.
- Conversion speed is low:
  - Confirm both input/output are on fast drive.
  - Tune `h5_chunk_t/x/y`.

## 8. Reproducibility

- Keep `meta` from output MAT.
- Record:
  - GPU/CPU/RAM
  - Python version
  - `numpy` / `h5py` versions
  - key opts (`orientation_fix`, chunk params)

## 9. Optional TIFF compatibility

If TIFF is used instead of H5, additional converter dependencies apply:

- `tifffile` package
- `1pMCRI-production/tiff_to_h5_fast.py`

Install:

```bash
pip install tifffile
```
