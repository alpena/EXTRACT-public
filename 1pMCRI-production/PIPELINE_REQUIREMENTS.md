# Pipeline Requirements (TIFF -> H5 -> EXTRACT)

This document summarizes environment requirements and machine-specific settings for:

- `1pMCRI-production/tiff_to_h5_fast.py`
- `1pMCRI-production/run_1pMCRI_pipeline.m`
- `1pMCRI-production/test_1pMCRI_minimal.m`

## 1. Required software

- MATLAB (with toolboxes required by EXTRACT)
- EXTRACT repository code on local disk
- Python (Anaconda/Miniconda recommended)
- Python packages:
  - `numpy`
  - `tifffile`
  - `h5py`

Install Python packages:

```bash
pip install numpy tifffile h5py
```

or with conda:

```bash
conda install numpy h5py
pip install tifffile
```

## 2. Fast drive requirement

The pipeline is designed to use a fast NVMe cache location:

- Windows: `E:\EXTRACT-cache`
- Linux/Ubuntu: `/mnt/nvme/EXTRACT-cache`

Both TIFF and H5 are processed on this fast drive to reduce I/O bottlenecks.

## 3. Python executable selection

`1pMCRI-production/run_1pMCRI_pipeline.m` supports:

- Auto-detect mode: `python_exe = ''`
- Manual override: set full path (recommended if auto-detect fails)

Examples:

- Windows: `C:\Users\<user>\anaconda3\python.exe`
- Ubuntu: `/home/<user>/miniconda3/bin/python`

## 4. Data format assumptions

- Input TIFF is expected to be `uint16`.
- H5 output dataset name is `/mov`.
- The converter supports:
  - Standard multipage TIFF
  - ImageJ stack TIFF

## 5. Performance-related defaults

- Python converter chunk target: `chunk_t` (recommended start: `96-256`)
- For XY-partitioned EXTRACT reads, set `chunk_x`/`chunk_y` (recommended start: `256/256`)
- If RAM is tight, reduce `chunk_t`.
- If RAM is abundant, increase `chunk_t` (and optionally `chunk_x/y`).
- HDF5 chunk limit (< 4GB per chunk) is handled automatically.

## 6. Typical execution flow

1. Run `run_1pMCRI_pipeline(input_tiff, opts)`.
2. Script copies TIFF to fast drive cache.
3. Script runs Python TIFF->H5 conversion.
4. Script runs EXTRACT from fast-drive H5 reference.
5. Script saves output `.mat` with `output`, `config_used`, `meta`.

## 7. Troubleshooting

- `python not found`:
  - Set `python_exe` manually in MATLAB script.
- `ModuleNotFoundError`:
  - Install missing Python packages in the same interpreter.
- Slow conversion:
  - Confirm source and destination are both on fast drive.
  - Increase `chunk_t` if RAM allows.
- HDF5 chunk-size errors:
  - Do not force extremely high chunk sizes; the script auto-clamps.

## 8. Notes for reproducibility

- Record machine info (CPU, RAM, NVMe model).
- Record Python version and package versions.
- Keep `meta` fields from output `.mat` for run provenance.
