# 1pMCRI Production Pipeline (H5-first)

This folder is managed as an H5-first production pipeline for EXTRACT.

- Primary input: masknmf registration output H5
- Primary runner: `1pMCRI-production/run_1pMCRI_pipeline.m`
- Requirements: `1pMCRI-production/PIPELINE_REQUIREMENTS.md`

## Standard flow (recommended)

1. Input masknmf H5 (`/motion_corrected`)
2. Read source H5 directly (no input copy)
3. Convert to EXTRACT-optimized H5 (`/mov`)
4. Run EXTRACT
5. Save output MAT (`output`, `config_used`, `meta`)

## Quick start (H5 input)

```matlab
opts = struct();
opts.input_h5 = 'E:\EXTRACT-cache\moco_results_extract.h5';
opts.python_exe = 'C:\Users\<user>\anaconda3\python.exe'; % optional
opts.orientation_fix = 'none'; % 'none' | 'transpose_xy'
R = run_1pMCRI_pipeline('', opts);
```

Notes:
- masknmf input dataset is fixed to `/motion_corrected`.
- EXTRACT always reads `/mov` from the optimized H5.
- If orientation looks transposed, set `opts.orientation_fix = 'transpose_xy'`.

## Key H5 options

- `opts.input_h5`: input masknmf H5 path
- `opts.h5_skip_if_exists` (default `true`)
- `opts.h5_chunk_t/x/y` (default `256/256/256`)
- `opts.h5_compression` (default `0`)
- `opts.orientation_fix` (default `none`)

## TIFF compatibility mode (optional)

TIFF input remains supported for backward compatibility:

```matlab
R = run_1pMCRI_pipeline('R:\data\movie.tif', struct());
```

This path uses TIFF->H5 conversion before EXTRACT.
