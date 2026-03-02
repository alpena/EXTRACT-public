# 1pMCRI Production Pipeline

Environment and machine requirements for the TIFF->H5->EXTRACT pipeline:
- `1pMCRI-production/PIPELINE_REQUIREMENTS.md`

## 1pMCRI minimal smoke test

1. Edit `1pMCRI-production/test_1pMCRI_minimal.m` and set `h5_path` + `dataset_name` for your movie.
2. (Optional) Adjust `quick_n_frames`, `avg_cell_radius`, and `gpu_id`.
3. Run `test_1pMCRI_minimal` from MATLAB.

The script runs EXTRACT with single-GPU settings and saves:
- `output`
- `config_used`
- `meta`

Default output file: `1pMCRI-production/test_1pMCRI_output_full.mat`.
