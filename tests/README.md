# Tests

## 1pMCRI minimal smoke test

1. Edit `tests/test_1pMCRI_minimal.m` and set `tiff_path` to your multi-page 1pMCRI TIFF.
2. (Optional) Adjust `quick_n_frames`, `avg_cell_radius`, and `gpu_id`.
3. Run `test_1pMCRI_minimal` from MATLAB.

The script runs EXTRACT with single-GPU settings and saves:
- `output`
- `config_used`
- `meta`

Default output file: `tests/test_1pMCRI_output.mat`.
