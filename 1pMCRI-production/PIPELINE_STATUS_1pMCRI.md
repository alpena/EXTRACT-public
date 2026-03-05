# 1pMCRI Pipeline Status (Work in Progress)

This document summarizes the currently intended analysis pipeline for 1pMCRI data.
It is a **WIP status note**, not a finalized SOP.

## Intended End-to-End Flow
1. `.dcimg` -> **masknmf** for non-rigid registration
2. Registered movie -> **EXTRACT** for trace extraction
3. Extracted traces -> **Cascade** for spike inference

## Stage 1: non-rigid registration (masknmf)
### Goal
- Convert raw Hamamatsu `.dcimg` recordings into motion-corrected (non-rigid registered) movies suitable for source extraction.

### Current status
- Treated as an upstream preprocessing step.
- Registered outputs are being consumed downstream (e.g., cropped TIFF/H5 inputs for EXTRACT).

### Typical output artifact
- Registered/cropped movie files (example naming pattern):
  - `*_reg*.tif`
  - `*_reg*.mat`

## Stage 2: trace extraction (EXTRACT)
### Goal
- Run EXTRACT on registered movies and obtain spatial footprints + temporal traces.

### Current scripts in this folder
- `run_1pMCRI_pipeline.m`
- `run_target_reach_250810.m`
- `test_1pMCRI_minimal.m`
- `visualize_1pMCRI_full.m`

### Current status
- Operational for production runs.
- `trace_output_option` variants in use:
  - `no_constraint`: closest to raw overlap-separated traces.
  - `baseline_adjusted`, `nonneg`: non-negative and denoised outputs.

### Example EXTRACT outputs
- `output_250206-UK6-1-F=4_power=5mW_reg_crop_[no_constraint].mat`
- `output_250206-UK6-1-F=4_power=5mW_reg_crop_[baseline_adjusted].mat`
- `output_250206-UK6-1-F=4_power=5mW_reg_crop_[nonneg].mat`

## Stage 3: spike inference (Cascade)
### Goal
- Convert EXTRACT temporal traces into inferred spike probabilities/rates.

### Current status
- Implemented with both repositories:
  - `R:\code\Cascade` (TensorFlow-based CASCADE)
  - `R:\code\CascadeTorch` (PyTorch-based CascadeTorch)
- 1pMCRI-specific scripts/notebooks have been added for:
  - subset-trace demos
  - all-cell inference

### Expected input to Cascade
- EXTRACT `output.temporal_weights`
- Converted to shape `(neurons, time)` before prediction.

## Known Open Items
- Formalize Stage 1 (`.oir` -> masknmf) command-level reproducibility notes.
- Finalize recommended model-selection policy in Cascade by frame rate/noise regime.
- Define standardized output schema and destination for downstream behavioral analyses.

## Notes
- This file is intentionally concise and meant to be updated as the pipeline stabilizes.
