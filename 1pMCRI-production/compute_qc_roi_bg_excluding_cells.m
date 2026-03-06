%% Compute QC traces: ROI raw and local background raw (excluding all cell masks)
% This script computes, for all valid ROIs:
% 1) Raw intensity from the same spatial ROI
% 2) Local background raw intensity from surrounding area (~30 px),
%    excluding all detected cell masks
% 3) ROI-BG raw trace
%
% Input source (default):
%   run_dir/artifacts/output_*_moco_direct.mat
%   run_dir/artifacts/*_moco_direct.h5 (dataset: /mov, shape: t x x x y)
%
% Output:
%   run_dir/qc_raw_roi_localbg_excluding_cells_allrois.mat

clearvars;

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

run_dir = 'R:/code/1pMCRI-pipeline/demo_data/output/target_reach_250810-Ras2-GC#78';
roi_thresh_frac = 0.20;
bg_radius_px = 30;
bg_radius_step_px = 10;
bg_radius_max_px = 80;
chunk_frames = 2000;

if ~isfolder(run_dir)
    error('run_dir not found: %s', run_dir);
end

artifact_dir = fullfile(run_dir, 'artifacts');
if ~isfolder(artifact_dir)
    error('artifacts directory not found: %s', artifact_dir);
end

extract_mat = find_single_file(artifact_dir, 'output_*_moco_direct.mat');
movie_h5 = find_single_file(artifact_dir, '*_moco_direct.h5');
movie_dataset = '/mov';

fprintf('EXTRACT MAT: %s\n', extract_mat);
fprintf('Movie H5   : %s (%s)\n', movie_h5, movie_dataset);
fprintf('Chunk frames: %d\n', chunk_frames);

L = load(extract_mat, 'output');
if ~isfield(L, 'output')
    error('No ''output'' struct found in: %s', extract_mat);
end
output = L.output;

if ~isfield(output, 'spatial_weights') || ~isfield(output, 'temporal_weights')
    error('output.spatial_weights / output.temporal_weights is missing.');
end

S = output.spatial_weights;
[S2, h, w, n_cells] = spatial_to_sparse2d(S);

T = as_cells_by_time(output.temporal_weights, n_cells);
[n_cells_t, n_frames_t] = size(T);
if n_cells_t ~= n_cells
    error('Cell count mismatch: spatial=%d temporal=%d', n_cells, n_cells_t);
end

info = h5info(movie_h5, movie_dataset);
mov_size = info.Dataspace.Size;
if numel(mov_size) ~= 3
    error('Dataset %s must be 3D, got size=%s', movie_dataset, mat2str(mov_size));
end

[time_dim, dim_h, dim_w] = infer_movie_dims(mov_size, h, w);
mov_t = mov_size(time_dim);

n_frames = min(mov_t, n_frames_t);
if n_frames < 1
    error('No frames available after alignment.');
end

fprintf('Movie size: h=%d, w=%d, t=%d\n', h, w, n_frames);
fprintf('Cells: %d\n', n_cells);
fprintf('H5 dims inferred: size=%s, time_dim=%d, h_dim=%d, w_dim=%d\n', ...
    mat2str(mov_size), time_dim, dim_h, dim_w);

% Build ROI masks and centroids directly from sparse columns
fprintf('Phase 1/5: build ROI masks and centroids\n');
roi_masks = cell(n_cells, 1);
cx = nan(n_cells, 1, 'single');
cy = nan(n_cells, 1, 'single');

for k = 1:n_cells
    if mod(k, 1000) == 0 || k == n_cells
        fprintf('  ROI masks/centroids: %d / %d (%.1f%%)\n', k, n_cells, 100 * k / max(n_cells, 1));
    end
    [pix_idx, ~, ww] = find(S2(:, k));
    if isempty(pix_idx)
        continue;
    end
    keep_pos = ww > 0;
    if ~any(keep_pos)
        continue;
    end
    pix_idx = pix_idx(keep_pos);
    ww = single(ww(keep_pos));

    mmax = max(ww);
    if mmax <= 0
        continue;
    end
    keep_roi = ww >= (roi_thresh_frac * mmax);
    if ~any(keep_roi)
        continue;
    end
    pix_idx = pix_idx(keep_roi);
    ww = ww(keep_roi);
    roi_masks{k} = pix_idx;

    s = sum(ww);
    if s <= 0
        continue;
    end
    [yy, xx] = ind2sub([h, w], pix_idx);
    xx = single(xx);
    yy = single(yy);
    cx(k) = sum(xx .* ww) / s;
    cy(k) = sum(yy .* ww) / s;
end

roi_id = find(~cellfun(@isempty, roi_masks) & isfinite(cx) & isfinite(cy));
n_roi = numel(roi_id);
if n_roi == 0
    error('No valid ROI found.');
end
fprintf('Valid ROI count: %d\n', n_roi);

% Build all-cell union mask (for background exclusion)
fprintf('Phase 2/5: build all-cell union mask\n');
all_cell_union = false(h, w);
for i = 1:n_roi
    all_cell_union(roi_masks{roi_id(i)}) = true;
end
all_cell_union_lin = all_cell_union(:);

% Build local background masks with radius expansion
fprintf('Phase 3/5: build local background masks\n');
bg_masks = cell(n_roi, 1);
bg_pixel_count = zeros(n_roi, 1);
bg_radius_used = zeros(n_roi, 1);
invalid_bg = false(n_roi, 1);

for i = 1:n_roi
    if mod(i, 1000) == 0 || i == n_roi
        fprintf('  Local BG masks: %d / %d (%.1f%%)\n', i, n_roi, 100 * i / max(n_roi, 1));
    end
    rid = roi_id(i);
    found = false;
    for rad = bg_radius_px:bg_radius_step_px:bg_radius_max_px
        idx = local_bg_indices(cx(rid), cy(rid), rad, h, w, all_cell_union_lin);
        if ~isempty(idx)
            bg_masks{i} = idx;
            bg_pixel_count(i) = numel(idx);
            bg_radius_used(i) = rad;
            found = true;
            break;
        end
    end
    if ~found
        invalid_bg(i) = true;
        bg_masks{i} = [];
        bg_pixel_count(i) = 0;
        bg_radius_used(i) = bg_radius_max_px;
    end
end

fprintf('Invalid BG ROI count: %d / %d\n', nnz(invalid_bg), n_roi);

% Build sparse selectors for fast mean extraction
fprintf('Phase 4/5: build sparse selectors\n');
A_roi = build_sparse_selector(roi_masks, roi_id, h * w, n_roi);
cnt_roi = full(sum(A_roi, 1))';
fprintf('  ROI selector nnz: %d\n', nnz(A_roi));

A_bg = build_sparse_selector(bg_masks, [], h * w, n_roi);
cnt_bg = full(sum(A_bg, 1))';
fprintf('  BG selector nnz : %d\n', nnz(A_bg));

trace_raw_roi = zeros(n_roi, n_frames, 'single');
trace_raw_bg = nan(n_roi, n_frames, 'single');

fprintf('Phase 5/5: read movie chunks and compute traces\n');
n_chunks = ceil(n_frames / chunk_frames);
for c = 1:n_chunks
    f_begin = (c - 1) * chunk_frames + 1;
    f_end = min(c * chunk_frames, n_frames);
    n_this = f_end - f_begin + 1;
    fprintf('  Chunk %d/%d (frames %d-%d, %.1f%%)\n', ...
        c, n_chunks, f_begin, f_end, 100 * c / max(n_chunks, 1));

    start = ones(1, 3);
    count = mov_size;
    start(time_dim) = f_begin;
    count(time_dim) = n_this;

    M = h5read(movie_h5, movie_dataset, start, count);
    M = permute(single(M), [dim_h, dim_w, time_dim]); % -> [h, w, t]
    Mr = reshape(M, h * w, n_this);

    sum_roi = double(A_roi') * double(Mr);
    trace_raw_roi(:, f_begin:f_end) = single(bsxfun(@rdivide, sum_roi, max(cnt_roi, 1)));

    valid_bg = cnt_bg > 0;
    if any(valid_bg)
        sum_bg = double(A_bg(:, valid_bg)') * double(Mr);
        bg_block = single(bsxfun(@rdivide, sum_bg, cnt_bg(valid_bg)));
        trace_raw_bg(valid_bg, f_begin:f_end) = bg_block;
    end
    fprintf('    completed chunk %d/%d\n', c, n_chunks);
end

trace_raw_minus_bg = trace_raw_roi - trace_raw_bg;
trace_temporal_weights = single(T(roi_id, 1:n_frames));

params = struct();
params.roi_thresh_frac = roi_thresh_frac;
params.bg_radius_px = bg_radius_px;
params.bg_radius_step_px = bg_radius_step_px;
params.bg_radius_max_px = bg_radius_max_px;
params.chunk_frames = chunk_frames;
params.movie_dataset = movie_dataset;

source_run_dir = run_dir;
source_extract_mat = extract_mat;
source_movie_h5 = movie_h5;

out_mat = fullfile(run_dir, 'qc_raw_roi_localbg_excluding_cells_allrois.mat');
fprintf('Saving output MAT: %s\n', out_mat);
save(out_mat, ...
    'trace_raw_roi', 'trace_raw_bg', 'trace_raw_minus_bg', 'trace_temporal_weights', ...
    'roi_id', 'bg_pixel_count', 'bg_radius_used', 'invalid_bg', ...
    'source_run_dir', 'source_extract_mat', 'source_movie_h5', 'params', '-v7.3');

fprintf('Saved: %s\n', out_mat);

%% ---- local functions ----
function p = find_single_file(dir_path, pattern)
D = dir(fullfile(dir_path, pattern));
if isempty(D)
    error('No file matched: %s/%s', dir_path, pattern);
end
if numel(D) > 1
    names = strjoin({D.name}, ', ');
    error('Multiple files matched (%s): %s', pattern, names);
end
p = fullfile(D(1).folder, D(1).name);
end

function T = as_cells_by_time(Tin, n_cells)
Tin = single(Tin);
if size(Tin, 2) == n_cells
    T = Tin';
elseif size(Tin, 1) == n_cells
    T = Tin;
else
    error('Could not infer temporal_weights orientation.');
end
end

function [S2, h, w, n_cells] = spatial_to_sparse2d(S)
if isa(S, 'ndSparse')
    sz = size(S);
    if numel(sz) ~= 3
        error('ndSparse spatial_weights must be 3D-like. Got size=%s', mat2str(sz));
    end
    h = sz(1);
    w = sz(2);
    n_cells = sz(3);
    S2 = sparse2d(S); % [h*w x n_cells]
    if ~issparse(S2)
        S2 = sparse(S2);
    end
else
    S = single(S);
    if ndims(S) ~= 3
        error('spatial_weights must be 3D or ndSparse.');
    end
    [h, w, n_cells] = size(S);
    S2 = sparse(reshape(S, h * w, n_cells));
end
end

function idx = local_bg_indices(cx, cy, rad, h, w, all_cell_union_lin)
xmin = max(1, floor(double(cx) - rad));
xmax = min(w, ceil(double(cx) + rad));
ymin = max(1, floor(double(cy) - rad));
ymax = min(h, ceil(double(cy) + rad));

[Yb, Xb] = ndgrid(ymin:ymax, xmin:xmax);
keep = ((double(Xb) - double(cx)).^2 + (double(Yb) - double(cy)).^2) <= double(rad)^2;
if ~any(keep(:))
    idx = [];
    return;
end
idx = sub2ind([h w], Yb(keep), Xb(keep));
idx = idx(~all_cell_union_lin(idx));
end

function A = build_sparse_selector(mask_cell, roi_ids, n_rows, n_cols)
if isempty(roi_ids)
    src_ids = (1:n_cols)';
else
    src_ids = roi_ids(:);
    if numel(src_ids) ~= n_cols
        error('ROI id count mismatch: expected %d, got %d', n_cols, numel(src_ids));
    end
end

counts = zeros(n_cols, 1);
for col = 1:n_cols
    counts(col) = numel(mask_cell{src_ids(col)});
end

n_total = sum(counts);
ii = zeros(n_total, 1);
jj = zeros(n_total, 1);
write_pos = 1;
for col = 1:n_cols
    idx = mask_cell{src_ids(col)};
    n_this = numel(idx);
    if n_this == 0
        continue;
    end
    span = write_pos:(write_pos + n_this - 1);
    ii(span) = idx(:);
    jj(span) = col;
    write_pos = write_pos + n_this;
end

if write_pos <= n_total
    ii(write_pos:end) = [];
    jj(write_pos:end) = [];
end
A = sparse(ii, jj, 1, n_rows, n_cols);
end

function [time_dim, dim_h, dim_w] = infer_movie_dims(mov_size, h, w)
for td = 1:3
    sd = setdiff(1:3, td, 'stable');
    a = mov_size(sd(1));
    b = mov_size(sd(2));
    if a == h && b == w
        time_dim = td;
        dim_h = sd(1);
        dim_w = sd(2);
        return;
    end
    if a == w && b == h
        time_dim = td;
        dim_h = sd(2);
        dim_w = sd(1);
        return;
    end
end

error(['Could not infer movie dimensions from H5 size and spatial_weights. ', ...
    'H5 size=%s, spatial [h,w]=[%d,%d].'], mat2str(mov_size), h, w);
end
