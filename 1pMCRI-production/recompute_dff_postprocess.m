function result = recompute_dff_postprocess(result_path, opts)
% Recompute post-hoc dF/F from EXTRACT output and save as canonical HDF5 artifact.
%
% Inputs
%   result_path : path to EXTRACT output MAT that contains struct `output`
%   opts        : struct with optional fields
%       frame_rate_hz              (default 30)
%       half_window_sec            (default 60)
%       percentile_q               (default 8)
%       baseline_mode              (default 'moving_percentile')
%                                   'moving_percentile' | 'global_percentile' | 'extract_temporal'
%       moving_percentile_impl     (default 'decimated')
%                                   'exact' | 'decimated'
%       moving_decimate_factor     (default 10)
%       moving_percentile_use_parfor (default true)
%       save_output                (default true)
%       save_path                  (default auto .h5 path)
%       plot_n_cells               (default 0; 0 disables plot)
%       rng_seed                   (default 1)
%       save_figure                (default false)
%       figure_path                (default auto when save_figure=true)
%       nan_inf_fail_ratio         (default 1e-3)
%
% Output
%   result: struct with fields dff, output_path, fig_path, meta

if nargin < 1 || isempty(result_path)
    error('result_path is required.');
end
if nargin < 2 || isempty(opts)
    opts = struct();
end

frame_rate_hz = get_opt(opts, 'frame_rate_hz', 30);
half_window_sec = get_opt(opts, 'half_window_sec', 60);
percentile_q = get_opt(opts, 'percentile_q', 8);
baseline_mode = lower(strtrim(char(get_opt(opts, 'baseline_mode', 'moving_percentile'))));
moving_percentile_impl = lower(strtrim(char(get_opt(opts, 'moving_percentile_impl', 'decimated'))));
moving_decimate_factor = get_opt(opts, 'moving_decimate_factor', 10);
moving_percentile_use_parfor = logical(get_opt(opts, 'moving_percentile_use_parfor', true));
save_output = logical(get_opt(opts, 'save_output', true));
plot_n_cells = get_opt(opts, 'plot_n_cells', 0);
rng_seed = get_opt(opts, 'rng_seed', 1);
save_figure = logical(get_opt(opts, 'save_figure', false));
nan_inf_fail_ratio = get_opt(opts, 'nan_inf_fail_ratio', 1e-3);

if ~isfile(result_path)
    error('Result file not found: %s', result_path);
end

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));

F0_cell = [];
F_est = [];
F0_t = [];
win = NaN;

if strcmp(baseline_mode, 'extract_temporal')
    fprintf(['Loading temporal_weights directly from MAT/HDF5 for extract_temporal mode ', ...
        '(avoids loading unrelated EXTRACT summary fields)...\n']);
    T = load_temporal_weights_only(result_path);
    [n_cells, n_frames] = size(T);
else
    L = load(result_path, 'output');
    if ~isfield(L, 'output')
        error('No output struct found in: %s', result_path);
    end
    output = L.output;

    required_fields = {'spatial_weights', 'temporal_weights', 'info'};
    for i = 1:numel(required_fields)
        if ~isfield(output, required_fields{i})
            error('output.%s is missing.', required_fields{i});
        end
    end
    if ~isfield(output.info, 'F_per_pixel') || isempty(output.info.F_per_pixel)
        error('output.info.F_per_pixel is missing or empty.');
    end

    fprintf('Loading spatial weights and temporal weights...\n');
    [S2, n_cells] = spatial_to_2d(output.spatial_weights);
    T = temporal_to_cells_by_time(output.temporal_weights, n_cells); % cells x frames
    [n_cells_t, n_frames] = size(T);
    if n_cells_t ~= n_cells
        error('Cell count mismatch: spatial=%d, temporal=%d', n_cells, n_cells_t);
    end

    Fpix = single(output.info.F_per_pixel(:)); % [h*w x 1]
    if size(S2, 1) ~= numel(Fpix)
        error('Spatial size mismatch: numel(F_per_pixel)=%d, size(S2,1)=%d', ...
            numel(Fpix), size(S2, 1));
    end

    fprintf('Computing per-cell baseline anchor F0_cell from F_per_pixel...\n');
    F0_cell = compute_f0_cell(S2, Fpix); % [n_cells x 1]
    F_est = bsxfun(@plus, T, F0_cell);   % [cells x frames]
end

switch baseline_mode
    case 'extract_temporal'
        fprintf('Using EXTRACT temporal_weights directly as dff (no recomputation).\n');
        dff = single(T);
    case 'global_percentile'
        fprintf('Computing global percentile baseline: q=%g over full trace\n', percentile_q);
        F0_const = single(prctile(F_est, percentile_q, 2));
        F0_t = bsxfun(@times, F0_const, ones(1, n_frames, 'single'));
    case 'moving_percentile'
        win = max(5, round(2 * half_window_sec * frame_rate_hz) + 1);
        if mod(win, 2) == 0
            win = win + 1;
        end
        fprintf(['Computing moving percentile baseline: q=%g, fs=%.3f Hz, ', ...
            'window=%d frames (~%.1f sec), impl=%s\n'], ...
            percentile_q, frame_rate_hz, win, win / frame_rate_hz, moving_percentile_impl);
        t_mp = tic;
        switch moving_percentile_impl
            case 'exact'
                F0_t = moving_percentile_exact(F_est, percentile_q, win, moving_percentile_use_parfor);
            case 'decimated'
                [F0_t, decim_info] = moving_percentile_decimated( ...
                    F_est, percentile_q, win, moving_decimate_factor, moving_percentile_use_parfor);
                fprintf(['Decimated moving percentile: factor=%d, frames_ds=%d, ', ...
                    'window_ds=%d\n'], decim_info.decimate_factor, ...
                    decim_info.n_frames_ds, decim_info.window_ds);
            otherwise
                error('Unsupported moving_percentile_impl: %s', moving_percentile_impl);
        end
        fprintf('Moving percentile finished in %.2f sec\n', toc(t_mp));
    otherwise
        error('Unsupported baseline_mode: %s', baseline_mode);
end

if ~strcmp(baseline_mode, 'extract_temporal')
    dff = (F_est - F0_t) ./ max(F0_t, eps('single'));
end
bad_ratio = mean(~isfinite(dff(:)));
if bad_ratio > nan_inf_fail_ratio
    error('dff contains too many non-finite values: ratio=%.6f > %.6f', bad_ratio, nan_inf_fail_ratio);
end
if size(dff, 1) ~= n_cells
    error('dff shape mismatch: expected N=%d, got %d', n_cells, size(dff, 1));
end

fprintf('Done: dFF size = [%d cells x %d frames]\n', size(dff, 1), size(dff, 2));

switch baseline_mode
    case 'moving_percentile'
        suffix = 'mp';
    case 'global_percentile'
        suffix = 'gp';
    case 'extract_temporal'
        suffix = 'tw';
    otherwise
        suffix = 'dff';
end

[out_dir, out_name, ~] = fileparts(result_path);
default_out_h5 = fullfile(out_dir, sprintf('%s_dff_%s.h5', out_name, suffix));
out_h5 = get_opt(opts, 'save_path', default_out_h5);
if ~endsWith(lower(char(out_h5)), '.h5')
    error('postprocess save_path must end with .h5: %s', out_h5);
end

fig_png = '';
if plot_n_cells > 0 || save_figure
    rng(rng_seed);
    ids = randperm(n_cells, min(plot_n_cells, n_cells));
    if isempty(ids)
        ids = 1:min(5, n_cells);
    end
    f = figure('Color', 'w', 'Position', [60 60 1800 1100], 'Visible', 'off');
    ncols = 2;
    if strcmp(baseline_mode, 'extract_temporal')
        ncols = 1;
    end
    tlo = tiledlayout(numel(ids), ncols, 'TileSpacing', 'compact', 'Padding', 'compact');
    if strcmp(baseline_mode, 'moving_percentile')
        title(tlo, sprintf('Post-hoc dF/F | mode=moving q=%g, +-%.0f sec', ...
            percentile_q, half_window_sec));
    elseif strcmp(baseline_mode, 'extract_temporal')
        title(tlo, 'Post-hoc dF/F | mode=extract_temporal (raw temporal weights)');
    else
        title(tlo, sprintf('Post-hoc dF/F | mode=global q=%g', percentile_q));
    end
    for r = 1:numel(ids)
        cid = ids(r);
        if ~strcmp(baseline_mode, 'extract_temporal')
            nexttile((r - 1) * 2 + 1);
            hold on;
            plot(F_est(cid, :), 'Color', [0.35 0.35 0.35], 'LineWidth', 0.8);
            plot(F0_t(cid, :), 'Color', [0.85 0.2 0.2], 'LineWidth', 1.0);
            hold off;
            grid on;
            xlabel('Frame');
            ylabel('a.u.');
            title(sprintf('Cell %d: F_{est} and F0', cid));
            tile_idx = (r - 1) * 2 + 2;
        else
            tile_idx = r;
        end
        nexttile(tile_idx);
        plot(dff(cid, :), 'Color', [0.1 0.45 0.9], 'LineWidth', 0.8);
        grid on;
        xlabel('Frame');
        if strcmp(baseline_mode, 'extract_temporal')
            ylabel('a.u.');
            title(sprintf('Cell %d: temporal\\_weights', cid));
        else
            ylabel('\DeltaF/F');
            title(sprintf('Cell %d: dF/F', cid));
        end
    end

    if strcmp(baseline_mode, 'moving_percentile')
        default_fig_png = fullfile(fileparts(result_path), 'recomputed_dff_moving_percentile.png');
    elseif strcmp(baseline_mode, 'extract_temporal')
        default_fig_png = fullfile(fileparts(result_path), 'recomputed_dff_extract_temporal.png');
    else
        default_fig_png = fullfile(fileparts(result_path), 'recomputed_dff_global_percentile.png');
    end
    fig_png = get_opt(opts, 'figure_path', default_fig_png);
    if save_figure || plot_n_cells > 0
        exportgraphics(f, fig_png, 'Resolution', 180);
        fprintf('Saved figure: %s\n', fig_png);
    end
    close(f);
end

meta = struct();
meta.source_result_path = result_path;
meta.frame_rate_hz = frame_rate_hz;
meta.half_window_sec = half_window_sec;
meta.percentile_q = percentile_q;
meta.window_frames = win;
meta.baseline_mode = baseline_mode;
meta.use_extract_temporal_direct = strcmp(baseline_mode, 'extract_temporal');
meta.moving_percentile_impl = moving_percentile_impl;
meta.moving_decimate_factor = moving_decimate_factor;
meta.nan_inf_fail_ratio = nan_inf_fail_ratio;
meta.bad_ratio = bad_ratio;
meta.n_cells = size(dff, 1);
meta.n_frames = size(dff, 2);
meta.orientation = 'cells_by_frames';
meta.output_format = 'postprocess_h5_v1';
meta.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));

if save_output
    write_postprocess_h5(out_h5, dff, F_est, F0_t, F0_cell, meta);
    fprintf('Saved dF/F HDF5: %s\n', out_h5);
end

result = struct();
result.dff = dff;
result.output_path = out_h5;
result.fig_path = fig_png;
result.meta = meta;
end

%% ---- Local helpers ----
function v = get_opt(s, key, default_v)
if isstruct(s) && isfield(s, key) && ~isempty(s.(key))
    v = s.(key);
else
    v = default_v;
end
end

function [S2, n_cells] = spatial_to_2d(S)
if isa(S, 'ndSparse')
    S2 = sparse2d(S);
    n_cells = size(S2, 2);
else
    if ndims(S) ~= 3
        error('spatial_weights must be 3D or ndSparse.');
    end
    [h, w, n_cells] = size(S);
    S2 = reshape(single(S), h * w, n_cells);
end
end

function T = temporal_to_cells_by_time(Tin, n_cells)
Tin = single(Tin);
if size(Tin, 2) == n_cells
    T = Tin';
elseif size(Tin, 1) == n_cells
    T = Tin;
else
    error('Could not infer temporal_weights orientation.');
end
end

function T = load_temporal_weights_only(result_path)
try
    L = load(result_path, 'output');
    if isfield(L, 'output') && isfield(L.output, 'temporal_weights')
        Tin = single(L.output.temporal_weights);
        n_cells_guess = max(size(Tin));
        T = temporal_to_cells_by_time(Tin, n_cells_guess);
        return;
    end
catch ME
    fprintf('load(..., ''output'') failed, falling back to h5read: %s\n', ME.message);
end

try
    Tin = h5read(result_path, '/output/temporal_weights');
catch ME
    error('Failed to read /output/temporal_weights from %s: %s', result_path, ME.message);
end

Tin = single(Tin);
if ndims(Tin) ~= 2
    error('output.temporal_weights must be 2D, got ndims=%d', ndims(Tin));
end
if size(Tin, 1) <= size(Tin, 2)
    T = Tin;
else
    T = Tin';
end
end

function F0_cell = compute_f0_cell(S2, Fpix)
if issparse(S2)
    [ii, jj, vv] = find(S2);
    vv(vv < 0) = 0;
    S2p = sparse(ii, jj, vv, size(S2, 1), size(S2, 2));
else
    S2p = max(single(S2), 0);
end
num = double(Fpix(:))' * double(S2p);
den = full(sum(S2p, 1));
F0_cell = single((num ./ max(den, eps))');
end

function F0_t = moving_percentile_exact(F_est, q, win, use_parfor)
F0_t = moving_percentile_exact_compat(F_est, q, win, use_parfor);
end

function [F0_t, info] = moving_percentile_decimated(F_est, q, win, decimate_factor, use_parfor)
if decimate_factor < 1
    decimate_factor = 1;
end
[n_cells, n_frames] = size(F_est);
if decimate_factor == 1
    F0_t = moving_percentile_exact(F_est, q, win, use_parfor);
    info = struct('decimate_factor', 1, 'n_frames_ds', n_frames, 'window_ds', win);
    return;
end
idx_ds = 1:decimate_factor:n_frames;
F_est_ds = F_est(:, idx_ds);
win_ds = max(5, round(win / decimate_factor));
if mod(win_ds, 2) == 0
    win_ds = win_ds + 1;
end
F0_ds = moving_percentile_exact(F_est_ds, q, win_ds, use_parfor);
F0_t = repelem(single(F0_ds), 1, decimate_factor);
if size(F0_t, 2) < n_frames
    pad_cols = n_frames - size(F0_t, 2);
    F0_t = [F0_t, repmat(F0_t(:, end), 1, pad_cols)];
elseif size(F0_t, 2) > n_frames
    F0_t = F0_t(:, 1:n_frames);
end
if ~isequal(size(F0_t), [n_cells, n_frames])
    error('Decimated baseline size mismatch.');
end
info = struct('decimate_factor', decimate_factor, ...
    'n_frames_ds', numel(idx_ds), 'window_ds', win_ds);
end

function Y = moving_percentile_exact_compat(X, q, win, use_parfor)
X = single(X);
[n_cells, n_frames] = size(X);
Y = zeros(n_cells, n_frames, 'single');
half = floor(win / 2);
if half < 0
    half = 0;
end

run_parallel = false;
if use_parfor && license('test', 'Distrib_Computing_Toolbox')
    try
        if isempty(gcp('nocreate'))
            parpool('threads');
        end
        run_parallel = true;
    catch
        run_parallel = false;
    end
end

chunk_cells = 64;
n_chunks = ceil(n_cells / chunk_cells);
for c = 1:n_chunks
    c_begin = (c - 1) * chunk_cells + 1;
    c_end = min(c * chunk_cells, n_cells);
    Xi = X(c_begin:c_end, :);
    Yi = zeros(size(Xi), 'single');

    if run_parallel
        parfor t = 1:n_frames
            idx_begin = max(1, t - half);
            idx_end = min(n_frames, t + half);
            Yi(:, t) = single(prctile(Xi(:, idx_begin:idx_end), q, 2));
        end
    else
        for t = 1:n_frames
            idx_begin = max(1, t - half);
            idx_end = min(n_frames, t + half);
            Yi(:, t) = single(prctile(Xi(:, idx_begin:idx_end), q, 2));
        end
    end
    Y(c_begin:c_end, :) = Yi;
end
end

function write_postprocess_h5(out_h5, dff, F_est, F0_t, F0_cell, meta)
if isfile(out_h5)
    delete(out_h5);
end

[n_cells, n_frames] = size(dff);
write_h5_2d_dataset(out_h5, '/dff', single(dff));
if ~isempty(F_est)
    write_h5_2d_dataset(out_h5, '/F_est', single(F_est));
end
if ~isempty(F0_t)
    write_h5_2d_dataset(out_h5, '/F0_t', single(F0_t));
end
if ~isempty(F0_cell)
    write_h5_vector_dataset(out_h5, '/F0_cell', single(F0_cell(:)));
end

write_h5_scalar_dataset(out_h5, '/meta/n_cells', int64(n_cells), 'int64');
write_h5_scalar_dataset(out_h5, '/meta/n_frames', int64(n_frames), 'int64');
write_h5_scalar_dataset(out_h5, '/meta/frame_rate_hz', single(meta.frame_rate_hz), 'single');
write_h5_scalar_dataset(out_h5, '/meta/half_window_sec', single(meta.half_window_sec), 'single');
write_h5_scalar_dataset(out_h5, '/meta/percentile_q', single(meta.percentile_q), 'single');
write_h5_scalar_dataset(out_h5, '/meta/moving_decimate_factor', int64(meta.moving_decimate_factor), 'int64');
write_h5_scalar_dataset(out_h5, '/meta/nan_inf_fail_ratio', single(meta.nan_inf_fail_ratio), 'single');
write_h5_scalar_dataset(out_h5, '/meta/bad_ratio', single(meta.bad_ratio), 'single');
write_h5_string_dataset(out_h5, '/meta/orientation', meta.orientation);
write_h5_string_dataset(out_h5, '/meta/output_format', meta.output_format);
write_h5_string_dataset(out_h5, '/meta/baseline_mode', meta.baseline_mode);
write_h5_string_dataset(out_h5, '/meta/source_result_path', meta.source_result_path);
write_h5_string_dataset(out_h5, '/meta/timestamp', meta.timestamp);

verify_postprocess_h5(out_h5, n_cells, n_frames, meta.baseline_mode);
end

function write_h5_scalar_dataset(h5_path, ds_path, value, dtype_name)
h5create(h5_path, ds_path, [1 1], 'Datatype', dtype_name, 'ChunkSize', [1 1]);
h5write(h5_path, ds_path, reshape(value, [1 1]));
end

function write_h5_string_dataset(h5_path, ds_path, txt)
txt = char(txt);
bytes = uint8(txt(:)');
if isempty(bytes)
    bytes = uint8(0);
end
h5create(h5_path, ds_path, [numel(bytes) 1], 'Datatype', 'uint8', 'ChunkSize', [numel(bytes) 1]);
h5write(h5_path, ds_path, reshape(bytes, [numel(bytes) 1]));
end

function write_h5_2d_dataset(h5_path, ds_path, arr)
arr = single(arr);
[store_rows, store_cols] = size(arr.');
chunk_rows = max(1, min(store_rows, 2048));
chunk_cols = max(1, min(store_cols, 256));
h5create(h5_path, ds_path, [store_rows store_cols], 'Datatype', 'single', ...
    'ChunkSize', [chunk_rows chunk_cols]);

store = arr.';
for row_begin = 1:chunk_rows:store_rows
    row_end = min(store_rows, row_begin + chunk_rows - 1);
    block = store(row_begin:row_end, :);
    h5write(h5_path, ds_path, block, [row_begin 1], size(block));
end
end

function write_h5_vector_dataset(h5_path, ds_path, arr)
arr = single(arr(:));
n = numel(arr);
chunk_len = max(1, min(n, 1048576));
h5create(h5_path, ds_path, [n 1], 'Datatype', 'single', 'ChunkSize', [chunk_len 1]);
h5write(h5_path, ds_path, reshape(arr, [n 1]));
end

function verify_postprocess_h5(h5_path, n_cells, n_frames, baseline_mode)
info = h5info(h5_path, '/dff');
if numel(info.Dataspace.Size) ~= 2
    error('postprocess /dff must be 2D after write.');
end
orientation = read_h5_string_dataset(h5_path, '/meta/orientation');
if ~strcmp(orientation, 'cells_by_frames')
    error('postprocess /meta/orientation mismatch: %s', orientation);
end
stored_cells = h5read(h5_path, '/meta/n_cells');
stored_frames = h5read(h5_path, '/meta/n_frames');
if stored_cells ~= n_cells || stored_frames ~= n_frames
    error('postprocess metadata shape mismatch after write.');
end
if strcmp(baseline_mode, 'extract_temporal')
    return;
end
h5info(h5_path, '/F0_t');
h5info(h5_path, '/F_est');
end

function txt = read_h5_string_dataset(h5_path, ds_path)
raw = h5read(h5_path, ds_path);
txt = char(raw(:)');
null_idx = find(txt == char(0), 1, 'first');
if ~isempty(null_idx)
    txt = txt(1:null_idx - 1);
end
end
