%% Recompute dF/F from EXTRACT temporal_weights using percentile baseline
% Uses output.info.F_per_pixel as per-cell baseline anchor:
%   F0_cell(i) = weighted mean of F_per_pixel with spatial_weights(:, :, i)
% Then computes:
%   F_est(i,t)   = temporal_weights(i,t) + F0_cell(i)
%   F0_t(i,t)    = percentile baseline of F_est(i,:)
%   dFF(i,t)     = (F_est(i,t) - F0_t(i,t)) / max(F0_t(i,t), eps)
%
% Notes:
% - This is a post-hoc approximation from extracted traces, not raw-pixel dF/F.
% - For strict raw dF/F, recompute from movie using ROI masks.

result_path = fullfile(fileparts(mfilename('fullpath')), ...
    'output_250810-Ras2-GC#78_reg_s_crop.mat');

frame_rate_hz = 30;
half_window_sec = 90;     % +-60 s window
percentile_q = 20;         % baseline percentile
baseline_mode = 'moving_percentile';  % 'global_percentile'(default) | 'moving_percentile'
moving_percentile_impl = 'decimated'; % 'exact' | 'decimated'
moving_decimate_factor = 10;          % used when impl='decimated'
save_output = true;
plot_n_cells = 5;
rng_seed = 1;
moving_percentile_use_parfor = true;

% Ensure ndSparse class can be resolved during MAT load.
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));

if ~isfile(result_path)
    error('Result file not found: %s', result_path);
end

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

% Build F estimate
F_est = bsxfun(@plus, T, F0_cell);

baseline_mode = lower(strtrim(char(baseline_mode)));
win = NaN;
switch baseline_mode
    case 'global_percentile'
        fprintf('Computing global percentile baseline: q=%g over full trace\n', percentile_q);
        F0_const = single(prctile(F_est, percentile_q, 2)); % [cells x 1]
        F0_t = bsxfun(@times, F0_const, ones(1, n_frames, 'single'));
    case 'moving_percentile'
        win = max(5, round(2 * half_window_sec * frame_rate_hz) + 1);
        if mod(win, 2) == 0
            win = win + 1;
        end
        impl = lower(strtrim(char(moving_percentile_impl)));
        fprintf(['Computing moving percentile baseline: q=%g, fs=%.3f Hz, ', ...
            'window=%d frames (~%.1f sec), impl=%s\n'], ...
            percentile_q, frame_rate_hz, win, win / frame_rate_hz, impl);
        t_mp = tic;
        switch impl
            case 'exact'
                F0_t = moving_percentile_exact(F_est, percentile_q, win, moving_percentile_use_parfor);
            case 'decimated'
                [F0_t, decim_info] = moving_percentile_decimated( ...
                    F_est, percentile_q, win, moving_decimate_factor, moving_percentile_use_parfor);
                fprintf(['Decimated moving percentile: factor=%d, frames_ds=%d, ', ...
                    'window_ds=%d\n'], decim_info.decimate_factor, ...
                    decim_info.n_frames_ds, decim_info.window_ds);
            otherwise
                error('Unsupported moving_percentile_impl: %s', impl);
        end
        fprintf('Moving percentile finished in %.2f sec\n', toc(t_mp));
    otherwise
        error('Unsupported baseline_mode: %s', baseline_mode);
end

dff = (F_est - F0_t) ./ max(F0_t, eps('single'));

fprintf('Done: dFF size = [%d cells x %d frames]\n', size(dff, 1), size(dff, 2));

% Quick visualization
rng(rng_seed);
ids = randperm(n_cells, min(plot_n_cells, n_cells));
f = figure('Color', 'w', 'Position', [60 60 1800 1100]);
tlo = tiledlayout(numel(ids), 2, 'TileSpacing', 'compact', 'Padding', 'compact');
if strcmp(baseline_mode, 'moving_percentile')
    title(tlo, sprintf('Post-hoc dF/F | mode=moving q=%g, +-%.0f sec', ...
        percentile_q, half_window_sec));
else
    title(tlo, sprintf('Post-hoc dF/F | mode=global q=%g', percentile_q));
end

for r = 1:numel(ids)
    cid = ids(r);
    nexttile((r - 1) * 2 + 1);
    hold on;
    plot(F_est(cid, :), 'Color', [0.35 0.35 0.35], 'LineWidth', 0.8);
    plot(F0_t(cid, :), 'Color', [0.85 0.2 0.2], 'LineWidth', 1.0);
    hold off;
    grid on;
    if strcmp(baseline_mode, 'moving_percentile')
        title(sprintf('Cell %d: F_{est} and moving F0', cid));
    else
        title(sprintf('Cell %d: F_{est} and global F0', cid));
    end
    xlabel('Frame');
    ylabel('a.u.');
    if r == 1
        legend({'F_{est}', 'F0(t)'}, 'Location', 'best');
    end

    nexttile((r - 1) * 2 + 2);
    plot(dff(cid, :), 'Color', [0.1 0.45 0.9], 'LineWidth', 0.8);
    grid on;
    title(sprintf('Cell %d: dF/F', cid));
    xlabel('Frame');
    ylabel('\DeltaF/F');
end

switch baseline_mode
    case 'moving_percentile'
        suffix = 'mp';
        fig_png = fullfile(fileparts(result_path), 'recomputed_dff_moving_percentile.png');
    case 'global_percentile'
        suffix = 'gp';
        fig_png = fullfile(fileparts(result_path), 'recomputed_dff_global_percentile.png');
    otherwise
        suffix = 'dff';
        fig_png = fullfile(fileparts(result_path), 'recomputed_dff.png');
end
exportgraphics(f, fig_png, 'Resolution', 180);
fprintf('Saved figure: %s\n', fig_png);

if save_output
    [out_dir, out_name, ~] = fileparts(result_path);
    out_mat = fullfile(out_dir, sprintf('%s_dff_%s.mat', out_name, suffix));
    meta = struct();
    meta.source_result_path = result_path;
    meta.frame_rate_hz = frame_rate_hz;
    meta.half_window_sec = half_window_sec;
    meta.percentile_q = percentile_q;
    meta.window_frames = win;
    meta.baseline_mode = char(baseline_mode);
    meta.moving_percentile_impl = char(moving_percentile_impl);
    meta.moving_decimate_factor = moving_decimate_factor;
    meta.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
    save(out_mat, 'dff', 'F_est', 'F0_t', 'F0_cell', 'ids', 'meta', '-v7.3');
    fprintf('Saved dF/F mat: %s\n', out_mat);
end

%% ---- Local functions ----
function [S2, n_cells] = spatial_to_2d(S)
% Returns S2 = [num_pixels x n_cells], preserving sparsity when possible.
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
    T = Tin';  % time x cells -> cells x time
elseif size(Tin, 1) == n_cells
    T = Tin;   % already cells x time
else
    error('Could not infer temporal_weights orientation.');
end
end

function F0_cell = compute_f0_cell(S2, Fpix)
% F0_cell(i) = sum_j(S(j,i)*Fpix(j)) / sum_j(S(j,i)), with S clipped at >=0.
if issparse(S2)
    [ii, jj, vv] = find(S2);
    vv(vv < 0) = 0;
    S2p = sparse(ii, jj, vv, size(S2, 1), size(S2, 2));
else
    S2p = max(single(S2), 0);
end

num = double(Fpix(:))' * double(S2p);   % [1 x n_cells]
den = full(sum(S2p, 1));                 % [1 x n_cells]
F0_cell = single((num ./ max(den, eps))');
end

function F0_t = moving_percentile_exact(F_est, q, win, use_parfor)
% Primary path: always use in-script exact implementation.
F0_t = moving_percentile_exact_compat(F_est, q, win, use_parfor);
end

function [F0_t, info] = moving_percentile_decimated(F_est, q, win, decimate_factor, use_parfor)
% Fast approximation:
% 1) decimate in time
% 2) compute moving percentile on decimated timeline
% 3) upsample by repelem (stair-step baseline)
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
% Exact moving-percentile implementation with shrink endpoints.
% X: [cells x frames]
% q: percentile (0..100)
% win: odd/even supported
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

% Process cells in chunks to reduce peak temp allocations inside prctile.
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
    if run_parallel
        fprintf('Moving percentile(parfor): cell-chunk %d/%d done\n', c, n_chunks);
    else
        fprintf('Moving percentile(serial): cell-chunk %d/%d done\n', c, n_chunks);
    end
end
end
