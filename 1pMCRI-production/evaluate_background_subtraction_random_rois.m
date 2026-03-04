%% Evaluate EXTRACT background subtraction quality on random ROIs
% For random ROIs, plot:
% 1) EXTRACT temporal_weights
% 2) Raw intensity from the same spatial ROI
% 3) Local background raw intensity from surrounding area (~30 px)
%
% Inputs (default):
%   EXTRACT output: output_250206-UK6-1-F=4_power=5mW_reg_crop_[no_constraint].mat
%   Raw movie:      250206-UK6-1-F=4_power=5mW_reg_crop.tiff

clearvars;

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

result_path = fullfile(script_dir, 'output_250810-Ras2-GC#78_reg_s_crop.mat');
tiff_path = find_tiff_path(script_dir, '250810-Ras2-GC#78_reg_s_crop.tif');
% result_path = fullfile(script_dir, 'output_250206-UK6-1-F=4_power=5mW_reg_crop_[no_constraint].mat');
% tiff_path = find_tiff_path(script_dir, '250206-UK6-1-F=4_power=5mW_reg_crop.tiff');

n_query = 10;
n_overlap_pairs = 10;
rng_seed = 7;
bg_radius_px = 30;     % neighborhood radius for local background
roi_thresh_frac = 0.20; % ROI mask threshold relative to per-ROI max
chunk_frames = 2000;

if ~isfile(result_path)
    error('Result MAT not found: %s', result_path);
end
if ~isfile(tiff_path)
    error('Raw TIFF not found: %s', tiff_path);
end

L = load(result_path, 'output');
if ~isfield(L, 'output')
    error('No ''output'' found in %s', result_path);
end
output = L.output;

if ~isfield(output, 'spatial_weights') || ~isfield(output, 'temporal_weights')
    error('output.spatial_weights / output.temporal_weights is missing.');
end

S = output.spatial_weights;
if isa(S, 'ndSparse')
    S = full(S);
end
S = single(S);
[h, w, n_cells] = size(S);

T = as_cells_by_time(output.temporal_weights, n_cells); % cells x frames
[n_cells_t, n_frames_t] = size(T);
if n_cells_t ~= n_cells
    error('Cell count mismatch between spatial (%d) and temporal (%d).', n_cells, n_cells_t);
end

info = imfinfo(tiff_path);
n_ifd = numel(info);
n_frames_raw = numel(info);
if n_frames_raw == 1 && isfield(info(1), 'ImageDescription')
    tok = regexp(info(1).ImageDescription, 'images=(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        n_frames_raw = str2double(tok{1});
    end
end
n_frames = min(n_frames_raw, n_frames_t);
if n_frames < 10
    error('Too few frames: %d', n_frames);
end

fprintf('Movie size: %d x %d x %d frames\n', h, w, n_frames);
fprintf('Cells: %d\n', n_cells);

% Build ROI masks and centroids
[Yg, Xg] = ndgrid(single(1:h), single(1:w));
cx = nan(n_cells, 1, 'single');
cy = nan(n_cells, 1, 'single');
roi_masks = cell(n_cells, 1);

for k = 1:n_cells
    wk = S(:, :, k);
    wk(wk < 0) = 0;
    mmax = max(wk(:));
    if mmax <= 0
        continue;
    end
    mk = wk >= (roi_thresh_frac * mmax);
    if ~any(mk(:))
        continue;
    end
    roi_masks{k} = find(mk);
    ww = wk(mk);
    xx = Xg(mk);
    yy = Yg(mk);
    s = sum(ww);
    if s > 0
        cx(k) = sum(xx .* ww) / s;
        cy(k) = sum(yy .* ww) / s;
    end
end

valid = find(~cellfun(@isempty, roi_masks) & isfinite(cx) & isfinite(cy));
if isempty(valid)
    error('No valid ROI masks.');
end

rng(rng_seed);
n_pick = min(n_query, numel(valid));
pick = valid(randperm(numel(valid), n_pick));

% Build local background masks for picked ROIs
bg_masks = cell(n_pick, 1);
for i = 1:n_pick
    rid = pick(i);
    dist2 = (Xg - cx(rid)).^2 + (Yg - cy(rid)).^2;
    ring = (dist2 <= bg_radius_px^2);
    ring(roi_masks{rid}) = false;
    if ~any(ring(:))
        error('Empty background mask for ROI %d. Increase bg_radius_px.', rid);
    end
    bg_masks{i} = find(ring);
end

% Extract raw traces in chunks
raw_roi = zeros(n_pick, n_frames, 'single');
raw_bg = zeros(n_pick, n_frames, 'single');

n_chunks = ceil(n_frames / chunk_frames);
for c = 1:n_chunks
    f_begin = (c - 1) * chunk_frames + 1;
    f_end = min(c * chunk_frames, n_frames);
    n_this = f_end - f_begin + 1;
    fprintf('Reading TIFF chunk %d/%d (frames %d-%d)\n', c, n_chunks, f_begin, f_end);
    M = read_tiff_chunk_general(tiff_path, info, n_ifd, n_frames_raw, f_begin, n_this); % h x w x n_this
    Mr = reshape(M, h * w, n_this);
    for i = 1:n_pick
        rid = pick(i);
        raw_roi(i, f_begin:f_end) = mean(Mr(roi_masks{rid}, :), 1);
        raw_bg(i, f_begin:f_end) = mean(Mr(bg_masks{i}, :), 1);
    end
end

Tpick = T(pick, 1:n_frames);
raw_sub = raw_roi - raw_bg;

% Metrics: correlation with temporal_weights
r_raw = nan(n_pick, 1);
r_bg = nan(n_pick, 1);
r_sub = nan(n_pick, 1);
for i = 1:n_pick
    tw = zscore_safe(Tpick(i, :));
    rr = zscore_safe(raw_roi(i, :));
    rb = zscore_safe(raw_bg(i, :));
    rs = zscore_safe(raw_sub(i, :));
    r_raw(i) = corr(tw(:), rr(:), 'rows', 'complete');
    r_bg(i) = corr(tw(:), rb(:), 'rows', 'complete');
    r_sub(i) = corr(tw(:), rs(:), 'rows', 'complete');
end

% Plot
fig = figure('Color', 'w', 'Position', [80 60 1900 2200]);
tlo = tiledlayout(n_pick, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf(['Random ROI background-subtraction evaluation (n=%d, radius=%d px)\n', ...
    'Median corr: TW-raw=%.3f, TW-bg=%.3f, TW-(raw-bg)=%.3f'], ...
    n_pick, bg_radius_px, median(r_raw, 'omitnan'), median(r_bg, 'omitnan'), median(r_sub, 'omitnan')));

if isfield(output, 'info') && isfield(output.info, 'summary_image') && ~isempty(output.info.summary_image)
    summary_img = single(output.info.summary_image);
else
    summary_img = max(S, [], 3);
end
summary_img = summary_img - min(summary_img(:));
if max(summary_img(:)) > 0
    summary_img = summary_img ./ max(summary_img(:));
end

crop_half = max(bg_radius_px, 40);

for i = 1:n_pick
    rid = pick(i);
    tw = zscore_safe(Tpick(i, :));
    rr = zscore_safe(raw_roi(i, :));
    rb = zscore_safe(raw_bg(i, :));
    rs = zscore_safe(raw_sub(i, :));

    % Col 1: spatial weight overlay on summary image
    nexttile((i - 1) * 4 + 1);
    qx = double(cx(rid));
    qy = double(cy(rid));
    xmin = max(1, floor(qx - crop_half));
    xmax = min(w, ceil(qx + crop_half));
    ymin = max(1, floor(qy - crop_half));
    ymax = min(h, ceil(qy + crop_half));

    ax_sp = gca;
    imagesc(xmin:xmax, ymin:ymax, summary_img(ymin:ymax, xmin:xmax));
    colormap(ax_sp, gray(256));
    hold on;
    wk = S(:, :, rid);
    th = roi_thresh_frac * max(wk(:));
    if th > 0
        contour(xmin:xmax, ymin:ymax, wk(ymin:ymax, xmin:xmax), [th th], ...
            'Color', [1 0.2 0.1], 'LineWidth', 1.4);
    end
    plot(cx(rid), cy(rid), 'o', 'Color', [1 0.2 0.1], ...
        'MarkerFaceColor', [1 0.2 0.1], 'MarkerSize', 4);
    hold off;
    axis image off;
    set(gca, 'YDir', 'reverse');
    title(sprintf('ROI %d spatial', rid));

    nexttile((i - 1) * 4 + 2);
    plot(tw, 'k-', 'LineWidth', 0.9);
    grid on;
    ylabel('z');
    title(sprintf('ROI %d temporal weight', rid));
    if i == n_pick, xlabel('Frame'); end

    nexttile((i - 1) * 4 + 3);
    hold on;
    plot(rr, 'Color', [0.1 0.45 0.9], 'LineWidth', 0.9);
    plot(rb, 'Color', [0.85 0.25 0.2], 'LineWidth', 0.9);
    hold off;
    grid on;
    title(sprintf('Raw ROI / BG (corr TW: %.2f / %.2f)', r_raw(i), r_bg(i)));
    if i == 1
        legend({'ROI raw', 'BG raw'}, 'Location', 'best');
    end
    if i == n_pick, xlabel('Frame'); end

    nexttile((i - 1) * 4 + 4);
    hold on;
    plot(tw, 'k-', 'LineWidth', 0.8);
    plot(rs, 'Color', [0.2 0.7 0.2], 'LineWidth', 0.9);
    hold off;
    grid on;
    title(sprintf('TW vs (ROI-BG), corr=%.2f', r_sub(i)));
    if i == 1
        legend({'TW', 'ROI-BG raw'}, 'Location', 'best');
    end
    if i == n_pick, xlabel('Frame'); end
end

out_png = fullfile(script_dir, 'background_subtraction_eval_random10.png');
exportgraphics(fig, out_png, 'Resolution', 180);
fprintf('Saved figure: %s\n', out_png);

% Save metrics table
Tmetrics = table(pick(:), r_raw, r_bg, r_sub, ...
    'VariableNames', {'roi_id', 'corr_tw_raw', 'corr_tw_bg', 'corr_tw_raw_minus_bg'});
out_mat = fullfile(script_dir, 'background_subtraction_eval_random10.mat');
save(out_mat, 'Tmetrics', 'pick', 'r_raw', 'r_bg', 'r_sub', 'bg_radius_px', 'roi_thresh_frac', ...
    'result_path', 'tiff_path');
fprintf('Saved metrics: %s\n', out_mat);

disp(Tmetrics);

%% Figure 2: Overlapping ROI pairs
fprintf('Finding overlapping ROI pairs...\n');
valid_n = numel(valid);
nnz_est = sum(cellfun(@numel, roi_masks(valid)));
B = spalloc(h * w, valid_n, nnz_est);
for j = 1:valid_n
    B(roi_masks{valid(j)}, j) = true;
end
O = spones(B' * B);
O = triu(O, 1);
[ii, jj] = find(O);

if isempty(ii)
    warning('No overlapping ROI pairs found with current roi_thresh_frac=%.3f.', roi_thresh_frac);
else
    rng(rng_seed + 1);
    n_pair = min(n_overlap_pairs, numel(ii));
    sel = randperm(numel(ii), n_pair);
    pair_a = valid(ii(sel));
    pair_b = valid(jj(sel));
    pair_ids = unique([pair_a; pair_b]);

    % Background masks for pair ROI IDs
    n_pair_ids = numel(pair_ids);
    pair_bg_masks = cell(n_pair_ids, 1);
    for k = 1:n_pair_ids
        rid = pair_ids(k);
        dist2 = (Xg - cx(rid)).^2 + (Yg - cy(rid)).^2;
        ring = (dist2 <= bg_radius_px^2);
        ring(roi_masks{rid}) = false;
        if ~any(ring(:))
            ring = ~false(size(ring));
            ring(roi_masks{rid}) = false;
        end
        pair_bg_masks{k} = find(ring);
    end

    % Read movie again and compute raw traces for pair ROI IDs
    raw_pair = zeros(n_pair_ids, n_frames, 'single');
    raw_pair_bg = zeros(n_pair_ids, n_frames, 'single');
    for c = 1:n_chunks
        f_begin = (c - 1) * chunk_frames + 1;
        f_end = min(c * chunk_frames, n_frames);
        n_this = f_end - f_begin + 1;
        fprintf('Reading TIFF chunk %d/%d for overlap-pair figure (frames %d-%d)\n', ...
            c, n_chunks, f_begin, f_end);
        M = read_tiff_chunk_general(tiff_path, info, n_ifd, n_frames_raw, f_begin, n_this);
        Mr = reshape(M, h * w, n_this);
        for k = 1:n_pair_ids
            rid = pair_ids(k);
            raw_pair(k, f_begin:f_end) = mean(Mr(roi_masks{rid}, :), 1);
            raw_pair_bg(k, f_begin:f_end) = mean(Mr(pair_bg_masks{k}, :), 1);
        end
    end

    id_to_local = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for k = 1:n_pair_ids
        id_to_local(double(pair_ids(k))) = k;
    end

    fig2 = figure('Color', 'w', 'Position', [100 40 2000 2000]);
    tlo2 = tiledlayout(n_pair, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tlo2, sprintf('Overlapping ROI pairs evaluation (n=%d, radius=%d px)', n_pair, bg_radius_px));

    pair_metrics = table('Size', [n_pair 8], ...
        'VariableTypes', {'double','double','double','double','double','double','double','double'}, ...
        'VariableNames', {'roi_a','roi_b','overlap_px', ...
        'corr_a_tw_sub','corr_b_tw_sub','corr_tw_pair','corr_raw_pair','corr_sub_pair'});

    for p = 1:n_pair
        a = pair_a(p);
        b = pair_b(p);
        ka = id_to_local(double(a));
        kb = id_to_local(double(b));

        tw_a = zscore_safe(T(a, 1:n_frames));
        tw_b = zscore_safe(T(b, 1:n_frames));
        rr_a = zscore_safe(raw_pair(ka, :));
        rr_b = zscore_safe(raw_pair(kb, :));
        rb_a = zscore_safe(raw_pair_bg(ka, :));
        rb_b = zscore_safe(raw_pair_bg(kb, :));
        rs_a = zscore_safe(raw_pair(ka, :) - raw_pair_bg(ka, :));
        rs_b = zscore_safe(raw_pair(kb, :) - raw_pair_bg(kb, :));

        overlap_px = numel(intersect(roi_masks{a}, roi_masks{b}));
        pair_metrics.roi_a(p) = a;
        pair_metrics.roi_b(p) = b;
        pair_metrics.overlap_px(p) = overlap_px;
        pair_metrics.corr_a_tw_sub(p) = corr(tw_a(:), rs_a(:), 'rows', 'complete');
        pair_metrics.corr_b_tw_sub(p) = corr(tw_b(:), rs_b(:), 'rows', 'complete');
        pair_metrics.corr_tw_pair(p) = corr(tw_a(:), tw_b(:), 'rows', 'complete');
        pair_metrics.corr_raw_pair(p) = corr(rr_a(:), rr_b(:), 'rows', 'complete');
        pair_metrics.corr_sub_pair(p) = corr(rs_a(:), rs_b(:), 'rows', 'complete');

        % Col 1: spatial overlay of overlapping pair
        nexttile((p - 1) * 4 + 1);
        mx = mean([cx(a), cx(b)]);
        my = mean([cy(a), cy(b)]);
        xmin = max(1, floor(mx - crop_half));
        xmax = min(w, ceil(mx + crop_half));
        ymin = max(1, floor(my - crop_half));
        ymax = min(h, ceil(my + crop_half));
        ax = gca;
        imagesc(xmin:xmax, ymin:ymax, summary_img(ymin:ymax, xmin:xmax));
        colormap(ax, gray(256));
        hold on;
        wa = S(:, :, a); wb = S(:, :, b);
        tha = roi_thresh_frac * max(wa(:));
        thb = roi_thresh_frac * max(wb(:));
        if tha > 0
            contour(xmin:xmax, ymin:ymax, wa(ymin:ymax, xmin:xmax), [tha tha], ...
                'Color', [1 0.2 0.1], 'LineWidth', 1.4);
        end
        if thb > 0
            contour(xmin:xmax, ymin:ymax, wb(ymin:ymax, xmin:xmax), [thb thb], ...
                'Color', [0.1 0.7 1.0], 'LineWidth', 1.4);
        end
        plot(cx(a), cy(a), 'o', 'Color', [1 0.2 0.1], 'MarkerFaceColor', [1 0.2 0.1], 'MarkerSize', 4);
        plot(cx(b), cy(b), 'o', 'Color', [0.1 0.7 1.0], 'MarkerFaceColor', [0.1 0.7 1.0], 'MarkerSize', 4);
        hold off;
        axis image off;
        set(gca, 'YDir', 'reverse');
        title(sprintf('ROI %d vs %d (overlap=%d px)', a, b, overlap_px));

        % Col 2: temporal weights
        nexttile((p - 1) * 4 + 2);
        hold on;
        plot(tw_a, 'Color', [1 0.2 0.1], 'LineWidth', 0.9);
        plot(tw_b, 'Color', [0.1 0.7 1.0], 'LineWidth', 0.9);
        hold off;
        grid on;
        title(sprintf('TW corr=%.2f', pair_metrics.corr_tw_pair(p)));
        if p == 1
            legend({sprintf('TW ROI %d', a), sprintf('TW ROI %d', b)}, 'Location', 'best');
        end
        if p == n_pair, xlabel('Frame'); end

        % Col 3: raw ROI and local BG
        nexttile((p - 1) * 4 + 3);
        hold on;
        plot(rr_a, '-', 'Color', [0.8 0.1 0.1], 'LineWidth', 0.8);
        plot(rb_a, '--', 'Color', [0.8 0.1 0.1], 'LineWidth', 0.8);
        plot(rr_b, '-', 'Color', [0.1 0.5 0.9], 'LineWidth', 0.8);
        plot(rb_b, '--', 'Color', [0.1 0.5 0.9], 'LineWidth', 0.8);
        hold off;
        grid on;
        title(sprintf('Raw pair corr=%.2f', pair_metrics.corr_raw_pair(p)));
        if p == 1
            legend({'A raw','A bg','B raw','B bg'}, 'Location', 'best');
        end
        if p == n_pair, xlabel('Frame'); end

        % Col 4: TW vs (raw-bg) for both
        nexttile((p - 1) * 4 + 4);
        hold on;
        plot(tw_a, 'k-', 'LineWidth', 0.6);
        plot(rs_a, '-', 'Color', [1 0.2 0.1], 'LineWidth', 0.9);
        plot(tw_b, 'k--', 'LineWidth', 0.6);
        plot(rs_b, '-', 'Color', [0.1 0.7 1.0], 'LineWidth', 0.9);
        hold off;
        grid on;
        title(sprintf('A:%.2f  B:%.2f  subPair:%.2f', ...
            pair_metrics.corr_a_tw_sub(p), pair_metrics.corr_b_tw_sub(p), pair_metrics.corr_sub_pair(p)));
        if p == 1
            legend({'A TW','A raw-bg','B TW','B raw-bg'}, 'Location', 'best');
        end
        if p == n_pair, xlabel('Frame'); end
    end

    out_png2 = fullfile(script_dir, 'background_subtraction_eval_overlap_pairs.png');
    exportgraphics(fig2, out_png2, 'Resolution', 180);
    fprintf('Saved overlap-pair figure: %s\n', out_png2);

    out_mat2 = fullfile(script_dir, 'background_subtraction_eval_overlap_pairs.mat');
    save(out_mat2, 'pair_metrics', 'pair_a', 'pair_b', 'pair_ids', 'bg_radius_px', ...
        'roi_thresh_frac', 'result_path', 'tiff_path');
    fprintf('Saved overlap-pair metrics: %s\n', out_mat2);
    disp(pair_metrics);
end

%% ---- local functions ----
function T = as_cells_by_time(Tin, n_cells)
Tin = single(Tin);
if size(Tin, 2) == n_cells
    T = Tin'; % time x cells -> cells x time
elseif size(Tin, 1) == n_cells
    T = Tin;
else
    error('Could not infer temporal_weights orientation.');
end
end

function z = zscore_safe(x)
x = single(x(:)');
mu = mean(x, 'omitnan');
sd = std(x, 0, 'omitnan');
if sd < eps('single')
    z = zeros(size(x), 'single');
else
    z = (x - mu) / sd;
end
end

function p = find_tiff_path(script_dir, fname)
cands = {
    fullfile(script_dir, fname), ...
    fullfile(fileparts(script_dir), '1pMCRI-demo', fname), ...
    fullfile(pwd, fname)
};
p = cands{1};
for i = 1:numel(cands)
    if isfile(cands{i})
        p = cands{i};
        return;
    end
end
end

function M = read_tiff_chunk_general(tiff_path, info, n_ifd, total_frames, f_begin, n_this)
% Supports both multi-page TIFF and ImageJ single-IFD stack TIFF.
h = info(1).Height;
w = info(1).Width;

if n_ifd > 1
    M = single(read_from_tif(tiff_path, f_begin, n_this));
    return;
end

if total_frames <= 1
    M = single(imread(tiff_path, 1));
    M = reshape(M, h, w, 1);
    return;
end

% ImageJ single-IFD path
t = Tiff(tiff_path, 'r');
bits_per_sample = double(t.getTag('BitsPerSample'));
samples_per_pixel = double(t.getTag('SamplesPerPixel'));
compression = double(t.getTag('Compression'));
strip_offset = double(t.getTag('StripOffsets'));
frame_bytes = double(t.getTag('StripByteCounts'));
close(t);

if bits_per_sample ~= 16 || samples_per_pixel ~= 1 || compression ~= 1
    error('ImageJ single-IFD path supports uncompressed uint16 single-channel TIFF only.');
end
if frame_bytes ~= (h * w * 2)
    error('Unexpected frame byte size in TIFF: %d (expected %d).', frame_bytes, h * w * 2);
end

machinefmt = detect_tiff_byte_order_local(tiff_path);
fid = fopen(tiff_path, 'r', machinefmt);
if fid < 0
    error('Failed to open TIFF for binary read: %s', tiff_path);
end
cleanup_obj = onCleanup(@() fclose(fid));

offset = strip_offset + (f_begin - 1) * frame_bytes;
fseek(fid, offset, 'bof');
vec = fread(fid, h * w * n_this, '*uint16');
if numel(vec) ~= h * w * n_this
    error('Failed to read chunk frames %d-%d from TIFF.', f_begin, f_begin + n_this - 1);
end

% TIFF raster is row-major; transpose x/y for MATLAB ordering.
M = single(permute(reshape(vec, [w, h, n_this]), [2 1 3]));
clear cleanup_obj
end

function machinefmt = detect_tiff_byte_order_local(tiff_path)
fid = fopen(tiff_path, 'r');
if fid < 0
    error('Failed to open TIFF header: %s', tiff_path);
end
cleanup_obj = onCleanup(@() fclose(fid));
sig = fread(fid, 2, '*char')';
if strcmp(sig, 'II')
    machinefmt = 'ieee-le';
elseif strcmp(sig, 'MM')
    machinefmt = 'ieee-be';
else
    error('Invalid TIFF byte-order signature.');
end
clear cleanup_obj
end
