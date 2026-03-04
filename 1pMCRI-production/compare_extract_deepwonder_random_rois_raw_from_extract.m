%% Compare EXTRACT vs DeepWonder on random ROIs using EXTRACT masks for raw/bg
% This script compares temporal components on matched cells while forcing
% raw-signal/background extraction to use EXTRACT spatial masks.
%
% Figure (random ROIs):
% 1) EXTRACT ROI mask over raw summary image
% 2) Temporal components (EXTRACT vs DeepWonder)
% 3) Raw ROI / Raw background (both from EXTRACT mask geometry)
% 4) Temporal vs (Raw-BG) overlay

clearvars;

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

extract_mat = fullfile(script_dir, 'output_250810-Ras2-GC#78_reg_s_crop.mat');
deep_dir = 'R:\data\manipulandum\target_reach\demo_data\RSM_250810-Ras2-GC#78_reg_s_crop_20260304-1115\mat';
tiff_path = fullfile(repo_root, '1pMCRI-demo', '250810-Ras2-GC#78_reg_s_crop.tif');

n_query = 10;
rng_seed = 1;
roi_thresh_frac = 0.20;
bg_radius_px = 30;
chunk_frames = 2000;
max_match_dist_px = 5;
temporal_norm_mode = 'robust_zscore'; % 'none'|'zscore'|'robust_zscore'
frame_start = 10001;
frame_end = 15000;

if ~isfile(extract_mat)
    error('EXTRACT result not found: %s', extract_mat);
end
if ~isfolder(deep_dir)
    error('DeepWonder dir not found: %s', deep_dir);
end
if ~isfile(tiff_path)
    error('TIFF not found: %s', tiff_path);
end

%% Load EXTRACT
E = load(extract_mat, 'output');
if ~isfield(E, 'output') || ~isfield(E.output, 'spatial_weights') || ~isfield(E.output, 'temporal_weights')
    error('Invalid EXTRACT output in %s', extract_mat);
end
S_ex = E.output.spatial_weights;
if isa(S_ex, 'ndSparse')
    S_ex = full(S_ex);
end
S_ex = single(S_ex);
[h, w, n_ex] = size(S_ex);
T_ex = single(E.output.temporal_weights');
if isfield(E.output, 'info') && isfield(E.output.info, 'summary_image') && ~isempty(E.output.info.summary_image)
    summary_img = single(E.output.info.summary_image);
else
    summary_img = [];
end

[Yg, Xg] = ndgrid(single(1:h), single(1:w));
cx_ex = nan(n_ex, 1, 'single');
cy_ex = nan(n_ex, 1, 'single');
roi_mask_idx = cell(n_ex, 1);

for i = 1:n_ex
    m = max(S_ex(:, :, i), 0);
    vmax = max(m(:));
    if vmax <= 0
        continue;
    end
    bw = m >= (roi_thresh_frac * vmax);
    if ~any(bw(:))
        continue;
    end
    roi_mask_idx{i} = find(bw);
    ww = m(bw);
    cx_ex(i) = sum(Xg(bw) .* ww) / sum(ww);
    cy_ex(i) = sum(Yg(bw) .* ww) / sum(ww);
end
valid_ex = find(isfinite(cx_ex) & isfinite(cy_ex) & ~cellfun(@isempty, roi_mask_idx));

%% Load DeepWonder centroids + traces
files = dir(fullfile(deep_dir, 'results_*.mat'));
if isempty(files)
    error('No results_*.mat in %s', deep_dir);
end
[~, ix] = sort({files.name});
files = files(ix);

cx_dw = [];
cy_dw = [];
trace_dw = {};
pos_dw = {};
val_dw = {};
coord_offset = [];

for fi = 1:numel(files)
    D = load(fullfile(files(fi).folder, files(fi).name), 'final_mask_list');
    L = D.final_mask_list;
    for k = 1:numel(L)
        e = L{k};
        if ~isfield(e, 'position') || isempty(e.position) || ~isfield(e, 'trace') || isempty(e.trace)
            continue;
        end
        p = double(e.position);
        if isempty(coord_offset)
            if min(p(:)) <= 0
                coord_offset = 1;
            else
                coord_offset = 0;
            end
        end
        if isfield(e, 'centroid') && ~isempty(e.centroid) && numel(e.centroid) >= 2
            c = double(e.centroid(:)');
            cy = c(1) + coord_offset;
            cx = c(2) + coord_offset;
        else
            y = p(:, 1) + coord_offset;
            x = p(:, 2) + coord_offset;
            cy = mean(y);
            cx = mean(x);
        end
        if cx < 1 || cx > w || cy < 1 || cy > h
            continue;
        end
        y = p(:, 1) + coord_offset;
        x = p(:, 2) + coord_offset;
        keep = y >= 1 & y <= h & x >= 1 & x <= w;
        y = y(keep);
        x = x(keep);
        if isempty(y)
            continue;
        end
        if isfield(e, 'value') && ~isempty(e.value) && numel(e.value) == size(p, 1)
            vv = single(e.value(keep));
        else
            vv = ones(numel(y), 1, 'single');
        end

        cx_dw(end + 1, 1) = single(cx); %#ok<AGROW>
        cy_dw(end + 1, 1) = single(cy); %#ok<AGROW>
        trace_dw{end + 1, 1} = single(e.trace(:)'); %#ok<AGROW>
        pos_dw{end + 1, 1} = [y(:), x(:)]; %#ok<AGROW>
        val_dw{end + 1, 1} = vv(:); %#ok<AGROW>
    end
end

n_dw = numel(cx_dw);
if n_dw == 0
    error('No valid DeepWonder cells found in FOV.');
end

%% Match EXTRACT -> DeepWonder by nearest centroid (one-to-one greedy)
dwx = double(cx_dw(:)');
dwy = double(cy_dw(:)');
ex_idx = valid_ex(:);
nearest_dw = zeros(numel(ex_idx), 1);
nearest_d = inf(numel(ex_idx), 1);
for i = 1:numel(ex_idx)
    exi = ex_idx(i);
    d2 = (dwx - double(cx_ex(exi))).^2 + (dwy - double(cy_ex(exi))).^2;
    [v, j] = min(d2);
    nearest_dw(i) = j;
    nearest_d(i) = sqrt(v);
end

[~, ord] = sort(nearest_d, 'ascend');
used_dw = false(n_dw, 1);
match_ex = [];
match_dw = [];
for ii = 1:numel(ord)
    i = ord(ii);
    if nearest_d(i) > max_match_dist_px
        break;
    end
    j = nearest_dw(i);
    if used_dw(j)
        continue;
    end
    used_dw(j) = true;
    match_ex(end + 1, 1) = ex_idx(i); %#ok<AGROW>
    match_dw(end + 1, 1) = j; %#ok<AGROW>
end
if numel(match_ex) < 1
    error('No matched cells under %.1f px', max_match_dist_px);
end
fprintf('Matched EXTRACT->DeepWonder cells: %d\n', numel(match_ex));

%% Random selection from matched cells
rng(rng_seed);
n_pick = min(n_query, numel(match_ex));
sel = randperm(numel(match_ex), n_pick);
pick_ex = match_ex(sel);
pick_dw = match_dw(sel);

%% Raw/BG masks from EXTRACT geometry
bg_masks = cell(n_pick, 1);
for i = 1:n_pick
    rid = pick_ex(i);
    dist2 = (Xg - cx_ex(rid)).^2 + (Yg - cy_ex(rid)).^2;
    ring = (dist2 <= bg_radius_px^2);
    ring(roi_mask_idx{rid}) = false;
    if ~any(ring(:))
        error('Empty background mask for EXTRACT ROI %d', rid);
    end
    bg_masks{i} = find(ring);
end

%% Read TIFF and extract raw traces
info = imfinfo(tiff_path);
n_ifd = numel(info);
n_frames_raw = numel(info);
if n_frames_raw == 1 && isfield(info(1), 'ImageDescription')
    tok = regexp(info(1).ImageDescription, 'images=(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        n_frames_raw = str2double(tok{1});
    end
end
n_frames = n_frames_raw;
frame_start = max(1, frame_start);
frame_end = min(n_frames_raw, frame_end);
if frame_end < frame_start
    error('Invalid frame range: start=%d end=%d', frame_start, frame_end);
end
n_frames = frame_end - frame_start + 1;

raw_roi = zeros(n_pick, n_frames, 'single');
raw_bg = zeros(n_pick, n_frames, 'single');
if isempty(summary_img)
    summary_acc = zeros(h, w, 'double');
else
    summary_acc = [];
end

n_chunks = ceil((frame_end - frame_start + 1) / chunk_frames);
for c = 1:n_chunks
    g_begin = frame_start + (c - 1) * chunk_frames;
    g_end = min(frame_start + c * chunk_frames - 1, frame_end);
    n_this = g_end - g_begin + 1;
    l_begin = g_begin - frame_start + 1;
    l_end = g_end - frame_start + 1;
    fprintf('Reading TIFF chunk %d/%d (frames %d-%d)\n', c, n_chunks, g_begin, g_end);
    M = read_tiff_chunk_general(tiff_path, info, n_ifd, n_frames_raw, g_begin, n_this);
    if isempty(summary_img)
        summary_acc = summary_acc + sum(double(M), 3);
    end
    Mr = reshape(M, h * w, n_this);
    for i = 1:n_pick
        rid = pick_ex(i);
        raw_roi(i, l_begin:l_end) = mean(Mr(roi_mask_idx{rid}, :), 1);
        raw_bg(i, l_begin:l_end) = mean(Mr(bg_masks{i}, :), 1);
    end
end
if isempty(summary_img)
    summary_img = single(summary_acc / n_frames);
end
summary_img = summary_img - min(summary_img(:));
if max(summary_img(:)) > 0, summary_img = summary_img ./ max(summary_img(:)); end

%% Build normalized traces and metrics
r_ex_raw = nan(n_pick, 1);
r_dw_raw = nan(n_pick, 1);
r_ex_sub = nan(n_pick, 1);
r_dw_sub = nan(n_pick, 1);
r_ex_dw = nan(n_pick, 1);

TexN = zeros(n_pick, n_frames, 'single');
TdwN = zeros(n_pick, n_frames, 'single');
RoiN = zeros(n_pick, n_frames, 'single');
BgN = zeros(n_pick, n_frames, 'single');
SubN = zeros(n_pick, n_frames, 'single');

for i = 1:n_pick
    tr_ex_full = single(T_ex(pick_ex(i), :));
    tr_ex = extract_trace_window(tr_ex_full, frame_start, frame_end, n_frames_raw);
    tr_dw_full = single(trace_dw{pick_dw(i)});
    tr_dw = extract_trace_window(tr_dw_full, frame_start, frame_end, n_frames_raw);
    tr_ex = normalize_trace(tr_ex, temporal_norm_mode);
    tr_dw = normalize_trace(tr_dw, temporal_norm_mode);
    rr = zscore_safe(raw_roi(i, :));
    rb = zscore_safe(raw_bg(i, :));
    rs = zscore_safe(raw_roi(i, :) - raw_bg(i, :));

    TexN(i, :) = tr_ex;
    TdwN(i, :) = tr_dw;
    RoiN(i, :) = rr;
    BgN(i, :) = rb;
    SubN(i, :) = rs;

    r_ex_raw(i) = corr(tr_ex(:), rr(:), 'rows', 'complete');
    r_dw_raw(i) = corr(tr_dw(:), rr(:), 'rows', 'complete');
    r_ex_sub(i) = corr(tr_ex(:), rs(:), 'rows', 'complete');
    r_dw_sub(i) = corr(tr_dw(:), rs(:), 'rows', 'complete');
    r_ex_dw(i) = corr(tr_ex(:), tr_dw(:), 'rows', 'complete');
end

%% Figure
crop_half = max(bg_radius_px, 40);
fig = figure('Color', 'w', 'Position', [80 60 2100 2200]);
tlo = tiledlayout(n_pick, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf(['EXTRACT vs DeepWonder (raw/bg from EXTRACT masks) | n=%d\n', ...
    'Frames=%d:%d | Median corr EX-DW=%.3f | EX-(raw-bg)=%.3f | DW-(raw-bg)=%.3f'], ...
    n_pick, frame_start, frame_end, median(r_ex_dw, 'omitnan'), median(r_ex_sub, 'omitnan'), median(r_dw_sub, 'omitnan')));

for i = 1:n_pick
    rid = pick_ex(i);

    nexttile((i - 1) * 4 + 1);
    qx = double(cx_ex(rid));
    qy = double(cy_ex(rid));
    xmin = max(1, floor(qx - crop_half));
    xmax = min(w, ceil(qx + crop_half));
    ymin = max(1, floor(qy - crop_half));
    ymax = min(h, ceil(qy + crop_half));
    ax = gca;
    imagesc(xmin:xmax, ymin:ymax, summary_img(ymin:ymax, xmin:xmax));
    colormap(ax, gray(256));
    hold on;
    roi_im = zeros(ymax - ymin + 1, xmax - xmin + 1, 'single');
    [yy, xx] = ind2sub([h, w], roi_mask_idx{rid});
    yy = yy - ymin + 1;
    xx = xx - xmin + 1;
    good = yy >= 1 & yy <= size(roi_im, 1) & xx >= 1 & xx <= size(roi_im, 2);
    lin = sub2ind(size(roi_im), yy(good), xx(good));
    roi_im(lin) = 1;
    contour(xmin:xmax, ymin:ymax, roi_im, [0.5 0.5], 'Color', [1 0.2 0.1], 'LineWidth', 1.4);
    % Draw matched DeepWonder ROI contour on the same panel.
    did = pick_dw(i);
    pdw = pos_dw{did};
    vdw = val_dw{did};
    roi_im_dw = zeros(ymax - ymin + 1, xmax - xmin + 1, 'single');
    ydw = round(pdw(:, 1)) - ymin + 1;
    xdw = round(pdw(:, 2)) - xmin + 1;
    good_dw = ydw >= 1 & ydw <= size(roi_im_dw, 1) & xdw >= 1 & xdw <= size(roi_im_dw, 2);
    ydw = ydw(good_dw); xdw = xdw(good_dw); vdw = vdw(good_dw);
    lin_dw = sub2ind(size(roi_im_dw), ydw, xdw);
    roi_im_dw(lin_dw) = max(roi_im_dw(lin_dw), vdw);
    th_dw = max(roi_im_dw(:)) * 0.3;
    if th_dw > 0
        contour(xmin:xmax, ymin:ymax, roi_im_dw, [th_dw th_dw], 'Color', [0.1 0.8 1.0], 'LineWidth', 1.2);
    end
    hold off;
    axis image off;
    set(gca, 'YDir', 'reverse');
    title(sprintf('EX ROI %d / DW ROI %d', rid, did));

    nexttile((i - 1) * 4 + 2);
    hold on;
    plot(TexN(i, :), 'Color', [0.1 0.35 0.9], 'LineWidth', 1.0);
    plot(TdwN(i, :), 'Color', [0.85 0.25 0.2], 'LineWidth', 1.0);
    hold off;
    grid on;
    title(sprintf('Temporal corr(EX,DW)=%.2f', r_ex_dw(i)));
    if i == 1
        legend({'EXTRACT', 'DeepWonder'}, 'Location', 'best');
    end
    if i == n_pick, xlabel('Frame'); end

    nexttile((i - 1) * 4 + 3);
    hold on;
    plot(RoiN(i, :), 'Color', [0.2 0.6 0.2], 'LineWidth', 0.9);
    plot(BgN(i, :), 'Color', [0.7 0.2 0.7], 'LineWidth', 0.9);
    hold off;
    grid on;
    title(sprintf('Raw/BG corr EX=%.2f DW=%.2f', r_ex_raw(i), r_dw_raw(i)));
    if i == 1
        legend({'Raw ROI', 'Raw BG'}, 'Location', 'best');
    end
    if i == n_pick, xlabel('Frame'); end

    nexttile((i - 1) * 4 + 4);
    hold on;
    plot(SubN(i, :), 'k-', 'LineWidth', 0.8);
    plot(TexN(i, :), 'Color', [0.1 0.35 0.9], 'LineWidth', 1.0);
    plot(TdwN(i, :), 'Color', [0.85 0.25 0.2], 'LineWidth', 1.0);
    hold off;
    grid on;
    title(sprintf('corr vs raw-bg EX=%.2f DW=%.2f', r_ex_sub(i), r_dw_sub(i)));
    if i == 1
        legend({'Raw-BG', 'EXTRACT', 'DeepWonder'}, 'Location', 'best');
    end
    if i == n_pick, xlabel('Frame'); end
end

out_png = fullfile(script_dir, 'compare_extract_deepwonder_random_rois_raw_from_extract.png');
exportgraphics(fig, out_png, 'Resolution', 180);
fprintf('Saved figure: %s\n', out_png);

Tmetrics = table(pick_ex(:), pick_dw(:), r_ex_dw, r_ex_raw, r_dw_raw, r_ex_sub, r_dw_sub, ...
    'VariableNames', {'extract_roi_id', 'deepwonder_roi_id', 'corr_ex_dw', ...
    'corr_ex_raw', 'corr_dw_raw', 'corr_ex_raw_minus_bg', 'corr_dw_raw_minus_bg'});
out_mat = fullfile(script_dir, 'compare_extract_deepwonder_random_rois_raw_from_extract.mat');
save(out_mat, 'Tmetrics', 'pick_ex', 'pick_dw', 'extract_mat', 'deep_dir', 'tiff_path', ...
    'bg_radius_px', 'roi_thresh_frac', 'max_match_dist_px', 'temporal_norm_mode', ...
    'frame_start', 'frame_end');
fprintf('Saved metrics: %s\n', out_mat);
disp(Tmetrics);

%% ---- local functions ----
function y = normalize_trace(x, mode)
x = single(x(:)');
switch lower(mode)
    case 'none'
        y = x - median(x, 'omitnan');
    case 'zscore'
        y = zscore_safe(x);
    case 'robust_zscore'
        med = median(x, 'omitnan');
        mad_val = median(abs(x - med), 'omitnan');
        scl = max(1.4826 * mad_val, eps('single'));
        y = (x - med) ./ scl;
    otherwise
        error('Unknown temporal_norm_mode: %s', mode);
end
end

function tr = extract_trace_window(tr_full, frame_start, frame_end, n_frames_ref)
tr_full = single(tr_full(:)');
if numel(tr_full) == n_frames_ref
    tr = tr_full(frame_start:frame_end);
else
    xi = linspace(1, n_frames_ref, numel(tr_full));
    xq = frame_start:frame_end;
    tr = interp1(xi, tr_full, xq, 'linear', 'extrap');
end
tr = single(tr);
end

function z = zscore_safe(x)
x = single(x(:)');
mu = mean(x, 'omitnan');
sd = std(x, 0, 'omitnan');
if sd < eps('single')
    z = zeros(size(x), 'single');
else
    z = (x - mu) ./ sd;
end
end

function M = read_tiff_chunk_general(tiff_path, info, n_ifd, total_frames, f_begin, n_this)
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
