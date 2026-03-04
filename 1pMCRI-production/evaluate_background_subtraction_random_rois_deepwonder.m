%% Evaluate DeepWonder background subtraction quality on random ROIs
% For random ROIs, plot:
% 1) ROI spatial mask over raw summary image
% 2) DeepWonder temporal trace
% 3) Raw intensity from the same ROI and local background
% 4) Temporal trace vs (raw - background)
%
% Also generates an overlap-pair figure for ROI pairs with spatial overlap.

clearvars;

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

deep_dir = 'R:\data\manipulandum\target_reach\demo_data\RSM_250810-Ras2-GC#78_reg_s_crop_20260304-1115\mat';
tiff_path = find_tiff_path(script_dir, '250810-Ras2-GC#78_reg_s_crop.tif');

n_query = 10;
n_overlap_pairs = 10;
rng_seed = 7;
bg_radius_px = 30;
chunk_frames = 2000;
mask_level = 0.30; % contour threshold fraction of ROI max
roi_core_fraction = 0.20; % keep nearest XX of ROI pixels from centroid

if ~isfolder(deep_dir)
    error('DeepWonder directory not found: %s', deep_dir);
end
if ~isfile(tiff_path)
    error('Raw TIFF not found: %s', tiff_path);
end

files = dir(fullfile(deep_dir, 'results_*.mat'));
if isempty(files)
    error('No results_*.mat found in %s', deep_dir);
end
[~, ix] = sort({files.name});
files = files(ix);

info = imfinfo(tiff_path);
n_ifd = numel(info);
h = info(1).Height;
w = info(1).Width;
n_frames_raw = numel(info);
if n_frames_raw == 1 && isfield(info(1), 'ImageDescription')
    tok = regexp(info(1).ImageDescription, 'images=(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        n_frames_raw = str2double(tok{1});
    end
end
if n_frames_raw < 10
    error('Too few frames in TIFF: %d', n_frames_raw);
end

fprintf('Movie size: %d x %d x %d frames\n', h, w, n_frames_raw);
fprintf('Loading DeepWonder ROIs from %d files...\n', numel(files));

% Pass 1: load ROI metadata
roi = struct('file_id', {}, 'local_id', {}, 'position', {}, 'value', {}, ...
    'trace', {}, 'cx', {}, 'cy', {}, 'mask_idx', {});
coord_offset = [];
for fi = 1:numel(files)
    fpath = fullfile(files(fi).folder, files(fi).name);
    D = load(fpath, 'final_mask_list');
    L = D.final_mask_list;
    valid_count = 0;
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
            fprintf('Detected DeepWonder coordinate offset = +%d\n', coord_offset);
        end
        y = p(:, 1) + coord_offset;
        x = p(:, 2) + coord_offset;
        keep = y >= 1 & y <= h & x >= 1 & x <= w;
        if ~any(keep)
            continue;
        end
        y = y(keep);
        x = x(keep);
        if isfield(e, 'value') && ~isempty(e.value) && numel(e.value) == size(p, 1)
            v = single(e.value(keep));
        else
            v = ones(numel(y), 1, 'single');
        end
        tr = single(e.trace(:)');
        idx = sub2ind([h, w], round(y), round(x));

        r.file_id = fi;
        r.local_id = k;
        r.position = [y(:), x(:)];
        r.value = v(:);
        r.trace = tr;
        if isfield(e, 'centroid') && ~isempty(e.centroid) && numel(e.centroid) >= 2
            c = double(e.centroid(:)');
            cy0 = single(c(1) + coord_offset);
            cx0 = single(c(2) + coord_offset);
        else
            cy0 = single(mean(y));
            cx0 = single(mean(x));
        end

        % Restrict ROI to centroid-nearest core pixels (e.g. 20% area).
        [idx, y, x, v] = restrict_roi_core_by_centroid(idx, y, x, v, cy0, cx0, roi_core_fraction);
        if isempty(idx)
            continue;
        end

        % Update centroid after core restriction.
        if all(v == 0)
            r.cy = single(mean(y));
            r.cx = single(mean(x));
        else
            wsum = sum(v);
            r.cy = single(sum(y .* v) / max(wsum, eps('single')));
            r.cx = single(sum(x .* v) / max(wsum, eps('single')));
        end
        r.position = [y(:), x(:)];
        r.value = v(:);
        r.mask_idx = idx(:);
        roi(end + 1) = r; %#ok<AGROW>
        valid_count = valid_count + 1;
    end
    fprintf('  %s: %d valid cells\n', files(fi).name, valid_count);
end

n_cells = numel(roi);
if n_cells == 0
    error('No valid DeepWonder cells were loaded.');
end
fprintf('Total valid DeepWonder cells: %d\n', n_cells);

% Temporal trace matrix (cells x frames)
n_frames = min(n_frames_raw, min(cellfun(@numel, {roi.trace})));
T = zeros(n_cells, n_frames, 'single');
cx = zeros(n_cells, 1, 'single');
cy = zeros(n_cells, 1, 'single');
for i = 1:n_cells
    T(i, :) = roi(i).trace(1:n_frames);
    cx(i) = roi(i).cx;
    cy(i) = roi(i).cy;
end

mask_nonempty = ~cellfun(@isempty, {roi.mask_idx})';
valid = find(isfinite(cx) & isfinite(cy) & mask_nonempty);
rng(rng_seed);
n_pick = min(n_query, numel(valid));
pick = valid(randperm(numel(valid), n_pick));

% Background masks for random ROIs
[Yg, Xg] = ndgrid(single(1:h), single(1:w));
bg_masks = cell(n_pick, 1);
for i = 1:n_pick
    rid = pick(i);
    dist2 = (Xg - cx(rid)).^2 + (Yg - cy(rid)).^2;
    ring = (dist2 <= bg_radius_px^2);
    ring(roi(rid).mask_idx) = false;
    if ~any(ring(:))
        error('Empty background mask for ROI %d. Increase bg_radius_px.', rid);
    end
    bg_masks{i} = find(ring);
end

% Raw extraction for random ROIs
raw_roi = zeros(n_pick, n_frames, 'single');
raw_bg = zeros(n_pick, n_frames, 'single');
summary_acc = zeros(h, w, 'double');

n_chunks = ceil(n_frames / chunk_frames);
for c = 1:n_chunks
    f_begin = (c - 1) * chunk_frames + 1;
    f_end = min(c * chunk_frames, n_frames);
    n_this = f_end - f_begin + 1;
    fprintf('Reading TIFF chunk %d/%d (frames %d-%d)\n', c, n_chunks, f_begin, f_end);
    M = read_tiff_chunk_general(tiff_path, info, n_ifd, n_frames_raw, f_begin, n_this);
    summary_acc = summary_acc + sum(double(M), 3);
    Mr = reshape(M, h * w, n_this);
    for i = 1:n_pick
        rid = pick(i);
        raw_roi(i, f_begin:f_end) = mean(Mr(roi(rid).mask_idx, :), 1);
        raw_bg(i, f_begin:f_end) = mean(Mr(bg_masks{i}, :), 1);
    end
end
summary_img = single(summary_acc / n_frames);
summary_img = summary_img - min(summary_img(:));
if max(summary_img(:)) > 0
    summary_img = summary_img ./ max(summary_img(:));
end

Tpick = T(pick, 1:n_frames);
raw_sub = raw_roi - raw_bg;

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

%% Figure 1: random ROIs
crop_half = max(bg_radius_px, 40);
fig1 = figure('Color', 'w', 'Position', [80 60 1900 2200]);
tlo1 = tiledlayout(n_pick, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo1, sprintf(['DeepWonder random ROI background-subtraction eval (n=%d, radius=%d px)\n', ...
    'Median corr: TW-raw=%.3f, TW-bg=%.3f, TW-(raw-bg)=%.3f'], ...
    n_pick, bg_radius_px, median(r_raw, 'omitnan'), median(r_bg, 'omitnan'), median(r_sub, 'omitnan')));

for i = 1:n_pick
    rid = pick(i);
    tw = zscore_safe(Tpick(i, :));
    rr = zscore_safe(raw_roi(i, :));
    rb = zscore_safe(raw_bg(i, :));
    rs = zscore_safe(raw_sub(i, :));

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
    roi_im = zeros(ymax - ymin + 1, xmax - xmin + 1, 'single');
    yp = round(roi(rid).position(:, 1)) - ymin + 1;
    xp = round(roi(rid).position(:, 2)) - xmin + 1;
    vv = roi(rid).value;
    good = yp >= 1 & yp <= size(roi_im, 1) & xp >= 1 & xp <= size(roi_im, 2);
    yp = yp(good); xp = xp(good); vv = vv(good);
    lin = sub2ind(size(roi_im), yp, xp);
    roi_im(lin) = max(roi_im(lin), vv);
    th = max(roi_im(:)) * mask_level;
    if th > 0
        contour(xmin:xmax, ymin:ymax, roi_im, [th th], 'Color', [1 0.2 0.1], 'LineWidth', 1.4);
    end
    plot(cx(rid), cy(rid), 'o', 'Color', [1 0.2 0.1], 'MarkerFaceColor', [1 0.2 0.1], 'MarkerSize', 4);
    hold off;
    axis image off;
    set(gca, 'YDir', 'reverse');
    title(sprintf('ROI %d spatial', rid));

    nexttile((i - 1) * 4 + 2);
    plot(tw, 'k-', 'LineWidth', 0.9);
    grid on; ylabel('z');
    title(sprintf('ROI %d temporal trace', rid));
    if i == n_pick, xlabel('Frame'); end

    nexttile((i - 1) * 4 + 3);
    hold on;
    plot(rr, 'Color', [0.1 0.45 0.9], 'LineWidth', 0.9);
    plot(rb, 'Color', [0.85 0.25 0.2], 'LineWidth', 0.9);
    hold off;
    grid on;
    title(sprintf('Raw ROI / BG (corr TW: %.2f / %.2f)', r_raw(i), r_bg(i)));
    if i == 1, legend({'ROI raw', 'BG raw'}, 'Location', 'best'); end
    if i == n_pick, xlabel('Frame'); end

    nexttile((i - 1) * 4 + 4);
    hold on;
    plot(tw, 'k-', 'LineWidth', 0.8);
    plot(rs, 'Color', [0.2 0.7 0.2], 'LineWidth', 0.9);
    hold off;
    grid on;
    title(sprintf('TW vs (ROI-BG), corr=%.2f', r_sub(i)));
    if i == 1, legend({'TW', 'ROI-BG raw'}, 'Location', 'best'); end
    if i == n_pick, xlabel('Frame'); end
end

out_png1 = fullfile(script_dir, 'deepwonder_background_subtraction_eval_random10.png');
exportgraphics(fig1, out_png1, 'Resolution', 180);
fprintf('Saved figure: %s\n', out_png1);

Tmetrics = table(pick(:), r_raw, r_bg, r_sub, ...
    'VariableNames', {'roi_id', 'corr_tw_raw', 'corr_tw_bg', 'corr_tw_raw_minus_bg'});
out_mat1 = fullfile(script_dir, 'deepwonder_background_subtraction_eval_random10.mat');
save(out_mat1, 'Tmetrics', 'pick', 'r_raw', 'r_bg', 'r_sub', 'bg_radius_px', ...
    'deep_dir', 'tiff_path');
fprintf('Saved metrics: %s\n', out_mat1);
disp(Tmetrics);

%% Figure 2: overlapping ROI pairs
fprintf('Finding overlapping ROI pairs...\n');
valid_n = numel(valid);
nnz_est = sum(cellfun(@numel, {roi(valid).mask_idx}));
B = spalloc(h * w, valid_n, nnz_est);
for j = 1:valid_n
    B(roi(valid(j)).mask_idx, j) = true;
end
O = spones(B' * B);
O = triu(O, 1);
[ii, jj] = find(O);

if isempty(ii)
    warning('No overlapping ROI pairs found.');
else
    rng(rng_seed + 1);
    n_pair = min(n_overlap_pairs, numel(ii));
    sel = randperm(numel(ii), n_pair);
    pair_a = valid(ii(sel));
    pair_b = valid(jj(sel));
    pair_ids = unique([pair_a; pair_b]);
    n_pair_ids = numel(pair_ids);

    pair_bg_masks = cell(n_pair_ids, 1);
    for k = 1:n_pair_ids
        rid = pair_ids(k);
        dist2 = (Xg - cx(rid)).^2 + (Yg - cy(rid)).^2;
        ring = (dist2 <= bg_radius_px^2);
        ring(roi(rid).mask_idx) = false;
        if ~any(ring(:))
            ring = true(size(ring));
            ring(roi(rid).mask_idx) = false;
        end
        pair_bg_masks{k} = find(ring);
    end

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
            raw_pair(k, f_begin:f_end) = mean(Mr(roi(rid).mask_idx, :), 1);
            raw_pair_bg(k, f_begin:f_end) = mean(Mr(pair_bg_masks{k}, :), 1);
        end
    end

    id_to_local = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for k = 1:n_pair_ids
        id_to_local(double(pair_ids(k))) = k;
    end

    fig2 = figure('Color', 'w', 'Position', [100 40 2000 2000]);
    tlo2 = tiledlayout(n_pair, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tlo2, sprintf('DeepWonder overlapping ROI pairs eval (n=%d, radius=%d px)', n_pair, bg_radius_px));

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

        overlap_px = numel(intersect(roi(a).mask_idx, roi(b).mask_idx));
        pair_metrics.roi_a(p) = a;
        pair_metrics.roi_b(p) = b;
        pair_metrics.overlap_px(p) = overlap_px;
        pair_metrics.corr_a_tw_sub(p) = corr(tw_a(:), rs_a(:), 'rows', 'complete');
        pair_metrics.corr_b_tw_sub(p) = corr(tw_b(:), rs_b(:), 'rows', 'complete');
        pair_metrics.corr_tw_pair(p) = corr(tw_a(:), tw_b(:), 'rows', 'complete');
        pair_metrics.corr_raw_pair(p) = corr(rr_a(:), rr_b(:), 'rows', 'complete');
        pair_metrics.corr_sub_pair(p) = corr(rs_a(:), rs_b(:), 'rows', 'complete');

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
        draw_roi_contour(roi(a), xmin, xmax, ymin, ymax, [1 0.2 0.1], mask_level);
        draw_roi_contour(roi(b), xmin, xmax, ymin, ymax, [0.1 0.7 1.0], mask_level);
        plot(cx(a), cy(a), 'o', 'Color', [1 0.2 0.1], 'MarkerFaceColor', [1 0.2 0.1], 'MarkerSize', 4);
        plot(cx(b), cy(b), 'o', 'Color', [0.1 0.7 1.0], 'MarkerFaceColor', [0.1 0.7 1.0], 'MarkerSize', 4);
        hold off;
        axis image off;
        set(gca, 'YDir', 'reverse');
        title(sprintf('ROI %d vs %d (overlap=%d)', a, b, overlap_px));

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

    out_png2 = fullfile(script_dir, 'deepwonder_background_subtraction_eval_overlap_pairs.png');
    exportgraphics(fig2, out_png2, 'Resolution', 180);
    fprintf('Saved overlap-pair figure: %s\n', out_png2);

    out_mat2 = fullfile(script_dir, 'deepwonder_background_subtraction_eval_overlap_pairs.mat');
    save(out_mat2, 'pair_metrics', 'pair_a', 'pair_b', 'pair_ids', 'bg_radius_px', ...
        'deep_dir', 'tiff_path');
    fprintf('Saved overlap-pair metrics: %s\n', out_mat2);
    disp(pair_metrics);
end

%% ---- local functions ----
function draw_roi_contour(r, xmin, xmax, ymin, ymax, color_rgb, mask_level)
roi_im = zeros(ymax - ymin + 1, xmax - xmin + 1, 'single');
yp = round(r.position(:, 1)) - ymin + 1;
xp = round(r.position(:, 2)) - xmin + 1;
vv = r.value;
good = yp >= 1 & yp <= size(roi_im, 1) & xp >= 1 & xp <= size(roi_im, 2);
yp = yp(good); xp = xp(good); vv = vv(good);
lin = sub2ind(size(roi_im), yp, xp);
roi_im(lin) = max(roi_im(lin), vv);
th = max(roi_im(:)) * mask_level;
if th > 0
    contour(xmin:xmax, ymin:ymax, roi_im, [th th], 'Color', color_rgb, 'LineWidth', 1.4);
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

function [idx_out, y_out, x_out, v_out] = restrict_roi_core_by_centroid(idx, y, x, v, cy, cx, frac)
% Keep the nearest frac (0..1] of ROI pixels from centroid.
frac = max(0, min(1, frac));
if isempty(idx) || frac >= 1
    idx_out = idx;
    y_out = y;
    x_out = x;
    v_out = v;
    return;
end

d2 = (single(y) - single(cy)).^2 + (single(x) - single(cx)).^2;
n_keep = max(1, round(numel(d2) * frac));
[~, ord] = sort(d2, 'ascend');
sel = ord(1:n_keep);

idx_out = idx(sel);
y_out = y(sel);
x_out = x(sel);
v_out = v(sel);
end
