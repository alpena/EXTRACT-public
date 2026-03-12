function out = compare_extract_runs_spatial_overlay(varargin)
% compare_extract_runs_spatial_overlay
% Compare two EXTRACT output MAT files from the same FOV and export
% aggregate / contour overlay figures.

ensure_extract_paths();

p = inputParser;
addParameter(p, 'base_mat', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'comp_mat', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'out_dir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'sample_count', 12, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'max_match_dist_px', 12, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'contour_level_frac', 0.20, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
parse(p, varargin{:});

base_mat = char(p.Results.base_mat);
comp_mat = char(p.Results.comp_mat);
out_dir = char(p.Results.out_dir);
sample_count = double(p.Results.sample_count);
max_match_dist_px = double(p.Results.max_match_dist_px);
contour_level_frac = double(p.Results.contour_level_frac);

if isempty(base_mat) || ~isfile(base_mat)
    error('base_mat not found: %s', base_mat);
end
if isempty(comp_mat) || ~isfile(comp_mat)
    error('comp_mat not found: %s', comp_mat);
end
if isempty(out_dir)
    out_dir = fileparts(base_mat);
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

base = load_extract_spatial(base_mat);
comp = load_extract_spatial(comp_mat);
if base.h ~= comp.h || base.w ~= comp.w
    error('FOV mismatch: base=[%d %d], comp=[%d %d]', base.h, base.w, comp.h, comp.w);
end

matched = match_cells_by_centroid(base.cx, base.cy, comp.cx, comp.cy, max_match_dist_px);

fig1 = figure('Color', 'w', 'Position', [60 60 1800 600]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
imagesc(base.map);
axis image off;
colormap(gray(256));
title(sprintf('Baseline aggregate (n=%d)', base.n_cells), 'Interpreter', 'none');

nexttile;
imagesc(comp.map);
axis image off;
colormap(gray(256));
title(sprintf('6x6 aggregate (n=%d)', comp.n_cells), 'Interpreter', 'none');

nexttile;
overlay = zeros(base.h, base.w, 3, 'single');
overlay(:, :, 1) = base.map;
overlay(:, :, 2) = comp.map;
imagesc(overlay);
axis image off;
title(sprintf('Overlay: red=baseline, green=6x6, matched=%d', numel(matched.base_ids)), ...
    'Interpreter', 'none');

agg_png = fullfile(out_dir, 'extract_spatial_overlay_aggregate.png');
exportgraphics(fig1, agg_png, 'Resolution', 180);
close(fig1);

fig2 = figure('Color', 'w', 'Position', [60 60 1800 1500]);
n_show = min(sample_count, numel(matched.base_ids));
n_cols = 4;
n_rows = max(1, ceil(n_show / n_cols));
tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

base_bg = base.map;
for k = 1:n_show
    i = matched.base_ids(k);
    j = matched.comp_ids(k);
    cx = round((base.cx(i) + comp.cx(j)) / 2);
    cy = round((base.cy(i) + comp.cy(j)) / 2);
    half_w = 80;
    xmin = max(1, cx - half_w);
    xmax = min(base.w, cx + half_w);
    ymin = max(1, cy - half_w);
    ymax = min(base.h, cy + half_w);

    nexttile;
    imagesc(xmin:xmax, ymin:ymax, base_bg(ymin:ymax, xmin:xmax));
    axis image off;
    colormap(gray(256));
    hold on;
    draw_roi_contour(base.S(:, :, i), contour_level_frac, [1.0 0.15 0.10], 1.5);
    draw_roi_contour(comp.S(:, :, j), contour_level_frac, [0.10 0.80 0.90], 1.3);
    title(sprintf('B%d vs P%d  d=%.2f px', i, j, matched.dist_px(k)), 'Interpreter', 'none');
    xlim([xmin xmax]);
    ylim([ymin ymax]);
    set(gca, 'YDir', 'reverse');
    hold off;
end

sample_png = fullfile(out_dir, 'extract_spatial_overlay_matched_rois.png');
exportgraphics(fig2, sample_png, 'Resolution', 180);
close(fig2);

out = struct();
out.aggregate_png = agg_png;
out.sample_png = sample_png;
out.n_base_cells = base.n_cells;
out.n_comp_cells = comp.n_cells;
out.n_matched = numel(matched.base_ids);
fprintf('Saved aggregate overlay: %s\n', agg_png);
fprintf('Saved matched ROI overlay: %s\n', sample_png);
fprintf('Matched cells within %.1f px: %d\n', max_match_dist_px, out.n_matched);
end

function data = load_extract_spatial(mat_path)
tmp = load(mat_path, 'output');
if ~isfield(tmp, 'output') || ~isfield(tmp.output, 'spatial_weights')
    error('Invalid EXTRACT output: %s', mat_path);
end

S = tmp.output.spatial_weights;
if isa(S, 'ndSparse')
    S = full(S);
end
S = single(S);

[h, w, n_cells] = size(S);
map = sum(max(S, 0), 3);
map = map - min(map(:));
if max(map(:)) > 0
    map = map ./ max(map(:));
end

[yy, xx] = ndgrid(single(1:h), single(1:w));
cx = nan(n_cells, 1, 'single');
cy = nan(n_cells, 1, 'single');
for i = 1:n_cells
    roi = max(S(:, :, i), 0);
    roi_sum = sum(roi(:));
    if roi_sum <= 0
        continue;
    end
    cx(i) = sum(roi(:) .* xx(:)) / roi_sum;
    cy(i) = sum(roi(:) .* yy(:)) / roi_sum;
end

valid = isfinite(cx) & isfinite(cy);
data = struct( ...
    'path', mat_path, ...
    'S', S, ...
    'h', h, ...
    'w', w, ...
    'n_cells', n_cells, ...
    'map', map, ...
    'cx', cx, ...
    'cy', cy, ...
    'valid', valid);
end

function matched = match_cells_by_centroid(base_cx, base_cy, comp_cx, comp_cy, max_match_dist_px)
base_valid = find(isfinite(base_cx) & isfinite(base_cy));
comp_valid = find(isfinite(comp_cx) & isfinite(comp_cy));

nearest_comp = zeros(numel(base_valid), 1);
nearest_dist = inf(numel(base_valid), 1);
comp_x = double(comp_cx(comp_valid)');
comp_y = double(comp_cy(comp_valid)');

for ii = 1:numel(base_valid)
    i = base_valid(ii);
    d2 = (comp_x - double(base_cx(i))).^2 + (comp_y - double(base_cy(i))).^2;
    [v, j] = min(d2);
    nearest_comp(ii) = comp_valid(j);
    nearest_dist(ii) = sqrt(v);
end

[~, order] = sort(nearest_dist, 'ascend');
used_comp = false(numel(comp_cx), 1);
base_ids = zeros(0, 1);
comp_ids = zeros(0, 1);
dist_px = zeros(0, 1);
for kk = 1:numel(order)
    ii = order(kk);
    if nearest_dist(ii) > max_match_dist_px
        break;
    end
    j = nearest_comp(ii);
    if used_comp(j)
        continue;
    end
    used_comp(j) = true;
    base_ids(end + 1, 1) = base_valid(ii); %#ok<AGROW>
    comp_ids(end + 1, 1) = j; %#ok<AGROW>
    dist_px(end + 1, 1) = nearest_dist(ii); %#ok<AGROW>
end

matched = struct('base_ids', base_ids, 'comp_ids', comp_ids, 'dist_px', dist_px);
end

function draw_roi_contour(roi, level_frac, color_rgb, line_width)
level = max(roi(:)) * level_frac;
if level <= 0
    return;
end
contour(1:size(roi, 2), 1:size(roi, 1), roi, [level level], ...
    'Color', color_rgb, 'LineWidth', line_width);
end

function ensure_extract_paths()
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(repo_root));
end
