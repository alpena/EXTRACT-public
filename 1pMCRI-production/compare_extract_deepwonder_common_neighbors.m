%% Compare EXTRACT vs DeepWonder using registered spatial maps and neighbor analysis
% 1) Register EXTRACT crop to DeepWonder full FOV using aggregate spatial maps
% 2) Match likely common cells by transformed centroid proximity
% 3) For random query cells, compare query+neighbors (EXTRACT style) across methods
%
% Output:
% - extract_deepwonder_registration.png
% - extract_deepwonder_common_neighbors.png

extract_mat = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-production', 'output_250810-Ras2-GC#78_reg_s_crop.mat');
deep_dir = 'R:\data\manipulandum\target_reach\250810-Ras2-GC#78\RSM_250810-Ras2-GC#78_reg_20250812-1449\mat';
out_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '1pMCRI-production');

rng_seed = 1;
n_query = 5;
n_neighbors = 8;
max_match_dist_px = 15;
fixed_half_window = 120;

if ~isfile(extract_mat)
    error('EXTRACT output not found: %s', extract_mat);
end
deep_files = dir(fullfile(deep_dir, 'results_*.mat'));
if isempty(deep_files)
    error('No DeepWonder results_*.mat found: %s', deep_dir);
end
[~, sidx] = sort({deep_files.name});
deep_files = deep_files(sidx);

%% Load EXTRACT outputs
E = load(extract_mat, 'output');
if ~isfield(E, 'output') || ~isfield(E.output, 'spatial_weights') || ~isfield(E.output, 'temporal_weights')
    error('Invalid EXTRACT output format in %s', extract_mat);
end
S_ex = E.output.spatial_weights;
if isa(S_ex, 'ndSparse')
    S_ex = full(S_ex);
end
S_ex = single(S_ex);
T_ex = single(E.output.temporal_weights'); % cells x time
[h_ex, w_ex, n_ex] = size(S_ex);

% EXTRACT aggregate map + centroids
map_ex = sum(max(S_ex, 0), 3);
map_ex = map_ex - min(map_ex(:));
if max(map_ex(:)) > 0
    map_ex = map_ex ./ max(map_ex(:));
end
[Yex, Xex] = ndgrid(single(1:h_ex), single(1:w_ex));
cx_ex = nan(n_ex, 1, 'single');
cy_ex = nan(n_ex, 1, 'single');
for i = 1:n_ex
    m = max(S_ex(:, :, i), 0);
    sm = sum(m(:));
    if sm > 0
        cx_ex(i) = sum(m(:) .* Xex(:)) / sm;
        cy_ex(i) = sum(m(:) .* Yex(:)) / sm;
    end
end
valid_ex = find(isfinite(cx_ex) & isfinite(cy_ex));

%% Load DeepWonder metadata + aggregate map
% Deep positions are 0-based in these files. Convert to MATLAB 1-based.
h_dw = 1700;
w_dw = 1900;
map_dw = zeros(h_dw, w_dw, 'single');
cx_dw = [];
cy_dw = [];
file_id_dw = [];
local_id_dw = [];

fprintf('Building DeepWonder aggregate map and centroid table...\n');
for fi = 1:numel(deep_files)
    fpath = fullfile(deep_files(fi).folder, deep_files(fi).name);
    D = load(fpath, 'final_mask_list');
    L = D.final_mask_list;
    nloc = numel(L);
    cx_loc = nan(nloc, 1, 'single');
    cy_loc = nan(nloc, 1, 'single');

    for k = 1:nloc
        e = L{k};
        p = double(e.position);
        if isempty(p)
            continue;
        end
        y = p(:, 1) + 1;
        x = p(:, 2) + 1;
        valid = y >= 1 & y <= h_dw & x >= 1 & x <= w_dw;
        if ~any(valid)
            continue;
        end
        y = y(valid);
        x = x(valid);
        if isfield(e, 'value') && ~isempty(e.value) && numel(e.value) == size(p, 1)
            v = single(e.value(valid));
        else
            v = ones(numel(y), 1, 'single');
        end

        % Aggregate map accumulation
        idx = sub2ind([h_dw, w_dw], round(y), round(x));
        map_dw(idx) = map_dw(idx) + v;

        if isfield(e, 'centroid') && ~isempty(e.centroid) && numel(e.centroid) >= 2
            c = double(e.centroid(:)');
            cy_loc(k) = single(c(1) + 1);
            cx_loc(k) = single(c(2) + 1);
        else
            cy_loc(k) = single(mean(y));
            cx_loc(k) = single(mean(x));
        end
    end

    valid_loc = isfinite(cx_loc) & isfinite(cy_loc);
    cx_dw = [cx_dw; cx_loc(valid_loc)]; %#ok<AGROW>
    cy_dw = [cy_dw; cy_loc(valid_loc)]; %#ok<AGROW>
    file_id_dw = [file_id_dw; fi * ones(sum(valid_loc), 1, 'uint16')]; %#ok<AGROW>
    local_id_dw = [local_id_dw; uint32(find(valid_loc))]; %#ok<AGROW>
    fprintf('  %s: %d valid cells\n', deep_files(fi).name, sum(valid_loc));
end
map_dw = map_dw - min(map_dw(:));
if max(map_dw(:)) > 0
    map_dw = map_dw ./ max(map_dw(:));
end
n_dw = numel(cx_dw);
if n_dw == 0
    error('No valid DeepWonder cells found.');
end

%% Registration: find EXTRACT crop location in DeepWonder FOV by NCC
c = normxcorr2(map_ex, map_dw);
[~, imax] = max(abs(c(:)));
[ypeak, xpeak] = ind2sub(size(c), imax);
yoff = ypeak - size(map_ex, 1);
xoff = xpeak - size(map_ex, 2);
row0 = yoff + 1;
col0 = xoff + 1;
fprintf('Registration offset: row0=%d, col0=%d\n', row0, col0);

% Transformed EXTRACT centroids in DeepWonder coordinates
cx_ex_dw = cx_ex + single(col0 - 1);
cy_ex_dw = cy_ex + single(row0 - 1);

%% Cell matching by centroid proximity (one-to-one greedy)
% Compute nearest DeepWonder cell for each valid EXTRACT cell.
ex_idx = valid_ex(:);
exx = double(cx_ex_dw(ex_idx));
exy = double(cy_ex_dw(ex_idx));
dwx = double(cx_dw(:)');
dwy = double(cy_dw(:)');

nearest_dw = zeros(numel(ex_idx), 1);
nearest_d = inf(numel(ex_idx), 1);
for i = 1:numel(ex_idx)
    d2 = (dwx - exx(i)).^2 + (dwy - exy(i)).^2;
    [v, j] = min(d2);
    nearest_dw(i) = j;
    nearest_d(i) = sqrt(v);
end

[~, ord] = sort(nearest_d, 'ascend');
used_dw = false(n_dw, 1);
match_ex = [];
match_dw = [];
match_d = [];
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
    match_ex(end+1, 1) = ex_idx(i); %#ok<AGROW>
    match_dw(end+1, 1) = j; %#ok<AGROW>
    match_d(end+1, 1) = nearest_d(i); %#ok<AGROW>
end
fprintf('Matched cells: %d (threshold %.1f px)\n', numel(match_ex), max_match_dist_px);
if numel(match_ex) < 2
    error('Too few matched cells (%d).', numel(match_ex));
end

% Mapping from EXTRACT cell index -> DeepWonder global index
ex_to_dw = containers.Map('KeyType', 'uint32', 'ValueType', 'uint32');
for i = 1:numel(match_ex)
    ex_to_dw(uint32(match_ex(i))) = uint32(match_dw(i));
end

%% Select random matched query cells and neighbors in EXTRACT space
rng(rng_seed);
nq = min(n_query, numel(match_ex));
q_ex = match_ex(randperm(numel(match_ex), nq));

% Cache for deep cells we need to draw/plot
needed_dw = [];
neighbor_ex_sets = cell(nq, 1);
neighbor_dw_sets = cell(nq, 1);
for r = 1:nq
    q = q_ex(r);
    exx0 = double(cx_ex_dw(match_ex));
    exy0 = double(cy_ex_dw(match_ex));
    d2 = (exx0 - double(cx_ex_dw(q))).^2 + (exy0 - double(cy_ex_dw(q))).^2;
    [~, od] = sort(d2, 'ascend');
    nn = match_ex(od(2:min(n_neighbors + 1, numel(od))));
    ids_ex = [q; nn(:)];
    ids_dw = zeros(size(ids_ex), 'uint32');
    for k = 1:numel(ids_ex)
        ids_dw(k) = ex_to_dw(uint32(ids_ex(k)));
    end
    neighbor_ex_sets{r} = ids_ex;
    neighbor_dw_sets{r} = double(ids_dw);
    needed_dw = [needed_dw; double(ids_dw(:))]; %#ok<AGROW>
end
needed_dw = unique(needed_dw);

%% Load DeepWonder positions/traces only for needed matched cells
deep_cache = struct('position', cell(n_dw, 1), 'trace', cell(n_dw, 1));
for fi = unique(file_id_dw(needed_dw))'
    fpath = fullfile(deep_files(fi).folder, deep_files(fi).name);
    D = load(fpath, 'final_mask_list');
    L = D.final_mask_list;
    ids = needed_dw(file_id_dw(needed_dw) == fi);
    for ii = 1:numel(ids)
        g = ids(ii);
        k = double(local_id_dw(g));
        e = L{k};
        p = double(e.position);
        deep_cache(g).position = [p(:,1)+1, p(:,2)+1];
        deep_cache(g).trace = single(e.trace(:)');
    end
end

%% Figure 1: registration overview
f_reg = figure('Color', 'w', 'Position', [80 80 1400 700]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
imagesc(map_dw); axis image off; colormap(gray);
title('DeepWonder aggregate spatial map');
hold on;
rectangle('Position', [col0, row0, w_ex, h_ex], 'EdgeColor', [1 0 0], 'LineWidth', 2);
hold off;
nexttile;
overlay = zeros(h_dw, w_dw, 3, 'single');
overlay(:, :, 2) = map_dw; % green
r1 = max(1, row0); r2 = min(h_dw, row0 + h_ex - 1);
c1 = max(1, col0); c2 = min(w_dw, col0 + w_ex - 1);
ex_crop = map_ex(1:(r2-r1+1), 1:(c2-c1+1));
overlay(r1:r2, c1:c2, 1) = ex_crop; % red
imagesc(overlay); axis image off;
title('Registration overlay (red=EXTRACT, green=DeepWonder)');
out_reg = fullfile(out_dir, 'extract_deepwonder_registration.png');
exportgraphics(f_reg, out_reg, 'Resolution', 180);

%% Figure 2: matched query + neighbors comparison
f_cmp = figure('Color', 'w', 'Position', [40 40 2200 2100]);
tlo = tiledlayout(5, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf('Matched EXTRACT vs DeepWonder (query=%d, neighbors=%d)', nq, n_neighbors));

for r = 1:n_query
    if r > nq
        for cc = 1:4
            nexttile((r - 1) * 4 + cc); axis off; text(0.1,0.5,'No match');
        end
        continue;
    end

    ids_ex = neighbor_ex_sets{r};
    ids_dw = neighbor_dw_sets{r};
    nset = numel(ids_ex);
    cmap = lines(nset);
    cmap(1,:) = [1.0 0.2 0.1];

    q = ids_ex(1);
    qx = double(cx_ex_dw(q));
    qy = double(cy_ex_dw(q));
    xmin = max(1, floor(qx - fixed_half_window));
    xmax = min(w_dw, ceil(qx + fixed_half_window));
    ymin = max(1, floor(qy - fixed_half_window));
    ymax = min(h_dw, ceil(qy + fixed_half_window));

    % Col1: spatial overlay (EXTRACT contours in solid, DeepWonder dots in dashed color)
    ax1 = nexttile((r - 1) * 4 + 1);
    imagesc(xmin:xmax, ymin:ymax, map_dw(ymin:ymax, xmin:xmax)); axis image off;
    set(ax1, 'YDir', 'reverse');
    colormap(ax1, gray(256));
    hold on;
    for j = 1:nset
        roi_ex = S_ex(:, :, ids_ex(j));
        % Draw EXTRACT contour in deep coordinates
        t = max(roi_ex(:)) * 0.2;
        if t > 0
            contour((1:w_ex) + (col0 - 1), (1:h_ex) + (row0 - 1), roi_ex, [t t], ...
                'Color', cmap(j,:), 'LineWidth', 1.4);
        end
        p_dw = deep_cache(ids_dw(j)).position;
        scatter(p_dw(:,2), p_dw(:,1), 6, 'MarkerEdgeColor', cmap(j,:), ...
            'MarkerFaceColor', cmap(j,:), 'MarkerFaceAlpha', 0.35, 'MarkerEdgeAlpha', 0.35);
    end
    xlim([xmin xmax]); ylim([ymin ymax]);
    title(sprintf('Spatial overlay (Q EX=%d / DW=%d)', ids_ex(1), ids_dw(1)));
    hold off;

    % Col2: temporal overlay (EXTRACT solid, DeepWonder dashed)
    ax2 = nexttile((r - 1) * 4 + 2);
    hold(ax2, 'on');
    Tex = zeros(nset, size(T_ex,2), 'single');
    Tdw = zeros(nset, size(T_ex,2), 'single');
    for j = 1:nset
        tr_ex = single(T_ex(ids_ex(j), :));
        tr_dw = single(deep_cache(ids_dw(j)).trace);
        if numel(tr_dw) ~= numel(tr_ex)
            tr_dw = interp1(1:numel(tr_dw), tr_dw, linspace(1, numel(tr_dw), numel(tr_ex)), 'linear', 'extrap');
        end
        tr_ex = tr_ex - median(tr_ex);
        tr_dw = tr_dw - median(tr_dw);
        Tex(j,:) = tr_ex;
        Tdw(j,:) = tr_dw;
        plot(tr_ex, '-', 'Color', cmap(j,:), 'LineWidth', 1.1);
        plot(tr_dw, '--', 'Color', cmap(j,:), 'LineWidth', 1.0);
    end
    grid on; box on;
    xlabel('Frame'); ylabel('a.u.');
    title('Temporal (solid=EXTRACT, dashed=DeepWonder)');
    hold(ax2, 'off');

    % Col3: EXTRACT correlation
    ax3 = nexttile((r - 1) * 4 + 3);
    Cex = corrcoef(double(Tex'));
    imagesc(Cex, [-1 1]); axis image;
    colormap(ax3, parula(256)); clim(ax3, [-1 1]);
    title('EXTRACT corr');
    xticks(1:nset); yticks(1:nset);

    % Col4: DeepWonder correlation
    ax4 = nexttile((r - 1) * 4 + 4);
    Cdw = corrcoef(double(Tdw'));
    imagesc(Cdw, [-1 1]); axis image;
    colormap(ax4, parula(256)); clim(ax4, [-1 1]);
    title('DeepWonder corr');
    xticks(1:nset); yticks(1:nset);
end

out_cmp = fullfile(out_dir, 'extract_deepwonder_common_neighbors.png');
exportgraphics(f_cmp, out_cmp, 'Resolution', 180);

fprintf('Saved: %s\n', out_reg);
fprintf('Saved: %s\n', out_cmp);
