%% DeepWonder: random 5 ROI + nearest 5 neighbors (spatial + temporal)
% Source folder:
% R:\data\manipulandum\target_reach\250810-Ras2-GC#78\RSM_250810-Ras2-GC#78_reg_20250812-1449\mat
%
% Output layout:
% - nexttile(5,3)
% - col 1: spatial (zoomed around target ROI and neighbors)
% - col 2: temporal traces
% - col 3: correlation matrix (target + neighbors)

src_dir = 'R:\data\manipulandum\target_reach\250810-Ras2-GC#78\RSM_250810-Ras2-GC#78_reg_20250812-1449\mat';
out_png = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-production', 'deepwonder_random5_nearest5_spatial_temporal.png');
rng_seed = 1;
n_query = 5;
n_neighbors = 8;
pad_px = 20;

files = dir(fullfile(src_dir, 'results_*.mat'));
if isempty(files)
    error('No results_*.mat found in %s', src_dir);
end
[~, ix] = sort({files.name});
files = files(ix);

fprintf('Scanning centroids from %d files...\n', numel(files));

% Pass 1: collect global centroid table and location map (file/local index).
cx = [];
cy = [];
file_idx = [];
local_idx = [];
for fi = 1:numel(files)
    fpath = fullfile(files(fi).folder, files(fi).name);
    D = load(fpath, 'final_mask_list');
    L = D.final_mask_list;
    n = numel(L);
    cxi = nan(n, 1);
    cyi = nan(n, 1);
    for k = 1:n
        e = L{k};
        if isfield(e, 'centroid') && ~isempty(e.centroid)
            c = double(e.centroid(:)');
            if numel(c) >= 2
                cyi(k) = c(1);
                cxi(k) = c(2);
                continue;
            end
        end
        if isfield(e, 'position') && ~isempty(e.position)
            p = double(e.position);
            cyi(k) = mean(p(:, 1));
            cxi(k) = mean(p(:, 2));
        end
    end
    valid = isfinite(cxi) & isfinite(cyi);
    cx = [cx; cxi(valid)]; %#ok<AGROW>
    cy = [cy; cyi(valid)]; %#ok<AGROW>
    file_idx = [file_idx; fi * ones(sum(valid), 1)]; %#ok<AGROW>
    local_idx = [local_idx; find(valid)]; %#ok<AGROW>
    clear D L
    fprintf('  %s: %d valid cells\n', files(fi).name, sum(valid));
end

n_cells = numel(cx);
if n_cells == 0
    error('No valid cells were found in final_mask_list.');
end
fprintf('Total valid cells: %d\n', n_cells);

rng(rng_seed);
nq = min(n_query, n_cells);
q_global = randperm(n_cells, nq);

% Neighbor selection table (global IDs in centroid table).
neighbor_table = cell(nq, 1);
needed = [];
for r = 1:nq
    q = q_global(r);
    d2 = (cx - cx(q)).^2 + (cy - cy(q)).^2;
    [~, ord] = sort(d2, 'ascend');
    nn = ord(2:min(n_neighbors + 1, n_cells));
    ids = [q; nn(:)];
    neighbor_table{r} = ids;
    needed = [needed; ids(:)]; %#ok<AGROW>
end
needed = unique(needed);

% Pass 2: load only needed cells' position/value/trace.
roi_data = struct('position', cell(n_cells, 1), ...
                  'value', cell(n_cells, 1), ...
                  'trace', cell(n_cells, 1));
for fi = unique(file_idx(needed))'
    fpath = fullfile(files(fi).folder, files(fi).name);
    D = load(fpath, 'final_mask_list');
    L = D.final_mask_list;
    gidx = needed(file_idx(needed) == fi);
    for ii = 1:numel(gidx)
        g = gidx(ii);
        k = local_idx(g);
        e = L{k};
        roi_data(g).position = double(e.position);
        if isfield(e, 'value') && ~isempty(e.value) && numel(e.value) == size(e.position, 1)
            roi_data(g).value = single(e.value(:));
        else
            roi_data(g).value = ones(size(e.position, 1), 1, 'single');
        end
        roi_data(g).trace = single(e.trace(:)');
    end
    clear D L
end

fig = figure('Color', 'w', 'Position', [60 60 1650 2100]);
tlo = tiledlayout(5, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf('DeepWonder random %d ROI + nearest %d', nq, n_neighbors));

for r = 1:n_query
    if r > nq
        nexttile((r - 1) * 3 + 1); axis off; text(0.1, 0.5, 'No ROI available');
        nexttile((r - 1) * 3 + 2); axis off; text(0.1, 0.5, 'No ROI available');
        nexttile((r - 1) * 3 + 3); axis off; text(0.1, 0.5, 'No ROI available');
        continue;
    end

    ids = neighbor_table{r};
    cmap = lines(numel(ids));
    cmap(1, :) = [1.0, 0.2, 0.1]; % query ROI

    % Local spatial window around selected ROIs.
    x_all = [];
    y_all = [];
    for j = 1:numel(ids)
        p = roi_data(ids(j)).position;
        y_all = [y_all; p(:, 1)]; %#ok<AGROW>
        x_all = [x_all; p(:, 2)]; %#ok<AGROW>
    end
    xmin = floor(min(x_all)) - pad_px;
    xmax = ceil(max(x_all)) + pad_px;
    ymin = floor(min(y_all)) - pad_px;
    ymax = ceil(max(y_all)) + pad_px;

    ax_sp = nexttile((r - 1) * 3 + 1);
    hold on;
    win_h = ymax - ymin + 1;
    win_w = xmax - xmin + 1;
    imagesc([xmin xmax], [ymin ymax], zeros(win_h, win_w, 'single'));
    colormap(ax_sp, gray(256));
    for j = 1:numel(ids)
        p = roi_data(ids(j)).position;
        v = roi_data(ids(j)).value;
        % Build local ROI image from sparse pixel list, then draw contour.
        roi_im = zeros(win_h, win_w, 'single');
        yi = round(p(:, 1)) - ymin + 1;
        xi = round(p(:, 2)) - xmin + 1;
        valid_px = yi >= 1 & yi <= win_h & xi >= 1 & xi <= win_w;
        yi = yi(valid_px);
        xi = xi(valid_px);
        vv = v(valid_px);
        lin = sub2ind([win_h, win_w], yi, xi);
        roi_im(lin) = max(roi_im(lin), vv);
        level = max(roi_im(:)) * 0.30;
        if level > 0
            contour(xmin:xmax, ymin:ymax, roi_im, [level level], ...
                'Color', cmap(j, :), 'LineWidth', 1.6);
        end
        ccx = mean(p(:, 2)); ccy = mean(p(:, 1));
        plot(ccx, ccy, 'o', 'Color', cmap(j, :), 'MarkerFaceColor', cmap(j, :), 'MarkerSize', 4);
    end
    set(gca, 'YDir', 'reverse');
    xlim([xmin xmax]); ylim([ymin ymax]);
    axis image;
    grid on;
    title(sprintf('ROI %d + nearest %d (zoom)', ids(1), numel(ids) - 1), 'Interpreter', 'none');
    hold off;

    nexttile((r - 1) * 3 + 2);
    hold on;
    Tmat = zeros(numel(ids), numel(roi_data(ids(1)).trace), 'single');
    for j = 1:numel(ids)
        tr = roi_data(ids(j)).trace;
        tr = tr - median(tr);
        Tmat(j, :) = tr;
        plot(tr, 'Color', cmap(j, :), 'LineWidth', 1.0);
    end
    grid on; box on;
    xlabel('Frame'); ylabel('a.u.');
    title('temporal_weight (trace)');
    legend(strcat("ROI ", string(ids)), 'Location', 'northeastoutside');
    hold off;

    ax_corr = nexttile((r - 1) * 3 + 3);
    C = corrcoef(double(Tmat'));
    imagesc(C, [-1 1]);
    axis image;
    colormap(ax_corr, parula(256));
    clim(ax_corr, [-1 1]);
    title('trace correlation');
    xticks(1:numel(ids));
    yticks(1:numel(ids));
    labels = strcat("R", string(ids));
    xticklabels(labels);
    yticklabels(labels);
    xtickangle(45);
end

exportgraphics(fig, out_png, 'Resolution', 180);
fprintf('Saved: %s\n', out_png);
