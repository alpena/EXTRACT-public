%% Plot random ROI + nearest neighbors (spatial + temporal + correlation)
% Layout: nexttile(5,3), col1=spatial, col2=temporal, col3=correlation

result_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-production', 'output_250810-Ras2-GC#78_reg_s_crop.mat');
rng_seed = 1;
n_query = 5;
n_neighbors = 8;
fixed_half_window = 20; % 2*120+1 = 241 px window

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

if ~isfile(result_path)
    error('Result file not found: %s', result_path);
end

L = load(result_path, 'output');
if ~isfield(L, 'output')
    error('No ''output'' found in: %s', result_path);
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

T = output.temporal_weights'; % cells x frames
T = single(T);

[h, w, n_cells] = size(S);
% Background image for spatial overlays.
if isfield(output, 'info') && isfield(output.info, 'summary_image') && ~isempty(output.info.summary_image)
    bg = single(output.info.summary_image);
else
    bg = max(S, [], 3);
end
bg = bg - min(bg(:));
if max(bg(:)) > 0
    bg = bg ./ max(bg(:));
end

% Weighted centroids for each ROI.
[Ygrid, Xgrid] = ndgrid(single(1:h), single(1:w));
cx = nan(n_cells, 1, 'single');
cy = nan(n_cells, 1, 'single');
for k = 1:n_cells
    m = S(:, :, k);
    m(m < 0) = 0;
    s = sum(m(:));
    if s > 0
        cx(k) = sum(m(:) .* Xgrid(:)) / s;
        cy(k) = sum(m(:) .* Ygrid(:)) / s;
    end
end

valid = find(isfinite(cx) & isfinite(cy));
rng(rng_seed);
q_idx = valid(randperm(numel(valid), min(n_query, numel(valid))));

fig = figure('Color', 'w', 'Position', [80 80 1500 2200]);
tlo = tiledlayout(5, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf('Random %d ROIs with %d nearest neighbors', numel(q_idx), n_neighbors));

for r = 1:n_query
    if r > numel(q_idx)
        nexttile((r - 1) * 3 + 1); axis off; text(0.1, 0.5, 'No ROI available');
        nexttile((r - 1) * 3 + 2); axis off; text(0.1, 0.5, 'No ROI available');
        nexttile((r - 1) * 3 + 3); axis off; text(0.1, 0.5, 'No ROI available');
        continue;
    end

    q = q_idx(r);
    d2 = (cx(valid) - cx(q)).^2 + (cy(valid) - cy(q)).^2;
    [~, ord] = sort(d2, 'ascend');
    nbr = valid(ord(2:min(n_neighbors + 1, numel(ord))));
    ids = [q; nbr(:)];

    cmap = lines(numel(ids));
    cmap(1, :) = [1.0, 0.2, 0.1]; % query ROI (red-ish)

    % Left: spatial overlay
    % Force fixed-size zoom window around query ROI.
    qx = double(cx(q));
    qy = double(cy(q));
    xmin = max(1, floor(qx - fixed_half_window));
    xmax = min(w, ceil(qx + fixed_half_window));
    ymin = max(1, floor(qy - fixed_half_window));
    ymax = min(h, ceil(qy + fixed_half_window));

    ax_sp = nexttile((r - 1) * 3 + 1);
    hold on;
    colormap(ax_sp, gray(256));
    bg_crop = bg(ymin:ymax, xmin:xmax);
    imagesc(xmin:xmax, ymin:ymax, bg_crop);
    for j = 1:numel(ids)
        roi = S(:, :, ids(j));
        roi_crop = roi(ymin:ymax, xmin:xmax);
        t = max(roi_crop(:)) * 0.20;
        if t <= 0
            continue;
        end
        contour(xmin:xmax, ymin:ymax, roi_crop, [t t], ...
            'Color', cmap(j, :), 'LineWidth', 1.5);
        plot(cx(ids(j)), cy(ids(j)), 'o', 'Color', cmap(j, :), ...
            'MarkerFaceColor', cmap(j, :), 'MarkerSize', 4);
    end
    xlim([xmin xmax]);
    ylim([ymin ymax]);
    set(ax_sp, 'YDir', 'reverse');
    axis image off;
    title(sprintf('ROI %d + nearest %d', q, numel(ids) - 1), 'Interpreter', 'none');
    hold off;

    % Right: temporal traces
    nexttile((r - 1) * 3 + 2);
    hold on;
    Tmat = zeros(numel(ids), size(T, 2), 'single');
    for j = 1:numel(ids)
        tr = T(ids(j), :);
        tr = tr - median(tr);
        Tmat(j, :) = tr;
        plot(tr, 'Color', cmap(j, :), 'LineWidth', 1.0);
    end
    grid on; box on;
    xlabel('Frame');
    ylabel('a.u.');
    title('temporal_weights');
    legend_labels = strcat("ROI ", string(ids));
    legend(legend_labels, 'Location', 'northeastoutside');
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

out_png = fullfile(fileparts(result_path), 'random5_nearest5_spatial_temporal.png');
exportgraphics(fig, out_png, 'Resolution', 180);
fprintf('Saved: %s\n', out_png);
