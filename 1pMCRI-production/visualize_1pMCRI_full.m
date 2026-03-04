%% Visualize EXTRACT output for 1pMCRI run
% This script loads EXTRACT output (.mat) and generates summary figures.

result_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-production', 'output_250810-Ras2-GC#78_reg.mat');

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

if ~isfile(result_path)
    error('Result file not found: %s', result_path);
end

R = load(result_path);
if ~isfield(R, 'output')
    error('Result file does not contain ''output'': %s', result_path);
end
output = R.output;

if ~isfield(output, 'temporal_weights') || ~isfield(output, 'spatial_weights')
    error('Output structure is missing temporal_weights/spatial_weights.');
end

T = output.temporal_weights'; % cells x time
S = output.spatial_weights;   % h x w x cells
outdir = fileparts(result_path);

fprintf('Loaded: %s\n', result_path);
fprintf('Cells: %d, Frames: %d\n', size(T, 1), size(T, 2));

% 1) Cell map overlay
f1 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 900 900]);
plot_output_cellmap(output, 0);
save_cellmap = fullfile(outdir, 'test_1pMCRI_full_cellmap.png');
exportgraphics(f1, save_cellmap, 'Resolution', 200);
close(f1);

% 2) Top traces
n_show = min(30, size(T, 1));
act = max(T, [], 2) - min(T, [], 2);
[~, idx] = sort(act, 'descend');
idx = idx(1:n_show);
Tsel = T(idx, :);
Tsel = Tsel ./ max(Tsel, [], 2);
Tsel(~isfinite(Tsel)) = 0;
offset = (0:n_show-1)' * 1.15;

f2 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1500 850]);
hold on;
for i = 1:n_show
    plot(Tsel(i, :) + offset(i), 'LineWidth', 0.8, 'Color', [0.05 0.35 0.75]);
end
hold off;
grid on;
xlabel('Frame');
ylabel('Cell index (offset)');
title(sprintf('Top %d normalized traces (full run)', n_show));
set(gca, 'YTick', offset, 'YTickLabel', string(idx));
save_traces = fullfile(outdir, 'test_1pMCRI_full_traces_top30.png');
exportgraphics(f2, save_traces, 'Resolution', 200);
close(f2);

% 3) Activity histogram
f3 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 900 520]);
histogram(act, 40, 'FaceColor', [0.15 0.55 0.25], 'EdgeColor', 'none');
xlabel('Trace dynamic range (max-min)');
ylabel('Cell count');
title(sprintf('Activity distribution across extracted cells (N=%d)', size(T, 1)));
grid on;
save_hist = fullfile(outdir, 'test_1pMCRI_full_activity_hist.png');
exportgraphics(f3, save_hist, 'Resolution', 200);
close(f3);

% 4) Weighted spatial activity map
w = mean(T, 2);
w = w - min(w);
if max(w) > 0
    w = w ./ max(w);
end
map = zeros(size(S, 1), size(S, 2), 'single');
for k = 1:size(S, 3)
    map = map + single(full(S(:, :, k))) .* single(w(k));
end

f4 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 900 850]);
imagesc(map);
axis image off;
colormap hot;
colorbar;
title('Weighted spatial activity map (full run)');
save_map = fullfile(outdir, 'test_1pMCRI_full_weighted_map.png');
exportgraphics(f4, save_map, 'Resolution', 200);
close(f4);

fprintf('Saved: %s\n', save_cellmap);
fprintf('Saved: %s\n', save_traces);
fprintf('Saved: %s\n', save_hist);
fprintf('Saved: %s\n', save_map);
