%% Visualize what correct_baseline estimates and removes
% This script reproduces the core logic of correct_baseline on temporal_weights
% and visualizes:
%   1) raw trace
%   2) estimated static background component (s*t)
%   3) estimated slow baseline drift (subt)
%   4) corrected trace
%
% Notes:
% - In EXTRACT, correct_baseline is applied to movie pixels before extraction.
% - Here we apply the same model to ROI traces for interpretability.

result_mat = fullfile(fileparts(mfilename('fullpath')), ...
    'output_250810-Ras2-GC#78_reg_s_crop.mat');

tau = 72;
remove_background = true;
n_plot_cells = 5;
rng_seed = 1;

if ~isfile(result_mat)
    error('Result file not found: %s', result_mat);
end

L = load(result_mat, 'output');
if ~isfield(L, 'output') || ~isfield(L.output, 'temporal_weights') || isempty(L.output.temporal_weights)
    error('output.temporal_weights not found in %s', result_mat);
end

T = single(L.output.temporal_weights'); % cells x frames
[n_cells, n_frames] = size(T);
if n_cells < 1 || n_frames < 10
    error('Unexpected temporal_weights size: [%d %d]', n_cells, n_frames);
end

fprintf('Loaded traces: %d cells x %d frames\n', n_cells, n_frames);
fprintf('Applying baseline decomposition with tau=%g, remove_background=%d\n', ...
    tau, remove_background);

[Tcorr, Tbg, Tdrift] = decompose_baseline_like_extract(T, tau, remove_background);

rng(rng_seed);
plot_ids = randperm(n_cells, min(n_plot_cells, n_cells));

f = figure('Color', 'w', 'Position', [60 60 1900 1100]);
tlo = tiledlayout(numel(plot_ids), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf('correct\\_baseline decomposition (tau=%g, remove\\_background=%d)', ...
    tau, remove_background));

for r = 1:numel(plot_ids)
    cid = plot_ids(r);
    tr_raw = T(cid, :);
    tr_bg = Tbg(cid, :);
    tr_drift = Tdrift(cid, :);
    tr_corr = Tcorr(cid, :);

    nexttile((r - 1) * 3 + 1);
    hold on;
    plot(tr_raw, 'k-', 'LineWidth', 0.8);
    plot(tr_bg, 'Color', [0.85 0.2 0.2], 'LineWidth', 0.8);
    plot(tr_drift, 'Color', [0.2 0.45 0.85], 'LineWidth', 0.8);
    hold off;
    grid on;
    title(sprintf('Cell %d: raw/bg/drift', cid));
    xlabel('Frame');
    ylabel('a.u.');
    if r == 1
        legend({'Raw', 'Background s*t', 'Drift subt'}, 'Location', 'best');
    end

    nexttile((r - 1) * 3 + 2);
    hold on;
    plot(tr_raw, 'Color', [0.55 0.55 0.55], 'LineWidth', 0.8);
    plot(tr_corr, 'Color', [0.1 0.65 0.2], 'LineWidth', 0.9);
    hold off;
    grid on;
    title(sprintf('Cell %d: corrected', cid));
    xlabel('Frame');
    ylabel('a.u.');
    if r == 1
        legend({'Raw', 'Corrected'}, 'Location', 'best');
    end

    nexttile((r - 1) * 3 + 3);
    c = corrcoef(double([tr_raw(:), tr_bg(:), tr_drift(:), tr_corr(:)]));
    imagesc(c, [-1, 1]);
    axis image;
    colormap(gca, turbo(256));
    colorbar;
    set(gca, 'XTick', 1:4, 'XTickLabel', {'Raw','BG','Drift','Corr'});
    set(gca, 'YTick', 1:4, 'YTickLabel', {'Raw','BG','Drift','Corr'});
    title(sprintf('Cell %d: corr matrix', cid));
end

out_png = fullfile(fileparts(result_mat), 'correct_baseline_decomposition_temporal_weights.png');
exportgraphics(f, out_png, 'Resolution', 180);
fprintf('Saved: %s\n', out_png);

%% ---- Local functions ----
function [Mcorr, Mbg, Mdrift] = decompose_baseline_like_extract(Min, tau, remove_background)
% Min: components x frames
% Returns decomposition aligned with correct_baseline inner behavior.

ABS_TOL = 1e-6;
[num_components, num_frames] = size(Min);
M = single(Min);

k = max(1, round(tau * 50)); % same as correct_baseline
ss = max(1, round(k * 2));
smoothing_hlen = 5;

Mbg = zeros(size(M), 'single');
if remove_background
    ppp = std(M, 0, 2);
    if any(ppp > 0)
        s = ppp;
        t = (s' * M) / max(1e-6, sum(s.^2));
        s = max((M * t') / max(1e-6, sum(t.^2)), 0);
        t = medfilt1(t);
        Mbg = s * t;
        M = M - Mbg;
    end
end

% zero-mean in time
M = bsxfun(@minus, M, mean(M, 2));

stds = std(M, 1, 2);
stds = stds(stds > ABS_TOL);
if isempty(stds)
    Mcorr = zeros(size(M), 'single');
    Mdrift = zeros(size(M), 'single');
    return;
end
stat = median(stds);
step_size = max(stat / 5, ABS_TOL);
edges = (-10 * stat):step_size:(10 * stat);
if numel(edges) < 3
    edges = [-1, 0, 1] * max(stat, 1);
end

num_samples = max(3, ceil(num_frames / k));
sampled_indices = round(linspace(1, num_frames, num_samples));
baselines = zeros(num_components, num_samples, 'single');

for idx_sample = 1:num_samples
    idx_begin = max(1, sampled_indices(idx_sample) - round(ss / 2));
    idx_end = min(num_frames, sampled_indices(idx_sample) + round(ss / 2));
    data = M(:, idx_begin:idx_end);
    hist_counts = histc(data, edges, 2);
    [~, idx_max] = max(hist_counts, [], 2);
    most_freq_vals = edges(idx_max)' + step_size / 2;
    baselines(:, idx_sample) = single(most_freq_vals);
end

filt_out = zeros(num_components, num_samples, 'single');
for idx_sample = 0:num_samples - 1
    idx_begin = max(1, 1 + idx_sample - smoothing_hlen);
    idx_end = min(num_samples, idx_sample + smoothing_hlen + 1);
    b = baselines(:, idx_begin:idx_end);
    filt_out(:, idx_sample + 1) = mean(b, 2);
end

Mdrift = interp1(sampled_indices, filt_out', (1:num_frames)', 'linear')';
Mcorr = M - Mdrift;
Mcorr = bsxfun(@minus, Mcorr, mean(Mcorr, 2));
end
