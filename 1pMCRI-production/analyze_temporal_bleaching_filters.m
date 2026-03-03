%% Analyze bleaching-shift removal for EXTRACT temporal_weights (1p data)
% Sample input:
%   output_250810-Ras2-GC#78_reg_s_crop.mat
%
% Compares:
% 1) Butterworth high-pass
% 2) Moving-median baseline subtraction
% 3) Moving-percentile baseline subtraction
%
% Notes:
% - Please set frame_rate_hz correctly for your acquisition.
% - ±60 sec effect is modeled with ~120 sec window/cutoff timescale.

result_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-production', 'output_250810-Ras2-GC#78_reg_s_crop.mat');

frame_rate_hz = 30;   % TODO: set to your true frame rate
window_sec = 120;     % ~±60 sec neighborhood
hp_cutoff_hz = 1 / window_sec;
percentile_q = 8;    % moving quantile baseline

rng_seed = 1;
n_plot_cells = 5;
n_eval_cells = 200;

if ~isfile(result_path)
    error('Result file not found: %s', result_path);
end

L = load(result_path, 'output');
if ~isfield(L, 'output') || ~isfield(L.output, 'temporal_weights')
    error('output.temporal_weights not found in %s', result_path);
end

Traw = single(L.output.temporal_weights'); % cells x frames
[n_cells, n_frames] = size(Traw);
if n_cells < 1 || n_frames < 10
    error('Unexpected temporal_weights size: [%d %d]', n_cells, n_frames);
end

fprintf('Loaded temporal_weights: %d cells x %d frames\n', n_cells, n_frames);
fprintf('frame_rate_hz=%.3f, window_sec=%.1f, hp_cutoff_hz=%.5f\n', ...
    frame_rate_hz, window_sec, hp_cutoff_hz);

win = max(round(window_sec * frame_rate_hz), 5);
if mod(win, 2) == 0
    win = win + 1;
end

rng(rng_seed);
plot_ids = randperm(n_cells, min(n_plot_cells, n_cells));
eval_ids = randperm(n_cells, min(n_eval_cells, n_cells));

Tp = Traw(plot_ids, :);
Te = Traw(eval_ids, :);

% 1) Butterworth high-pass
[b_hp, a_hp] = butter(2, hp_cutoff_hz / (frame_rate_hz / 2), 'high');
Tp_hp = filtfilt(b_hp, a_hp, double(Tp)')';
Te_hp = filtfilt(b_hp, a_hp, double(Te)')';

% 2) Moving median baseline subtraction
Tp_med_base = movmedian(Tp, win, 2, 'Endpoints', 'shrink');
Te_med_base = movmedian(Te, win, 2, 'Endpoints', 'shrink');
Tp_med = single(Tp - Tp_med_base);
Te_med = single(Te - Te_med_base);

% 3) Moving percentile-like baseline subtraction
% Prefer movprctile when available; otherwise use a robust low-envelope fallback.
if exist('movprctile', 'file') == 2
    Tp_q_base = movprctile(Tp, percentile_q, win, 2, 'Endpoints', 'shrink');
    Te_q_base = movprctile(Te, percentile_q, win, 2, 'Endpoints', 'shrink');
    q_label = sprintf('MovQ%d', percentile_q);
else
    win2 = max(5, round(win / 4));
    if mod(win2, 2) == 0
        win2 = win2 + 1;
    end
    Tp_q_base = movmedian(movmin(Tp, win, 2, 'Endpoints', 'shrink'), ...
        win2, 2, 'Endpoints', 'shrink');
    Te_q_base = movmedian(movmin(Te, win, 2, 'Endpoints', 'shrink'), ...
        win2, 2, 'Endpoints', 'shrink');
    q_label = 'LowEnvelope';
    warning('movprctile not available; using low-envelope baseline fallback.');
end
Tp_q = single(Tp - Tp_q_base);
Te_q = single(Te - Te_q_base);

% Low-frequency residual metric (smaller is better)
lf_cut_hz = hp_cutoff_hz;
ratio_raw = lowfreq_ratio_batch(Te, frame_rate_hz, lf_cut_hz);
ratio_hp = lowfreq_ratio_batch(single(Te_hp), frame_rate_hz, lf_cut_hz);
ratio_med = lowfreq_ratio_batch(Te_med, frame_rate_hz, lf_cut_hz);
ratio_q = lowfreq_ratio_batch(Te_q, frame_rate_hz, lf_cut_hz);

medians = [median(ratio_raw), median(ratio_hp), median(ratio_med), median(ratio_q)];
labels = {'Raw', 'HP-Butter', 'MovMedian', q_label};

fprintf('Median low-freq power ratio (< %.5f Hz):\n', lf_cut_hz);
for i = 1:numel(labels)
    fprintf('  %-10s : %.4f\n', labels{i}, medians(i));
end

% Plot comparison
f = figure('Color', 'w', 'Position', [60 60 1800 1200]);
tlo = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf(['Bleaching-shift filtering comparison | fs=%.2f Hz, window=%.1f s, ', ...
    'cells eval=%d'], frame_rate_hz, window_sec, numel(eval_ids)));

nexttile(1);
b1 = bar(medians);
b1.FaceColor = [0.2 0.4 0.8];
set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels);
ylabel('Median low-freq power ratio');
grid on;
title('Lower is better');

nexttile(2);
hold on;
for i = 1:numel(plot_ids)
    tr = Tp(i, :) - median(Tp(i, :));
    plot(tr + (i - 1) * 4, 'Color', [0.2 0.2 0.2], 'LineWidth', 0.8);
end
hold off;
title('Raw traces (5 random cells)');
xlabel('Frame'); ylabel('offset');
grid on;

nexttile(3);
hold on;
for i = 1:numel(plot_ids)
    tr = Tp_hp(i, :) - median(Tp_hp(i, :));
    plot(tr + (i - 1) * 4, 'Color', [0.85 0.2 0.2], 'LineWidth', 0.8);
end
hold off;
title('High-pass Butterworth');
xlabel('Frame'); ylabel('offset');
grid on;

nexttile(4);
hold on;
for i = 1:numel(plot_ids)
    tr = Tp_med(i, :) - median(Tp_med(i, :));
    plot(tr + (i - 1) * 4, 'Color', [0.15 0.6 0.2], 'LineWidth', 0.8);
end
hold off;
title('Moving median subtraction');
xlabel('Frame'); ylabel('offset');
grid on;

nexttile(5);
hold on;
for i = 1:numel(plot_ids)
    tr = Tp_q(i, :) - median(Tp_q(i, :));
    plot(tr + (i - 1) * 4, 'Color', [0.6 0.2 0.85], 'LineWidth', 0.8);
end
hold off;
if strcmp(q_label, 'LowEnvelope')
    title('Low-envelope baseline subtraction');
else
    title(sprintf('Moving percentile subtraction (q=%d)', percentile_q));
end
xlabel('Frame'); ylabel('offset');
grid on;

nexttile(6);
boxplot([ratio_raw(:), ratio_hp(:), ratio_med(:), ratio_q(:)], labels);
ylabel('Low-freq power ratio');
title('Distribution across evaluated cells');
grid on;

out_png = fullfile(fileparts(result_path), 'temporal_bleaching_filter_comparison.png');
exportgraphics(f, out_png, 'Resolution', 180);
fprintf('Saved: %s\n', out_png);

%% Local function
function r = lowfreq_ratio_batch(T, fs, fcut)
% T: cells x frames
T = single(T);
n = size(T, 2);
T = T - mean(T, 2);
Y = fft(double(T), [], 2);
P = abs(Y).^2;
f = (0:n-1) * (fs / n);
half = 1:floor(n/2);
f = f(half);
P = P(:, half);
lf = f <= fcut;
tot = f > 0;
num = sum(P(:, lf), 2);
den = sum(P(:, tot), 2) + eps;
r = single(num ./ den);
end
