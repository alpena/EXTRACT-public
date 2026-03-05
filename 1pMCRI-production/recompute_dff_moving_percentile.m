%% Recompute dF/F (compat wrapper)
% This script remains for backward compatibility.
% Preferred API for pipeline use:
%   R = recompute_dff_postprocess(result_path, opts)

result_path = fullfile(fileparts(mfilename('fullpath')), ...
    'output_250810-Ras2-GC#78_reg_s_crop.mat');

opts = struct();
opts.frame_rate_hz = 30;
opts.half_window_sec = 60;
opts.percentile_q = 8;
opts.baseline_mode = 'moving_percentile';   % 'moving_percentile' | 'global_percentile'
opts.moving_percentile_impl = 'decimated';  % 'exact' | 'decimated'
opts.moving_decimate_factor = 10;
opts.moving_percentile_use_parfor = true;
opts.save_output = true;
opts.plot_n_cells = 5;
opts.rng_seed = 1;
opts.save_figure = true;

R = recompute_dff_postprocess(result_path, opts);
disp(R);
