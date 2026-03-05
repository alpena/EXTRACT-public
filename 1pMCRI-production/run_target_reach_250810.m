%% Run 1pMCRI standard pipeline for target_reach 250810 dataset (H5-first)
% Edit options below, then run this script.
%
% Standard input is masknmf output H5 with dataset '/motion_corrected'.
% run_1pMCRI_pipeline will convert it to optimized '/mov' H5 for EXTRACT.

input_h5 = 'R:\code\masknmf-toolbox\demo_data\output\250810-Ras2-GC#78_moco_first100_smoke_direct.h5';
% input_h5 = 'E:\EXTRACT-cache\moco_results_extract.h5';
[~, src_name, ~] = fileparts(input_h5);

opts = struct();
opts.dataset_name = '/mov';
opts.input_h5 = input_h5;
opts.input_h5_preoptimized = true; % true: use input_h5:/mov directly (skip /motion_corrected -> /mov conversion)
opts.python_exe = ''; % empty -> auto-detect (Conda/Anaconda preferred)
opts.h5_skip_if_exists = true;  % ignored when input_h5_preoptimized=true
opts.h5_chunk_t = 256;
opts.h5_chunk_x = 256;
opts.h5_chunk_y = 256;
opts.h5_compression = 0;        % ignored when input_h5_preoptimized=true
opts.orientation_fix = 'transpose_xy'; % ignored when input_h5_preoptimized=true
opts.quick_n_frames = inf;
opts.avg_cell_radius = 6;
opts.gpu_id = 1;
opts.use_gpu = true;
opts.parallel_cpu = false;
opts.force_rebuild_h5 = false;
opts.cellfind_max_steps = 2000; % max ROI candidates per partition
opts.trace_output_option = 'no_constraint'; % e.g., 'baseline_adjusted','no_constraint','nonneg'
% NOTE: 'no_constraint' is closest to the raw overlap-separated trace output.
% 'baseline_adjusted' and 'nonneg' are not only non-negative but also denoised.

% opts.avg_event_tau = 72;
% opts.remove_background = true;
% opts.save_path = fullfile(fileparts(mfilename('fullpath')), ['output_' src_name '_correct_baseline.mat']);

%opts.num_partitions_x = 2; opts.num_partitions_y = 2;

% Optional threshold overrides
opts.thresholds = struct();
opts.thresholds.eccent_thresh = 2;
opts.thresholds.size_lower_limit = 0.2;
opts.thresholds.size_upper_limit = 2;

R = run_1pMCRI_pipeline('', opts);
fprintf('Pipeline complete. Result file: %s\n', R.save_path);
