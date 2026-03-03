%% Run 1pMCRI standard pipeline for target_reach 250810 dataset
% Edit options here if needed, then run this script.

%input_tiff = 'R:\data\manipulandum\target_reach\250810-Ras2-GC#78\250810-Ras2-GC#78_reg.tif';
input_tiff = 'R:\code\EXTRACT-public\1pMCRI-demo\250810-Ras2-GC#78_reg_s_crop.tif';

opts = struct();
opts.dataset_name = '/mov';
% For faster XY-partition reads in EXTRACT, use tiled chunks:
opts.chunk_t = 256;
opts.chunk_x = 256;
opts.chunk_y = 256;
opts.use_python_converter = true;
opts.python_exe = ''; % empty -> auto-detect (Conda/Anaconda preferred)
opts.quick_n_frames = inf;
opts.avg_cell_radius = 6;
opts.gpu_id = 1;
opts.use_gpu = true;
opts.parallel_cpu = false;
opts.force_rebuild_h5 = false;
opts.cellfind_max_steps = 1500; % max ROI candidates per partition
% opts.trace_output_option = 'baseline_adjusted'; % e.g., 'none','nonneg'

%opts.num_partitions_x = 5; opts.num_partitions_y = 5;

% Optional threshold overrides
opts.thresholds = struct();
opts.thresholds.eccent_thresh = 2;
opts.thresholds.size_lower_limit = 0.2;
opts.thresholds.size_upper_limit = 2;

R = run_1pMCRI_pipeline(input_tiff, opts);
fprintf('Pipeline complete. Result file: %s\n', R.save_path);
