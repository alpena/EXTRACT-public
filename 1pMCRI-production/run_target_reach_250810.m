%% Run 1pMCRI standard pipeline for target_reach 250810 dataset
% Edit options here if needed, then run this script.

input_tiff = 'R:\data\manipulandum\target_reach\250810-Ras2-GC#78\250810-Ras2-GC#78_reg.tif';

opts = struct();
opts.dataset_name = '/mov';
opts.chunk_frames = 1000;
opts.use_python_converter = true;
opts.python_exe = ''; % empty -> auto-detect (Conda/Anaconda preferred)
opts.quick_n_frames = inf;
opts.avg_cell_radius = 6;
opts.gpu_id = 1;
opts.use_gpu = true;
opts.parallel_cpu = false;
opts.force_rebuild_h5 = false;

% Optional threshold overrides
opts.thresholds = struct();
opts.thresholds.eccent_thresh = 2;
opts.thresholds.size_lower_limit = 0.2;
opts.thresholds.size_upper_limit = 2;

R = run_1pMCRI_pipeline(input_tiff, opts);
fprintf('Pipeline complete. Result file: %s\n', R.save_path);
