%% Minimal EXTRACT test for 1pMCRI H5 movie (single GPU)
% Edit only the parameters in this section before running.
h5_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-demo', '250206-UK6-1-F=4_power=5mW_reg_t_crop_s_full.h5');
dataset_name = '/mov';
if ispc
    fast_h5_dir = fullfile('E:\', 'EXTRACT-cache');
else
    fast_h5_dir = fullfile(filesep, 'mnt', 'nvme', 'EXTRACT-cache');
end
quick_n_frames = inf;
avg_cell_radius = 6;
gpu_id = 1;
save_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-production', 'test_1pMCRI_output_full.mat');

%% Initialize paths from repo root
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));
addpath(genpath(fullfile(repo_root, 'Learning-materials')));

%% Input checks and frame count detection
if ~isfile(h5_path)
    error('Input H5 not found: %s', h5_path);
end

% Copy H5 to fast NVMe drive if available.
if ~exist(fast_h5_dir, 'dir')
    mkdir(fast_h5_dir);
end
[~, h5_name, h5_ext] = fileparts(h5_path);
h5_fast_path = fullfile(fast_h5_dir, [h5_name h5_ext]);
src_info = dir(h5_path);
need_copy = true;
if isfile(h5_fast_path)
    dst_info = dir(h5_fast_path);
    need_copy = ~(dst_info.bytes == src_info.bytes);
end
if need_copy
    fprintf('Copying H5 to fast drive...\nFrom: %s\nTo:   %s\n', h5_path, h5_fast_path);
    copyfile(h5_path, h5_fast_path, 'f');
else
    fprintf('Using existing fast-drive copy: %s\n', h5_fast_path);
end

info = h5info(h5_fast_path, dataset_name);
movie_size = info.Dataspace.Size;
if numel(movie_size) ~= 3
    error('Expected 3D movie dataset in %s:%s', h5_fast_path, dataset_name);
end
total_frames = movie_size(3);
if total_frames < 1
    error('No frames found in H5 dataset: %s:%s', h5_fast_path, dataset_name);
end
n_frames = min(quick_n_frames, total_frames);
M = [h5_fast_path ':' dataset_name];

%% GPU preflight (single GPU assumption)
if gpuDeviceCount < gpu_id
    error('Requested gpu_id=%d is not available. Detected GPUs: %d', gpu_id, gpuDeviceCount);
end
gpuDevice(gpu_id);

%% Run EXTRACT
fprintf('Reading H5 by reference: %s\n', M);
fprintf('Frames used: %d / %d\n', n_frames, total_frames);

config = get_defaults([]);
config.preprocess = true;

config.use_gpu = true;
config.multi_gpu = false;
config.pick_gpu = gpu_id;
config.use_default_gpu = false;

% config.parallel_cpu = true;
% config.use_gpu = false;

config.num_frames = n_frames;
config.downsample_time_by = 5;
config.avg_cell_radius = avg_cell_radius;
config.max_iter = 6;
config.verbose = 2;
config.thresholds.eccent_thresh = 2;
config.thresholds.size_lower_limit = 0.2;
config.thresholds.size_upper_limit = 2;
config.use_sparse_arrays = 1;

fprintf('Starting EXTRACT...\n');
tic;
output = extractor(M, config);
elapsed_sec = toc;

if isfield(output, 'temporal_weights') && ~isempty(output.temporal_weights)
    n_cells = size(output.temporal_weights, 2);
else
    n_cells = 0;
end

[h, w, ~] = get_movie_size(M);
fprintf('EXTRACT complete.\n');
fprintf('Movie size: %d x %d x %d\n', h, w, n_frames);
fprintf('Detected cells: %d\n', n_cells);
fprintf('Elapsed time (sec): %.2f\n', elapsed_sec);

%% Save output
save_dir = fileparts(save_path);
if ~isempty(save_dir) && ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

config_used = output.config;
meta = struct();
meta.h5_path = h5_path;
meta.h5_fast_path = h5_fast_path;
meta.dataset_name = dataset_name;
meta.frames_used = n_frames;
meta.total_frames = total_frames;
meta.movie_height = h;
meta.movie_width = w;
meta.gpu_id = gpu_id;
meta.elapsed_sec = elapsed_sec;
meta.n_cells = n_cells;
meta.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));

save(save_path, 'output', 'config_used', 'meta', '-v7.3');
fprintf('Saved result: %s\n', save_path);
