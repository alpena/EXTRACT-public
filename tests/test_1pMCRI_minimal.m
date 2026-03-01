%% Minimal EXTRACT smoke test for 1pMCRI TIFF (single GPU)
% Edit only the parameters in this section before running.
tiff_path = 'PATH_TO_YOUR_1pMCRI_MOVIE.tif';
quick_n_frames = 1000;
avg_cell_radius = 6;
gpu_id = 1;
save_path = fullfile('tests', 'test_1pMCRI_output.mat');

%% Initialize paths from repo root
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));
addpath(genpath(fullfile(repo_root, 'Learning-materials')));

%% Input checks
if ~isfile(tiff_path)
    error('Input TIFF not found: %s', tiff_path);
end

tiff_info = imfinfo(tiff_path);
total_frames = numel(tiff_info);
if total_frames < 1
    error('No frames found in TIFF: %s', tiff_path);
end
n_frames = min(quick_n_frames, total_frames);

%% GPU preflight (single GPU assumption)
if gpuDeviceCount < gpu_id
    error('Requested gpu_id=%d is not available. Detected GPUs: %d', gpu_id, gpuDeviceCount);
end
gpuDevice(gpu_id);

%% Read movie and run EXTRACT
fprintf('Reading TIFF: %s\n', tiff_path);
fprintf('Frames used: %d / %d\n', n_frames, total_frames);
M = read_from_tif(tiff_path, 1, n_frames);

config = get_defaults([]);
config.preprocess = true;
config.use_gpu = true;
config.multi_gpu = false;
config.pick_gpu = gpu_id;
config.use_default_gpu = false;
config.num_partitions_x = 1;
config.num_partitions_y = 1;
config.avg_cell_radius = avg_cell_radius;
config.cellfind_max_steps = 120;
config.max_iter = 6;
config.verbose = 2;

fprintf('Starting EXTRACT...\n');
tic;
output = extractor(M, config);
elapsed_sec = toc;

if isfield(output, 'temporal_weights') && ~isempty(output.temporal_weights)
    n_cells = size(output.temporal_weights, 2);
else
    n_cells = 0;
end

[h, w, ~] = size(M);
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
meta.tiff_path = tiff_path;
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
