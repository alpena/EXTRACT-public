%% Minimal EXTRACT smoke test for 1pMCRI TIFF (single GPU)
% Edit only the parameters in this section before running.
tiff_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-demo', '250206-UK6-1-F=4_power=5mW_reg_crop.tiff');
quick_n_frames = 1000;
avg_cell_radius = 6;
gpu_id = 1;
save_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'tests', 'test_1pMCRI_output.mat');

%% Initialize paths from repo root
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));
addpath(genpath(fullfile(repo_root, 'Learning-materials')));

%% Input checks and frame count detection
if ~isfile(tiff_path)
    error('Input TIFF not found: %s', tiff_path);
end

tiff_info = imfinfo(tiff_path);
total_frames = detect_total_frames(tiff_info);
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
M = read_tiff_stack_subset(tiff_path, 1, n_frames, tiff_info);

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

function total_frames = detect_total_frames(tiff_info)
total_frames = numel(tiff_info);
if total_frames == 1 && isfield(tiff_info(1), 'ImageDescription')
    desc = tiff_info(1).ImageDescription;
    token = regexp(desc, 'images=(\d+)', 'tokens', 'once');
    if ~isempty(token)
        total_frames = str2double(token{1});
    end
end
end

function M = read_tiff_stack_subset(tiff_path, start_frame, n_frames, tiff_info)
% Supports both standard multipage TIFF and ImageJ single-IFD stack TIFF.
num_ifd = numel(tiff_info);
if num_ifd >= (start_frame + n_frames - 1)
    M = read_from_tif(tiff_path, start_frame, n_frames);
    return;
end

if ~(num_ifd == 1 && isfield(tiff_info(1), 'ImageDescription'))
    error('Unsupported TIFF layout for partial reading: %s', tiff_path);
end

desc = tiff_info(1).ImageDescription;
token = regexp(desc, 'images=(\d+)', 'tokens', 'once');
if isempty(token)
    error('Single-IFD TIFF detected but ImageJ images=... metadata not found.');
end
total_frames = str2double(token{1});
end_frame = start_frame + n_frames - 1;
if end_frame > total_frames
    error('Requested frame range exceeds TIFF stack length.');
end

t = Tiff(tiff_path, 'r');
height = double(t.getTag('ImageLength'));
width = double(t.getTag('ImageWidth'));
bits_per_sample = double(t.getTag('BitsPerSample'));
samples_per_pixel = double(t.getTag('SamplesPerPixel'));
compression = double(t.getTag('Compression'));
strip_offset = double(t.getTag('StripOffsets'));
frame_bytes = double(t.getTag('StripByteCounts'));
close(t);

if bits_per_sample ~= 16 || samples_per_pixel ~= 1 || compression ~= 1
    error('Only uncompressed 16-bit single-channel ImageJ stack TIFF is supported.');
end
if frame_bytes ~= (height * width * 2)
    error('Unexpected frame byte size in TIFF.');
end

machinefmt = detect_tiff_byte_order(tiff_path);
fid = fopen(tiff_path, 'r', machinefmt);
if fid < 0
    error('Failed to open TIFF for binary reading: %s', tiff_path);
end
cleanup_fid = onCleanup(@() fclose(fid));

M = zeros(height, width, n_frames, 'single');
for k = 1:n_frames
    frame_idx = start_frame + k - 1;
    offset = strip_offset + (frame_idx - 1) * frame_bytes;
    fseek(fid, offset, 'bof');
    frame = fread(fid, height * width, '*uint16');
    if numel(frame) ~= height * width
        error('Failed to read frame %d from TIFF.', frame_idx);
    end
    % TIFF raster is row-major; transpose after reshape for MATLAB order.
    M(:, :, k) = single(reshape(frame, [width, height])');
end
clear cleanup_fid
end

function machinefmt = detect_tiff_byte_order(tiff_path)
fid = fopen(tiff_path, 'r');
if fid < 0
    error('Failed to open TIFF header: %s', tiff_path);
end
cleanup_fid = onCleanup(@() fclose(fid));
sig = fread(fid, 2, '*char')';
if strcmp(sig, 'II')
    machinefmt = 'ieee-le';
elseif strcmp(sig, 'MM')
    machinefmt = 'ieee-be';
else
    error('Invalid TIFF byte-order signature.');
end
clear cleanup_fid
end
