%% End-to-end pipeline for target_reach dataset
% 1) Convert TIFF stack to H5 (/mov) in chunks (uint16)
% 2) Run EXTRACT using H5 reference input (single GPU by default)
%
% Target TIFF:
% R:\data\manipulandum\target_reach\250810-Ras2-GC#78\250810-Ras2-GC#78_reg.tif

%% User parameters
input_tiff = 'R:\data\manipulandum\target_reach\250810-Ras2-GC#78\250810-Ras2-GC#78_reg.tif';
dataset_name = '/mov';
% Recommended: 1000 for speed. Lower to ~500 if RAM is tight.
% If RAM allows, 1500-2000 may further improve throughput.
chunk_frames = 1000;
use_python_converter = true;
python_exe = ''; % empty -> auto-detect (Conda/Anaconda preferred)

% Fast drive cache for EXTRACT runtime
if ispc
    fast_h5_dir = fullfile('E:\', 'EXTRACT-cache');
else
    fast_h5_dir = fullfile(filesep, 'mnt', 'nvme', 'EXTRACT-cache');
end

% H5 output location (directly on fast drive)
[~, src_name, src_ext] = fileparts(input_tiff);
fast_tiff_path = fullfile(fast_h5_dir, [src_name src_ext]);
h5_path = fullfile(fast_h5_dir, [src_name '.h5']);

quick_n_frames = inf;
avg_cell_radius = 6;
gpu_id = 1;
save_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'tests', 'output_target_reach_250810.mat');

%% Initialize paths
script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));
addpath(genpath(fullfile(repo_root, 'Learning-materials')));

%% Step 1: Copy TIFF to fast drive, then TIFF -> H5 (uint16, chunked)
if ~isfile(input_tiff)
    error('Input TIFF not found: %s', input_tiff);
end
if ~exist(fast_h5_dir, 'dir')
    mkdir(fast_h5_dir);
end

src_info = dir(input_tiff);
need_tiff_copy = true;
if isfile(fast_tiff_path)
    dst_info = dir(fast_tiff_path);
    need_tiff_copy = ~(dst_info.bytes == src_info.bytes);
end
if need_tiff_copy
    fprintf('Copying TIFF to fast drive...\nFrom: %s\nTo:   %s\n', input_tiff, fast_tiff_path);
    copyfile(input_tiff, fast_tiff_path, 'f');
else
    fprintf('Using existing fast-drive TIFF: %s\n', fast_tiff_path);
end

if use_python_converter
    python_exe = resolve_python_exe(python_exe);
    run_python_tiff_to_h5(fast_tiff_path, h5_path, dataset_name, chunk_frames, python_exe, script_dir);
else
    convert_tiff_to_h5_chunked_uint16(fast_tiff_path, h5_path, dataset_name, chunk_frames);
end

%% Step 2: Run EXTRACT from H5 reference on fast drive
h5_fast_path = h5_path;
info = h5info(h5_fast_path, dataset_name);
movie_size = info.Dataspace.Size;
if numel(movie_size) ~= 3
    error('Expected 3D movie dataset in %s:%s', h5_fast_path, dataset_name);
end
total_frames = movie_size(3);
n_frames = min(quick_n_frames, total_frames);
M = [h5_fast_path ':' dataset_name];

if gpuDeviceCount < gpu_id
    error('Requested gpu_id=%d is not available. Detected GPUs: %d', gpu_id, gpuDeviceCount);
end
gpuDevice(gpu_id);

fprintf('Running EXTRACT on: %s\n', M);
fprintf('Frames used: %d / %d\n', n_frames, total_frames);

config = get_defaults([]);
config.preprocess = true;
config.use_gpu = true;
config.parallel_cpu = false;
config.multi_gpu = false;
config.pick_gpu = gpu_id;
config.use_default_gpu = false;
config.num_frames = n_frames;
config.downsample_time_by = 5;
config.avg_cell_radius = avg_cell_radius;
config.max_iter = 6;
config.verbose = 2;
config.thresholds.eccent_thresh = 2;
config.thresholds.size_lower_limit = 0.2;
config.thresholds.size_upper_limit = 2;
config.use_sparse_arrays = 1;

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

save_dir = fileparts(save_path);
if ~isempty(save_dir) && ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

config_used = output.config;
meta = struct();
meta.input_tiff = input_tiff;
meta.fast_tiff_path = fast_tiff_path;
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

%% ---- Local functions ----
function convert_tiff_to_h5_chunked_uint16(input_tiff, output_h5, dataset_name, chunk_frames)
tiff_info = imfinfo(input_tiff);
[height, width] = deal(tiff_info(1).Height, tiff_info(1).Width);
num_ifd = numel(tiff_info);
total_frames = detect_total_frames(tiff_info);

fprintf('TIFF->H5: %s\n', input_tiff);
fprintf('Size: %d x %d x %d\n', height, width, total_frames);

if isfile(output_h5)
    delete(output_h5);
end

% HDF5 requires chunk byte size < 4GB.
bytes_per_frame = double(height) * double(width) * 2; % uint16
max_chunk_frames = max(floor((4 * 1024^3 - 1) / bytes_per_frame), 1);
effective_chunk_frames = min(chunk_frames, max_chunk_frames);
if effective_chunk_frames < chunk_frames
    fprintf(['chunk_frames=%d is too large for HDF5 chunk limit. ', ...
        'Using %d instead.\n'], chunk_frames, effective_chunk_frames);
end
h5create(output_h5, dataset_name, [height, width, total_frames], ...
    'Datatype', 'uint16', ...
    'ChunkSize', [height, width, min(effective_chunk_frames, total_frames)]);

is_imagej_single_ifd = false;
if num_ifd == 1 && total_frames > 1
    is_imagej_single_ifd = true;
    t = Tiff(input_tiff, 'r');
    bits_per_sample = double(t.getTag('BitsPerSample'));
    samples_per_pixel = double(t.getTag('SamplesPerPixel'));
    compression = double(t.getTag('Compression'));
    strip_offset = double(t.getTag('StripOffsets'));
    frame_bytes = double(t.getTag('StripByteCounts'));
    close(t);
    if bits_per_sample ~= 16 || samples_per_pixel ~= 1 || compression ~= 1
        error('ImageJ single-IFD mode supports uncompressed 16-bit single-channel TIFF only.');
    end
    if frame_bytes ~= (height * width * 2)
        error('Unexpected frame byte size in TIFF.');
    end
    machinefmt = detect_tiff_byte_order(input_tiff);
    fid = fopen(input_tiff, 'r', machinefmt);
    if fid < 0
        error('Failed to open TIFF for binary reading: %s', input_tiff);
    end
    cleanup_fid = onCleanup(@() fclose(fid));
end

num_chunks = ceil(total_frames / effective_chunk_frames);
for i = 1:num_chunks
    f_begin = (i - 1) * effective_chunk_frames + 1;
    f_end = min(i * effective_chunk_frames, total_frames);
    n_this = f_end - f_begin + 1;
    if i == 1 || i == num_chunks || mod(i, 10) == 0
        fprintf('Chunk %d/%d: frames %d-%d\n', i, num_chunks, f_begin, f_end);
    end

    if is_imagej_single_ifd
        offset = strip_offset + (f_begin - 1) * frame_bytes;
        fseek(fid, offset, 'bof');
        frame_vec = fread(fid, height * width * n_this, '*uint16');
        if numel(frame_vec) ~= height * width * n_this
            error(['Failed to read TIFF chunk frames %d-%d. ', ...
                'Expected %d values, got %d.'], ...
                f_begin, f_end, height * width * n_this, numel(frame_vec));
        end
        % TIFF raster is row-major; transpose x/y for MATLAB ordering.
        block = permute(reshape(frame_vec, [width, height, n_this]), [2 1 3]);
    else
        block = uint16(read_from_tif(input_tiff, f_begin, n_this));
    end
    h5write(output_h5, dataset_name, block, [1, 1, f_begin], [height, width, n_this]);
end
fprintf('Wrote H5: %s\n', output_h5);
end

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

function run_python_tiff_to_h5(input_tiff, output_h5, dataset_name, chunk_frames, python_exe, script_dir)
py_script = fullfile(script_dir, 'tiff_to_h5_fast.py');
if ~isfile(py_script)
    error('Python converter script not found: %s', py_script);
end

cmd = sprintf('"%s" "%s" --input "%s" --output "%s" --dataset "%s" --chunk-frames %d', ...
    python_exe, py_script, input_tiff, output_h5, dataset_name, chunk_frames);
fprintf('Running Python converter:\n%s\n', cmd);
[status, out] = system(cmd);
fprintf('%s\n', out);
if status ~= 0
    error(['Python TIFF->H5 conversion failed. ', ...
        'Set python_exe to a valid interpreter with tifffile/h5py installed.']);
end
end

function python_exe = resolve_python_exe(python_exe_in)
% Auto-detect a usable Python executable. Prefer Conda/Anaconda.
if nargin >= 1 && ~isempty(python_exe_in)
    python_exe = python_exe_in;
    return;
end

candidates = {};
if ispc
    userprofile = getenv('USERPROFILE');
    localapp = getenv('LOCALAPPDATA');
    conda_prefix = getenv('CONDA_PREFIX');
    if ~isempty(conda_prefix)
        candidates{end+1} = fullfile(conda_prefix, 'python.exe'); %#ok<AGROW>
    end
    if ~isempty(userprofile)
        candidates{end+1} = fullfile(userprofile, 'anaconda3', 'python.exe'); %#ok<AGROW>
        candidates{end+1} = fullfile(userprofile, 'miniconda3', 'python.exe'); %#ok<AGROW>
    end
    if ~isempty(localapp)
        candidates{end+1} = fullfile(localapp, 'anaconda3', 'python.exe'); %#ok<AGROW>
        candidates{end+1} = fullfile(localapp, 'miniconda3', 'python.exe'); %#ok<AGROW>
    end
    candidates{end+1} = 'python'; %#ok<AGROW>
else
    conda_prefix = getenv('CONDA_PREFIX');
    if ~isempty(conda_prefix)
        candidates{end+1} = fullfile(conda_prefix, 'bin', 'python'); %#ok<AGROW>
    end
    candidates{end+1} = 'python3'; %#ok<AGROW>
    candidates{end+1} = 'python'; %#ok<AGROW>
end

for i = 1:numel(candidates)
    c = candidates{i};
    if contains(c, filesep) || contains(c, '\')
        if isfile(c)
            python_exe = c;
            fprintf('Using Python: %s\n', python_exe);
            return;
        end
    else
        [status, ~] = system(sprintf('"%s" --version', c));
        if status == 0
            python_exe = c;
            fprintf('Using Python: %s\n', python_exe);
            return;
        end
    end
end

error(['No usable Python executable found. Set python_exe manually to your ', ...
    'Anaconda python path (Windows example: C:\\Users\\<user>\\anaconda3\\python.exe).']);
end
