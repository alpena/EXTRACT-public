function result = run_1pMCRI_pipeline(input_tiff, opts)
% Standard 1pMCRI pipeline:
% 1) Copy TIFF to fast drive cache
% 2) Convert TIFF -> H5 (/mov)
% 3) Run EXTRACT and save output MAT
%
% Usage:
%   run_1pMCRI_pipeline('R:\path\movie.tif');
%   run_1pMCRI_pipeline('R:\path\movie.tif', struct('gpu_id', 1));

if nargin < 1 || isempty(input_tiff)
    error('input_tiff is required. Example: run_1pMCRI_pipeline(''R:\\data\\movie.tif'')');
end
if nargin < 2 || isempty(opts)
    opts = struct();
end

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);

dataset_name = get_opt(opts, 'dataset_name', '/mov');
chunk_t = get_opt(opts, 'chunk_t', 128);
chunk_x = get_opt(opts, 'chunk_x', []);
chunk_y = get_opt(opts, 'chunk_y', []);
target_chunk_mb = get_opt(opts, 'target_chunk_mb', 16);
use_python_converter = get_opt(opts, 'use_python_converter', true);
python_exe = get_opt(opts, 'python_exe', ''); % empty -> auto-detect
quick_n_frames = get_opt(opts, 'quick_n_frames', inf);
avg_cell_radius = get_opt(opts, 'avg_cell_radius', 6);
gpu_id = get_opt(opts, 'gpu_id', 1);
downsample_time_by = get_opt(opts, 'downsample_time_by', 5);
max_iter = get_opt(opts, 'max_iter', 6);
cellfind_max_steps = get_opt(opts, 'cellfind_max_steps', []);
verbose = get_opt(opts, 'verbose', 2);
trace_output_option = get_opt(opts, 'trace_output_option', '');
use_gpu = get_opt(opts, 'use_gpu', true);
parallel_cpu = get_opt(opts, 'parallel_cpu', false);
force_rebuild_h5 = get_opt(opts, 'force_rebuild_h5', false);
thresholds = get_opt(opts, 'thresholds', struct());
num_partitions_x = get_opt(opts, 'num_partitions_x', []);
num_partitions_y = get_opt(opts, 'num_partitions_y', []);
avg_event_tau = get_opt(opts, 'avg_event_tau', []);
remove_background = get_opt(opts, 'remove_background', []);

if ispc
    default_fast_h5_dir = fullfile('E:\', 'EXTRACT-cache');
else
    default_fast_h5_dir = fullfile(filesep, 'mnt', 'nvme', 'EXTRACT-cache');
end
fast_h5_dir = get_opt(opts, 'fast_h5_dir', default_fast_h5_dir);

[~, src_name, src_ext] = fileparts(input_tiff);
fast_tiff_path = fullfile(fast_h5_dir, [src_name src_ext]);
h5_path = get_opt(opts, 'h5_path', fullfile(fast_h5_dir, [src_name '.h5']));
default_save_path = fullfile(repo_root, '1pMCRI-production', ['output_' src_name '.mat']);
save_path = get_opt(opts, 'save_path', default_save_path);

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

if isfile(h5_path) && ~force_rebuild_h5
    fprintf('H5 already exists. Skipping conversion: %s\n', h5_path);
else
    if isfile(h5_path) && force_rebuild_h5
        delete(h5_path);
    end
    if use_python_converter
        python_exe = resolve_python_exe(python_exe);
        run_python_tiff_to_h5( ...
            fast_tiff_path, h5_path, dataset_name, ...
            chunk_t, chunk_x, chunk_y, target_chunk_mb, python_exe, script_dir);
    else
        convert_tiff_to_h5_chunked_uint16(fast_tiff_path, h5_path, dataset_name, chunk_t);
    end
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

if use_gpu
    if gpuDeviceCount < gpu_id
        error('Requested gpu_id=%d is not available. Detected GPUs: %d', gpu_id, gpuDeviceCount);
    end
    gpuDevice(gpu_id);
end

fprintf('Running EXTRACT on: %s\n', M);
fprintf('Frames used: %d / %d\n', n_frames, total_frames);

config = get_defaults([]);
config.preprocess = true;
config.use_gpu = use_gpu;
config.parallel_cpu = parallel_cpu;
config.multi_gpu = false;
config.pick_gpu = gpu_id;
config.use_default_gpu = false;
config.num_frames = n_frames;
config.downsample_time_by = downsample_time_by;
config.avg_cell_radius = avg_cell_radius;
config.max_iter = max_iter;
if ~isempty(cellfind_max_steps)
    config.cellfind_max_steps = cellfind_max_steps;
end
if ~isempty(trace_output_option)
    config.trace_output_option = trace_output_option;
end
if ~isempty(avg_event_tau)
    config.avg_event_tau = avg_event_tau;
end
if ~isempty(remove_background)
    config.remove_background = logical(remove_background);
end
config.verbose = verbose;
config.thresholds.eccent_thresh = get_opt(thresholds, 'eccent_thresh', 2);
config.thresholds.size_lower_limit = get_opt(thresholds, 'size_lower_limit', 0.2);
config.thresholds.size_upper_limit = get_opt(thresholds, 'size_upper_limit', 2);
config.use_sparse_arrays = 1;
if ~isempty(num_partitions_x) && ~isempty(num_partitions_y)
    config.num_partitions_x = num_partitions_x;
    config.num_partitions_y = num_partitions_y;
end

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

result = struct();
result.output = output;
result.config_used = config_used;
result.meta = meta;
result.save_path = save_path;
result.h5_path = h5_path;
end

%% ---- Local functions ----
function v = get_opt(s, key, default_v)
if isstruct(s) && isfield(s, key) && ~isempty(s.(key))
    v = s.(key);
else
    v = default_v;
end
end

function convert_tiff_to_h5_chunked_uint16(input_tiff, output_h5, dataset_name, chunk_t)
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
effective_chunk_frames = min(chunk_t, max_chunk_frames);
if effective_chunk_frames < chunk_t
    fprintf(['chunk_t=%d is too large for HDF5 chunk limit. ', ...
        'Using %d instead.\n'], chunk_t, effective_chunk_frames);
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

function run_python_tiff_to_h5(input_tiff, output_h5, dataset_name, ...
    chunk_t, chunk_x, chunk_y, target_chunk_mb, python_exe, script_dir)
py_script = fullfile(script_dir, 'tiff_to_h5_fast.py');
if ~isfile(py_script)
    error('Python converter script not found: %s', py_script);
end

% Use unbuffered Python + MATLAB echo mode so progress is shown live.
cmd = sprintf(['"%s" -u "%s" --input "%s" --output "%s" --dataset "%s" ', ...
    '--chunk-t %d --target-chunk-mb %.2f'], ...
    python_exe, py_script, input_tiff, output_h5, dataset_name, chunk_t, target_chunk_mb);
if ~isempty(chunk_x)
    cmd = sprintf('%s --chunk-x %d', cmd, chunk_x);
end
if ~isempty(chunk_y)
    cmd = sprintf('%s --chunk-y %d', cmd, chunk_y);
end
fprintf('Running Python converter:\n%s\n', cmd);

try
    status = system(cmd, '-echo');
catch
    % Fallback for older MATLAB versions that do not support '-echo'.
    [status, out] = system(cmd);
    fprintf('%s\n', out);
end

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
