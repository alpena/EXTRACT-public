function result = run_1pMCRI_pipeline(input_tiff, opts)
% Standard 1pMCRI pipeline:
% - TIFF input: Copy TIFF to fast drive -> convert TIFF->H5 (/mov) -> EXTRACT
% - H5 input  : Copy masknmf H5 to fast drive -> convert motion_corrected->/mov -> EXTRACT
%
% Usage:
%   run_1pMCRI_pipeline('R:\path\movie.tif');
%   run_1pMCRI_pipeline('', struct('input_h5', 'R:\path\moco.h5'));
%   run_1pMCRI_pipeline('', struct('input_h5', 'R:\path\extract_ready.h5', ...
%       'input_h5_preoptimized', true));

if nargin < 1
    input_tiff = '';
end
if nargin < 2 || isempty(opts)
    opts = struct();
end

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);

input_h5 = get_opt(opts, 'input_h5', '');
if isempty(input_tiff) && isempty(input_h5)
    error(['Either input_tiff or opts.input_h5 is required. ', ...
        'Example: run_1pMCRI_pipeline(''R:\\data\\movie.tif'')']);
end
if ~isempty(input_tiff) && ~isempty(input_h5)
    error('Provide only one input source: input_tiff or opts.input_h5.');
end

dataset_name = normalize_dataset_name(get_opt(opts, 'dataset_name', '/mov')); % EXTRACT input dataset
masknmf_dataset_name = '/motion_corrected'; % fixed by design
baseline_dataset_name = normalize_dataset_name(get_opt(opts, 'baseline_dataset_name', '/F_per_pixel'));

chunk_t = get_opt(opts, 'chunk_t', 128);
chunk_x = get_opt(opts, 'chunk_x', []);
chunk_y = get_opt(opts, 'chunk_y', []);
target_chunk_mb = get_opt(opts, 'target_chunk_mb', 16);
use_python_converter = get_opt(opts, 'use_python_converter', true);
python_exe = get_opt(opts, 'python_exe', ''); % empty -> auto-detect
quick_n_frames = get_opt(opts, 'quick_n_frames', inf);
avg_cell_radius = get_opt(opts, 'avg_cell_radius', 6);
partition_overlap = get_opt(opts, 'partition_overlap', []);
partition_core_margin = get_opt(opts, 'partition_core_margin', []);
gpu_id = get_opt(opts, 'gpu_id', 1);
downsample_time_by = get_opt(opts, 'downsample_time_by', 5);
max_iter = get_opt(opts, 'max_iter', 6);
cellfind_max_steps = get_opt(opts, 'cellfind_max_steps', []);
verbose = get_opt(opts, 'verbose', 2);
trace_output_option = get_opt(opts, 'trace_output_option', '');
preprocess = get_opt(opts, 'preprocess', true);
compact_output = get_opt(opts, 'compact_output', true);
use_gpu = get_opt(opts, 'use_gpu', true);
multi_gpu = get_opt(opts, 'multi_gpu', false);
debug_gpu_memory = get_opt(opts, 'debug_gpu_memory', false);
parallel_cpu = get_opt(opts, 'parallel_cpu', false);
num_workers = get_opt(opts, 'num_workers', []);
force_rebuild_h5 = get_opt(opts, 'force_rebuild_h5', false);
h5_skip_if_exists = get_opt(opts, 'h5_skip_if_exists', true);
h5_chunk_t = get_opt(opts, 'h5_chunk_t', 256);
h5_chunk_x = get_opt(opts, 'h5_chunk_x', 256);
h5_chunk_y = get_opt(opts, 'h5_chunk_y', 256);
h5_compression = get_opt(opts, 'h5_compression', 0);
orientation_fix = get_opt(opts, 'orientation_fix', 'transpose_xy'); % 'none' | 'transpose_xy'
input_h5_preoptimized = get_opt(opts, 'input_h5_preoptimized', false);
thresholds = get_opt(opts, 'thresholds', struct());
num_partitions_x = get_opt(opts, 'num_partitions_x', []);
num_partitions_y = get_opt(opts, 'num_partitions_y', []);
avg_event_tau = get_opt(opts, 'avg_event_tau', []);
remove_background = get_opt(opts, 'remove_background', []);
cleanup_fast_cache = get_opt(opts, 'cleanup_fast_cache', true);

if ispc
    default_fast_h5_dir = fullfile('E:\', 'EXTRACT-cache');
else
    default_fast_h5_dir = fullfile(filesep, 'mnt', 'nvme', 'EXTRACT-cache');
end
fast_h5_dir = get_opt(opts, 'fast_h5_dir', default_fast_h5_dir);

if ~exist(fast_h5_dir, 'dir')
    mkdir(fast_h5_dir);
end

if ~isempty(input_h5)
    source_input = input_h5;
else
    source_input = input_tiff;
end
[~, src_name, src_ext] = fileparts(source_input);
default_h5_path = fullfile(fast_h5_dir, [src_name '.h5']);
if ~isempty(input_h5)
    default_h5_path = fullfile(fast_h5_dir, [src_name '_extract_opt.h5']);
end
h5_path = get_opt(opts, 'h5_path', default_h5_path);
default_save_path = fullfile(repo_root, '1pMCRI-production', ['output_' src_name '.mat']);
save_path = get_opt(opts, 'save_path', default_save_path);

fast_source_path = fullfile(fast_h5_dir, [src_name src_ext]);
fast_tiff_path = '';
fast_input_h5_path = '';

addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));
addpath(genpath(fullfile(repo_root, 'Learning-materials')));

%% Step 1: Prepare /mov H5 on fast drive
if ~isempty(input_tiff)
    if ~isfile(input_tiff)
        error('Input TIFF not found: %s', input_tiff);
    end
    fast_tiff_path = copy_source_to_fast_drive(input_tiff, fast_source_path, 'TIFF');
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
else
    if ~isfile(input_h5)
        error('Input H5 not found: %s', input_h5);
    end
    if input_h5_preoptimized
        % Preoptimized flow: copy input H5 to fast drive, then use it directly.
        fast_input_h5_path = copy_source_to_fast_drive(input_h5, fast_source_path, 'H5');
        fprintf(['input_h5_preoptimized=true. Skipping H5 optimization and ', ...
            'using %s:%s directly.\n'], fast_input_h5_path, dataset_name);
        fprintf(['Ignoring optimization options: h5_skip_if_exists, h5_chunk_t/x/y, ', ...
            'h5_compression, orientation_fix.\n']);
        h5_path = fast_input_h5_path;
        try
            info_direct = h5info(h5_path, dataset_name);
        catch ME
            root_info = h5info(h5_path);
            names = collect_h5_dataset_paths(root_info);
            msg = sprintf(['input_h5_preoptimized=true requires dataset %s in %s\n', ...
                'Available datasets:\n  %s'], ...
                dataset_name, h5_path, strjoin(names, sprintf('\n  ')));
            cause = MException('run_1pMCRI_pipeline:MissingPreoptimizedDataset', msg);
            cause = addCause(cause, ME);
            throw(cause);
        end
        if numel(info_direct.Dataspace.Size) ~= 3
            error('Preoptimized input must be 3D at %s:%s', h5_path, dataset_name);
        end
    else
        % Non-preoptimized flow: read source H5 directly and convert to /mov.
        fast_input_h5_path = input_h5;
        fprintf('Using source H5 directly (no copy): %s\n', fast_input_h5_path);
    end
    if ~input_h5_preoptimized && isfile(h5_path) && h5_skip_if_exists && ~force_rebuild_h5
        fprintf('Optimized H5 already exists. Skipping conversion: %s\n', h5_path);
    elseif ~input_h5_preoptimized
        if isfile(h5_path)
            delete(h5_path);
        end
        try
            src_dims = h5info(fast_input_h5_path, masknmf_dataset_name).Dataspace.Size;
        catch ME
            root_info = h5info(fast_input_h5_path);
            names = collect_h5_dataset_paths(root_info);
            msg = sprintf(['Required dataset not found: %s in %s\nAvailable datasets:\n  %s'], ...
                masknmf_dataset_name, fast_input_h5_path, strjoin(names, sprintf('\n  ')));
            cause = MException('run_1pMCRI_pipeline:MissingMaskNmfDataset', msg);
            cause = addCause(cause, ME);
            throw(cause);
        end
        if numel(src_dims) ~= 3
            error('Expected 3D movie in %s:%s', fast_input_h5_path, masknmf_dataset_name);
        end
        python_exe = resolve_python_exe(python_exe);
        run_python_h5_to_h5( ...
            fast_input_h5_path, h5_path, masknmf_dataset_name, dataset_name, ...
            src_dims(1), src_dims(2), src_dims(3), ...
            h5_chunk_t, h5_chunk_x, h5_chunk_y, h5_compression, ...
            orientation_fix, python_exe, script_dir);
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
config.preprocess = logical(preprocess);
config.compact_output = logical(compact_output);
if ~config.preprocess
    try
        config.F_per_pixel = single(h5read(h5_fast_path, baseline_dataset_name));
        fprintf('Loaded baseline image from %s:%s\n', h5_fast_path, baseline_dataset_name);
    catch ME
        warning(['Failed to load baseline image for preprocessed movie from %s:%s\n', ...
            'EXTRACT will fall back to assuming dfofed input.\n%s'], ...
            h5_fast_path, baseline_dataset_name, ME.message);
    end
end
config.use_gpu = use_gpu;
config.parallel_cpu = parallel_cpu;
config.multi_gpu = multi_gpu;
config.debug_gpu_memory = logical(debug_gpu_memory);
config.pick_gpu = gpu_id;
config.use_default_gpu = false;
if ~isempty(num_workers)
    config.num_workers = num_workers;
end
config.num_frames = n_frames;
config.downsample_time_by = downsample_time_by;
config.avg_cell_radius = avg_cell_radius;
if ~isempty(partition_overlap)
    config.partition_overlap = partition_overlap;
end
if ~isempty(partition_core_margin)
    config.partition_core_margin = partition_core_margin;
end
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
meta.input_h5 = input_h5;
meta.input_h5_preoptimized = logical(input_h5_preoptimized);
meta.fast_tiff_path = fast_tiff_path;
meta.fast_input_h5_path = fast_input_h5_path;
meta.h5_path = h5_path;
meta.h5_fast_path = h5_fast_path;
meta.dataset_name = dataset_name;
meta.baseline_dataset_name = baseline_dataset_name;
meta.orientation_fix = orientation_fix;
meta.preprocess = logical(preprocess);
meta.compact_output = logical(compact_output);
meta.frames_used = n_frames;
meta.total_frames = total_frames;
meta.movie_height = h;
meta.movie_width = w;
meta.gpu_id = gpu_id;
meta.elapsed_sec = elapsed_sec;
meta.n_cells = n_cells;
meta.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));

cleanup_report = struct();
cleanup_report.enabled = logical(cleanup_fast_cache);
cleanup_report.deleted = {};
cleanup_report.skipped = {};
cleanup_report.missing = {};
cleanup_report.failed = {};
if cleanup_fast_cache
    cleanup_targets = {h5_path};
    if ~isempty(fast_tiff_path)
        cleanup_targets{end + 1} = fast_tiff_path; %#ok<AGROW>
    end
    if input_h5_preoptimized && ~isempty(fast_input_h5_path)
        cleanup_targets{end + 1} = fast_input_h5_path; %#ok<AGROW>
    end
    cleanup_report = cleanup_fast_cache_files(cleanup_targets, fast_h5_dir);
end
meta.cleanup_fast_cache = cleanup_report;

save(save_path, 'output', 'config_used', 'meta', '-v7.3', '-nocompression');
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

function p = normalize_dataset_name(p)
if ~startsWith(p, '/')
    p = ['/' p];
end
end

function fast_path = copy_source_to_fast_drive(src_path, fast_path, label)
src_info = dir(src_path);
need_copy = true;
if isfile(fast_path)
    dst_info = dir(fast_path);
    need_copy = ~(dst_info.bytes == src_info.bytes);
end
if need_copy
    fprintf('Copying %s to fast drive...\nFrom: %s\nTo:   %s\n', label, src_path, fast_path);
    copyfile(src_path, fast_path, 'f');
else
    fprintf('Using existing fast-drive %s: %s\n', label, fast_path);
end
end

function run_python_h5_to_h5(input_h5, output_h5, input_dataset_name, output_dataset_name, ...
    expected_h, expected_w, expected_t, chunk_t, chunk_x, chunk_y, compression, ...
    orientation_fix, python_exe, script_dir)
py_script = fullfile(script_dir, 'h5_to_h5_fast.py');
if ~isfile(py_script)
    error('Python H5 converter script not found: %s', py_script);
end
if ~strcmp(orientation_fix, 'none') && ~strcmp(orientation_fix, 'transpose_xy')
    error('Unsupported orientation_fix: %s (use ''none'' or ''transpose_xy'').', orientation_fix);
end

cmd = sprintf(['"%s" -u "%s" --input "%s" --output "%s" ', ...
    '--input-dataset "%s" --output-dataset "%s" ', ...
    '--expected-height %d --expected-width %d --expected-frames %d ', ...
    '--chunk-t %d --chunk-x %d --chunk-y %d --compression %d --orientation-fix %s'], ...
    python_exe, py_script, input_h5, output_h5, ...
    input_dataset_name, output_dataset_name, ...
    expected_h, expected_w, expected_t, ...
    chunk_t, chunk_x, chunk_y, compression, orientation_fix);
fprintf('Running Python H5 optimizer:\n%s\n', cmd);

try
    status = system(cmd, '-echo');
catch
    [status, out] = system(cmd);
    fprintf('%s\n', out);
end
if status ~= 0
    error('Python H5->H5 conversion failed.');
end

out_info = h5info(output_h5, output_dataset_name);
out_dims = out_info.Dataspace.Size;
if numel(out_dims) ~= 3
    error('Output dataset is not 3D: %s:%s', output_h5, output_dataset_name);
end
verify_expected_h = expected_h;
verify_expected_w = expected_w;
if strcmp(orientation_fix, 'transpose_xy')
    verify_expected_h = expected_w;
    verify_expected_w = expected_h;
end
if out_dims(1) ~= verify_expected_h || out_dims(2) ~= verify_expected_w || out_dims(3) ~= expected_t
    error(['H5 dataset shape mismatch after conversion.\nExpected [%d %d %d], found [%d %d %d] ', ...
        'for %s:%s'], ...
        verify_expected_h, verify_expected_w, expected_t, out_dims(1), out_dims(2), out_dims(3), ...
        output_h5, output_dataset_name);
end
end

function report = cleanup_fast_cache_files(targets, fast_h5_dir)
report = struct();
report.enabled = true;
report.deleted = {};
report.skipped = {};
report.missing = {};
report.failed = {};

fast_root_cmp = normalize_path_for_compare(fast_h5_dir);
seen = {};
for i = 1:numel(targets)
    target = char(targets{i});
    if isempty(target)
        continue;
    end
    target_cmp = normalize_path_for_compare(target);
    if any(strcmp(seen, target_cmp))
        continue;
    end
    seen{end + 1} = target_cmp; %#ok<AGROW>

    if ~startsWith(target_cmp, fast_root_cmp)
        report.skipped{end + 1} = target; %#ok<AGROW>
        continue;
    end
    if ~isfile(target)
        report.missing{end + 1} = target; %#ok<AGROW>
        continue;
    end

    try
        delete(target);
        report.deleted{end + 1} = target; %#ok<AGROW>
        fprintf('Deleted fast-drive cache file: %s\n', target);
    catch ME
        report.failed{end + 1} = sprintf('%s :: %s', target, ME.message); %#ok<AGROW>
        warning('Failed to delete fast-drive cache file: %s\n%s', target, ME.message);
    end
end
end

function norm_path = normalize_path_for_compare(path_in)
norm_path = char(path_in);
norm_path = strrep(norm_path, '/', filesep);
norm_path = strrep(norm_path, '\', filesep);
if ispc
    norm_path = lower(norm_path);
end
end

function names = collect_h5_dataset_paths(group_info)
names = {};
for i = 1:numel(group_info.Datasets)
    dname = group_info.Datasets(i).Name;
    gname = group_info.Name;
    if strcmp(gname, '/')
        names{end+1} = ['/' dname]; %#ok<AGROW>
    else
        names{end+1} = [gname '/' dname]; %#ok<AGROW>
    end
end
for i = 1:numel(group_info.Groups)
    child = collect_h5_dataset_paths(group_info.Groups(i));
    names = [names child]; %#ok<AGROW>
end
if isempty(names)
    names = {'<none>'};
end
end
