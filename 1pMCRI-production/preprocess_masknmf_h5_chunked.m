function result = preprocess_masknmf_h5_chunked(input_h5, opts)
% preprocess_masknmf_h5_chunked
% Global EXTRACT-style preprocessing for masknmf H5 output with chunked H5 write.
%
% This utility is intended for the workflow:
%   masknmf output H5 -> global F_per_pixel / df / spatial highpass ->
%   chunked H5 for downstream EXTRACT or inspection.
%
% Example:
%   opts = struct();
%   opts.input_dataset = '/motion_corrected';
%   opts.output_h5 = 'R:\data\movie_preprocessed.h5';
%   opts.output_dataset = '/mov';
%   opts.orientation_fix = 'transpose_xy';
%   opts.h5_chunk_t = 128;
%   opts.h5_chunk_x = 256;
%   opts.h5_chunk_y = 256;
%   opts.partition_size_time = 500;
%   R = preprocess_masknmf_h5_chunked('R:\data\masknmf_output.h5', opts);

if nargin < 1 || isempty(input_h5)
    error('input_h5 is required.');
end
if nargin < 2 || isempty(opts)
    opts = struct();
end
if ~isfile(input_h5)
    error('Input H5 not found: %s', input_h5);
end

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));

input_dataset = normalize_dataset_name(get_opt(opts, 'input_dataset', '/motion_corrected'));
output_dataset = normalize_dataset_name(get_opt(opts, 'output_dataset', '/mov'));
output_df_dataset = normalize_dataset_name(get_opt(opts, 'output_df_dataset', output_dataset));
orientation_fix = char(get_opt(opts, 'orientation_fix', 'transpose_xy'));
output_h5 = char(get_opt(opts, 'output_h5', default_output_path(input_h5, '', '')));
output_df_h5 = char(get_opt(opts, 'output_df_h5', ''));
h5_skip_if_exists = logical(get_opt(opts, 'h5_skip_if_exists', true));
force_rebuild = logical(get_opt(opts, 'force_rebuild_h5', false));
h5_chunk_t = double(get_opt(opts, 'h5_chunk_t', 256));
h5_chunk_x = double(get_opt(opts, 'h5_chunk_x', 256));
h5_chunk_y = double(get_opt(opts, 'h5_chunk_y', 256));
h5_compression = double(get_opt(opts, 'h5_compression', 0));
partition_size_time = double(get_opt(opts, 'partition_size_time', 10000));
avg_cell_radius = double(get_opt(opts, 'avg_cell_radius', 6));
spatial_highpass_cutoff = double(get_opt(opts, 'spatial_highpass_cutoff', 5));
use_gpu = logical(get_opt(opts, 'use_gpu', true));
frame_begin = double(get_opt(opts, 'frame_begin', 1));
frame_count = get_opt(opts, 'frame_count', []);
verbose = double(get_opt(opts, 'verbose', 1));

if ~strcmp(orientation_fix, 'none') && ~strcmp(orientation_fix, 'transpose_xy')
    error('Unsupported orientation_fix: %s', orientation_fix);
end
if partition_size_time < 1 || floor(partition_size_time) ~= partition_size_time
    error('partition_size_time must be a positive integer.');
end
if h5_chunk_t < 1 || h5_chunk_x < 1 || h5_chunk_y < 1
    error('h5_chunk_t/x/y must be >= 1.');
end
if h5_compression < 0 || h5_compression > 9 || floor(h5_compression) ~= h5_compression
    error('h5_compression must be an integer in [0, 9].');
end

info = h5info(input_h5, input_dataset);
src_size = info.Dataspace.Size;
if numel(src_size) ~= 3
    error('Expected 3D movie in %s:%s', input_h5, input_dataset);
end

src_h = src_size(1);
src_w = src_size(2);
src_t = src_size(3);
if frame_begin < 1 || floor(frame_begin) ~= frame_begin
    error('frame_begin must be a positive integer.');
end
if frame_begin > src_t
    error('frame_begin=%d exceeds total frames=%d.', frame_begin, src_t);
end
if isempty(frame_count)
    frame_count = src_t - frame_begin + 1;
else
    frame_count = double(frame_count);
    if frame_count < 1 || floor(frame_count) ~= frame_count
        error('frame_count must be a positive integer when provided.');
    end
end
frame_end = min(src_t, frame_begin + frame_count - 1);
out_t = frame_end - frame_begin + 1;

out_h = src_h;
out_w = src_w;
if strcmp(orientation_fix, 'transpose_xy')
    out_h = src_w;
    out_w = src_h;
end

if isempty(output_h5)
    output_h5 = default_output_path(input_h5, frame_begin, out_t);
end
save_df = ~isempty(strtrim(output_df_h5));
if save_df && isempty(output_df_h5)
    output_df_h5 = default_output_path(input_h5, frame_begin, out_t, '_df');
end

if h5_skip_if_exists && ~force_rebuild && isfile(output_h5) && (~save_df || isfile(output_df_h5))
    fprintf('Output already exists. Skipping preprocessing: %s\n', output_h5);
    result = struct();
    result.output_h5 = output_h5;
    result.output_df_h5 = output_df_h5;
    result.output_dataset = output_dataset;
    result.output_df_dataset = output_df_dataset;
    result.frame_begin = frame_begin;
    result.frame_end = frame_end;
    result.output_size = [out_h, out_w, out_t];
    result.skipped = true;
    return;
end

ensure_parent_dir(output_h5);
if save_df
    ensure_parent_dir(output_df_h5);
end
if force_rebuild || isfile(output_h5)
    delete_if_exists(output_h5);
end
if save_df && (force_rebuild || isfile(output_df_h5))
    delete_if_exists(output_df_h5);
end

try
    cfg = get_defaults([]);
    cfg.use_gpu = use_gpu;
    cfg = select_extract_gpu(cfg);
    use_gpu = logical(cfg.use_gpu);
catch
    use_gpu = false;
    fprintf('%s: No usable GPU detected, using CPU instead.\n', now_stamp());
end

chunk_cfg = resolve_output_chunk_size(out_h, out_w, out_t, h5_chunk_y, h5_chunk_x, h5_chunk_t);
create_output_h5(output_h5, output_dataset, out_h, out_w, out_t, chunk_cfg, h5_compression);
create_f_per_pixel_dataset(output_h5, out_h, out_w);
write_common_metadata(output_h5, input_h5, input_dataset, frame_begin, frame_end, orientation_fix, ...
    avg_cell_radius, spatial_highpass_cutoff, partition_size_time);

if save_df
    create_output_h5(output_df_h5, output_df_dataset, out_h, out_w, out_t, chunk_cfg, h5_compression);
    create_f_per_pixel_dataset(output_df_h5, out_h, out_w);
    write_common_metadata(output_df_h5, input_h5, input_dataset, frame_begin, frame_end, orientation_fix, ...
        avg_cell_radius, spatial_highpass_cutoff, partition_size_time);
end

fprintf('Input H5           : %s:%s\n', input_h5, input_dataset);
fprintf('Output H5          : %s:%s\n', output_h5, output_dataset);
fprintf('Frame range        : %d-%d (%d frames)\n', frame_begin, frame_end, out_t);
fprintf('Output movie size  : %d x %d x %d\n', out_h, out_w, out_t);
fprintf('Output chunk size  : [%d %d %d]\n', chunk_cfg.chunk_y, chunk_cfg.chunk_x, chunk_cfg.chunk_t);
fprintf('Compression        : %d\n', h5_compression);
fprintf('orientation_fix    : %s\n', orientation_fix);
fprintf('use_gpu            : %d\n', use_gpu);

sum_image = zeros(out_h, out_w, 'double');
[perframes_mean, startno_mean] = get_partition_starters(out_t, 2 * partition_size_time);
fprintf('%s: Calculating F_per_pixel ...\n', now_stamp());
for idx = 1:numel(startno_mean)
    src_start = frame_begin + startno_mean(idx) - 1;
    block = read_input_block(input_h5, input_dataset, src_h, src_w, src_start, perframes_mean(idx), orientation_fix);
    sum_image = sum_image + double(sum(block, 3));
    if verbose >= 1
        fprintf('\tmean pass %d/%d: frames %d-%d\n', idx, numel(startno_mean), src_start, src_start + perframes_mean(idx) - 1);
    end
end
F_per_pixel = single(sum_image / out_t);
h5write(output_h5, '/F_per_pixel', F_per_pixel);
if save_df
    h5write(output_df_h5, '/F_per_pixel', F_per_pixel);
end

fprintf('%s: Running df + spatial highpass ...\n', now_stamp());
[perframes_out, startno_out] = get_partition_starters(out_t, partition_size_time);
for idx = 1:numel(startno_out)
    src_start = frame_begin + startno_out(idx) - 1;
    dst_start = startno_out(idx);
    block = read_input_block(input_h5, input_dataset, src_h, src_w, src_start, perframes_out(idx), orientation_fix);
    block = bsxfun(@minus, block, F_per_pixel);
    if save_df
        h5write(output_df_h5, output_df_dataset, block, [1, 1, dst_start], [out_h, out_w, perframes_out(idx)]);
    end
    block = spatial_bandpass(block, avg_cell_radius, spatial_highpass_cutoff, inf, use_gpu);
    h5write(output_h5, output_dataset, block, [1, 1, dst_start], [out_h, out_w, perframes_out(idx)]);
    if verbose >= 1
        fprintf('\thighpass pass %d/%d: frames %d-%d\n', idx, numel(startno_out), src_start, src_start + perframes_out(idx) - 1);
    end
end

fprintf('%s: Preprocessing finished.\n', now_stamp());

result = struct();
result.output_h5 = output_h5;
result.output_df_h5 = output_df_h5;
result.output_dataset = output_dataset;
result.output_df_dataset = output_df_dataset;
result.frame_begin = frame_begin;
result.frame_end = frame_end;
result.output_size = [out_h, out_w, out_t];
result.chunk_size = [chunk_cfg.chunk_y, chunk_cfg.chunk_x, chunk_cfg.chunk_t];
result.F_per_pixel_path = '/F_per_pixel';
result.skipped = false;
end

function block = read_input_block(input_h5, input_dataset, src_h, src_w, src_start, n_frames, orientation_fix)
block = single(h5read(input_h5, input_dataset, [1, 1, src_start], [src_h, src_w, n_frames]));
if strcmp(orientation_fix, 'transpose_xy')
    block = permute(block, [2, 1, 3]);
end
end

function chunk_cfg = resolve_output_chunk_size(out_h, out_w, out_t, chunk_y, chunk_x, chunk_t)
chunk_cfg = struct();
chunk_cfg.chunk_y = min(max(round(chunk_y), 1), out_h);
chunk_cfg.chunk_x = min(max(round(chunk_x), 1), out_w);
chunk_cfg.chunk_t = min(max(round(chunk_t), 1), out_t);

bytes_per_voxel = 4; % single
max_chunk_bytes = 4 * 1024^3 - 1;
chunk_bytes = double(chunk_cfg.chunk_y) * double(chunk_cfg.chunk_x) * double(chunk_cfg.chunk_t) * bytes_per_voxel;

while chunk_bytes > max_chunk_bytes && chunk_cfg.chunk_t > 1
    chunk_cfg.chunk_t = max(floor(chunk_cfg.chunk_t / 2), 1);
    chunk_bytes = double(chunk_cfg.chunk_y) * double(chunk_cfg.chunk_x) * double(chunk_cfg.chunk_t) * bytes_per_voxel;
end
while chunk_bytes > max_chunk_bytes && (chunk_cfg.chunk_x > 1 || chunk_cfg.chunk_y > 1)
    if chunk_cfg.chunk_x >= chunk_cfg.chunk_y && chunk_cfg.chunk_x > 1
        chunk_cfg.chunk_x = max(floor(chunk_cfg.chunk_x / 2), 1);
    elseif chunk_cfg.chunk_y > 1
        chunk_cfg.chunk_y = max(floor(chunk_cfg.chunk_y / 2), 1);
    end
    chunk_bytes = double(chunk_cfg.chunk_y) * double(chunk_cfg.chunk_x) * double(chunk_cfg.chunk_t) * bytes_per_voxel;
end
end

function create_output_h5(output_h5, output_dataset, out_h, out_w, out_t, chunk_cfg, h5_compression)
args = {'Datatype', 'single', 'ChunkSize', [chunk_cfg.chunk_y, chunk_cfg.chunk_x, chunk_cfg.chunk_t]};
if h5_compression > 0
    args = [args, {'Deflate', h5_compression}];
end
h5create(output_h5, output_dataset, [out_h, out_w, out_t], args{:});
end

function create_f_per_pixel_dataset(output_h5, out_h, out_w)
h5create(output_h5, '/F_per_pixel', [out_h, out_w], 'Datatype', 'single');
end

function write_common_metadata(output_h5, input_h5, input_dataset, frame_begin, frame_end, orientation_fix, ...
    avg_cell_radius, spatial_highpass_cutoff, partition_size_time)
h5writeatt(output_h5, '/', 'source_h5', input_h5);
h5writeatt(output_h5, '/', 'source_dataset', input_dataset);
h5writeatt(output_h5, '/', 'frame_begin', int64(frame_begin));
h5writeatt(output_h5, '/', 'frame_end', int64(frame_end));
h5writeatt(output_h5, '/', 'orientation_fix', orientation_fix);
h5writeatt(output_h5, '/', 'avg_cell_radius', avg_cell_radius);
h5writeatt(output_h5, '/', 'spatial_highpass_cutoff', spatial_highpass_cutoff);
h5writeatt(output_h5, '/', 'partition_size_time', partition_size_time);
end

function output_h5 = default_output_path(input_h5, frame_begin, frame_count, suffix)
if nargin < 4
    suffix = '';
end
[parent_dir, stem, ~] = fileparts(input_h5);
frame_suffix = '';
if ~isempty(frame_begin) && ~isempty(frame_count)
    frame_suffix = sprintf('_f%d_n%d', frame_begin, frame_count);
end
output_h5 = fullfile(parent_dir, [stem '_extract_global_preprocessed' suffix frame_suffix '.h5']);
end

function ensure_parent_dir(path_in)
parent_dir = fileparts(path_in);
if ~isempty(parent_dir) && ~exist(parent_dir, 'dir')
    mkdir(parent_dir);
end
end

function delete_if_exists(path_in)
if isfile(path_in)
    delete(path_in);
end
end

function p = normalize_dataset_name(p)
if ~startsWith(p, '/')
    p = ['/' p];
end
end

function v = get_opt(s, key, default_v)
if isstruct(s) && isfield(s, key) && ~isempty(s.(key))
    v = s.(key);
else
    v = default_v;
end
end

function stamp = now_stamp()
stamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
end

function [perframes, startno] = get_partition_starters(totalnum, numFrame)
windowsize = min(totalnum, numFrame);
startno = 1:windowsize:totalnum;

if numel(startno) > 1
    perframes = ones(numel(startno), 1) * numFrame;
    lastframes = mod(totalnum, numFrame);
    if lastframes > 0
        perframes(end - 1) = perframes(end - 1) + lastframes;
        startno(end) = [];
    end
else
    perframes = totalnum;
end
end
