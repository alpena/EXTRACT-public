function out = export_viewer_bundle_python(varargin)
% export_viewer_bundle_python
% Convert EXTRACT output (with ndSparse spatial_weights) into a Python-friendly HDF5 bundle.
%
% Example:
%   export_viewer_bundle_python('run_dir', ...
%       'R:/code/1pMCRI-pipeline/demo_data/output/smoke_250810');
%
% Name-value options:
%   run_dir       : pipeline run directory (default smoke example)
%   extract_mat   : EXTRACT output MAT (required if auto-discovery fails)
%   cascade_h5    : optional cascade HDF5 with spike_prob
%   qc_mat        : optional QC MAT from compute_qc_roi_bg_excluding_cells
%   movie_h5      : optional source movie H5 path
%   movie_dataset : movie dataset path in H5 (default '/mov')
%   out_h5        : output bundle path (default <run_dir>/artifacts/viewer_bundle.h5)
%
% Output:
%   out: struct with resolved paths and summary.

opts = struct();
opts.run_dir = 'R:/code/1pMCRI-pipeline/demo_data/output/smoke_250810';
opts.extract_mat = '';
opts.cascade_h5 = '';
opts.qc_mat = '';
opts.movie_h5 = '';
opts.movie_dataset = '/mov';
opts.out_h5 = '';
opts.version = '1.0.0';
opts.verbose = true;
opts = parse_name_values(opts, varargin);

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

[extract_mat, cascade_h5, qc_mat, movie_h5, out_h5, run_dir] = resolve_paths(opts);

if opts.verbose
    fprintf('run_dir    : %s\n', run_dir);
    fprintf('extract_mat: %s\n', extract_mat);
    fprintf('cascade_h5 : %s\n', to_str(cascade_h5));
    fprintf('qc_mat     : %s\n', to_str(qc_mat));
    fprintf('movie_h5   : %s\n', to_str(movie_h5));
    fprintf('out_h5     : %s\n', out_h5);
end

L = load(extract_mat, 'output');
if ~isfield(L, 'output')
    error('No output struct in extract_mat: %s', extract_mat);
end
output = L.output;
required_fields = {'spatial_weights', 'temporal_weights'};
for i = 1:numel(required_fields)
    if ~isfield(output, required_fields{i})
        error('Missing output.%s in extract_mat', required_fields{i});
    end
end

S = output.spatial_weights;
[S2, h, w, n_cells] = spatial_to_sparse2d(S);
T = temporal_to_cells_by_time(output.temporal_weights, n_cells);
T = single(T);
T_base = size(T, 2);

if opts.verbose
    fprintf('Spatial shape: h=%d, w=%d, cells=%d\n', h, w, n_cells);
    fprintf('Temporal shape: [%d x %d]\n', size(T, 1), size(T, 2));
end

% Images (optional)
summary_image = [];
max_image = [];
f_per_pixel = [];
if isfield(output, 'info') && isstruct(output.info)
    if isfield(output.info, 'summary_image') && ~isempty(output.info.summary_image)
        summary_image = single(output.info.summary_image);
        if ~isequal(size(summary_image), [h, w])
            warning('summary_image size mismatch. Skipped.');
            summary_image = [];
        end
    end
    if isfield(output.info, 'max_image') && ~isempty(output.info.max_image)
        max_image = single(output.info.max_image);
        if ~isequal(size(max_image), [h, w])
            warning('max_image size mismatch. Skipped.');
            max_image = [];
        end
    end
    if isfield(output.info, 'F_per_pixel') && ~isempty(output.info.F_per_pixel)
        f_per_pixel = single(output.info.F_per_pixel(:));
        if numel(f_per_pixel) ~= h * w
            warning('F_per_pixel length mismatch. Skipped.');
            f_per_pixel = [];
        end
    end
end

% ROI centroids
centroid_xy = compute_centroids_xy(S2, h, w);

% Convert sparse2d to CSR components for scipy.sparse.csr_matrix
[csr_data, csr_indices, csr_indptr, csr_shape] = sparse_to_csr(S2);

% Optional traces
trace_spike_prob = [];
trace_raw_roi = [];
trace_raw_bg = [];
trace_raw_minus_bg = [];

if ~isempty(cascade_h5) && isfile(cascade_h5)
    sp = read_cascade_spike_prob_h5(cascade_h5, n_cells);
    if ~isempty(sp)
        trace_spike_prob = align_trace_time(sp, T_base);
    else
        warning('cascade_h5 has no spike_prob. Skipped.');
    end
end

if ~isempty(qc_mat) && isfile(qc_mat)
    Q = load(qc_mat);
    has_qc = isfield(Q, 'roi_id') && ...
             isfield(Q, 'trace_raw_roi') && ...
             isfield(Q, 'trace_raw_bg') && ...
             isfield(Q, 'trace_raw_minus_bg');
    if has_qc
        roi_id = double(Q.roi_id(:));
        src_raw_roi = orient_by_id_count(single(Q.trace_raw_roi), numel(roi_id));
        src_raw_bg = orient_by_id_count(single(Q.trace_raw_bg), numel(roi_id));
        src_raw_minus_bg = orient_by_id_count(single(Q.trace_raw_minus_bg), numel(roi_id));
        trace_raw_roi = map_roi_trace_to_all_cells(src_raw_roi, roi_id, n_cells, T_base);
        trace_raw_bg = map_roi_trace_to_all_cells(src_raw_bg, roi_id, n_cells, T_base);
        trace_raw_minus_bg = map_roi_trace_to_all_cells(src_raw_minus_bg, roi_id, n_cells, T_base);
    else
        warning('qc_mat missing required fields. Skipped QC traces.');
    end
end

if isfile(out_h5)
    delete(out_h5);
end
out_dir = fileparts(out_h5);
if ~isempty(out_dir) && ~isfolder(out_dir)
    mkdir(out_dir);
end

% Required datasets
write_string_dataset(out_h5, '/meta/version', opts.version);
write_string_dataset(out_h5, '/meta/source_extract_mat', extract_mat);

write_scalar_dataset(out_h5, '/roi/h', int64(h), 'int64');
write_scalar_dataset(out_h5, '/roi/w', int64(w), 'int64');
write_scalar_dataset(out_h5, '/roi/n_cells', int64(n_cells), 'int64');

write_numeric_vector_dataset(out_h5, '/roi/sparse_csr/data', csr_data, 'single');
write_numeric_vector_dataset(out_h5, '/roi/sparse_csr/indices', csr_indices, 'int32');
write_numeric_vector_dataset(out_h5, '/roi/sparse_csr/indptr', csr_indptr, 'int64');
write_numeric_vector_dataset(out_h5, '/roi/sparse_csr/shape', csr_shape, 'int64');

write_py2d_dataset(out_h5, '/trace/temporal_weights', T, 'single');

% Optional datasets
if ~isempty(summary_image)
    write_py2d_dataset(out_h5, '/image/summary_image', summary_image, 'single');
end
if ~isempty(max_image)
    write_py2d_dataset(out_h5, '/image/max_image', max_image, 'single');
end
if ~isempty(f_per_pixel)
    h5create(out_h5, '/image/F_per_pixel', size(f_per_pixel), 'Datatype', 'single');
    h5write(out_h5, '/image/F_per_pixel', f_per_pixel);
end

write_py2d_dataset(out_h5, '/roi/centroid_xy', centroid_xy, 'single');

if ~isempty(trace_spike_prob)
    write_py2d_dataset(out_h5, '/trace/spike_prob', trace_spike_prob, 'single');
end
if ~isempty(trace_raw_roi)
    write_py2d_dataset(out_h5, '/trace/raw_roi', trace_raw_roi, 'single');
end
if ~isempty(trace_raw_bg)
    write_py2d_dataset(out_h5, '/trace/raw_bg', trace_raw_bg, 'single');
end
if ~isempty(trace_raw_minus_bg)
    write_py2d_dataset(out_h5, '/trace/raw_minus_bg', trace_raw_minus_bg, 'single');
end

if ~isempty(movie_h5)
    write_string_dataset(out_h5, '/movie/path_h5', movie_h5);
    write_string_dataset(out_h5, '/movie/dataset', opts.movie_dataset);
end

required_paths = { ...
    '/meta/version', ...
    '/meta/source_extract_mat', ...
    '/roi/h', ...
    '/roi/w', ...
    '/roi/n_cells', ...
    '/roi/sparse_csr/data', ...
    '/roi/sparse_csr/indices', ...
    '/roi/sparse_csr/indptr', ...
    '/roi/sparse_csr/shape', ...
    '/roi/centroid_xy', ...
    '/trace/temporal_weights'};
if ~isempty(trace_spike_prob)
    required_paths{end + 1} = '/trace/spike_prob';
end
if ~isempty(trace_raw_roi)
    required_paths{end + 1} = '/trace/raw_roi';
end
if ~isempty(trace_raw_bg)
    required_paths{end + 1} = '/trace/raw_bg';
end
if ~isempty(trace_raw_minus_bg)
    required_paths{end + 1} = '/trace/raw_minus_bg';
end
verify_required_bundle_datasets(out_h5, required_paths);

out = struct();
out.run_dir = run_dir;
out.extract_mat = extract_mat;
out.cascade_h5 = cascade_h5;
out.qc_mat = qc_mat;
out.movie_h5 = movie_h5;
out.movie_dataset = opts.movie_dataset;
out.out_h5 = out_h5;
out.h = h;
out.w = w;
out.n_cells = n_cells;
out.n_frames = T_base;

if opts.verbose
    fprintf('Saved viewer bundle: %s\n', out_h5);
end

end

function [extract_mat, cascade_h5, qc_mat, movie_h5, out_h5, run_dir] = resolve_paths(opts)
extract_mat = strtrim(char(opts.extract_mat));
cascade_h5 = strtrim(char(opts.cascade_h5));
qc_mat = strtrim(char(opts.qc_mat));
movie_h5 = strtrim(char(opts.movie_h5));
out_h5 = strtrim(char(opts.out_h5));
run_dir = strtrim(char(opts.run_dir));

if isempty(run_dir) && ~isempty(extract_mat)
    run_dir = fileparts(fileparts(extract_mat));
end
if isempty(run_dir)
    error('run_dir is empty and could not be inferred.');
end
if ~isfolder(run_dir)
    error('run_dir not found: %s', run_dir);
end

artifact_dir = fullfile(run_dir, 'artifacts');
if ~isfolder(artifact_dir)
    error('artifacts dir not found: %s', artifact_dir);
end

if isempty(extract_mat)
    extract_mat = find_single_file(artifact_dir, 'output_*_moco_direct.mat');
end
if ~isfile(extract_mat)
    error('extract_mat not found: %s', extract_mat);
end

if isempty(movie_h5)
    cands = dir(fullfile(artifact_dir, '*_moco_direct.h5'));
    if numel(cands) == 1
        movie_h5 = fullfile(cands(1).folder, cands(1).name);
    end
end
if ~isempty(movie_h5) && ~isfile(movie_h5)
    error('movie_h5 not found: %s', movie_h5);
end

if isempty(cascade_h5)
    cands = dir(fullfile(artifact_dir, 'cascade_prediction_*.h5'));
    if numel(cands) == 1
        cascade_h5 = fullfile(cands(1).folder, cands(1).name);
    else
        cascade_h5 = '';
    end
end
if ~isempty(cascade_h5) && ~isfile(cascade_h5)
    error('cascade_h5 not found: %s', cascade_h5);
end

if isempty(qc_mat)
    cand = fullfile(run_dir, 'qc_raw_roi_localbg_excluding_cells_allrois.mat');
    if isfile(cand)
        qc_mat = cand;
    else
        qc_mat = '';
    end
end
if ~isempty(qc_mat) && ~isfile(qc_mat)
    error('qc_mat not found: %s', qc_mat);
end

if isempty(out_h5)
    out_h5 = fullfile(artifact_dir, 'viewer_bundle.h5');
end
end

function p = find_single_file(dir_path, pattern)
cands = dir(fullfile(dir_path, pattern));
if isempty(cands)
    error('No file matched: %s/%s', dir_path, pattern);
end
if numel(cands) > 1
    names = strjoin({cands.name}, ', ');
    error('Multiple files matched (%s): %s', pattern, names);
end
p = fullfile(cands(1).folder, cands(1).name);
end

function [S2, h, w, n_cells] = spatial_to_sparse2d(S)
if isa(S, 'ndSparse')
    sz = size(S);
    if numel(sz) ~= 3
        error('ndSparse spatial_weights must be 3D-like. Got size=%s', mat2str(sz));
    end
    h = sz(1);
    w = sz(2);
    n_cells = sz(3);
    S2 = sparse2d(S); % [h*w x n_cells]
    if ~issparse(S2)
        S2 = sparse(S2);
    end
else
    S = single(S);
    if ndims(S) ~= 3
        error('spatial_weights must be 3D or ndSparse.');
    end
    [h, w, n_cells] = size(S);
    S2 = sparse(reshape(S, h * w, n_cells));
end
end

function T = temporal_to_cells_by_time(Tin, n_cells)
Tin = single(Tin);
if size(Tin, 1) == n_cells
    T = Tin;
elseif size(Tin, 2) == n_cells
    T = Tin';
else
    error('Could not infer temporal_weights orientation: size=%s, n_cells=%d', ...
        mat2str(size(Tin)), n_cells);
end
end

function T = normalize_trace_orientation(Tin, n_cells)
Tin = single(Tin);
if size(Tin, 1) == n_cells
    T = Tin;
elseif size(Tin, 2) == n_cells
    T = Tin';
else
    error('Trace cell dimension mismatch: size=%s, n_cells=%d', mat2str(size(Tin)), n_cells);
end
end

function T = read_cascade_spike_prob_h5(cascade_h5, n_cells)
info = h5info(cascade_h5);
names = {info.Datasets.Name};
if ~ismember('spike_prob', names)
    error('spike_prob not found in CASCADE h5: %s', cascade_h5);
end
Tin = h5read(cascade_h5, '/spike_prob');
T = normalize_trace_orientation(single(Tin), n_cells);
end

function T_out = align_trace_time(T_in, t_base)
T_out = nan(size(T_in, 1), t_base, 'single');
t_keep = min(t_base, size(T_in, 2));
T_out(:, 1:t_keep) = T_in(:, 1:t_keep);
end

function T = orient_by_id_count(Tin, n_id)
Tin = single(Tin);
if size(Tin, 1) == n_id
    T = Tin;
elseif size(Tin, 2) == n_id
    T = Tin';
else
    error('Trace-ID count mismatch: trace size=%s, n_id=%d', mat2str(size(Tin)), n_id);
end
end

function full_trace = map_roi_trace_to_all_cells(src_trace, roi_id, n_cells, t_base)
full_trace = nan(n_cells, t_base, 'single');
if isempty(src_trace) || isempty(roi_id)
    return;
end
t_keep = min(t_base, size(src_trace, 2));

% keep valid IDs only
roi_id = round(double(roi_id(:)));
ok = roi_id >= 1 & roi_id <= n_cells;
roi_id = roi_id(ok);
src_trace = src_trace(ok, :);
if isempty(roi_id)
    return;
end

for i = 1:numel(roi_id)
    full_trace(roi_id(i), 1:t_keep) = src_trace(i, 1:t_keep);
end
end

function cent_xy = compute_centroids_xy(S2, h, w)
[ii, jj, vv] = find(S2);
vv(vv < 0) = 0;
S2p = sparse(ii, jj, vv, size(S2, 1), size(S2, 2));

[Y, X] = ndgrid(single(1:h), single(1:w));
xv = double(X(:));
yv = double(Y(:));
den = full(sum(S2p, 1));
num_x = xv' * double(S2p);
num_y = yv' * double(S2p);
cx = single((num_x ./ max(den, eps))');
cy = single((num_y ./ max(den, eps))');
cent_xy = [cx, cy];
end

function [data, indices, indptr, shape] = sparse_to_csr(S)
[ii, jj, vv] = find(S);
if isempty(ii)
    data = single([]);
    indices = int32([]);
    indptr = int64(zeros(size(S, 1) + 1, 1));
    shape = int64(size(S));
    return;
end

[~, ord] = sortrows([ii, jj], [1 2]);
ii = ii(ord);
jj = jj(ord);
vv = vv(ord);

n_rows = size(S, 1);
row_counts = accumarray(ii, 1, [n_rows, 1], @sum, 0);
indptr = int64([0; cumsum(row_counts)]);
indices = int32(jj - 1);
data = single(vv);
shape = int64(size(S));
end

function write_string_dataset(h5_path, ds_path, txt)
txt = char(txt);
bytes = uint8(txt(:)');
if isempty(bytes)
    bytes = uint8(0);
end
write_numeric_vector_dataset(h5_path, ds_path, bytes, 'uint8');
end

function write_scalar_dataset(h5_path, ds_path, value, dtype_name)
h5create(h5_path, ds_path, [1 1], 'Datatype', dtype_name, 'ChunkSize', [1 1]);
h5write(h5_path, ds_path, reshape(value, [1 1]));
verify_dataset_exists(h5_path, ds_path);
end

function write_numeric_vector_dataset(h5_path, ds_path, arr, dtype_name)
arr = arr(:);
n = numel(arr);
chunk_len = max(1, min(n, 1048576));
h5create(h5_path, ds_path, [n 1], 'Datatype', dtype_name, 'ChunkSize', [chunk_len 1]);
for start_idx = 1:chunk_len:n
    end_idx = min(n, start_idx + chunk_len - 1);
    chunk = arr(start_idx:end_idx);
    h5write(h5_path, ds_path, chunk, [start_idx 1], [numel(chunk) 1]);
end
verify_dataset_exists(h5_path, ds_path);
end

function write_py2d_dataset(h5_path, ds_path, arr_py, dtype_name)
if ndims(arr_py) ~= 2
    error('write_py2d_dataset expects 2D array. got size=%s', mat2str(size(arr_py)));
end
store_h = size(arr_py, 2);
store_w = size(arr_py, 1);
chunk_h = max(1, min(store_h, 256));
chunk_w = max(1, min(store_w, 256));
h5create(h5_path, ds_path, [store_h, store_w], 'Datatype', dtype_name, ...
    'ChunkSize', [chunk_h, chunk_w]);
for row_start = 1:chunk_h:store_h
    row_end = min(store_h, row_start + chunk_h - 1);
    chunk = arr_py(:, row_start:row_end).';
    h5write(h5_path, ds_path, chunk, [row_start 1], size(chunk));
end
verify_dataset_exists(h5_path, ds_path);
end

function verify_required_bundle_datasets(h5_path, ds_paths)
for i = 1:numel(ds_paths)
    verify_dataset_exists(h5_path, ds_paths{i});
end
end

function verify_dataset_exists(h5_path, ds_path)
try
    h5info(h5_path, ds_path);
catch ME
    error('Bundle write verification failed for %s (%s): %s', ds_path, h5_path, ME.message);
end
end

function opts = parse_name_values(opts, args)
if isempty(args)
    return;
end
if mod(numel(args), 2) ~= 0
    error('Name-value arguments must be paired.');
end
for i = 1:2:numel(args)
    key = char(args{i});
    val = args{i + 1};
    if ~isfield(opts, key)
        error('Unknown option: %s', key);
    end
    opts.(key) = val;
end
end

function s = to_str(x)
if isempty(x)
    s = '(none)';
else
    s = char(x);
end
end
