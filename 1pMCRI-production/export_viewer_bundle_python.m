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
%   postprocess_h5  : optional postprocess HDF5 with canonical dff/F0 metrics
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
opts.postprocess_h5 = '';
opts.qc_mat = '';
opts.movie_h5 = '';
opts.movie_dataset = '/mov';
opts.out_h5 = '';
opts.version = '1.1.0';
opts.verbose = true;
opts = parse_name_values(opts, varargin);

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

[extract_mat, cascade_h5, postprocess_h5, qc_mat, movie_h5, out_h5, run_dir] = resolve_paths(opts);

if opts.verbose
    fprintf('run_dir    : %s\n', run_dir);
    fprintf('extract_mat: %s\n', extract_mat);
    fprintf('cascade_h5 : %s\n', to_str(cascade_h5));
    fprintf('postprocess: %s\n', to_str(postprocess_h5));
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

metric_payload = build_metric_payload(output, S2, n_cells, postprocess_h5, trace_spike_prob);

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
write_string_list_dataset(out_h5, '/metric/names', metric_payload.names);
write_string_list_dataset(out_h5, '/metric/display_names', metric_payload.display_names);
write_string_list_dataset(out_h5, '/metric/source', metric_payload.source);
write_py2d_dataset(out_h5, '/metric/values', metric_payload.values, 'single');
write_py2d_dataset(out_h5, '/metric/valid', uint8(metric_payload.valid), 'uint8');

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
    '/metric/names', ...
    '/metric/display_names', ...
    '/metric/source', ...
    '/metric/values', ...
    '/metric/valid', ...
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
out.postprocess_h5 = postprocess_h5;
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

function [extract_mat, cascade_h5, postprocess_h5, qc_mat, movie_h5, out_h5, run_dir] = resolve_paths(opts)
extract_mat = strtrim(char(opts.extract_mat));
cascade_h5 = strtrim(char(opts.cascade_h5));
postprocess_h5 = strtrim(char(opts.postprocess_h5));
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
    extract_mat = find_single_file(artifact_dir, 'output_*_moco_*.mat');
end
if ~isfile(extract_mat)
    error('extract_mat not found: %s', extract_mat);
end

if isempty(movie_h5)
    cands = dir(fullfile(artifact_dir, '*_moco_*.h5'));
    if numel(cands) == 1
        movie_h5 = fullfile(cands(1).folder, cands(1).name);
    end
end
if ~isempty(movie_h5) && ~isfile(movie_h5)
    error('movie_h5 not found: %s', movie_h5);
end

if isempty(cascade_h5)
    cascade_h5 = read_artifact_from_manifest(run_dir, 'cascade', 'output_h5');
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

if isempty(postprocess_h5)
    postprocess_h5 = read_artifact_from_manifest(run_dir, 'postprocess', 'output_h5');
end
if isempty(postprocess_h5)
    cands = dir(fullfile(artifact_dir, 'output_*_dff_*.h5'));
    if numel(cands) == 1
        postprocess_h5 = fullfile(cands(1).folder, cands(1).name);
    else
        postprocess_h5 = '';
    end
end
if ~isempty(postprocess_h5) && ~isfile(postprocess_h5)
    error('postprocess_h5 not found: %s', postprocess_h5);
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

function payload = build_metric_payload(output, S2, n_cells, postprocess_h5, trace_spike_prob)
[names, display_names, source_names] = get_viewer_metric_spec();
metric_map = containers.Map(names, num2cell(1:numel(names)));
values = nan(numel(names), n_cells, 'single');
valid = false(numel(names), n_cells);

[values, valid] = assign_metric(values, valid, metric_map, ...
    'roi_area_px', compute_roi_area_px(S2, n_cells), true(1, n_cells));
[values, valid] = assign_metric(values, valid, metric_map, ...
    'roi_weight_max', compute_roi_weight_max(S2, n_cells), true(1, n_cells));

[values, valid] = fill_extract_metrics(values, valid, metric_map, output, n_cells);
[values, valid] = fill_postprocess_metrics(values, valid, metric_map, postprocess_h5, n_cells);
[values, valid] = fill_cascade_metrics(values, valid, metric_map, trace_spike_prob, n_cells);

payload = struct();
payload.names = names;
payload.display_names = display_names;
payload.source = source_names;
payload.values = values;
payload.valid = valid;
end

function [names, display_names, source_names] = get_viewer_metric_spec()
names = { ...
    'extract_T_maxval', ...
    'extract_S_corruption', ...
    'extract_S_eccent', ...
    'extract_S_area_1', ...
    'extract_S_smooth_area_1', ...
    'extract_S_area_2', ...
    'extract_S_max_corr', ...
    'extract_ST2_index_4', ...
    'extract_ST_corr_3', ...
    'extract_T_dup_val', ...
    'extract_is_rejected', ...
    'extract_is_tiny', ...
    'extract_is_huge', ...
    'extract_is_duplicate', ...
    'extract_is_spurious', ...
    'post_dff_std', ...
    'post_dff_p95', ...
    'post_dff_max', ...
    'post_F0_cell', ...
    'cascade_spike_mean', ...
    'cascade_spike_p95', ...
    'cascade_spike_max', ...
    'roi_area_px', ...
    'roi_weight_max'};

display_names = { ...
    'EXTRACT T max value', ...
    'EXTRACT spatial corruption', ...
    'EXTRACT eccentricity', ...
    'EXTRACT area 1', ...
    'EXTRACT smooth area 1', ...
    'EXTRACT area 2', ...
    'EXTRACT spatial max corr', ...
    'EXTRACT ST2 index 4', ...
    'EXTRACT ST corr 3', ...
    'EXTRACT T duplicate value', ...
    'EXTRACT rejected', ...
    'EXTRACT tiny', ...
    'EXTRACT huge', ...
    'EXTRACT duplicate', ...
    'EXTRACT spurious', ...
    'Post dF/F std', ...
    'Post dF/F p95', ...
    'Post dF/F max', ...
    'Post F0 cell', ...
    'Cascade spike mean', ...
    'Cascade spike p95', ...
    'Cascade spike max', ...
    'ROI area px', ...
    'ROI weight max'};

source_names = { ...
    'extract', ...
    'extract', ...
    'extract', ...
    'extract', ...
    'extract', ...
    'extract', ...
    'extract', ...
    'extract', ...
    'extract', ...
    'extract', ...
    'derived', ...
    'derived', ...
    'derived', ...
    'derived', ...
    'derived', ...
    'postprocess', ...
    'postprocess', ...
    'postprocess', ...
    'postprocess', ...
    'cascade', ...
    'cascade', ...
    'cascade', ...
    'derived', ...
    'derived'};
end

function [values, valid] = assign_metric(values, valid, metric_map, metric_name, metric_values, metric_valid)
if ~isKey(metric_map, metric_name)
    error('Unknown metric_name: %s', metric_name);
end
idx = metric_map(metric_name);
row = single(metric_values(:)');
if size(values, 2) ~= numel(row)
    error('Metric %s size mismatch: expected %d, got %d', metric_name, size(values, 2), numel(row));
end
mask = logical(metric_valid(:)');
if numel(mask) ~= numel(row)
    error('Metric %s validity size mismatch.', metric_name);
end
values(idx, :) = row;
valid(idx, :) = mask;
end

function [values, valid] = fill_extract_metrics(values, valid, metric_map, output, n_cells)
if ~isfield(output, 'info') || ~isstruct(output.info) || ...
        ~isfield(output.info, 'cellcheck') || ~isstruct(output.info.cellcheck) || ...
        ~isfield(output.info.cellcheck, 'metrics') || isempty(output.info.cellcheck.metrics)
    return;
end

metrics = orient_metric_table(output.info.cellcheck.metrics, n_cells);
[fmap, ~] = get_quality_metric_map();

raw_specs = { ...
    'extract_T_maxval', 'T_maxval'; ...
    'extract_S_corruption', 'S_corruption'; ...
    'extract_S_eccent', 'S_eccent'; ...
    'extract_S_area_1', 'S_area_1'; ...
    'extract_S_smooth_area_1', 'S_smooth_area_1'; ...
    'extract_S_area_2', 'S_area_2'; ...
    'extract_S_max_corr', 'S_max_corr'; ...
    'extract_ST2_index_4', 'ST2_index_4'; ...
    'extract_ST_corr_3', 'ST_corr_3'; ...
    'extract_T_dup_val', 'T_dup_val'};

for i = 1:size(raw_specs, 1)
    key_out = raw_specs{i, 1};
    key_in = raw_specs{i, 2};
    if isKey(fmap, key_in)
        row = single(metrics(fmap(key_in), :));
        row_valid = isfinite(row);
        [values, valid] = assign_metric(values, valid, metric_map, key_out, row, row_valid);
    end
end

derived = compute_extract_classification_metrics(output, metrics, fmap, n_cells);
derived_names = fieldnames(derived);
for i = 1:numel(derived_names)
    name = derived_names{i};
    entry = derived.(name);
    [values, valid] = assign_metric(values, valid, metric_map, name, entry.values, entry.valid);
end
end

function metrics = orient_metric_table(metric_table, n_cells)
metric_table = single(metric_table);
if size(metric_table, 2) == n_cells
    metrics = metric_table;
elseif size(metric_table, 1) == n_cells
    metrics = metric_table';
else
    error('Could not infer cellcheck metric orientation: size=%s, n_cells=%d', ...
        mat2str(size(metric_table)), n_cells);
end
end

function derived = compute_extract_classification_metrics(output, metrics, fmap, n_cells)
derived = struct();
default_false = false(1, n_cells);
default_nan = nan(1, n_cells, 'single');

required_metric_keys = { ...
    'T_maxval', 'S_corruption', 'S_eccent', 'S_area_1', 'S_smooth_area_1', ...
    'S_area_2', 'ST2_index_4', 'ST_corr_3', 'T_dup_val', 'S_max_corr'};
for i = 1:numel(required_metric_keys)
    if ~isKey(fmap, required_metric_keys{i})
        derived.extract_is_rejected = make_metric_entry(default_nan, default_false);
        derived.extract_is_tiny = make_metric_entry(default_nan, default_false);
        derived.extract_is_huge = make_metric_entry(default_nan, default_false);
        derived.extract_is_duplicate = make_metric_entry(default_nan, default_false);
        derived.extract_is_spurious = make_metric_entry(default_nan, default_false);
        return;
    end
end

if ~isfield(output, 'config') || ~isstruct(output.config) || ...
        ~isfield(output.config, 'thresholds') || ~isstruct(output.config.thresholds) || ...
        ~isfield(output.config, 'avg_cell_radius') || isempty(output.config.avg_cell_radius)
    derived.extract_is_rejected = make_metric_entry(default_nan, default_false);
    derived.extract_is_tiny = make_metric_entry(default_nan, default_false);
    derived.extract_is_huge = make_metric_entry(default_nan, default_false);
    derived.extract_is_duplicate = make_metric_entry(default_nan, default_false);
    derived.extract_is_spurious = make_metric_entry(default_nan, default_false);
    return;
end

th = output.config.thresholds;
avg_cell_area = pi * double(output.config.avg_cell_radius) .^ 2;

m_T_maxval = double(metrics(fmap('T_maxval'), :));
m_S_corr = double(metrics(fmap('S_corruption'), :));
m_S_ecc = double(metrics(fmap('S_eccent'), :));
m_S_area1 = double(metrics(fmap('S_area_1'), :));
m_S_smooth_area1 = double(metrics(fmap('S_smooth_area_1'), :));
m_S_area2 = double(metrics(fmap('S_area_2'), :));
m_ST2_4 = double(metrics(fmap('ST2_index_4'), :));
m_ST_corr3 = double(metrics(fmap('ST_corr_3'), :));
m_T_dup = double(metrics(fmap('T_dup_val'), :));
m_S_max_corr = double(metrics(fmap('S_max_corr'), :));

size_lower_px = double(th.size_lower_limit) * avg_cell_area;
size_upper_px = double(th.size_upper_limit) * avg_cell_area;

is_T_zeroed = (m_T_maxval <= double(th.T_min_snr));
is_S_tiny = (max(m_S_area1, m_S_smooth_area1) <= size_lower_px) | (m_S_smooth_area1 == 0);
is_S_huge = (m_S_area2 ./ max(1, -2 + m_S_ecc)) >= size_upper_px;
is_S_poor_looking = (m_S_corr >= double(th.spatial_corrupt_thresh));
is_S_poor_eccent = (m_S_ecc >= double(th.eccent_thresh));
is_T_duplicate = (m_T_dup >= double(th.T_dup_corr_thresh));
is_S_duplicate = (m_S_max_corr >= double(th.S_dup_corr_thresh));
is_ST_spurious = (m_ST2_4 <= double(th.low_ST_index_thresh)) | ...
                  (m_ST_corr3 < double(th.low_ST_corr_thresh));

is_duplicate = is_T_duplicate | is_S_duplicate;
is_spurious = is_ST_spurious | is_S_poor_looking | is_S_poor_eccent | is_T_zeroed;
is_rejected = is_T_zeroed | is_S_tiny | is_S_huge | ...
              is_S_poor_looking | is_S_poor_eccent | is_duplicate | is_ST_spurious;

all_valid = true(1, n_cells);
derived.extract_is_rejected = make_metric_entry(single(is_rejected), all_valid);
derived.extract_is_tiny = make_metric_entry(single(is_S_tiny), all_valid);
derived.extract_is_huge = make_metric_entry(single(is_S_huge), all_valid);
derived.extract_is_duplicate = make_metric_entry(single(is_duplicate), all_valid);
derived.extract_is_spurious = make_metric_entry(single(is_spurious), all_valid);
end

function [values, valid] = fill_postprocess_metrics(values, valid, metric_map, postprocess_h5, n_cells)
if isempty(postprocess_h5) || ~isfile(postprocess_h5)
    return;
end

[dff, F0_cell] = read_postprocess_data(postprocess_h5, n_cells);
if isempty(dff)
    return;
end

dff_std = rowwise_std(dff);
dff_p95 = rowwise_percentile(dff, 95);
dff_max = rowwise_max(dff);
f0 = single(F0_cell(:)');
if numel(f0) ~= n_cells
    f0 = nan(1, n_cells, 'single');
    f0_valid = false(1, n_cells);
else
    f0_valid = isfinite(f0);
end

[values, valid] = assign_metric(values, valid, metric_map, 'post_dff_std', dff_std, isfinite(dff_std));
[values, valid] = assign_metric(values, valid, metric_map, 'post_dff_p95', dff_p95, isfinite(dff_p95));
[values, valid] = assign_metric(values, valid, metric_map, 'post_dff_max', dff_max, isfinite(dff_max));
[values, valid] = assign_metric(values, valid, metric_map, 'post_F0_cell', f0, f0_valid);
end

function [dff, F0_cell] = read_postprocess_data(postprocess_h5, n_cells)
dff = [];
F0_cell = [];
[stored_cells, stored_frames, orientation] = read_postprocess_h5_meta(postprocess_h5);
if ~strcmp(orientation, 'cells_by_frames')
    error('postprocess_h5 orientation must be cells_by_frames: %s', orientation);
end

dff = single(h5read(postprocess_h5, '/dff'));
if isequal(size(dff), [stored_frames, stored_cells])
    dff = dff';
elseif ~isequal(size(dff), [stored_cells, stored_frames])
    error('postprocess_h5 /dff shape mismatch: got %s expected [%d %d]', ...
        mat2str(size(dff)), stored_cells, stored_frames);
end
if stored_cells ~= n_cells
    error('postprocess_h5 n_cells mismatch: expected %d got %d', n_cells, stored_cells);
end

if dataset_exists(postprocess_h5, '/F0_cell')
    F0_cell = single(h5read(postprocess_h5, '/F0_cell'));
else
    F0_cell = nan(n_cells, 1, 'single');
end
if isempty(F0_cell)
    F0_cell = nan(n_cells, 1, 'single');
else
    F0_cell = single(F0_cell(:));
end
end

function [n_cells, n_frames, orientation] = read_postprocess_h5_meta(postprocess_h5)
n_cells = double(h5read(postprocess_h5, '/meta/n_cells'));
n_frames = double(h5read(postprocess_h5, '/meta/n_frames'));
orientation = read_h5_string_dataset(postprocess_h5, '/meta/orientation');
end

function tf = dataset_exists(h5_path, ds_path)
try
    h5info(h5_path, ds_path);
    tf = true;
catch
    tf = false;
end
end

function [values, valid] = fill_cascade_metrics(values, valid, metric_map, trace_spike_prob, n_cells)
if isempty(trace_spike_prob)
    return;
end
spike = single(trace_spike_prob);
if size(spike, 1) ~= n_cells
    spike = normalize_trace_orientation(spike, n_cells);
end
spike_mean = rowwise_mean(spike);
spike_p95 = rowwise_percentile(spike, 95);
spike_max = rowwise_max(spike);

[values, valid] = assign_metric(values, valid, metric_map, 'cascade_spike_mean', spike_mean, isfinite(spike_mean));
[values, valid] = assign_metric(values, valid, metric_map, 'cascade_spike_p95', spike_p95, isfinite(spike_p95));
[values, valid] = assign_metric(values, valid, metric_map, 'cascade_spike_max', spike_max, isfinite(spike_max));
end

function area_px = compute_roi_area_px(S2, n_cells)
[~, jj, vv] = find(S2);
keep = vv > 0;
counts = accumarray(jj(keep), 1, [n_cells, 1], @sum, 0);
area_px = single(counts(:)');
end

function roi_weight_max = compute_roi_weight_max(S2, n_cells)
[~, jj, vv] = find(S2);
keep = vv > 0;
if ~any(keep)
    roi_weight_max = zeros(1, n_cells, 'single');
    return;
end
roi_weight_max = single(accumarray(jj(keep), vv(keep), [n_cells, 1], @max, 0));
roi_weight_max = roi_weight_max(:)';
end

function entry = make_metric_entry(values, valid)
entry = struct();
entry.values = single(values(:)');
entry.valid = logical(valid(:)');
end

function out = rowwise_mean(X)
X = double(X);
mask = isfinite(X);
tmp = X;
tmp(~mask) = 0;
den = sum(mask, 2);
out = single(sum(tmp, 2) ./ max(den, 1));
out(den == 0) = nan;
out = out(:)';
end

function out = rowwise_std(X)
X = double(X);
mask = isfinite(X);
tmp = X;
tmp(~mask) = 0;
den = sum(mask, 2);
mu = sum(tmp, 2) ./ max(den, 1);
sq = bsxfun(@minus, X, mu) .^ 2;
sq(~mask) = 0;
den2 = max(den - 1, 1);
out = single(sqrt(sum(sq, 2) ./ den2));
out(den <= 1) = nan;
out = out(:)';
end

function out = rowwise_max(X)
X = single(X);
mask = isfinite(X);
tmp = X;
tmp(~mask) = -inf;
out = max(tmp, [], 2);
out(~any(mask, 2)) = nan;
out = single(out(:)');
end

function out = rowwise_percentile(X, p)
n_rows = size(X, 1);
out = nan(1, n_rows, 'single');
for i = 1:n_rows
    vals = double(X(i, isfinite(X(i, :))));
    if isempty(vals)
        continue;
    end
    out(i) = single(prctile(vals, p));
end
end

function path_out = read_artifact_from_manifest(run_dir, stage_name, output_key)
path_out = '';
manifest_path = fullfile(run_dir, 'manifests', [stage_name '.json']);
if ~isfile(manifest_path)
    return;
end
try
    data = jsondecode(fileread(manifest_path));
catch
    return;
end
if ~isstruct(data) || ~isfield(data, 'status') || ~strcmpi(char(data.status), 'completed')
    return;
end
if ~isfield(data, 'outputs') || ~isstruct(data.outputs) || ~isfield(data.outputs, output_key)
    return;
end
value = data.outputs.(output_key);
if isempty(value)
    return;
end
candidate = char(value);
if is_absolute_path(candidate)
    path_out = candidate;
else
    path_out = fullfile(run_dir, candidate);
end
end

function tf = is_absolute_path(path_in)
if isempty(path_in)
    tf = false;
    return;
end
tf = startsWith(path_in, '/') || startsWith(path_in, '\\') || ...
    ~isempty(regexp(path_in, '^[A-Za-z]:[\\/]', 'once'));
end

function txt = read_h5_string_dataset(h5_path, ds_path)
raw = h5read(h5_path, ds_path);
txt = char(raw(:)');
null_idx = find(txt == char(0), 1, 'first');
if ~isempty(null_idx)
    txt = txt(1:null_idx - 1);
end
end

function write_string_list_dataset(h5_path, ds_path, values)
if isempty(values)
    error('write_string_list_dataset requires at least one value.');
end
max_len = max(cellfun(@numel, values));
max_len = max(1, max_len);
arr = zeros(numel(values), max_len, 'uint8');
for i = 1:numel(values)
    txt = char(values{i});
    bytes = uint8(txt(:)');
    if ~isempty(bytes)
        arr(i, 1:numel(bytes)) = bytes;
    end
end
write_py2d_dataset(h5_path, ds_path, arr, 'uint8');
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
