function result = replace_preprocessed_posF_with_extract(varargin)
% replace_preprocessed_posF_with_extract
% Replace preprocessed_data.mat pos/F using current pipeline artifacts.
%
% Inputs are resolved from a pipeline run_dir:
%   - EXTRACT: run_dir/artifacts/output_*_moco_direct.mat
%   - CASCADE: run_dir/artifacts/cascade_prediction_*.h5
%
% Name-value options:
%   run_dir           : pipeline output directory
%                       (default: demo_data/output/smoke_250810)
%   preprocessed_mat  : target preprocessed_data.mat to update (required)
%   extract_mat       : optional explicit EXTRACT MAT override
%   cascade_h5        : optional explicit CASCADE HDF5 override
%   write_mode        : 'matfile' (default) | 'append'
%
% Replaced variables:
%   pos : ROI centroids transformed using import_opt if present
%   F   : CASCADE spike_prob, stored as [cells x frames]

opts = struct();
opts.run_dir = 'R:/code/1pMCRI-pipeline/demo_data/output/smoke_250810';
opts.preprocessed_mat = '';
opts.extract_mat = '';
opts.cascade_h5 = '';
opts.write_mode = 'matfile';
opts = parse_name_values(opts, varargin);

run_dir = char(opts.run_dir);
preprocessed_mat = char(opts.preprocessed_mat);
extract_mat = char(opts.extract_mat);
cascade_h5 = char(opts.cascade_h5);
write_mode = lower(strtrim(char(opts.write_mode)));

if isempty(preprocessed_mat)
    error('preprocessed_mat is required.');
end
if ~isfolder(run_dir)
    error('run_dir not found: %s', run_dir);
end
if ~isfile(preprocessed_mat)
    error('preprocessed_data.mat not found: %s', preprocessed_mat);
end

artifact_dir = fullfile(run_dir, 'artifacts');
if ~isfolder(artifact_dir)
    error('artifacts dir not found: %s', artifact_dir);
end

if isempty(extract_mat)
    extract_mat = find_single_file(artifact_dir, 'output_*_moco_direct.mat');
end
if isempty(cascade_h5)
    cascade_h5 = find_single_file(artifact_dir, 'cascade_prediction_*.h5');
end
if ~isfile(extract_mat)
    error('EXTRACT output mat not found: %s', extract_mat);
end
if ~isfile(cascade_h5)
    error('CASCADE h5 not found: %s', cascade_h5);
end

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));

fprintf('run_dir          : %s\n', run_dir);
fprintf('preprocessed_mat : %s\n', preprocessed_mat);
fprintf('extract_mat      : %s\n', extract_mat);
fprintf('cascade_h5       : %s\n', cascade_h5);
fprintf('write_mode       : %s\n', write_mode);

import_opt = struct();
if ~isempty(whos('-file', preprocessed_mat, 'import_opt'))
    T0 = load(preprocessed_mat, 'import_opt');
    if isfield(T0, 'import_opt') && ~isempty(T0.import_opt)
        import_opt = T0.import_opt;
    end
end

E = load(extract_mat, 'output');
if ~isfield(E, 'output') || ~isfield(E.output, 'spatial_weights')
    error('Invalid EXTRACT output format in %s', extract_mat);
end

S = E.output.spatial_weights;
if isa(S, 'ndSparse')
    [h, w, n_cells] = size(S);
    S2 = sparse2d(S);
else
    S = double(S);
    if ndims(S) ~= 3
        error('spatial_weights must be 3D or ndSparse.');
    end
    [h, w, n_cells] = size(S);
    S2 = sparse(reshape(S, h * w, n_cells));
end
S2 = double(S2);

spk = read_cascade_spike_prob_h5(cascade_h5, n_cells);
if size(spk, 1) == n_cells
    F = spk;
elseif size(spk, 2) == n_cells
    F = spk';
else
    error(['Cell count mismatch: EXTRACT spatial cells=%d, ', ...
        'CASCADE spike_prob size=[%d %d]'], n_cells, size(spk, 1), size(spk, 2));
end

[Yg, Xg] = ndgrid(single(1:h), single(1:w));
Xv = double(Xg(:));
Yv = double(Yg(:));
S2 = max(S2, 0);
denom = full(sum(S2, 1));
sx = Xv' * S2;
sy = Yv' * S2;
denom = max(denom, eps);
pos_pix = [sx(:) ./ denom(:), sy(:) ./ denom(:)];

% Match existing downstream convention.
pos_pix(:, 2) = -pos_pix(:, 2);

[pos, transform_meta] = apply_import_opt_transform(pos_pix, import_opt);

switch write_mode
    case 'matfile'
        m = matfile(preprocessed_mat, 'Writable', true);
        m.pos = pos;
        m.F = F;
    case 'append'
        save(preprocessed_mat, 'pos', 'F', '-append');
    otherwise
        error('Unknown write_mode: %s', write_mode);
end

result = struct();
result.run_dir = run_dir;
result.preprocessed_mat = preprocessed_mat;
result.extract_mat = extract_mat;
result.cascade_h5 = cascade_h5;
result.pos_size = size(pos);
result.F_size = size(F);
result.transform_meta = transform_meta;

fprintf('Updated: %s\n', preprocessed_mat);
fprintf('  pos size = [%d %d]\n', size(pos, 1), size(pos, 2));
fprintf('  F   size = [%d %d]\n', size(F, 1), size(F, 2));

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

function spk = read_cascade_spike_prob_h5(cascade_h5, n_cells)
info = h5info(cascade_h5);
names = {info.Datasets.Name};
if ~ismember('spike_prob', names)
    error('spike_prob not found in CASCADE h5: %s', cascade_h5);
end
spk = single(h5read(cascade_h5, '/spike_prob'));
if size(spk, 1) ~= n_cells && size(spk, 2) == n_cells
    spk = spk';
end
end

function [pos_out, meta] = apply_import_opt_transform(pos_pix, import_opt)
meta.pix_to_dist = 1;
meta.center_pos = [0 0];
meta.angle_deg = 0;

if isstruct(import_opt)
    if isfield(import_opt, 'pix_to_dist') && ~isempty(import_opt.pix_to_dist)
        meta.pix_to_dist = double(import_opt.pix_to_dist);
    end
    if isfield(import_opt, 'center_pos') && ~isempty(import_opt.center_pos) && numel(import_opt.center_pos) >= 2
        cp = double(import_opt.center_pos(:)');
        meta.center_pos = cp(1:2);
    end
    if isfield(import_opt, 'angle_deg') && ~isempty(import_opt.angle_deg)
        meta.angle_deg = double(import_opt.angle_deg);
    end
end

pos = double(pos_pix) * meta.pix_to_dist;
pos(:, 1) = pos(:, 1) - meta.center_pos(1);
pos(:, 2) = pos(:, 2) - meta.center_pos(2);

theta = deg2rad(meta.angle_deg);
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
pos_out = (R * pos')';
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
