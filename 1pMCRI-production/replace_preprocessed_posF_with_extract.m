%% Replace preprocessed_data.mat pos/F with EXTRACT + CASCADE output
% - pos: ROI centroids in [x y] pixel coordinates (cells x 2)
% - F:   CASCADE spike_prob as (cells x frames)
% - Optional: reapply coordinate transform saved in import_opt
%   pos_mm = (pos_pix * pix_to_dist - center_pos) rotated by angle_deg.

clearvars;

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
addpath(genpath(fullfile(repo_root, 'EXTRACT')));

preprocessed_mat = fullfile(script_dir, 'preprocessed_data.mat');
extract_output_mat = fullfile(script_dir, 'output_250810-Ras2-GC#78_reg.mat');
cascade_mat = fullfile('R:\', 'code', 'Cascade', 'Example_datasets', ...
    '1pMCRI_demo_outputs', 'cascade_prediction_1pMCRI_temporal_weights_allcells.mat');
write_mode = 'matfile'; % 'matfile' (fast, recommended) | 'append'

if ~isfile(preprocessed_mat)
    error('preprocessed_data.mat not found: %s', preprocessed_mat);
end
if ~isfile(extract_output_mat)
    error('EXTRACT output mat not found: %s', extract_output_mat);
end
if ~isfile(cascade_mat)
    error('CASCADE mat not found: %s', cascade_mat);
end

% Load import options (if available) from existing preprocessed_data.mat
import_opt = struct();
if ~isempty(whos('-file', preprocessed_mat, 'import_opt'))
    T0 = load(preprocessed_mat, 'import_opt');
    if isfield(T0, 'import_opt') && ~isempty(T0.import_opt)
        import_opt = T0.import_opt;
    end
end

E = load(extract_output_mat, 'output');
if ~isfield(E, 'output') || ~isfield(E.output, 'spatial_weights') || ~isfield(E.output, 'temporal_weights')
    error('Invalid EXTRACT output format in %s', extract_output_mat);
end

S = E.output.spatial_weights;

% Build centroid positions from spatial weights
if isa(S, 'ndSparse')
    [h, w, n_cells] = size(S);
    S2 = sparse2d(S); % [h*w x n_cells]
else
    S = single(S);
    [h, w, n_cells] = size(S);
    S2 = reshape(S, h * w, n_cells);
    S2 = max(S2, 0);
end

% Load CASCADE spike_prob and convert to cells x frames.
C = load(cascade_mat, 'spike_prob');
if ~isfield(C, 'spike_prob') || isempty(C.spike_prob)
    error('spike_prob not found in CASCADE mat: %s', cascade_mat);
end
spk = single(C.spike_prob);
if size(spk, 1) == n_cells
    F = spk;
elseif size(spk, 2) == n_cells
    F = spk';
else
    error(['Cell count mismatch: EXTRACT spatial cells=%d, ', ...
        'CASCADE spike_prob size=[%d %d]'], n_cells, size(spk, 1), size(spk, 2));
end
F = single(F);

if size(F, 1) ~= n_cells
    error('Cell count mismatch: spatial=%d, temporal(F rows)=%d', n_cells, size(F, 1));
end

[Yg, Xg] = ndgrid(single(1:h), single(1:w));
Xv = double(Xg(:));
Yv = double(Yg(:));

if issparse(S2)
    S2 = max(S2, 0);
    denom = full(sum(S2, 1));
    sx = Xv' * S2;
    sy = Yv' * S2;
else
    S2 = double(max(S2, 0));
    denom = sum(S2, 1);
    sx = Xv' * S2;
    sy = Yv' * S2;
end

denom = max(denom, eps);
pos_pix = [sx(:) ./ denom(:), sy(:) ./ denom(:)]; % [x y] in pixels

% DeepWonder-style axis convention equivalent to:
% [centroid(2), -centroid(1)]  -> [x, -y]
pos_pix(:, 2) = -pos_pix(:, 2);

% Reproduce original coordinate transform if import_opt values exist.
[pos, transform_meta] = apply_import_opt_transform(pos_pix, import_opt);

% Replace variables in preprocessed_data.mat
switch lower(write_mode)
    case 'matfile'
        % Faster than save -append for large MAT files.
        m = matfile(preprocessed_mat, 'Writable', true);
        m.pos = pos;
        m.F = F;
    case 'append'
        save(preprocessed_mat, 'pos', 'F', '-append');
    otherwise
        error('Unknown write_mode: %s', write_mode);
end

fprintf('Updated: %s\n', preprocessed_mat);
fprintf('  pos size = [%d %d]\n', size(pos, 1), size(pos, 2));
fprintf('  F   size = [%d %d]\n', size(F, 1), size(F, 2));
fprintf('  write mode = %s\n', write_mode);
fprintf('  F source = %s (spike_prob)\n', cascade_mat);
fprintf('  transform: pix_to_dist=%g, center_pos=[%g %g], angle_deg=%g\n', ...
    transform_meta.pix_to_dist, transform_meta.center_pos(1), ...
    transform_meta.center_pos(2), transform_meta.angle_deg);

function [pos_out, meta] = apply_import_opt_transform(pos_pix, import_opt)
% Reproduce:
% pos = pos * pix_to_dist;
% pos(:,i) = pos(:,i) - center_pos(i);
% pos = (R * pos')', where R from angle_deg.

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
