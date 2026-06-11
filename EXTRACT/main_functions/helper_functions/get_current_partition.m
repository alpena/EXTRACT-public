function [M_out, fov_occupation, core_occupation_local] = get_current_partition(...
    M, npx, npy, npt, overlap, idx, core_margin)
% Slice the movie in the image dimensions to get current partition.
%   M: 3-D movie matrix
%   npx: number of partititons in the x dimension
%   npy: number of partititons in the y dimension
%   overlap: width of the overlap between adjacent partitions
%   idx: current partition index
%   core_margin: width of the non-admissible boundary band inside each
%       interior partition boundary
% returns:
%   M_out: output 3-D movie matrix, sliced according to inputs
%   fov_occupation: Binary 2-D array with 1's only for the current
%   rectangular partitioned region
%   core_occupation_local: Binary 2-D array in local partition coordinates
%       with 1's for the admissible core region
    [h, w, t] = get_movie_size(M);
    if nargin < 7 || isempty(core_margin)
        core_margin = 0;
    end
    % npt is either < t or =t
    if isempty(npt) || npt > t
        npt = t;
    end
    blocksize_x = ceil((w + (npx - 1) * overlap) / npx);
    blocksize_y = ceil((h + (npy - 1) * overlap) / npy);
    [idx_partition_y, idx_partition_x] = ind2sub([npy, npx], idx);
    x_begin = (idx_partition_x - 1) * (blocksize_x - overlap) + 1;
    x_end = min(x_begin + blocksize_x - 1, w);
    x_keep = x_begin:x_end;
    y_begin = (idx_partition_y - 1) * (blocksize_y - overlap) + 1;
    y_end = min(y_begin + blocksize_y - 1, h);
    y_keep = y_begin:y_end;
    
    % Get the desired block out of the movie
    ny = y_end - y_begin + 1;
    nx = x_end - x_begin + 1;
    if ischar(M) && numel(M) > 8 && strncmp(M, 'partdir:', 8)
        % Pre-split partition HDF5: contiguous layout, single read, no lock contention.
        part_dir = M(9:end);
        manifest = jsondecode(fileread(fullfile(part_dir, 'manifest.json')));
        if isfield(manifest, 'time_sharded') && manifest.time_sharded
            M_out = read_time_sharded_partition(part_dir, manifest, idx, ny, nx, npt);
        else
            part_path = fullfile(part_dir, sprintf('partition_%03d.h5', idx));
            M_out = h5read(part_path, '/mov', [1, 1, 1], [ny, nx, npt]);
        end
    elseif ischar(M) || iscell(M)
        [path, dataset] = parse_movie_name(M);
        M_out = h5read(path, dataset, [y_begin, x_begin, 1], [ny, nx, npt]);
    else
        M_out = M(y_keep, x_keep, :);
    end
    % Make sure output is single
    M_out = single(M_out);
    % Replace nan pixels with zeros
    M_out = replace_nans_with_zeros(M_out);
    % Trim zero edges (e.g. due to image registration ertifacts)
    try
        [M_out, nz_top, nz_bottom, nz_left, nz_right] = ...
            remove_zero_edge_pixels(M_out);
    catch
        nz_top=0;
        nz_bottom=0;
        nz_left=0;
        nz_right=0;
    end
    
    x_keep = x_keep(nz_left+1:end-nz_right);
    y_keep = y_keep(nz_top+1:end-nz_bottom);
    fov_occupation = false(h, w);
    fov_occupation(y_keep, x_keep) = true;
    core_x_begin = x_keep(1);
    core_x_end = x_keep(end);
    core_y_begin = y_keep(1);
    core_y_end = y_keep(end);
    if core_margin > 0
        if idx_partition_x > 1
            core_x_begin = min(core_x_begin + core_margin, core_x_end);
        end
        if idx_partition_x < npx
            core_x_end = max(core_x_end - core_margin, core_x_begin);
        end
        if idx_partition_y > 1
            core_y_begin = min(core_y_begin + core_margin, core_y_end);
        end
        if idx_partition_y < npy
            core_y_end = max(core_y_end - core_margin, core_y_begin);
        end
    end
    core_occupation_local = false(numel(y_keep), numel(x_keep));
    core_local_x_begin = core_x_begin - x_keep(1) + 1;
    core_local_x_end = core_x_end - x_keep(1) + 1;
    core_local_y_begin = core_y_begin - y_keep(1) + 1;
    core_local_y_end = core_y_end - y_keep(1) + 1;
    core_occupation_local(core_local_y_begin:core_local_y_end, core_local_x_begin:core_local_x_end) = true;
    %fprintf('\t \t \t Discarding a [%d px top, %d px bottom, %d px left, %d px right] inactive movie region. \n'...
    %    ,nz_top, nz_bottom, nz_left, nz_right);
    
end

function M_out = read_time_sharded_partition(part_dir, manifest, idx, ny, nx, npt)
    shards = manifest.time_shards;
    M_out = zeros(ny, nx, npt, 'single');
    dst_start = 1;
    remaining = npt;
    for s = 1:numel(shards)
        if remaining <= 0
            break;
        end
        shard = shards(s);
        shard_nt = double(shard.nt);
        read_nt = min(remaining, shard_nt);
        shard_part_dir = char(shard.partition_dir);
        if ~isfolder(shard_part_dir)
            shard_part_dir = fullfile(part_dir, shard_part_dir);
        end
        part_path = fullfile(shard_part_dir, sprintf('partition_%03d.h5', idx));
        block = h5read(part_path, '/mov', [1, 1, 1], [ny, nx, read_nt]);
        M_out(:, :, dst_start:(dst_start + read_nt - 1)) = single(block);
        dst_start = dst_start + read_nt;
        remaining = remaining - read_nt;
    end
    if remaining > 0
        error('Time-sharded partition manifest ended before npt=%d frames were read.', npt);
    end
end
