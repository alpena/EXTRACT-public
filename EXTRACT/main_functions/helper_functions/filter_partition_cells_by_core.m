function [keep_idx, centroid_xy] = filter_partition_cells_by_core(S, core_mask_local)
% Keep only cells whose weighted centroid lies inside the admissible core.

    n_cells = size(S, 2);
    keep_idx = true(1, n_cells);
    centroid_xy = nan(n_cells, 2);
    if isempty(S) || n_cells == 0
        return;
    end

    [h, w] = size(core_mask_local);
    [yy, xx] = ndgrid(single(1:h), single(1:w));
    xx = xx(:);
    yy = yy(:);

    for idx_cell = 1:n_cells
        weights = full(single(S(:, idx_cell)));
        total_weight = sum(weights);
        if total_weight <= 0
            keep_idx(idx_cell) = false;
            continue;
        end

        cx = sum(xx .* weights) / total_weight;
        cy = sum(yy .* weights) / total_weight;
        centroid_xy(idx_cell, :) = [cx, cy];

        cx_round = min(max(round(cx), 1), w);
        cy_round = min(max(round(cy), 1), h);
        keep_idx(idx_cell) = logical(core_mask_local(cy_round, cx_round));
    end
end
