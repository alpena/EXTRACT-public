function scale = set_extract_gpu_memory_budget_scale(config)
% Configure the per-worker GPU memory budget scale for EXTRACT helpers.
    scale = 1;
    if nargin >= 1 && isstruct(config) && ...
            isfield(config, 'gpu_memory_budget_scale') && ...
            ~isempty(config.gpu_memory_budget_scale)
        scale = double(config.gpu_memory_budget_scale);
    end
    if ~isfinite(scale) || scale < 1
        scale = 1;
    end
    global EXTRACT_GPU_MEMORY_BUDGET_SCALE;
    EXTRACT_GPU_MEMORY_BUDGET_SCALE = scale;
end
