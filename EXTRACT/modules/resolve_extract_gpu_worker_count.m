function [num_workers, details] = resolve_extract_gpu_worker_count(gpu_count, config)
%RESOLVE_EXTRACT_GPU_WORKER_COUNT Resolve EXTRACT parpool size for multi-GPU runs.

if nargin < 2 || isempty(config)
    config = struct();
end

gpu_count = max(0, round(gpu_count));
if gpu_count < 1
    error('gpu_count must be >= 1.');
end

oversubscribe = isfield(config, 'gpu_oversubscribe') && config.gpu_oversubscribe;
if oversubscribe
    if isfield(config, 'gpu_workers_per_device') && ~isempty(config.gpu_workers_per_device)
        workers_per_gpu = max(1, round(config.gpu_workers_per_device));
    else
        workers_per_gpu = 1;
    end
    max_workers = gpu_count * workers_per_gpu;
    if isfield(config, 'num_workers') && ~isempty(config.num_workers)
        requested_workers = round(config.num_workers);
    else
        requested_workers = max_workers;
    end
else
    workers_per_gpu = 1;
    max_workers = gpu_count;
    if isfield(config, 'num_workers') && ~isempty(config.num_workers)
        requested_workers = round(config.num_workers);
    else
        requested_workers = gpu_count;
    end
end

if requested_workers < 1
    error('Requested EXTRACT worker count must be >= 1.');
end

num_workers = min(requested_workers, max_workers);
details = struct();
details.gpu_count = gpu_count;
details.oversubscribe = oversubscribe;
details.workers_per_gpu = workers_per_gpu;
details.requested_workers = requested_workers;
details.max_workers = max_workers;
details.was_capped = requested_workers > max_workers;
end
