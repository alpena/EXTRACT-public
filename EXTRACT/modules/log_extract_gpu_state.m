function log_extract_gpu_state(config, stage, detail)
% Emit GPU memory diagnostics for EXTRACT when debug_gpu_memory is enabled.
if nargin < 2 || isempty(stage)
    stage = 'state';
end
if nargin < 3
    detail = '';
end
if ~isstruct(config) || ~isfield(config, 'debug_gpu_memory') || ~config.debug_gpu_memory
    return;
end
if ~isfield(config, 'use_gpu') || ~config.use_gpu
    return;
end

partition_str = '';
if isfield(config, 'partition_id') && ~isempty(config.partition_id)
    partition_str = sprintf(' partition=%d', config.partition_id);
end

worker_str = '';
try
    task = getCurrentTask();
    if ~isempty(task)
        worker_str = sprintf(' worker=%d', task.ID);
    end
catch
    worker_str = '';
end

detail_str = '';
if ~isempty(detail)
    detail_str = sprintf(' %s', detail);
end

try
    device = gpuDevice();
    used_mib = (double(device.TotalMemory) - double(device.AvailableMemory)) / 2^20;
    avail_mib = double(device.AvailableMemory) / 2^20;
    total_mib = double(device.TotalMemory) / 2^20;
    fprintf(['%s: GPU_STATE%s%s gpu=%d (%s) used=%.1f MiB ', ...
        'avail=%.1f MiB total=%.1f MiB stage=%s%s\n'], ...
        datestr(now), partition_str, worker_str, device.Index, device.Name, ...
        used_mib, avail_mib, total_mib, stage, detail_str);
catch ME
    fprintf('%s: GPU_STATE%s%s stage=%s unavailable=%s%s\n', ...
        datestr(now), partition_str, worker_str, stage, ME.identifier, detail_str);
end
end
