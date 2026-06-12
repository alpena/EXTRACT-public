function [config, selected_gpu, gpu_name] = select_extract_gpu(config)
% Select a GPU for the current EXTRACT execution context.
if nargin < 1 || isempty(config)
    config = struct();
end

selected_gpu = [];
gpu_name = '';

if ~isfield(config, 'use_gpu') || ~config.use_gpu
    return;
end

gpu_count = gpuDeviceCount;
if gpu_count < 1
    config.use_gpu = 0;
    return;
end

if isfield(config, 'multi_gpu') && config.multi_gpu
    task = [];
    try
        task = getCurrentTask();
    catch
        task = [];
    end
    if isempty(task)
        worker_id = 1;
    else
        worker_id = task.ID;
    end
    selected_gpu = mod(worker_id - 1, gpu_count) + 1;
else
    worker_id = [];
    if isfield(config, 'pick_gpu') && ~isempty(config.pick_gpu)
        selected_gpu = config.pick_gpu;
    else
        current_device = [];
        try
            current_device = gpuDevice;
        catch
            current_device = [];
        end
        if isempty(current_device)
            selected_gpu = 1;
        else
            selected_gpu = current_device.Index;
        end
    end
end

device = gpuDevice(selected_gpu);
gpu_name = device.Name;
available_memory_gb = device.AvailableMemory / 2^30;
config.pick_gpu = selected_gpu;
config.assigned_gpu_id = selected_gpu;
config.assigned_gpu_available_memory_gb = available_memory_gb;
if exist('worker_id', 'var')
    config.assigned_gpu_worker_id = worker_id;
end

if isfield(config, 'gpu_oversubscribe') && config.gpu_oversubscribe
    guard_gb = 6;
    if isfield(config, 'gpu_memory_guard_gb') && ~isempty(config.gpu_memory_guard_gb)
        guard_gb = config.gpu_memory_guard_gb;
    end
    if guard_gb > 0 && available_memory_gb < guard_gb
        error(['EXTRACT GPU memory guard failed on GPU %d (%s): ', ...
            'available %.2f GiB is below gpu_memory_guard_gb=%.2f GiB.'], ...
            selected_gpu, gpu_name, available_memory_gb, guard_gb);
    end
end

if isfield(config, 'verbose') && config.verbose ~= 0
    if isfield(config, 'multi_gpu') && config.multi_gpu
        fprintf('%s: EXTRACT worker %d assigned GPU %d (%s), available memory %.2f GiB\n', ...
            datestr(now), worker_id, selected_gpu, gpu_name, available_memory_gb);
    else
        fprintf('%s: EXTRACT assigned GPU %d (%s), available memory %.2f GiB\n', ...
            datestr(now), selected_gpu, gpu_name, available_memory_gb);
    end
end
end
