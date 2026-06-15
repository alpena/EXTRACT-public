function available_memory = get_effective_gpu_available_memory()
% Return GPU memory budget available to the current EXTRACT worker.
%
% EXTRACT historically sized GPU chunks from gpuDevice().AvailableMemory.
% When multiple MATLAB workers share one GPU, each process can see the same
% free memory and choose chunks that are too large collectively.  The optional
% EXTRACT_GPU_MEMORY_BUDGET_SCALE global lets oversubscribed workers size their
% chunks from a conservative per-worker budget.
    device = gpuDevice();
    scale = 1;
    global EXTRACT_GPU_MEMORY_BUDGET_SCALE;
    if ~isempty(EXTRACT_GPU_MEMORY_BUDGET_SCALE)
        scale = double(EXTRACT_GPU_MEMORY_BUDGET_SCALE);
    end
    if ~isfinite(scale) || scale < 1
        scale = 1;
    end
    available_memory = max(1, double(device.AvailableMemory) / scale);
end
