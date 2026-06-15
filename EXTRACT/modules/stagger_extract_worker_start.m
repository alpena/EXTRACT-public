function stagger_extract_worker_start(config)
% Stagger the first partition load on each parallel worker.
%
% Large pre-split movies can allocate tens of GiB per worker while each worker
% loads its first partition.  Starting all workers at exactly the same time can
% create a transient RAM spike that kills a worker before EXTRACT reaches GPU
% work.  The persistent flag keeps this delay to one time per worker process.
    persistent did_stagger;
    if ~isempty(did_stagger) && did_stagger
        return;
    end
    did_stagger = true;

    if nargin < 1 || ~isstruct(config) || ...
            ~isfield(config, 'gpu_worker_start_stagger_sec') || ...
            isempty(config.gpu_worker_start_stagger_sec)
        return;
    end
    stagger_sec = double(config.gpu_worker_start_stagger_sec);
    if ~isfinite(stagger_sec) || stagger_sec <= 0
        return;
    end

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
    pause_sec = max(0, worker_id - 1) * stagger_sec;
    if isfield(config, 'debug_gpu_memory') && config.debug_gpu_memory
        fprintf('%s: EXTRACT worker %d initial start stagger %.1f sec\n', ...
            datestr(now), worker_id, pause_sec);
    end
    if pause_sec > 0
        pause(pause_sec);
    end
end
