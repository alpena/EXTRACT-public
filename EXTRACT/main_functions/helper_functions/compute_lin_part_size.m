function num_chunks = compute_lin_part_size(M, use_gpu, gpu_slack_factor)
    if use_gpu
        [m, n] = size(M);
        avail_size = get_effective_gpu_available_memory() / 4 / gpu_slack_factor;
        num_chunks = ceil(m * n / avail_size);
    else
        num_chunks = 1;
    end
end
