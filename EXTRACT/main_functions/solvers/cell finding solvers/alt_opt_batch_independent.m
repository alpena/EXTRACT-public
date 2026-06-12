function [S, T, T_corr, S_corr, S_change, T_change] = ...
        alt_opt_batch_independent(Mt, f_2d_init_stack, noise_std, size_limit, use_gpu, kappa_t, kappa_s, max_iter)
%ALT_OPT_BATCH_INDEPENDENT Fit non-conflicting seed candidates in one GPU batch.
%
% This is an experimental batched analogue of alt_opt_single for cell-finding
% mini-batches. Candidates are treated as independent fits against the same
% residual movie. Supports may differ, so local movies are padded to the largest
% support in the batch.

if nargin < 8
    max_iter = 10;
end
if ~isnumeric(kappa_t) || ~isnumeric(kappa_s)
    error('alt_opt_batch_independent only supports numeric kappa values.');
end

[h, w, batch_size] = size(f_2d_init_stack);
n_frames = size(Mt, 1);
n_pixels = h * w;

n_iter_in = max_iter;
n_iter_out = max_iter;
TOL = 1e-1;
scale_lambda = 0;
extend_radius_low = 3;
extend_radius_high = 6;

S = maybe_gpu(use_gpu, reshape(f_2d_init_stack, n_pixels, batch_size));
T = maybe_gpu(use_gpu, ones(batch_size, n_frames, 'single') * 1e6);
S_corr = maybe_gpu(use_gpu, zeros(n_pixels, batch_size, 'single'));
T_corr = maybe_gpu(use_gpu, zeros(batch_size, n_frames, 'single'));

S_change = zeros(batch_size, n_iter_out, 'single');
T_change = zeros(batch_size, n_iter_out, 'single');
support_idx_cache = cell(1, batch_size);
is_component_done = false(1, batch_size);

for i = 1:n_iter_out
    S_before = S;
    T_before = T;

    [support_idx_cache, M_pack, S_pack, support_mask] = pack_support_movies(...
        Mt, S, h, w, use_gpu, extend_radius_low, extend_radius_high, support_idx_cache);

    if isempty(M_pack)
        break;
    end

    lambda_t = sum(S_pack, 1) * scale_lambda;
    T_batch = solve_single_source_adaptive_batched(...
        S_pack, M_pack, support_mask, n_iter_in, lambda_t, noise_std, kappa_t);
    T = reshape(T_batch', batch_size, n_frames);

    if sum(T(:)) == 0
        break;
    end

    t_noise_limit = zeros(1, batch_size, 'single');
    for b = 1:batch_size
        t_noise_limit(b) = sqrt(2) * 3 * gather(estimate_noise_std(T(b, :)));
    end
    t_noise_limit = maybe_gpu(use_gpu, t_noise_limit);
    valid_t = T' > t_noise_limit;

    M_time_pack = permute(M_pack, [2, 1, 3]);
    S_pack_next = solve_single_source_adaptive_batched(...
        T', M_time_pack, valid_t, n_iter_in, zeros(1, batch_size, 'single'), noise_std, kappa_s);

    S = maybe_gpu(use_gpu, zeros(n_pixels, batch_size, 'single'));
    for b = 1:batch_size
        idx = support_idx_cache{b};
        if isempty(idx)
            continue;
        end
        s_sub = S_pack_next(1:numel(idx), b);
        S(idx, b) = s_sub;
    end
    S = S ./ max(max(S, [], 1), 1e-10);

    if any(is_component_done)
        S(:, is_component_done) = S_before(:, is_component_done);
        T(is_component_done, :) = T_before(is_component_done, :);
    end

    support_size = cellfun(@numel, support_idx_cache);
    is_component_done = is_component_done | (support_size > size_limit);

    S_change(:, i) = gather(sum(abs(S - S_before), 1) ./ max(sum(S + S_before, 1), 1e-10) * 2)';
    T_change(:, i) = gather(sum(abs(T - T_before), 2) ./ max(sum(T + T_before, 2), 1e-10) * 2);

    is_component_done = is_component_done | gather(S_change(:, i)' < TOL & T_change(:, i)' < TOL);

    if all(is_component_done)
        break;
    end
end

% Correlation image and trace, using each component's final positive support.
for b = 1:batch_size
    s = S(:, b);
    t_row = T(b, :);
    t = t_row';
    is_pos_s = s > 0;
    if ~any(gather(is_pos_s)) || sum(t) == 0
        continue;
    end

    M_sub = maybe_gpu(use_gpu, Mt(:, is_pos_s))';
    sn = s / sum(s.^2);
    T_corr(b, :) = max(0, sn(is_pos_s)' * M_sub);

    idx_support = support_idx_cache{b};
    if isempty(idx_support)
        continue;
    end
    M_support = maybe_gpu(use_gpu, Mt(:, idx_support))';
    t_noise_limit = sqrt(2) * 3 * gather(estimate_noise_std(t_row));
    idx_valid_t = find(t_row > t_noise_limit);
    if numel(idx_valid_t) > 1
        t_sub = t(idx_valid_t);
        M_subsub = M_support(:, idx_valid_t);
        s_sub = M_subsub * (t_sub / (t_sub' * t_sub));
        s_corr = maybe_gpu(use_gpu, zeros(n_pixels, 1, 'single'));
        s_corr(idx_support) = max(0, s_sub);
        S_corr(:, b) = s_corr;
    end
end
end

function [support_idx, M_pack, S_pack, support_mask] = pack_support_movies(Mt, S, h, w, use_gpu, extend_radius_low, extend_radius_high, support_idx)
batch_size = size(S, 2);
support_lengths = zeros(1, batch_size);

for b = 1:batch_size
    s_2d = reshape(S(:, b), h, w);
    is_s_minimal = get_support_gpu_compatible(s_2d, use_gpu, extend_radius_low);
    if ~any(gather(is_s_minimal(:)))
        support_idx{b} = [];
    else
        old_idx = support_idx{b};
        is_s_minimal_cpu = gather(is_s_minimal(:));
        refresh_support = isempty(old_idx) || any(~ismember(find(is_s_minimal_cpu), old_idx));
        if refresh_support
            is_s = get_support_gpu_compatible(s_2d, use_gpu, extend_radius_high);
            support_idx{b} = gather(find(is_s));
        end
    end
    support_lengths(b) = numel(support_idx{b});
end

max_support = max(support_lengths);
if max_support == 0
    M_pack = [];
    S_pack = [];
    support_mask = [];
    return;
end

n_frames = size(Mt, 1);
M_pack = maybe_gpu(use_gpu, zeros(max_support, n_frames, batch_size, 'single'));
S_pack = maybe_gpu(use_gpu, zeros(max_support, batch_size, 'single'));
support_mask = false(max_support, batch_size);

for b = 1:batch_size
    idx = support_idx{b};
    n_idx = numel(idx);
    if n_idx == 0
        continue;
    end
    M_pack(1:n_idx, :, b) = maybe_gpu(use_gpu, Mt(:, idx))';
    S_pack(1:n_idx, b) = S(idx, b);
    support_mask(1:n_idx, b) = true;
end
support_mask = maybe_gpu(use_gpu, support_mask);
end

function is_in_support = get_support_gpu_compatible(s_2d, use_gpu, extend_radius)
if use_gpu
    filt = double(fspecial('disk', extend_radius) > 0);
    is_in_support = imfilter(s_2d > 0.1 * max(s_2d(:)), filt, 'replicate');
    is_in_support = is_in_support(:) > 0;
else
    se = strel('disk', extend_radius);
    is_in_support = imdilate(s_2d > 0.1 * max(s_2d(:)), se);
    is_in_support = is_in_support(:);
end
end
