function x = solve_single_source_adaptive_batched(a, B, data_mask, max_iter, lambda, noise_std, kappa)
%SOLVE_SINGLE_SOURCE_ADAPTIVE_BATCHED Batched independent robust single-source solve.
%
% B is n_obs x n_targets x batch_size. a is n_obs x batch_size. data_mask is
% either n_obs x batch_size or n_obs x n_targets x batch_size. The returned x is
% n_targets x batch_size.

use_gpu = isa(B, 'gpuArray');
TOL = 1e-4;
[n_obs, n_targets, batch_size] = size(B);

a = maybe_gpu(use_gpu, single(a));
data_mask = maybe_gpu(use_gpu, logical(data_mask));
lambda = maybe_gpu(use_gpu, reshape(single(lambda), 1, 1, batch_size));

if size(a, 2) ~= batch_size
    if size(a, 1) == batch_size && size(a, 2) == n_obs
        a = a';
    else
        error('Batched solver dimension mismatch: a=%s B=%s.', mat2str(size(a)), mat2str(size(B)));
    end
end
if isequal(size(data_mask), [n_obs, batch_size])
    data_mask = reshape(data_mask, n_obs, 1, batch_size);
end

a3 = reshape(a, n_obs, 1, batch_size);
valid = single(data_mask);
a3_valid = a3 .* valid;
a2_valid = (a3 .^ 2) .* valid;

den0 = sum(a2_valid, 1);
den0 = max(den0, 1e-10);
x = squeeze(sum(B .* a3_valid, 1) ./ den0);
x = reshape(x, n_targets, batch_size);
x = max(x, 0);

scaled_kappa = maybe_gpu(use_gpu, single(kappa * noise_std));
is_done = false(1, batch_size);

for i = 1:max_iter
    x_before = x;
    x3 = reshape(x, 1, n_targets, batch_size);
    res = B - a3 .* x3;
    mask = (res >= scaled_kappa) & data_mask;
    mask_not = (~mask) & data_mask;

    numerator = sum(a3 .* B .* single(mask_not), 1) + ...
        scaled_kappa .* (sum(a3 .* single(mask), 1) - lambda);
    denominator = sum((a3 .^ 2) .* single(mask_not), 1);
    denominator = max(denominator, 1e-10);
    x_new = squeeze(numerator ./ denominator);
    x_new = reshape(x_new, n_targets, batch_size);
    x_new = max(x_new, 0);
    if any(is_done)
        x_new(:, is_done) = x(:, is_done);
    end

    rel_change = gather(sum(abs(x_new - x_before), 1) ./ max(sum(abs(x_before), 1), 1e-10));
    is_done = is_done | (rel_change < TOL);
    x = x_new;

    if all(is_done)
        break;
    end
end
end
