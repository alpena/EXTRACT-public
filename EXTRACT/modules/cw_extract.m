function [S, T, summary] = cw_extract(M, config)
% Extracts cells one by one using one-sided Huber estimator

[h, w, n] = size(M);
use_gpu = config.use_gpu;

% Defaults
init_radius = 5;

avg_radius = config.avg_cell_radius;
max_spread = 2;
% imopen_radius = ceil(init_radius/config.init_cellsize_tol);
elim_size_thresh = config.cellfind_numpix_threshold;
avg_cell_area = pi * avg_radius ^ 2;
min_num_pixels = avg_cell_area *config.thresholds.size_lower_limit;
max_num_pixels = avg_cell_area * config.thresholds.size_upper_limit;
avg_yield_threshold = config.avg_yield_threshold;
yield_averaging_window = round(1/avg_yield_threshold);

show_each_cell = 0;


%%%%
%%% TEMP CODE!!!
%%%%
% M = reshape(M, h * w, n);
% M = bsxfun(@minus, M, mean(M, 1));
% M = bsxfun(@minus, M, mean(M, 2));
% M = reshape(M, h, w, n);


ind = reshape((max(M,[],3) > 1e-6),1,[]);

% Reduce noise in movie with a spatial filter
switch config.cellfind_filter_type
    case 'butter'
        M = spatial_bandpass(M, avg_radius, config.cellfind_spatial_highpass_cutoff, ...
            config.spatial_lowpass_cutoff, use_gpu, config.smoothing_ratio_x2y);
    case 'gauss'
        M = spatial_gauss_lowpass(M, avg_radius, use_gpu);
    case 'wiener'
        M = imwiener(M, use_gpu);
    case 'movavg'
        moving_rad_spatial=max(floor(config.moving_radius_spatial),2);
        moving_rad_temporal=max(floor(config.moving_radius_temporal),1);
        X=ones(moving_rad_spatial,moving_rad_spatial,moving_rad_temporal)/(moving_rad_temporal*moving_rad_spatial^2); 
        M=convn(M,X,'same');
    case 'median'
        M= medfilt3(M);
    case 'gaus'
        M= imgaussfilt3(M,config.filter_sigma);
    case 'none'
    otherwise
        error('Filter type not supported.');
end




% Flatten for subsequent processing
M = reshape(M, h * w, n);

% More efficient to use M transposed (cheaper to index in space this way)
Mt = M';
noise_per_pixel = estimate_noise_std(Mt, 1, use_gpu);
% Apply movie mask to noise if it exists
if ~isempty(config.movie_mask)
    noise_per_pixel = noise_per_pixel(config.movie_mask(:));
    ind = ind(config.movie_mask(:));
end
try
    noise_std = median(noise_per_pixel(ind));
catch
    noise_std = 1e-6;
end


% Get a stack of 2 ims (max im + im of max idx) -- used to get seed pixels
summary_stack = get_summary_stack(Mt, [h, w], max_spread, []);
summary.summary_im = reshape(summary_stack(:, 1), h, w);

if config.visualize_cellfinding
    is_bad=1;
    
    str = sprintf('\t \t \t Using cell finding visualization tool...\n');
    dispfun(str, config.verbose ==2);
    
    %max_im = max(M,[],3);
    max_im = summary.summary_im;


    trace_snr_all = [];
    mov_snr_all = [];
    
    subplot(121)
    if config.visualize_cellfinding_full_range
        imshow(max_im,[ ])
    else
        clims = quantile(max_im(:), [config.visualize_cellfinding_min config.visualize_cellfinding_max]);
        imshow(max_im,clims)
    end
    drawnow;
    subplot(222)
    histogram(trace_snr_all)
    xlabel('Trace snr')
    ylabel('Number of cells')
    drawnow;
    subplot(224)
    histogram(mov_snr_all)
    xlabel('Cellfind min snr')
    ylabel('Number of cells')
    drawnow;
    
end

% Stop finding cells if signal maximum is below a certain value
% Bias func is the underestimating bias of mis-specified robust estimation under no
% non-negative contamination (actually upper bound on it)
bias_func = @(k)  2 * (normpdf(k) + k.*normcdf(k) - k)./normcdf(k);
noise_limit = noise_std * config.cellfind_min_snr + ...
    noise_std * bias_func(config.cellfind_kappa_std_ratio);

% Second threshold is based on how much dimmer is the current pixel 
% compared to most bright region in the FOV
im_summary = summary_stack(:, 1);
dim_limit = quantile(im_summary(:), 0.999) / ...
    config.high2low_brightness_ratio;

min_magnitude = max(noise_limit, dim_limit);

dispfun(sprintf(...
    '\t \t \t \t noise std: %.4f \n\t \t \t \t minimum magnitude: %.4f \n',...
    noise_std, min_magnitude), config.verbose==2);

max_steps = config.cellfind_max_steps;

% Set an absolute minimum for noise threshold based on theoretical max of
% gaussians
mu = norminv(1 - 1/n) * (1 - 0.577) + 0.577 * norminv(1 - 1/n / exp(1));
% Mu is the mean of gumbel, and gumbel is very concentrated
abs_noise_threshold = 0;%noise_std * mu * 1.2;
% Initialize variables
S = zeros(h * w, max_steps, 'single');
T = zeros(max_steps, n, 'single');
% quality check related arrays
metrics = zeros(max_steps, 4, 'single');
is_attr_bad = false(max_steps, 5);

S_trash = S;
T_trash = T;
S_change = [];
T_change = [];

is_good = false(1, max_steps);
init_stop_reason = 'max_iter';

% Create image template
s_proto = fspecial('gaussian', 1 + 2 * [init_radius, init_radius], ...
    init_radius / 2.5);
% Scale so that maximum is at 1
s_proto = s_proto / max(s_proto(:));
maxes = [];
vals_max = [];

kappa_s = config.cellfind_kappa_std_ratio;
% Adaptive kappa for t if asked
if config.cellfind_adaptive_kappa
    kappa_t = @(d, k, v, alpha) kappa_of_epsilon(eps_func(d, k, v, alpha));
else
    kappa_t = kappa_s;
end

cellfind_batch_size = max(1, round(config.cellfind_batch_size));
cellfind_batch_distance_px = max(1, config.cellfind_batch_distance_factor * avg_radius);
cellfind_batch_trace_corr_thresh = config.cellfind_batch_trace_corr_thresh;
cellfind_batch_max_candidates = max(cellfind_batch_size, round(config.cellfind_batch_max_candidates));
cellfind_batch_fit_mode = lower(strtrim(config.cellfind_batch_fit_mode));
cellfind_batch_refine_iter = max(0, round(config.cellfind_batch_refine_iter));
cellfind_batch_enabled = cellfind_batch_size > 1 && ...
    config.cellfind_check_min_magnitude && ~config.visualize_cellfinding;

num_good_cells = 0;
if cellfind_batch_enabled
    dispfun(sprintf(['\t\t\t Using conflict-aware cell finding mini-batches: ', ...
        'batch_size=%d, distance=%.2f px, trace_corr<%.2f \n'], ...
        cellfind_batch_size, cellfind_batch_distance_px, ...
        cellfind_batch_trace_corr_thresh), config.verbose == 2);

    i = 0;
    debug_batch_seed_indices = {};
    debug_batch_seed_values = {};
    debug_batch_is_good = {};
    debug_batch_step_range = zeros(0, 2);
    while i < max_steps
        mod_im_summary = modify_summary_image(summary_stack(:, 1), h, w, ...
            min_magnitude, elim_size_thresh);
        [batch_inds, batch_vals] = select_seed_batch(mod_im_summary, h, w, ...
            min(cellfind_batch_size, max_steps - i), cellfind_batch_distance_px, ...
            cellfind_batch_trace_corr_thresh, cellfind_batch_max_candidates, Mt);

        if isempty(batch_inds)
            init_stop_reason = 'min_magnitude';
            break;
        end

        batch_idx_s_for_summary = [];
        batch_bad_pix_idx = [];
        batch_s_corr = {};
        batch_t_corr = {};
        batch_idx_s = {};
        batch_idx_t = {};
        batch_good_flags = false(1, numel(batch_inds));
        batch_start_step = i + 1;

        s_init_stack = zeros(h, w, numel(batch_inds), 'single');
        for idx_batch = 1:numel(batch_inds)
            ind_max = batch_inds(idx_batch);
            [y_max, x_max] = ind2sub([h, w], ind_max);
            if config.init_with_gaussian
                s_init = generate_images_from_centroids(h, w, s_proto, ...
                    [y_max; x_max], init_radius);
            else
                s_init = generate_init_image(Mt, h, w, ind_max, 0.5, floor(avg_radius*1.5));
            end
            s_init_stack(:, :, idx_batch) = single(reshape(s_init, h, w));
        end

        use_batched_fit = strcmp(cellfind_batch_fit_mode, 'batched') && ...
            isnumeric(kappa_t) && isnumeric(kappa_s);
        if use_batched_fit
            try
                [s_batch, t_batch, t_corr_batch, s_corr_batch, s_change_batch, t_change_batch] = ...
                    alt_opt_batch_independent(Mt, maybe_gpu(use_gpu, s_init_stack), noise_std, ...
                    max_num_pixels, use_gpu, kappa_t, kappa_s, config.cellfind_max_iter);
            catch ME
                warning('Batched cell finding fit failed; falling back to single fits. %s', ME.message);
                use_batched_fit = false;
            end
        end

        for idx_batch = 1:numel(batch_inds)
            if i >= max_steps
                break;
            end
            i = i + 1;
            ind_max = batch_inds(idx_batch);
            val_max = batch_vals(idx_batch);

            [y_max, x_max] = ind2sub([h, w], ind_max);
            maxes = [maxes, gather([y_max; x_max])]; %#ok<*AGROW>

            if use_batched_fit
                s = s_batch(:, idx_batch);
                t = t_batch(idx_batch, :);
                t_corr = t_corr_batch(idx_batch, :);
                s_corr = s_corr_batch(:, idx_batch);
                s_change = s_change_batch(idx_batch, :);
                t_change = t_change_batch(idx_batch, :);
                if cellfind_batch_refine_iter > 0
                    s_2d_init = reshape(s, h, w);
                    try
                        [s, t, t_corr, s_corr, s_change_refine, t_change_refine] = ...
                            alt_opt_single(Mt, s_2d_init, noise_std, max_num_pixels, use_gpu, ...
                            kappa_t, kappa_s, cellfind_batch_refine_iter);
                    catch
                        s_2d_init = gather(s_2d_init);
                        [s, t, t_corr, s_corr, s_change_refine, t_change_refine] = ...
                            alt_opt_single(Mt, s_2d_init, noise_std, max_num_pixels, 0, ...
                            kappa_t, kappa_s, cellfind_batch_refine_iter);
                    end
                    s_change = [s_change, s_change_refine]; %#ok<AGROW>
                    t_change = [t_change, t_change_refine]; %#ok<AGROW>
                end
            else
                s_2d_init = maybe_gpu(use_gpu, s_init_stack(:, :, idx_batch));
                try
                    [s, t, t_corr, s_corr, s_change, t_change] = ...
                        alt_opt_single(Mt, s_2d_init, noise_std, max_num_pixels, use_gpu, kappa_t, kappa_s, config.cellfind_max_iter);
                catch
                    s_2d_init = gather(s_2d_init);
                    [s, t, t_corr, s_corr, s_change, t_change] = ...
                        alt_opt_single(Mt, s_2d_init, noise_std, max_num_pixels, 0, kappa_t, kappa_s, config.cellfind_max_iter);
                end
            end

            S_change = [S_change; s_change];
            T_change = [T_change; t_change];

            cell_area = gather(get_cell_areas(s));
            metrics(i, 1) = cell_area;
            is_attr_bad(i, 1) = cell_area < min_num_pixels;
            is_attr_bad(i, 2) = cell_area > max_num_pixels;
            max_t = gather(max(t));
            metrics(i, 2) = max_t;
            is_attr_bad(i, 3) = max_t < max(abs_noise_threshold, min_magnitude);
            trace_snr = max(medfilt1(gather(t))) / estimate_noise_std(t) / sqrt(2);
            metrics(i, 3) = trace_snr;
            is_attr_bad(i, 4) = trace_snr < config.thresholds.T_min_snr;
            is_this_duplicate = is_duplicate(t, T, s, S);
            metrics(i, 4) = is_this_duplicate;
            is_attr_bad(i, 5) = is_this_duplicate;
            is_bad = any(is_attr_bad(i, :));

            idx_s = gather(find(s_corr > 0));
            idx_t = gather(find(t_corr > 0));
            batch_idx_s_for_summary = [batch_idx_s_for_summary; idx_s(:)];
            batch_s_corr{end + 1} = gather(s_corr); %#ok<AGROW>
            batch_t_corr{end + 1} = gather(t_corr); %#ok<AGROW>
            batch_idx_s{end + 1} = idx_s; %#ok<AGROW>
            batch_idx_t{end + 1} = idx_t; %#ok<AGROW>

            if is_bad
                batch_bad_pix_idx = [batch_bad_pix_idx; get_seed_suppression_pixels(ind_max, h, w, max_spread)]; %#ok<AGROW>
                T_trash(i, :) = gather(t);
                S_trash(:, i) = gather(s);
            else
                batch_good_flags(idx_batch) = true;
                num_good_cells = num_good_cells + 1;
                is_good(i) = true;
                vals_max = [vals_max, val_max];
                T(i, :) = gather(t);
                S(:, i) = gather(s);
            end
        end

        for idx_component = 1:numel(batch_s_corr)
            idx_s = batch_idx_s{idx_component};
            idx_t = batch_idx_t{idx_component};
            if isempty(idx_s) || isempty(idx_t)
                continue;
            end
            s_corr_this = batch_s_corr{idx_component};
            t_corr_this = batch_t_corr{idx_component};
            try
                Mt(idx_t, idx_s) = Mt(idx_t, idx_s) - ...
                    gather(1.0 * t_corr_this(idx_t)' * s_corr_this(idx_s)');
            catch
                Mt = Mt - max(gather(1.0 * t_corr_this' * s_corr_this'), 0);
            end
        end

        batch_idx_s_for_summary = unique(batch_idx_s_for_summary);
        if ~isempty(batch_idx_s_for_summary)
            summary_stack = get_summary_stack(...
                Mt, [h, w], max_spread, summary_stack, batch_idx_s_for_summary);
        end

        batch_bad_pix_idx = unique(batch_bad_pix_idx);
        if ~isempty(batch_bad_pix_idx)
            Mt(:, batch_bad_pix_idx) = Mt(:, batch_bad_pix_idx) * 0;
            summary_stack(batch_bad_pix_idx, 1) = summary_stack(batch_bad_pix_idx, 1) * 0;
        end

        debug_batch_seed_indices{end + 1} = gather(batch_inds(:)'); %#ok<AGROW>
        debug_batch_seed_values{end + 1} = gather(batch_vals(:)'); %#ok<AGROW>
        debug_batch_is_good{end + 1} = batch_good_flags; %#ok<AGROW>
        debug_batch_step_range(end + 1, :) = [batch_start_step, i]; %#ok<AGROW>

        n = yield_averaging_window;
        avg_yield = mean(is_good(max(1, i-n+1):i));
        if i > 2 * n && avg_yield <= avg_yield_threshold
            init_stop_reason = 'yield';
            break;
        end

        if mod(i, 100) < numel(batch_inds)
            dispfun(sprintf('\t\t\t Step #%d, found %d cells... \n', ...
                i, num_good_cells), config.verbose == 2);
        end
    end
else
for i = 1:max_steps
    if (config.visualize_cellfinding && i>1 && ~is_bad)
        
            subplot(121)
            plot_cells_overlay(reshape(gather(s),h,w),[0,1,0],[],0.2)
            drawnow;
        
    end
    % Select seed pixel for next init cell
    mod_im_summary = modify_summary_image(summary_stack(:, 1), h, w, ...
        min_magnitude, elim_size_thresh);
%     mod_im_summary = mod_im_summary .*Cn;
    [val_max, ind_max] = max(mod_im_summary(:));
    % Check min magnitude condition
    if config.cellfind_check_min_magnitude
        if val_max < min_magnitude %max(abs_noise_threshold, min_magnitude)
            init_stop_reason = 'min_magnitude';
            break;
        end
    end

    % Initialize image
    [y_max, x_max] = ind2sub([h, w], ind_max);
    maxes = [maxes, gather([y_max; x_max])]; %#ok<*AGROW>
    if config.init_with_gaussian
        s_init = generate_images_from_centroids(h, w, s_proto, ...
                [y_max; x_max], init_radius);
    else
        s_init = generate_init_image(Mt, h, w, ind_max, 0.5, floor(avg_radius*1.5));
    end
    s_2d_init = reshape(s_init, h, w);
    s_2d_init = single(s_2d_init);
    s_2d_init = maybe_gpu(use_gpu, s_2d_init);

    


    % Robust cell finding
    try
    [s, t, t_corr, s_corr, s_change, t_change] = ...
        alt_opt_single(Mt, s_2d_init, noise_std, max_num_pixels, use_gpu, kappa_t, kappa_s,config.cellfind_max_iter);
    catch
        s_2d_init = gather(s_2d_init);
    [s, t, t_corr, s_corr, s_change, t_change] = ...
        alt_opt_single(Mt, s_2d_init, noise_std, max_num_pixels, 0, kappa_t, kappa_s,config.cellfind_max_iter);
    end

    S_change = [S_change; s_change];
    T_change = [T_change; t_change];

    % Check attributes
    % check image isn't too small
    cell_area = gather(get_cell_areas(s));
    metrics(i, 1) = cell_area;
    is_attr_bad(i, 1) = cell_area < min_num_pixels;
    % Check image isn't too big
    is_attr_bad(i, 2) = cell_area > max_num_pixels;
    % Check trace magnitude
    max_t = gather(max(t));
    metrics(i, 2) = max_t;
    is_attr_bad(i, 3) = max_t < max(abs_noise_threshold, min_magnitude);
    trace_snr = max(medfilt1(gather(t))) / estimate_noise_std(t) / sqrt(2);
    % Check trace snr
    metrics(i, 3) = trace_snr;
    is_attr_bad(i, 4) = trace_snr < config.thresholds.T_min_snr;
    is_this_duplicate = is_duplicate(t, T, s, S);
    metrics(i, 4) = is_this_duplicate;
    % Check trace isn't duplicate
    is_attr_bad(i, 5) = is_this_duplicate;
    % Check trace snr
    is_bad = any(is_attr_bad(i, :));
%     fprintf('%d, %d, %d, %d \n', is_good_spatial1, is_good_spatial2, is_good_temporal1, is_good_temporal2);
    if show_each_cell
        subplot(321);
        imagesc(reshape(s, h, w));colormap jet; axis image;
        title(sprintf('Step: %d, is_good: %d', i, ~is_bad));
        subplot(322);
        imagesc(reshape(summary_stack(:, 1), h, w));axis image; colormap jet;colorbar;
        subplot(323);
        mod_im_summary = modify_summary_image(summary_stack(:, 1), h, w, ...
            min_magnitude, elim_size_thresh);
        imagesc(mod_im_summary);axis image; colormap jet;colorbar;
        subplot(324);
%         mmax_im = sqrt(reshape(sum(Mt.^2, 1)'/size(M, 2), h, w));% clim:[0, (noise_std*2)]
        mmax_im = s_2d_init;
        imagesc(mmax_im);axis image; colormap jet;colorbar;
        subplot(3, 2, [5, 6]);
        plot(t);
        pause;
    end

    

    % Subtract s * t
    idx_s = find(s_corr > 0);
    idx_t = find(t_corr > 0);
    try
        Mt(idx_t, idx_s) = Mt(idx_t, idx_s) - gather(1.0 * t_corr(idx_t)' * s_corr(idx_s)');
    catch
        Mt = Mt - max(gather(1.0 * t_corr' * s_corr'),0);
    end

    summary_stack = get_summary_stack(...
        Mt, [h, w], max_spread, summary_stack, idx_s);
    
    if is_bad
        pix_idx_lookup = reshape(1:h*w, h, w);
        [y, x] = ind2sub([h, w], ind_max);
        y_range = max(1, y-max_spread):min(h, y+max_spread);
        x_range = max(1, x-max_spread):min(w, x+max_spread);
        pix_idx = pix_idx_lookup(y_range, x_range);
        pix_idx = pix_idx(:);
        Mt(:, pix_idx) = Mt(:, pix_idx) * 0;
        summary_stack(pix_idx, 1) = summary_stack(pix_idx, 1) * 0;
        T_trash(i, :) = gather(t);
        S_trash(:, i) = gather(s);

        if config.visualize_cellfinding 
            if config.visualize_cellfinding_show_bad_cells

                
                subplot(121)
                plot_cells_overlay(reshape(gather(s),h,w),[1,0,0],[],0.2)
                title(['Cell finding in process. ' num2str(i) ' iterations ' num2str(num_good_cells) ' found.'])
                drawnow;
            end
    
        end

    else
        num_good_cells = num_good_cells + 1;
        is_good(i) = true;
        vals_max = [vals_max, val_max];
        T(i, :) = gather(t);
        S(:, i) = gather(s);
        if config.visualize_cellfinding

            trace_snr_all = [trace_snr_all, gather(trace_snr)];
            mov_snr_all = [mov_snr_all, gather(max_t/noise_std - bias_func(config.cellfind_kappa_std_ratio))];

            subplot(121)
            plot_cells_overlay(reshape(gather(s),h,w),[1,0,0],[],0.2)
            title(['Cell finding in process. ' num2str(i) ' iterations ' num2str(num_good_cells) ' found.'])
            drawnow;
            subplot(222)
            histogram(trace_snr_all,ceil(i/10))
            xlabel('Trace snr')
            ylabel('Number of cells')
            drawnow;
            subplot(224)
            histogram(mov_snr_all,ceil(i/10))
            xlabel('Cellfind min snr')
            ylabel('Number of cells')
            drawnow;
    
        end
    end
    
    % Stopping criterion based on the running yield of cells
    n = yield_averaging_window;
    avg_yield = mean(is_good(max(1, i-n+1):i));
    if i > 2 * n && avg_yield <= avg_yield_threshold
        init_stop_reason = 'yield';
        if (config.visualize_cellfinding && i>1 && ~is_bad)
        
            subplot(121)
            plot_cells_overlay(reshape(gather(s),h,w),[0,1,0],[],0.2)
            drawnow;
        
        end
        break;
    end
    
    if mod(i, 100)==0
        dispfun(sprintf('\t\t\t Step #%d, found %d cells... \n', ...
            i, num_good_cells), config.verbose == 2);
    end
end
end

if config.visualize_cellfinding
    subplot(121)
    title(['Cell finding completed. ' num2str(i) ' iterations ' num2str(num_good_cells) ' found.'])
    drawnow;
end

% Organize S & T matrices
S = S(:, is_good);
T = T(is_good, :);
S_trash = S_trash(:, ~is_good);
T_trash = T_trash(~is_good, :);

metrics = metrics(1:i, :);
is_attr_bad = is_attr_bad(1:i, :);
summary.metrics = metrics;
summary.is_attr_bad = is_attr_bad;
is_good = is_good(1:i);
summary.is_good = is_good;
summary.max_locations = maxes;
summary.max_values = vals_max;
summary.S_trash = S_trash;
summary.T_trash = T_trash;
summary.init_stop_reason = init_stop_reason;
summary.S_change = S_change;
summary.T_change = T_change;
summary.noise_per_pixel = noise_per_pixel;
if exist('debug_batch_seed_indices', 'var')
    summary.batch_seed_indices = debug_batch_seed_indices;
    summary.batch_seed_values = debug_batch_seed_values;
    summary.batch_is_good = debug_batch_is_good;
    summary.batch_step_range = debug_batch_step_range;
end

dispfun(sprintf(...
    '\t \t \t %d cells found after a total of %d steps... \n', ...
    size(S, 2), i), config.verbose ==2);

%----
% Helper functions
%----
    

    function m2 = modify_summary_image(m, h, w, t, elim_size_thresh)
    % Get an adjusted summary image
        m = reshape(m, h, w);
        % Eliminate small valued pixels
        m2 = m .* (m > t);
        % Opening with a disk of a cell radius eliminates small peaks
        if elim_size_thresh > 0
            mo = bwareaopen(m2>0, elim_size_thresh);
            m2 = m2 .* (mo > 0);
        end
    end
%     function m2 = modify_summary_image(m, h, w, t, imopen_radius)
%     % Get an adjusted summary image
%         m = reshape(m, h, w);
%         % Eliminate small valued pixels
%         m2 = m .* (m > t);
%         % Opening with a disk of a cell radius eliminates small peaks
%         if isfinite(imopen_radius)
%             mo = imopen(uint8(m2/max(m2(:))*255), ...
%                 strel('disk', imopen_radius));
%             m2 = m2 .* (mo > 0);
%         end
%     end
    
    function is_it = is_duplicate(t, T, s, S)
        corr_thresh = 0.7;
        idx_valid = find(s>0);
        s = s(idx_valid);
        S = S(idx_valid, :);
        prox = s' * S / sum(s.^2);
        idx_look = find(prox > 0.1);
        is_it = 0;
        if ~isempty(idx_look)
            num_frames = length(t);
            T_prox = T(idx_look, :);
            tz = zscore(t, 0) / sqrt(num_frames);
            Tz = zscore(T_prox, 0, 2) / sqrt(num_frames);
            if any(tz * Tz' > corr_thresh)
                is_it = 1;
            end
        end
    end

    function [batch_inds, batch_vals] = select_seed_batch(m, h, w, batch_size, ...
            min_dist_px, trace_corr_thresh, max_candidates, Mt)
        m = reshape(gather(m), h, w);
        is_candidate = imregionalmax(m) & (m > 0);
        candidate_inds = find(is_candidate(:));
        if isempty(candidate_inds)
            batch_inds = [];
            batch_vals = [];
            return;
        end

        candidate_vals = m(candidate_inds);
        [candidate_vals, order] = sort(candidate_vals, 'descend');
        candidate_inds = candidate_inds(order);
        n_candidates = min(numel(candidate_inds), max_candidates);
        candidate_inds = candidate_inds(1:n_candidates);
        candidate_vals = candidate_vals(1:n_candidates);

        batch_inds = zeros(1, batch_size);
        batch_vals = zeros(1, batch_size, 'like', candidate_vals);
        batch_y = zeros(1, batch_size);
        batch_x = zeros(1, batch_size);
        batch_trace_z = [];
        n_selected = 0;

        for idx_candidate = 1:n_candidates
            ind = candidate_inds(idx_candidate);
            [y, x] = ind2sub([h, w], ind);

            if n_selected > 0
                dist2 = (batch_y(1:n_selected) - y).^2 + ...
                    (batch_x(1:n_selected) - x).^2;
                if any(dist2 < min_dist_px^2)
                    continue;
                end
            end

            trace_z = normalized_seed_trace(Mt(:, ind));
            if ~isempty(batch_trace_z) && isfinite(trace_corr_thresh)
                if any(abs(trace_z' * batch_trace_z) >= trace_corr_thresh)
                    continue;
                end
            end

            n_selected = n_selected + 1;
            batch_inds(n_selected) = ind;
            batch_vals(n_selected) = candidate_vals(idx_candidate);
            batch_y(n_selected) = y;
            batch_x(n_selected) = x;
            batch_trace_z(:, n_selected) = trace_z; %#ok<AGROW>

            if n_selected >= batch_size
                break;
            end
        end

        batch_inds = batch_inds(1:n_selected);
        batch_vals = batch_vals(1:n_selected);
    end

    function trace_z = normalized_seed_trace(trace)
        trace_z = single(gather(trace(:)));
        trace_z = trace_z - mean(trace_z);
        denom = sqrt(sum(trace_z.^2));
        if denom > 0
            trace_z = trace_z / denom;
        else
            trace_z(:) = 0;
        end
    end

    function pix_idx = get_seed_suppression_pixels(ind_seed, h, w, max_spread)
        pix_idx_lookup = reshape(1:h*w, h, w);
        [y, x] = ind2sub([h, w], ind_seed);
        y_range = max(1, y-max_spread):min(h, y+max_spread);
        x_range = max(1, x-max_spread):min(w, x+max_spread);
        pix_idx = pix_idx_lookup(y_range, x_range);
        pix_idx = pix_idx(:);
    end

end
