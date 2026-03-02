function roi_threshold_gui()
% Interactive ROI threshold GUI for EXTRACT outputs.
% Run:
%   roi_threshold_gui

result_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-production', 'test_1pMCRI_output_full.mat');

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repo_root, 'EXTRACT')));
addpath(genpath(fullfile(repo_root, 'External algorithms')));

if ~isfile(result_path)
    error('Result file not found: %s', result_path);
end

L = load(result_path, 'output');
output = L.output;
if ~isfield(output, 'info') || ~isfield(output.info, 'cellcheck')
    error('output.info.cellcheck is missing. Run extractor and save full output first.');
end
if isempty(output.info.cellcheck.metrics)
    error('output.info.cellcheck.metrics is empty.');
end

if isa(output.spatial_weights, 'ndSparse')
    ims = full(output.spatial_weights);
else
    ims = output.spatial_weights;
end
summary_image = output.info.summary_image;
metrics = output.info.cellcheck.metrics;
[fmap, ~] = get_quality_metric_map();
avg_cell_area = pi * output.config.avg_cell_radius^2;

th = output.config.thresholds;

fig = uifigure('Name', 'EXTRACT ROI Threshold GUI', 'Position', [60 60 1450 860]);
gl = uigridlayout(fig, [1 2]);
gl.ColumnWidth = {380, '1x'};

panel = uipanel(gl, 'Title', 'Threshold Controls');
panel.Layout.Row = 1;
panel.Layout.Column = 1;
pgl = uigridlayout(panel, [14 2]);
pgl.RowHeight = repmat({28}, 1, 14);
pgl.ColumnWidth = {210, '1x'};

ef_Tmin = make_numeric_field(pgl, 1, 'T_min_snr', th.T_min_snr);
ef_spcor = make_numeric_field(pgl, 2, 'spatial_corrupt_thresh', th.spatial_corrupt_thresh);
ef_ecc = make_numeric_field(pgl, 3, 'eccent_thresh', th.eccent_thresh);
ef_sz_lo = make_numeric_field(pgl, 4, 'size_lower_limit', th.size_lower_limit);
ef_sz_hi = make_numeric_field(pgl, 5, 'size_upper_limit', th.size_upper_limit);
ef_tdup = make_numeric_field(pgl, 6, 'T_dup_corr_thresh', th.T_dup_corr_thresh);
ef_sdup = make_numeric_field(pgl, 7, 'S_dup_corr_thresh', th.S_dup_corr_thresh);
ef_sti = make_numeric_field(pgl, 8, 'low_ST_index_thresh', th.low_ST_index_thresh);
ef_stc = make_numeric_field(pgl, 9, 'low_ST_corr_thresh', th.low_ST_corr_thresh);

chk_show_rej = uicheckbox(pgl, 'Text', 'Show rejected ROIs', 'Value', true);
chk_show_rej.Layout.Row = 10;
chk_show_rej.Layout.Column = [1 2];
btn_update = uibutton(pgl, 'Text', 'Update', 'ButtonPushedFcn', @(~,~) refresh_plot());
btn_update.Layout.Row = 11;
btn_update.Layout.Column = [1 2];
lbl_counts = uilabel(pgl, 'Text', 'Accepted: -, Rejected: -', 'FontWeight', 'bold');
lbl_counts.Layout.Row = 12;
lbl_counts.Layout.Column = [1 2];
lbl_note = uilabel(pgl, 'Text', 'Note: type values then press Update.');
lbl_note.Layout.Row = 13;
lbl_note.Layout.Column = [1 2];
lbl_path = uilabel(pgl, 'Text', sprintf('Loaded: %s', result_path), 'Interpreter', 'none');
lbl_path.Layout.Row = 14;
lbl_path.Layout.Column = [1 2];

ax = uiaxes(gl);
ax.Layout.Row = 1;
ax.Layout.Column = 2;

chk_show_rej.ValueChangedFcn = @(~,~) refresh_plot();
ef_Tmin.ValueChangedFcn = @(~,~) refresh_plot();
ef_spcor.ValueChangedFcn = @(~,~) refresh_plot();
ef_ecc.ValueChangedFcn = @(~,~) refresh_plot();
ef_sz_lo.ValueChangedFcn = @(~,~) refresh_plot();
ef_sz_hi.ValueChangedFcn = @(~,~) refresh_plot();
ef_tdup.ValueChangedFcn = @(~,~) refresh_plot();
ef_sdup.ValueChangedFcn = @(~,~) refresh_plot();
ef_sti.ValueChangedFcn = @(~,~) refresh_plot();
ef_stc.ValueChangedFcn = @(~,~) refresh_plot();

refresh_plot();

    function refresh_plot()
        cur = get_current_thresholds();
        [is_bad, dbg] = classify_from_metrics(metrics, fmap, cur, avg_cell_area);
        is_good = ~is_bad;

        cla(ax);
        imagesc(ax, summary_image);
        axis(ax, 'image');
        axis(ax, 'off');
        colormap(ax, flipud(brewermap(64, 'rdgy')));
        hold(ax, 'on');
        axes(ax); %#ok<LAXES> ensure plot_cells_overlay draws on this UI axes
        if any(is_good)
            plot_cells_overlay(ims(:, :, is_good), [0, 0.7, 0.1], 1.0, 0.2);
        end
        if chk_show_rej.Value && any(is_bad)
            plot_cells_overlay(ims(:, :, is_bad), [0.85, 0.1, 0.1], 0.8, 0.2);
        end
        hold(ax, 'off');
        title(ax, sprintf(['Accepted %d | Rejected %d | Total %d\n' ...
            'eccent=%.3g, T_min_snr=%.3g, size=[%.3g, %.3g]'], ...
            sum(is_good), sum(is_bad), numel(is_bad), ...
            cur.eccent_thresh, cur.T_min_snr, cur.size_lower_limit, cur.size_upper_limit));
        lbl_counts.Text = sprintf('Accepted: %d, Rejected: %d', sum(is_good), sum(is_bad));

        fprintf('[ROI GUI] tiny=%d huge=%d lowSNR=%d badShape=%d eccent=%d Tdup=%d Sdup=%d ST=%d\n', ...
            sum(dbg.is_S_tiny), sum(dbg.is_S_huge), sum(dbg.is_T_zeroed), ...
            sum(dbg.is_S_poor_looking), sum(dbg.is_S_poor_eccent), ...
            sum(dbg.is_T_duplicate), sum(dbg.is_S_duplicate), sum(dbg.is_ST_spurious));
    end

    function cur = get_current_thresholds()
        cur = struct();
        cur.T_min_snr = ef_Tmin.Value;
        cur.spatial_corrupt_thresh = ef_spcor.Value;
        cur.eccent_thresh = ef_ecc.Value;
        cur.size_lower_limit = ef_sz_lo.Value;
        cur.size_upper_limit = ef_sz_hi.Value;
        cur.T_dup_corr_thresh = ef_tdup.Value;
        cur.S_dup_corr_thresh = ef_sdup.Value;
        cur.low_ST_index_thresh = ef_sti.Value;
        cur.low_ST_corr_thresh = ef_stc.Value;
    end
end

function ef = make_numeric_field(parent_layout, row_idx, label_text, initv)
lbl = uilabel(parent_layout, 'Text', label_text);
lbl.Layout.Row = row_idx;
lbl.Layout.Column = 1;
ef = uieditfield(parent_layout, 'numeric', 'Value', initv, ...
    'RoundFractionalValues', false);
ef.Layout.Row = row_idx;
ef.Layout.Column = 2;
end

function [is_bad, out] = classify_from_metrics(metrics, fmap, th, avg_cell_area)
% Post-hoc reproduction of key checks from remove_redundant().
m_T_maxval = metrics(fmap('T_maxval'), :);
m_S_corr = metrics(fmap('S_corruption'), :);
m_S_ecc = metrics(fmap('S_eccent'), :);
m_S_area1 = metrics(fmap('S_area_1'), :);
m_S_smooth_area1 = metrics(fmap('S_smooth_area_1'), :);
m_S_area2 = metrics(fmap('S_area_2'), :);
m_ST2_4 = metrics(fmap('ST2_index_4'), :);
m_ST_corr3 = metrics(fmap('ST_corr_3'), :);
m_T_dup = metrics(fmap('T_dup_val'), :);
m_S_max_corr = metrics(fmap('S_max_corr'), :);

size_lower_px = th.size_lower_limit * avg_cell_area;
size_upper_px = th.size_upper_limit * avg_cell_area;

is_T_zeroed = (m_T_maxval <= th.T_min_snr);
is_S_tiny = (max(m_S_area1, m_S_smooth_area1) <= size_lower_px) | (m_S_smooth_area1 == 0);
is_S_huge = (m_S_area2 ./ max(1, -2 + m_S_ecc)) >= size_upper_px;
is_S_poor_looking = (m_S_corr >= th.spatial_corrupt_thresh);
is_S_poor_eccent = (m_S_ecc >= th.eccent_thresh);
is_T_duplicate = (m_T_dup >= th.T_dup_corr_thresh);      % approximation
is_S_duplicate = (m_S_max_corr >= th.S_dup_corr_thresh); % approximation
is_ST_spurious = (m_ST2_4 <= th.low_ST_index_thresh) | (m_ST_corr3 < th.low_ST_corr_thresh);

is_bad = is_T_zeroed | is_S_tiny | is_S_huge | ...
         is_S_poor_looking | is_S_poor_eccent | ...
         is_T_duplicate | is_S_duplicate | is_ST_spurious;

out = struct();
out.is_T_zeroed = is_T_zeroed;
out.is_S_tiny = is_S_tiny;
out.is_S_huge = is_S_huge;
out.is_S_poor_looking = is_S_poor_looking;
out.is_S_poor_eccent = is_S_poor_eccent;
out.is_T_duplicate = is_T_duplicate;
out.is_S_duplicate = is_S_duplicate;
out.is_ST_spurious = is_ST_spurious;
end
