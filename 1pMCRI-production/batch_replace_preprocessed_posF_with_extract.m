function summary = batch_replace_preprocessed_posF_with_extract(varargin)
% batch_replace_preprocessed_posF_with_extract
% Batch wrapper around replace_preprocessed_posF_with_extract.
%
% Each item must define one pipeline run_dir and one target
% preprocessed_data.mat. Before replacement, this script copies the source
% MAT file to a sibling backup named preprocessed_data_DW.mat by default.
%
% Input modes:
%   1) batch_yaml / batch_yamls:
%        ops/batches/*.yaml with a sessions: list. For each session, this
%        function resolves:
%          run_dir = paths.work_dir in the matching dataset config
%          preprocessed_mat = <parent of run_dir>/preprocessed_data.mat
%
%   2) manifest_csv:
%        CSV with required columns:
%          run_dir, preprocessed_mat
%        Optional columns:
%          extract_mat, cascade_h5, backup_mat, enabled
%
%   3) explicit arrays:
%        run_dirs, preprocessed_mats
%        Optional matching arrays:
%          extract_mats, cascade_h5s, backup_mats
%
% Name-value options:
%   batch_yaml          : one batch YAML path under ops/batches
%   batch_yamls         : one or more batch YAML paths
%   manifest_csv       : CSV manifest path
%   run_dirs           : cellstr/string/char array of run_dir paths
%   preprocessed_mats  : cellstr/string/char array of MAT paths
%   extract_mats       : optional array or scalar empty value
%   cascade_h5s        : optional array or scalar empty value
%   backup_mats        : optional array or scalar empty value
%   repo_root          : repo root for resolving configs (auto by default)
%   config_dir         : config directory (default: <repo_root>/configs/datasets)
%   config_preference  : 'windows' (default) | 'ubuntu'
%   backup_suffix      : default '_DW'
%   overwrite_backup   : false by default
%   write_mode         : 'matfile' (default) | 'append'
%   continue_on_error  : true by default
%   report_csv         : optional CSV path for per-item results
%
% Example:
%   batch_replace_preprocessed_posF_with_extract( ...
%       'manifest_csv', 'R:/tmp/replace_manifest.csv');
%
%   batch_replace_preprocessed_posF_with_extract( ...
%       'run_dirs', {'R:/code/1pMCRI-pipeline/demo_data/output/smoke_250810'}, ...
%       'preprocessed_mats', {'R:/data/example/preprocessed_data.mat'});

opts = struct();
opts.batch_yaml = '';
opts.batch_yamls = {};
opts.manifest_csv = '';
opts.run_dirs = {};
opts.preprocessed_mats = {};
opts.extract_mats = {};
opts.cascade_h5s = {};
opts.backup_mats = {};
opts.repo_root = '';
opts.config_dir = '';
opts.config_preference = 'windows';
opts.backup_suffix = '_DW';
opts.overwrite_backup = false;
opts.write_mode = 'matfile';
opts.continue_on_error = true;
opts.report_csv = '';
opts = parse_name_values(opts, varargin);

items = resolve_items(opts);
n_items = numel(items);

if n_items == 0
    error('No batch items were resolved.');
end

results = repmat(empty_result_row(), n_items, 1);
success_count = 0;
failure_count = 0;
skip_count = 0;

fprintf('Batch items       : %d\n', n_items);
fprintf('write_mode        : %s\n', char(opts.write_mode));
fprintf('overwrite_backup  : %d\n', logical(opts.overwrite_backup));
fprintf('continue_on_error : %d\n', logical(opts.continue_on_error));

for idx = 1:n_items
    item = items(idx);
    fprintf('\n[%d/%d] run_dir=%s\n', idx, n_items, item.run_dir);
    fprintf('        preprocessed_mat=%s\n', item.preprocessed_mat);

    row = empty_result_row();
    row.index = idx;
    row.run_dir = string(item.run_dir);
    row.preprocessed_mat = string(item.preprocessed_mat);
    row.extract_mat = string(item.extract_mat);
    row.cascade_h5 = string(item.cascade_h5);
    row.backup_mat = string(item.backup_mat);

    if ~item.enabled
        row.status = "skipped";
        row.message = "disabled";
        results(idx) = row;
        skip_count = skip_count + 1;
        fprintf('        skipped (disabled)\n');
        continue;
    end

    try
        ensure_backup_copy(item.preprocessed_mat, item.backup_mat, logical(opts.overwrite_backup));

        replace_result = replace_preprocessed_posF_with_extract( ...
            'run_dir', item.run_dir, ...
            'preprocessed_mat', item.preprocessed_mat, ...
            'extract_mat', item.extract_mat, ...
            'cascade_h5', item.cascade_h5, ...
            'write_mode', opts.write_mode);

        row.status = "ok";
        row.message = "";
        row.pos_rows = size_or_zero(replace_result.pos_size, 1);
        row.pos_cols = size_or_zero(replace_result.pos_size, 2);
        row.F_rows = size_or_zero(replace_result.F_size, 1);
        row.F_cols = size_or_zero(replace_result.F_size, 2);
        results(idx) = row;
        success_count = success_count + 1;
        fprintf('        ok\n');
    catch ME
        row.status = "error";
        row.message = string(ME.message);
        results(idx) = row;
        failure_count = failure_count + 1;
        fprintf('        error: %s\n', ME.message);
        if ~logical(opts.continue_on_error)
            break;
        end
    end
end

result_table = struct2table(results);

if ~isempty(strtrim(char(opts.report_csv)))
    write_report_csv(char(opts.report_csv), result_table);
    fprintf('\nReport written: %s\n', char(opts.report_csv));
end

summary = struct();
summary.n_items = n_items;
summary.success_count = success_count;
summary.failure_count = failure_count;
summary.skip_count = skip_count;
summary.report_csv = string(opts.report_csv);
summary.results = result_table;

fprintf('\nSummary: success=%d failure=%d skipped=%d total=%d\n', ...
    success_count, failure_count, skip_count, n_items);

if failure_count > 0 && ~logical(opts.continue_on_error)
    error('Batch stopped after the first failure.');
end

end

function items = resolve_items(opts)
input_mode_count = 0;

if ~isempty(strtrim(char(opts.batch_yaml))) || ~isempty(opts.batch_yamls)
    input_mode_count = input_mode_count + 1;
end

manifest_csv = strtrim(char(opts.manifest_csv));
if ~isempty(manifest_csv)
    input_mode_count = input_mode_count + 1;
end

run_dirs = normalize_path_list(opts.run_dirs, 'run_dirs');
preprocessed_mats = normalize_path_list(opts.preprocessed_mats, 'preprocessed_mats');
if ~isempty(run_dirs) || ~isempty(preprocessed_mats)
    input_mode_count = input_mode_count + 1;
end

if input_mode_count == 0
    error(['Provide one input mode: batch_yaml/batch_yamls, ', ...
        'manifest_csv, or run_dirs/preprocessed_mats.']);
end
if input_mode_count > 1
    error(['Use only one input mode at a time: batch_yaml/batch_yamls, ', ...
        'manifest_csv, or run_dirs/preprocessed_mats.']);
end

batch_yamls = normalize_batch_yaml_list(opts);
if ~isempty(batch_yamls)
    items = read_items_from_batch_yamls(batch_yamls, opts);
    return;
end

if ~isempty(manifest_csv)
    items = read_items_from_manifest(manifest_csv, opts);
    return;
end

n_items = numel(run_dirs);

if n_items == 0
    error('run_dirs/preprocessed_mats input mode resolved zero items.');
end
if numel(preprocessed_mats) ~= n_items
    error('run_dirs and preprocessed_mats must have the same length.');
end

extract_mats = normalize_optional_path_list(opts.extract_mats, n_items);
cascade_h5s = normalize_optional_path_list(opts.cascade_h5s, n_items);
backup_mats = normalize_optional_path_list(opts.backup_mats, n_items);

items = repmat(empty_item(), n_items, 1);
for idx = 1:n_items
    items(idx).run_dir = run_dirs{idx};
    items(idx).preprocessed_mat = preprocessed_mats{idx};
    items(idx).extract_mat = extract_mats{idx};
    items(idx).cascade_h5 = cascade_h5s{idx};
    items(idx).backup_mat = resolve_backup_path(preprocessed_mats{idx}, backup_mats{idx}, char(opts.backup_suffix));
    items(idx).enabled = true;
end
end

function batch_yamls = normalize_batch_yaml_list(opts)
batch_yamls = {};

if ~isempty(strtrim(char(opts.batch_yaml)))
    batch_yamls{end + 1, 1} = strtrim(char(opts.batch_yaml));
end

if ~isempty(opts.batch_yamls)
    extra = normalize_path_list(opts.batch_yamls, 'batch_yamls');
    batch_yamls = [batch_yamls; extra(:)];
end
end

function items = read_items_from_batch_yamls(batch_yamls, opts)
repo_root = resolve_repo_root(opts.repo_root);
config_dir = resolve_config_dir(opts.config_dir, repo_root);
config_preference = lower(strtrim(char(opts.config_preference)));

if ~ismember(config_preference, {'windows', 'ubuntu'})
    error('config_preference must be ''windows'' or ''ubuntu''.');
end

items = repmat(empty_item(), 0, 1);

for batch_idx = 1:numel(batch_yamls)
    batch_yaml = batch_yamls{batch_idx};
    if ~isfile(batch_yaml)
        error('batch_yaml not found: %s', batch_yaml);
    end

    session_ids = read_batch_session_ids(batch_yaml);
    if isempty(session_ids)
        error('No sessions found in batch_yaml: %s', batch_yaml);
    end

    for idx = 1:numel(session_ids)
        session_id = session_ids{idx};
        config_path = resolve_session_config_path(session_id, config_dir, config_preference);
        run_dir = read_yaml_scalar(config_path, 'work_dir');
        if isempty(run_dir)
            error('work_dir not found in config: %s', config_path);
        end

        preprocessed_mat = fullfile(fileparts(run_dir), 'preprocessed_data.mat');
        item = empty_item();
        item.run_dir = run_dir;
        item.preprocessed_mat = preprocessed_mat;
        item.backup_mat = resolve_backup_path(preprocessed_mat, '', char(opts.backup_suffix));
        item.enabled = true;
        items(end + 1, 1) = item; %#ok<AGROW>
    end
end
end

function items = read_items_from_manifest(manifest_csv, opts)
if ~isfile(manifest_csv)
    error('manifest_csv not found: %s', manifest_csv);
end

import_opts = detectImportOptions(manifest_csv, 'TextType', 'string');
T = readtable(manifest_csv, import_opts);
vars = string(T.Properties.VariableNames);

required = ["run_dir", "preprocessed_mat"];
missing = required(~ismember(required, vars));
if ~isempty(missing)
    error('manifest_csv is missing required columns: %s', strjoin(cellstr(missing), ', '));
end

n_items = height(T);
items = repmat(empty_item(), n_items, 1);

for idx = 1:n_items
    items(idx).run_dir = strtrim(char(T.run_dir(idx)));
    items(idx).preprocessed_mat = strtrim(char(T.preprocessed_mat(idx)));
    items(idx).extract_mat = read_table_text(T, vars, 'extract_mat', idx);
    items(idx).cascade_h5 = read_table_text(T, vars, 'cascade_h5', idx);
    backup_override = read_table_text(T, vars, 'backup_mat', idx);
    items(idx).backup_mat = resolve_backup_path(items(idx).preprocessed_mat, backup_override, char(opts.backup_suffix));
    items(idx).enabled = read_enabled_flag(T, vars, idx);
end
end

function value = read_table_text(T, vars, var_name, idx)
if ~ismember(string(var_name), vars)
    value = '';
    return;
end

raw = T.(var_name)(idx);
if ismissing(raw)
    value = '';
else
    value = strtrim(char(raw));
end
end

function tf = read_enabled_flag(T, vars, idx)
if ~ismember("enabled", vars)
    tf = true;
    return;
end

raw = T.enabled(idx);
if islogical(raw)
    tf = logical(raw);
    return;
end

if isnumeric(raw)
    tf = raw ~= 0;
    return;
end

if iscell(raw)
    raw = raw{1};
end

if isstring(raw)
    token = lower(strtrim(char(raw)));
elseif ischar(raw)
    token = lower(strtrim(raw));
else
    tf = true;
    return;
end

if isempty(token)
    tf = true;
    return;
end

tf = ~ismember(token, {'0', 'false', 'no', 'off', 'skip'});
end

function ensure_backup_copy(source_path, backup_path, overwrite_backup)
if ~isfile(source_path)
    error('preprocessed_data.mat not found: %s', source_path);
end

if strcmpi(source_path, backup_path)
    error('backup_mat must differ from preprocessed_mat: %s', source_path);
end

backup_dir = fileparts(backup_path);
if ~isempty(backup_dir) && ~isfolder(backup_dir)
    mkdir(backup_dir);
end

if isfile(backup_path) && ~overwrite_backup
    error('Backup already exists. Set overwrite_backup=true to replace it: %s', backup_path);
end

if isfile(backup_path)
    ok = copyfile(source_path, backup_path, 'f');
else
    ok = copyfile(source_path, backup_path);
end

if ~ok || ~isfile(backup_path)
    error('Failed to create backup copy: %s', backup_path);
end

fprintf('        backup=%s\n', backup_path);
end

function backup_path = resolve_backup_path(source_path, backup_override, backup_suffix)
if ~isempty(strtrim(char(backup_override)))
    backup_path = strtrim(char(backup_override));
    return;
end

[parent_dir, name, ext] = fileparts(source_path);
backup_name = [name backup_suffix ext];
backup_path = fullfile(parent_dir, backup_name);
end

function repo_root = resolve_repo_root(repo_root_opt)
repo_root = strtrim(char(repo_root_opt));
if ~isempty(repo_root)
    return;
end

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(script_dir)));
end

function config_dir = resolve_config_dir(config_dir_opt, repo_root)
config_dir = strtrim(char(config_dir_opt));
if ~isempty(config_dir)
    return;
end

config_dir = fullfile(repo_root, 'configs', 'datasets');
end

function session_ids = read_batch_session_ids(batch_yaml)
lines = readlines(batch_yaml);
session_ids = {};
in_sessions = false;

for idx = 1:numel(lines)
    line = char(lines(idx));
    trimmed = strtrim(line);

    if isempty(trimmed) || startsWith(trimmed, '#')
        continue;
    end

    if ~in_sessions
        if strcmp(trimmed, 'sessions:')
            in_sessions = true;
        end
        continue;
    end

    if startsWith(line, ' ') || startsWith(line, sprintf('\t'))
        token = regexp(trimmed, '^-+\s*(.+?)\s*$', 'tokens', 'once');
        if ~isempty(token)
            session_ids{end + 1, 1} = strip_yaml_scalar(token{1}); %#ok<AGROW>
            continue;
        end
    end

    if ~startsWith(trimmed, '-')
        break;
    end
end
end

function config_path = resolve_session_config_path(session_id, config_dir, config_preference)
token = regexprep(session_id, '[^A-Za-z0-9]+', '_');
token = regexprep(token, '^_+|_+$', '');
if isempty(token)
    error('Could not derive config token from session_id: %s', session_id);
end

preferred = fullfile(config_dir, ['target_reach_' token '_' config_preference '.yaml']);
if isfile(preferred)
    config_path = preferred;
    return;
end

fallback_suffix = 'windows';
if strcmp(config_preference, 'windows')
    fallback_suffix = 'ubuntu';
end

fallback = fullfile(config_dir, ['target_reach_' token '_' fallback_suffix '.yaml']);
if isfile(fallback)
    config_path = fallback;
    return;
end

error('Config YAML not found for session_id=%s under %s', session_id, config_dir);
end

function value = read_yaml_scalar(yaml_path, key_name)
text = fileread(yaml_path);
pattern = ['(?m)^\s*' regexptranslate('escape', key_name) '\s*:\s*(.+?)\s*$'];
token = regexp(text, pattern, 'tokens', 'once');
if isempty(token)
    value = '';
    return;
end

value = strip_yaml_scalar(token{1});
end

function value = strip_yaml_scalar(value_in)
value = strtrim(char(value_in));
if numel(value) >= 2
    if (value(1) == '"' && value(end) == '"') || (value(1) == '''' && value(end) == '''')
        value = value(2:end - 1);
    end
end
end

function items = normalize_path_list(value, field_name)
if isstring(value)
    items = cellstr(value(:));
elseif ischar(value)
    if isempty(strtrim(value))
        items = {};
    else
        items = {value};
    end
elseif iscell(value)
    items = value(:);
else
    error('%s must be char, string, or cell array.', field_name);
end

for idx = 1:numel(items)
    if ~(ischar(items{idx}) || (isstring(items{idx}) && isscalar(items{idx})))
        error('%s entries must be text scalars.', field_name);
    end
    items{idx} = strtrim(char(items{idx}));
end
end

function items = normalize_optional_path_list(value, n_items)
if isempty(value)
    items = repmat({''}, n_items, 1);
    return;
end

items = normalize_path_list(value, 'optional path list');

if numel(items) == 1 && n_items > 1
    items = repmat(items, n_items, 1);
    return;
end

if numel(items) ~= n_items
    error('Optional path list must have length 1 or match the item count (%d).', n_items);
end
end

function write_report_csv(report_csv, T)
report_dir = fileparts(report_csv);
if ~isempty(report_dir) && ~isfolder(report_dir)
    mkdir(report_dir);
end
writetable(T, report_csv);
end

function row = empty_result_row()
row = struct( ...
    'index', 0, ...
    'status', "", ...
    'message', "", ...
    'run_dir', "", ...
    'preprocessed_mat', "", ...
    'extract_mat', "", ...
    'cascade_h5', "", ...
    'backup_mat', "", ...
    'pos_rows', 0, ...
    'pos_cols', 0, ...
    'F_rows', 0, ...
    'F_cols', 0);
end

function item = empty_item()
item = struct( ...
    'run_dir', '', ...
    'preprocessed_mat', '', ...
    'extract_mat', '', ...
    'cascade_h5', '', ...
    'backup_mat', '', ...
    'enabled', true);
end

function value = size_or_zero(sz, idx)
value = 0;
if isnumeric(sz) && numel(sz) >= idx
    value = sz(idx);
end
end

function opts = parse_name_values(opts, args)
if isempty(args)
    return;
end
if mod(numel(args), 2) ~= 0
    error('Name-value arguments must be paired.');
end
for i = 1:2:numel(args)
    key = char(args{i});
    val = args{i + 1};
    if ~isfield(opts, key)
        error('Unknown option: %s', key);
    end
    opts.(key) = val;
end
end
