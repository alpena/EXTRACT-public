%% Convert TIFF stack to H5 in chunks for large-scale EXTRACT runs
% Supports:
% 1) Standard multi-page TIFF
% 2) ImageJ single-IFD stack TIFF (ImageDescription contains images=...)
%
% Edit parameters in this section.
input_tiff = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-demo', '250206-UK6-1-F=4_power=5mW_reg_t_crop_s_full.tiff');
output_h5 = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '1pMCRI-demo', '250206-UK6-1-F=4_power=5mW_reg_t_crop_s_full.h5');
dataset_name = '/mov';
chunk_t = 200;

if ~isfile(input_tiff)
    error('Input TIFF not found: %s', input_tiff);
end
if chunk_t < 1 || floor(chunk_t) ~= chunk_t
    error('chunk_t must be a positive integer.');
end

tiff_info = imfinfo(input_tiff);
[height, width] = deal(tiff_info(1).Height, tiff_info(1).Width);
num_ifd = numel(tiff_info);
total_frames = detect_total_frames(tiff_info);

fprintf('Input TIFF: %s\n', input_tiff);
fprintf('Size: %d x %d x %d\n', height, width, total_frames);
fprintf('Layout: %s\n', detect_layout_name(num_ifd, total_frames));
fprintf('Output H5: %s:%s\n', output_h5, dataset_name);

if isfile(output_h5)
    delete(output_h5);
end

% Use chunk size [h,w,chunk_t] for streaming write.
h5create(output_h5, dataset_name, [height, width, total_frames], ...
    'Datatype', 'uint16', ...
    'ChunkSize', [height, width, min(chunk_t, total_frames)]);

machinefmt = detect_tiff_byte_order(input_tiff);
strip_offset = [];
frame_bytes = [];
is_imagej_single_ifd = false;

if num_ifd == 1 && total_frames > 1
    is_imagej_single_ifd = true;
    t = Tiff(input_tiff, 'r');
    bits_per_sample = double(t.getTag('BitsPerSample'));
    samples_per_pixel = double(t.getTag('SamplesPerPixel'));
    compression = double(t.getTag('Compression'));
    strip_offset = double(t.getTag('StripOffsets'));
    frame_bytes = double(t.getTag('StripByteCounts'));
    close(t);
    if bits_per_sample ~= 16 || samples_per_pixel ~= 1 || compression ~= 1
        error(['ImageJ single-IFD mode currently supports only ', ...
            'uncompressed 16-bit single-channel TIFF.']);
    end
    expected_frame_bytes = height * width * 2;
    if frame_bytes ~= expected_frame_bytes
        error('Unexpected frame byte size in TIFF (got %d, expected %d).', ...
            frame_bytes, expected_frame_bytes);
    end
end

if is_imagej_single_ifd
    fid = fopen(input_tiff, 'r', machinefmt);
    if fid < 0
        error('Failed to open TIFF for binary reading: %s', input_tiff);
    end
    cleanup_fid = onCleanup(@() fclose(fid));
end

num_chunks = ceil(total_frames / chunk_t);
for i = 1:num_chunks
    f_begin = (i - 1) * chunk_t + 1;
    f_end = min(i * chunk_t, total_frames);
    n_this = f_end - f_begin + 1;

    fprintf('Chunk %d/%d: frames %d-%d\n', i, num_chunks, f_begin, f_end);

    if is_imagej_single_ifd
        block = zeros(height, width, n_this, 'uint16');
        for k = 1:n_this
            frame_idx = f_begin + k - 1;
            offset = strip_offset + (frame_idx - 1) * frame_bytes;
            fseek(fid, offset, 'bof');
            frame = fread(fid, height * width, '*uint16');
            if numel(frame) ~= height * width
                error('Failed to read frame %d from TIFF.', frame_idx);
            end
            % TIFF raster is row-major; transpose for MATLAB.
            block(:, :, k) = reshape(frame, [width, height])';
        end
    else
        block = uint16(read_from_tif(input_tiff, f_begin, n_this));
    end

    h5write(output_h5, dataset_name, block, [1, 1, f_begin], [height, width, n_this]);
end

out_info = h5info(output_h5, dataset_name);
out_size = out_info.Dataspace.Size;
if ~isequal(out_size, [height, width, total_frames])
    error('Output H5 shape mismatch. Expected [%d %d %d], got [%d %d %d].', ...
        height, width, total_frames, out_size(1), out_size(2), out_size(3));
end
fprintf('H5 dataset shape (h,w,t): [%d %d %d]\n', out_size(1), out_size(2), out_size(3));

fprintf('Done. Wrote H5: %s\n', output_h5);

function total_frames = detect_total_frames(tiff_info)
total_frames = numel(tiff_info);
if total_frames == 1 && isfield(tiff_info(1), 'ImageDescription')
    desc = tiff_info(1).ImageDescription;
    token = regexp(desc, 'images=(\d+)', 'tokens', 'once');
    if ~isempty(token)
        total_frames = str2double(token{1});
    end
end
end

function machinefmt = detect_tiff_byte_order(tiff_path)
fid = fopen(tiff_path, 'r');
if fid < 0
    error('Failed to open TIFF header: %s', tiff_path);
end
cleanup_fid = onCleanup(@() fclose(fid));
sig = fread(fid, 2, '*char')';
if strcmp(sig, 'II')
    machinefmt = 'ieee-le';
elseif strcmp(sig, 'MM')
    machinefmt = 'ieee-be';
else
    error('Invalid TIFF byte-order signature.');
end
clear cleanup_fid
end

function name = detect_layout_name(num_ifd, total_frames)
if num_ifd > 1
    name = 'multi-page TIFF';
elseif num_ifd == 1 && total_frames > 1
    name = 'ImageJ single-IFD stack';
else
    name = 'single-frame TIFF';
end
end
