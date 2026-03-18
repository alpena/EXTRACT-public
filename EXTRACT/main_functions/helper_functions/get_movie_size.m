function [h, w, t] = get_movie_size(M)
    if ischar(M) && numel(M) > 8 && strncmp(M, 'partdir:', 8)
        info = jsondecode(fileread(fullfile(M(9:end), 'manifest.json')));
        h = double(info.h_fov);
        w = double(info.w_fov);
        t = double(info.nt);
        return;
    end
    if ischar(M) || iscell(M)
        [path, dataset] = parse_movie_name(M);
        movie_info = h5info(path, dataset);
        movie_size = num2cell(movie_info.Dataspace.Size);
        [h, w, t] = deal(movie_size{:});
    else
        [h, w, t] = size(M);
    end
end