
cache_file_path = {{inputs.parameters.max-parallel-steps}}
    if is_s3_path(cache_file_path):
        local_cache_file_path = "/tmp/cachedb.db"
        fs = get_filesystem(cache_file_path, anon=False)
        fs.get(cache_file_path, local_cache_file_path)
        log.info(
            f"Downloaded cache file from {cache_file_path} to {local_cache_file_path}"
        )
        cache = dscache.open_ro(local_cache_file_path)
    else:
        cache = dscache.open_ro(cache_file_path)