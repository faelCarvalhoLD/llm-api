from huggingface_hub import snapshot_download

if __name__ == '__main__':

    snapshot_download(repo_id="microsoft/phi-4", cache_dir="./hf-models/phi-4")
    snapshot_download(
        repo_id="microsoft/phi-2",
        cache_dir="./hf-models/phi-2",
        local_dir_use_symlinks=False
    )