from modelscope.hub.snapshot_download import snapshot_download

model_name = "qwen/Qwen1.5-1.8B"
model_dir = snapshot_download(model_name, cache_dir='.', revision='master')