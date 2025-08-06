import os
from huggingface_hub import snapshot_download

# 如果你使用 v2rayN 的 HTTP 代理端口（127.0.0.1:10809），请启用以下两行：
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

# 模型列表
models = [
    ("bert-base-uncased", "offline_models/bert-base-uncased"),
    ("sentence-transformers/all-MiniLM-L6-v2", "offline_models/all-MiniLM-L6-v2")
]

# 下载并保存模型
for repo_id, target_dir in models:
    print(f"\n📥 Downloading {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

print("\n✅ All models downloaded to 'offline_models/' folder successfully.")
