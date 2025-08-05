from pathlib import Path
import tarfile, shutil
from huggingface_hub import snapshot_download

# ——如需代理，可把 'http://IP:PORT' 换成你的代理地址后再运行——
# import os; os.environ["HTTP_PROXY"]=os.environ["HTTPS_PROXY"]="http://127.0.0.1:7890"

REPOS = [
    "colbert-ir/colbertv2.0",              # ColBERT v2（文本检索）
    "bert-base-uncased",                   # ColBERT backbone
    "sentence-transformers/all-MiniLM-L6-v2"  # 负样本挖掘（训练时用）
]

# 1) 逐个下载到本机 cache
for repo in REPOS:
    print(f"⏬  downloading {repo} …")
    snapshot_download(repo, local_files_only=False, resume_download=True)

# 2) 打包对应 cache 目录
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
tar_path = Path("hf_models.tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    for repo in REPOS:
        folder = HF_CACHE / repo.replace("/", "--")        # ← snapshot_download 的默认目录名
        assert folder.exists(), f"{folder} not found"
        tar.add(folder, arcname=folder.name)
print("✅  打包完成：", tar_path.resolve())