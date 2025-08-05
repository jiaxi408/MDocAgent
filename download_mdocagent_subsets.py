# download_mdocagent_subsets.py
import os, requests, json

subsets = {
  "mmlb": "MMLongBench",
  "ptab": "PaperTab",
  "ptext": "PaperText",
  "feta": "FetaTab",
  "longdocurl": "LongDocURL"
}
base_url = "https://huggingface.co/datasets/Lillianwei/Mdocagent-dataset-/resolve/main/"

for local, hf in subsets.items():
  os.makedirs(f"data/{local}/documents", exist_ok=True)
  # 假设每个子集 samples.json 列出 doc_id
  sample = json.load(open(f"data/{local}/samples.json"))
  for item in sample[:1]:
    doc = item["doc_id"]
    url = base_url + f"{hf}/documents/{doc}"
    r = requests.get(url)
    open(f"data/{local}/documents/{doc}", "wb").write(r.content)
