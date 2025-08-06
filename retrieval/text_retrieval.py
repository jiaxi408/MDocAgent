import os
import json
from tqdm import tqdm
from ragatouille import RAGPretrainedModel

from retrieval.base_retrieval import BaseRetrieval
from mydatasets.base_dataset import BaseDataset

class ColbertRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config

    def prepare(self, dataset: BaseDataset):
        samples = dataset.load_data(use_retreival=True)
        model_path = getattr(self.config, "pretrained_model_path", "colbert-ir/colbertv2.0")

        print(f"[INFO] Attempting to load RAG model from: {model_path}")

        # 设置是否使用离线模式。如果提供的路径是目录且存在模型权重，则优先使用本地模型。
        hf_identifier = model_path  # 默认直接使用配置中的路径
        if os.path.isdir(model_path):
            weight_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(weight_file):
                print(f"[INFO] Detected local model directory: {model_path}")
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            else:
                # 目录存在但未包含模型文件，尝试根据目录名推断出HF仓库名称并从线上加载
                hf_identifier = os.path.basename(model_path).replace("__", "/")
                print(
                    f"[WARNING] Local directory '{model_path}' missing model files. Falling back to HuggingFace repo '{hf_identifier}'."
                )
        else:
            print(
                f"[WARNING] Model path '{model_path}' is not a local directory. Will try to connect to HuggingFace Hub."
            )

        # 安全加载模型
        try:
            RAG = RAGPretrainedModel.from_pretrained(hf_identifier)
        except TypeError as e:
            print(f"[ERROR] Unexpected TypeError while loading model: {e}")
            RAG = RAGPretrainedModel.from_pretrained(hf_identifier)
        except Exception as e:
            print(f"[ERROR] Failed to load RAG model from: {hf_identifier}")
            raise e

        doc_index: dict = {}
        error = 0

        for sample in tqdm(samples):
            if self.config.r_text_index_key in sample and os.path.exists(sample[self.config.r_text_index_key]):
                continue
            if sample[self.config.doc_key] in doc_index:
                sample[self.config.r_text_index_key] = doc_index[sample[self.config.doc_key]]
                continue

            content_list = dataset.load_processed_content(sample)
            text = [content.txt.replace("\n", "") for content in content_list]

            try:
                index_path = RAG.index(
                    index_name=f"{dataset.config.name}-{self.config.text_question_key}-{sample[self.config.doc_key]}",
                    collection=text
                )
                doc_index[sample[self.config.doc_key]] = index_path
                sample[self.config.r_text_index_key] = index_path
            except Exception as e:
                error += 1
                print(f"[ERROR] Error processing {sample[self.config.doc_key]}: {e}")
                sample[self.config.r_text_index_key] = ""
                if error > len(samples) / 100:
                    print("[FATAL] Too many error cases. Exiting process.")
                    import sys
                    sys.exit(1)

        dataset.dump_data(samples, use_retreival=True)
        print(f"[INFO] Retrieval indexing completed. Total samples: {len(samples)}")

        return samples

    def find_sample_top_k(self, sample, top_k: int, page_id_key: str):
        if not os.path.exists(sample[self.config.r_text_index_key]+"/pid_docid_map.json"):
            print(f"Index not found for {sample[self.config.r_text_index_key]}/pid_docid_map.json.")
            return [], []
        with open(sample[self.config.r_text_index_key]+"/pid_docid_map.json",'r') as f:
            pid_map_data = json.load(f)
        unique_values = list(dict.fromkeys(pid_map_data.values()))
        value_to_rank = {val: idx for idx, val in enumerate(unique_values)}
        pid_map = {int(key): value_to_rank[value] for key, value in pid_map_data.items()}
        
        query = sample[self.config.text_question_key]
        RAG = RAGPretrainedModel.from_index(sample[self.config.r_text_index_key])
        results = RAG.search(query, k=len(pid_map))
        
        top_page_indices = [pid_map[page['passage_id']] for page in results]
        top_page_scores = [page['score'] for page in results]
        
        if page_id_key in sample:
            page_id_list = sample[page_id_key]
            assert isinstance(page_id_list, list)
            filtered_indices = []
            filtered_scores = []
            for idx, score in zip(top_page_indices, top_page_scores):
                if idx in page_id_list:
                    filtered_indices.append(idx)
                    filtered_scores.append(score)
            return filtered_indices[:top_k], filtered_scores[:top_k]
        
        return top_page_indices[:top_k], top_page_scores[:top_k]
        
    def find_top_k(self, dataset: BaseDataset, force_prepare=False):
        top_k = self.config.top_k
        samples = dataset.load_data(use_retreival=True)
        
        if self.config.r_text_index_key not in samples[0] or force_prepare:
            samples = self.prepare(dataset)
                
        for sample in tqdm(samples):
            top_page_indices, top_page_scores = self.find_sample_top_k(sample, top_k=top_k, page_id_key = dataset.config.page_id_key)
            sample[self.config.r_text_key] = top_page_indices
            sample[self.config.r_text_key+"_score"] = top_page_scores
        path = dataset.dump_data(samples, use_retreival=True)
        print(f"Save retrieval results at {path}.")