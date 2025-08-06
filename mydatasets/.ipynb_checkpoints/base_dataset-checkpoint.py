import json
import re
from dataclasses import dataclass
from PIL import Image
import os
import pymupdf
from tqdm import tqdm
from datetime import datetime
import glob
from typing import Dict, List  # 类型提示
from hydra.utils import to_absolute_path
@dataclass
class Content:
    image: Image            # 当前页的图像对象
    image_path: str         # 图像路径
    txt: str                # 当前页的文字内容


class BaseDataset():
    def __init__(self, config):
        self.config = config  # 配置加载

        # 将配置中的相对路径转换为绝对路径，避免 Hydra 改变工作目录导致路径错误
        for key in [
            "extract_path",
            "document_path",
            "sample_path",
            "sample_with_retrieval_path",
            "result_dir",
            "data_dir",
        ]:
            if hasattr(self.config, key):
                setattr(self.config, key, to_absolute_path(getattr(self.config, key)))

        # <extract_path>/<doc_id>_<page_index>.png
        self.IM_FILE = (
            lambda doc_name, index: f"{self.config.extract_path}/{doc_name}_{index}.png"
        )  # 直接按页存图片
        self.TEXT_FILE = (
            lambda doc_name, index: f"{self.config.extract_path}/{doc_name}_{index}.txt"
        )  # 按页存路径（也是论文中的说法：对文字按页再按段落）
        self.EXTRACT_DOCUMENT_ID = lambda sample: re.sub(
            "\\.pdf$", "", sample["doc_id"]
        ).split("/")[-1]
        current_time = datetime.now()
        self.time = current_time.strftime("%Y-%m-%d-%H-%M")  # 时间戳用于保存输出结果

    # 数据加载
    def load_data(self, use_retreival=True):
        path = self.config.sample_path
        if use_retreival:
            try:
                if os.path.exists(self.config.sample_with_retrieval_path):
                    path = self.config.sample_with_retrieval_path
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                print("Use original sample path!")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Sample file not found: {path}")
            with open(path, "r") as f:
                samples = json.load(f)

        return samples

    # 保存样本数据，中间过程输出
    def dump_data(self, samples, use_retreival=True):
        if use_retreival:
            path = self.config.sample_with_retrieval_path
        else:
            path = self.config.sample_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)

        return path

    # 加载最近一次保存的结果文件
    def load_latest_results(self):
        print(self.config.result_dir)
        path = find_latest_json(self.config.result_dir)
        with open(path, 'r') as f:
            samples = json.load(f)
        return samples, path

    # 保存本次结果为时间戳命名的文件
    def dump_reults(self, samples):
        os.makedirs(self.config.result_dir, exist_ok=True)
        path = os.path.join(self.config.result_dir, self.time + ".json")
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)
        return path

    # 加载带检索字段（r_text/r_image）的数据，并提取相应文本图像路径
    def load_retrieval_data(self):
        if not os.path.exists(self.config.sample_with_retrieval_path):
            raise FileNotFoundError(
                f"Sample file not found: {self.config.sample_with_retrieval_path}"
            )
        with open(self.config.sample_with_retrieval_path, "r") as f:
            samples = json.load(f)
        for sample in tqdm(samples):
            _, sample["texts"], sample["images"] = self.load_sample_retrieval_data(sample)
        return samples
    # 处理单个 sample 的检索数据
    def load_sample_retrieval_data(self, sample):
        content_list = self.load_processed_content(sample, disable_load_image=True)
        question:str = sample[self.config.question_key]
        texts = []
        images = []
        if self.config.use_mix:
            if self.config.r_mix_key in sample:
                for page in sample[self.config.r_mix_key][:self.config.top_k]:
                    if page in sample[self.config.r_image_key]:
                        origin_image_path = ""
                        origin_image_path = content_list[page].image_path
                        images.append(origin_image_path)
                    if page in sample[self.config.r_text_key]:
                        texts.append(content_list[page].txt.replace("\n", ""))
        else:
            if self.config.r_text_key in sample:
                for page in sample[self.config.r_text_key][:self.config.top_k]:
                    texts.append(content_list[page].txt.replace("\n", ""))
            if self.config.r_image_key in sample:
                for page in sample[self.config.r_image_key][:self.config.top_k]:
                    origin_image_path = ""
                    origin_image_path = content_list[page].image_path
                    images.append(origin_image_path)

        return question, texts, images
    # 加载整个文档的所有页面内容（非检索）
    def load_full_data(self):
        samples = self.load_data(use_retreival=False)
        for sample in tqdm(samples):
            _, sample["texts"], sample["images"] = self.load_sample_full_data(sample)
        return samples

    def load_sample_full_data(self, sample):
        content_list = self.load_processed_content(sample, disable_load_image=True)
        question:str = sample[self.config.question_key]
        texts = []
        images = []

        if self.config.page_id_key in sample:
            sample_no_list = sample[self.config.page_id_key]
        else:
            sample_no_list = [i for i in range(0,min(len(content_list),self.config.vlm_max_page))]
        for page in sample_no_list:
            texts.append(content_list[page].txt.replace("\n", ""))
            origin_image_path = ""
            origin_image_path = content_list[page].image_path
            images.append(origin_image_path)

        return question, texts, images
    # 从 .txt 和 .png 文件中加载结构化内容
    def load_processed_content(self, sample: Dict, disable_load_image=True) -> List[Content]:
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        content_list = []
        for page_idx in range(self.config.max_page):
            im_file = self.IM_FILE(doc_name, page_idx)
            text_file = self.TEXT_FILE(doc_name, page_idx)
            if not os.path.exists(im_file):
                break
            img = None
            if not disable_load_image:
                img = self.load_image(im_file)
            txt = self.load_txt(text_file)
            content_list.append(Content(image=img, image_path=im_file, txt=txt))
        return content_list
    # 加载图像 / 文本文件内容
    def load_image(self, file):
        pil_im = Image.open(file)
        return pil_im

    def load_txt(self, file):
        max_length = self.config.max_character_per_page
        with open(file, 'r') as file:
            content = file.read()
        content = content.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        return content[:max_length]
    # 提取 PDF 内容为图像和文本（写入磁盘）
    def extract_content(self, resolution=144):
        samples = self.load_data()
        for sample in tqdm(samples):
            self._extract_content(sample, resolution=resolution)

    def _extract_content(self, sample, resolution=144):
        max_pages = self.config.max_page
        os.makedirs(self.config.extract_path, exist_ok=True)
        image_list = []
        text_list = []
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)

        pdf_path = os.path.join(self.config.document_path, sample["doc_id"])
        if not os.path.exists(pdf_path):
            print(f"[跳过] 找不到文件: {pdf_path}")
            return  # 或者 return [], [] 取决于你下游是否需要 image_list, text_list

        try:
            with pymupdf.open(pdf_path) as pdf:
                for index, page in enumerate(pdf[:max_pages]):
                    # 保存图像
                    im_file = self.IM_FILE(doc_name, index)
                    if not os.path.exists(im_file):
                        im = page.get_pixmap(dpi=resolution)
                        im.save(im_file)
                    image_list.append(im_file)

                    # 保存文本
                    txt_file = self.TEXT_FILE(doc_name, index)
                    if not os.path.exists(txt_file):
                        text = page.get_text("text")
                        with open(txt_file, 'w') as f:
                            f.write(text)
                    text_list.append(txt_file)

        except Exception as e:
            print(f"[错误] 处理文件失败: {pdf_path}, 错误信息: {e}")

        return image_list, text_list
def extract_time(file_path):
    file_name = os.path.basename(file_path)
    time_str = file_name.split(".json")[0]
    return datetime.strptime(time_str, "%Y-%m-%d-%H-%M")

def find_latest_json(result_dir):
    pattern = os.path.join(result_dir, "*-*-*-*-*.json")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith('_results.json')]
    if not files:
        print(f"Json file not found at {result_dir}")
        return None
    latest_file = max(files, key=extract_time)
    return latest_file