import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
import hydra

@hydra.main(config_path="../config", config_name="base", version_base="1.2") # 传配置
def main(cfg):
    dataset = BaseDataset(cfg.dataset) # 该类负责数据加载、图像/文本抽取、OCR和PDF处理等逻辑
    dataset.extract_content()# 入口点

if __name__ == "__main__":
    main()