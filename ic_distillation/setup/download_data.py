import os

import hydra
from datasets import load_dataset
from dotenv import load_dotenv

from ic_distillation.utils import get_env


@hydra.main(config_path="../../config", config_name="config")
def main(cfg):
    DATA_PATH = get_env("DATA_PATH")
    data_path = DATA_PATH / cfg.data.path.split("/")[-1]
    dataset = load_dataset(cfg.data.path)
    dataset.save_to_disk(data_path)
    print(f"Dataset saved to {data_path}")


if __name__ == "__main__":
    load_dotenv()
    main()
