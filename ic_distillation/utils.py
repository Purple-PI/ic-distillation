import os
from pathlib import Path
from typing import Optional

import torch


def get_env(name: str) -> Optional[Path]:
    env_name = os.getenv(name)
    if env_name is None:
        return None
    return Path(os.getenv(name))


def get_dtype(type_str):
    if type_str == "bf16":
        return torch.bfloat16
    elif type_str == "fp16":
        return torch.float16
    else:
        return torch.float32
