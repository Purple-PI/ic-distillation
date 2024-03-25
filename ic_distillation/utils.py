import os
from pathlib import Path
from typing import Optional


def get_env(name: str) -> Optional[Path]:
    env_name = os.getenv(name)
    if env_name is None:
        return None
    return Path(os.getenv(name))
