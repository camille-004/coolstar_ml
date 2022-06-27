"""Utility module."""
import os
from typing import Any, Dict

import yaml  # type: ignore

CONFIG_PATH = "config"


def load_config(config_name: str) -> Dict[Any, Any]:
    """Load the contents of the config file."""
    with open(
        os.path.join(CONFIG_PATH, config_name), encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    return config
