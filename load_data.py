"""Load dataset in the correct format."""
import os
from typing import Tuple

import pandas as pd  # type: ignore

from utils import load_config

config = load_config("config.yaml")
DATA_DIR = config["data_dir"]
FILE_NAME = config["file_name"]
SINGLE_KEY = config["single_key"]
BINARY_KEY = config["binary_key"]
TARGET_COL = config["target_col"]


def load_spectral_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load single and binary spectra."""
    single_df = pd.read_hdf(os.path.join(DATA_DIR, FILE_NAME), key=SINGLE_KEY)
    single_df.drop(columns=config["single_drop_cols"], inplace=True)
    binary_df = pd.read_hdf(os.path.join(DATA_DIR, FILE_NAME), key=BINARY_KEY)
    binary_df.drop(columns=config["binary_drop_cols"], inplace=True)
    binary_df.rename(
        columns={config["binary_rename_col"]: config["spectral_type_col"]},
        inplace=True,
    )
    return single_df, binary_df


# %%
single, binary = load_spectral_data()
