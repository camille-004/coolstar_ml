"""Module for generating plots."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess_data import feature_engineering, get_binary_single_dfs
from utils import load_config

config = load_config("config.yaml")

plt.rcParams["figure.dpi"] = config["figure.dpi"]
plt.rcParams["savefig.dpi"] = config["savefig.dpi"]

sns.set_style(config["seaborn_style"])
sns.set_palette(config["seaborn_palette"])


def visualize_spectral_type(_df: pd.DataFrame) -> None:
    """
    Plot s histogram of the spectral/system types by binaries vs. singles.
    :param _df: DataFrame with spectral type columns
    :return: None
    """
    _df[config["spectral_type_col"]].hist(
        by=_df[config["target_col"]], figsize=tuple(config["seaborn_figsize"])
    )
    plt.suptitle("Spectral Type Distribution by Class")
    plt.show()


singles, binaries = get_binary_single_dfs(
    os.path.join(config["data_dir"], config["fp_july15"])
)
df = feature_engineering(singles, binaries, config["noise_scale"])
visualize_spectral_type(df)
