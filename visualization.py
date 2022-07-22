"""Module for generating plots."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
