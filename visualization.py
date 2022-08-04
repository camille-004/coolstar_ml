"""Module for generating plots."""
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from test_clf import filter_snr, prepare_data, split_data, test_rf
from utils import load_config

config = load_config("config.yaml")

plt.rcParams["figure.dpi"] = config["figure.dpi"]
plt.rcParams["savefig.dpi"] = config["savefig.dpi"]

sns.set_style(config["seaborn_style"])
sns.set_palette(config["seaborn_palette"])


def plot_spectral_type(_df: pd.DataFrame) -> None:
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


def plot_snr_class_dist(_snr_ranges: List[List[int]]) -> None:
    """
    Plot singles-vs.-binaries class distribution by SNR bins.

    :param _snr_ranges: List of list of SNR ranges to plot
    :return:
    """
    class_dist = pd.DataFrame()
    for i in _snr_ranges:
        df = prepare_data(
            f_name=config["fp_july15"],
            binaries_filter=["M0", "T9", "M0", "T9"],
            singles_filter=["M0", "T9"],
            _add_noise=True,
            snr=True,
            template_diffs=False,
            chisq=False,
        )
        snr_min, snr_max = i
        df = filter_snr(df, snr_min, snr_max)
        counts = df[config["target_col"]].value_counts()
        counts_df = counts.to_frame().rename(
            columns={config["target_col"]: "counts"}
        )
        counts_df[config["target_col"]] = [0, 1]
        counts_df["snr_range"] = str(i)
        class_dist = pd.concat([class_dist, counts_df])

    class_dist.reset_index(drop=True, inplace=True)
    ax = sns.barplot(x="snr_range", y="counts", hue="single", data=class_dist)
    ax.set_title("Class Distribution in SNR Bins")
    plt.show()


def plot_model_metrics_snr(
    _snr_ranges: List[List[int]], **kwargs: Union[List[str], bool, str, float]
) -> None:
    """
    Run an RF model and plot the testing performance metrics for each SNR range.

    :param _snr_ranges: List of list of SNR ranges to plot
    :return:
    """
    metrics_df = pd.DataFrame()
    n_test_singles = []
    for i in _snr_ranges:
        df = prepare_data(**kwargs)  # pylint: disable=E1120
        snr_min, snr_max = i
        df = filter_snr(df, snr_min, snr_max)
        X_train, X_test, y_train, y_test = split_data(df)
        n_test_singles.append(
            y_test.sum()
        )  # Get number of singles in test set
        _, prec, rec, f1 = test_rf(
            X_train, X_test, y_train, y_test, report=False
        )
        row = {
            "snr_range": str(i),
            "test_precision": [prec],
            "test_recall": [rec],
            "test_f1": [f1],
        }
        row = pd.DataFrame.from_dict(row)
        metrics_df = pd.concat([metrics_df, row])

    metrics_dfm = metrics_df.melt(
        "snr_range", var_name="metrics", value_name="vals"
    )
    sns.catplot(
        x="snr_range", y="vals", hue="metrics", data=metrics_dfm, kind="point"
    ).set(title="RF Test Set Performance vs. SNR Range")
    plt.show()
    sns.barplot(
        x=metrics_df["snr_range"],
        y=n_test_singles,
        color=sns.color_palette(config["seaborn_palette"])[0],
    ).set(title="Number of Singles in Testing Set vs. SNR Range")
    plt.show()
