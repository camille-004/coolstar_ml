"""Load the data with specified preprocessing/FE steps and test a classifier."""
import os

import pandas as pd

from preprocess_data import (  # filter_binaries,; filter_singles,
    feature_engineering,
    get_binary_single_dfs,
)
from utils import load_config

config = load_config("config.yaml")

singles, binaries = get_binary_single_dfs(
    os.path.join(config["data_dir"], config["fp_july15"])
)
df = feature_engineering(singles, binaries, config["noise_scale"])


def prepare_data(
    f_name: str,
    add_noise: bool,
    scale: float,
    snr: bool,
    template_diffs: bool,
    chisq: bool,
    bartolez_chisq: bool,
    chisq_std: bool,
) -> pd.DataFrame:
    """
    Load the fully cleaned and preprocessed dataset.

    :param f_name: Filename of raw dataset
    :param add_noise: Whether to add noise
    :param scale: Noise scaling (and chi-squared if using)
    :param snr: Whether to calculate signal-to-noise ratio
    :param template_diffs: Whether to add in the difference spectra as a feature
    :param chisq: Whether to calculate the chi-squared statistic for spectra
    :param bartolez_chisq: Whether to use Bartolez binning for chi-squared calculation
    :param chisq_std: Whether to add standard deviation feature (if using Bartolez binning for
    chi-squared calculation)
    :return: Prepared DataFrame
    """
    assert f_name in os.listdir(config["data_dir"])
    _singles, _binaries = get_binary_single_dfs(
        os.path.join(config["data_dir"], f_name)
    )
    _df = feature_engineering(
        _singles,
        _binaries,
        scale=scale,
        _add_noise=add_noise,
        snr=snr,
        add_template_diffs=template_diffs,
        chisq=chisq,
        bartolez_chisq=bartolez_chisq,
        chisq_std=chisq_std,
    )
    return _df
