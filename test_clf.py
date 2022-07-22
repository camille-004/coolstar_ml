"""Load the data with specified preprocessing/FE steps and test a classifier."""
import os
from typing import Optional, Tuple

import pandas as pd

from preprocess_data import (
    feature_engineering,
    filter_binaries,
    filter_singles,
    get_binary_single_dfs,
)
from utils import load_config

config = load_config("config.yaml")


# %%
def prepare_data(
    f_name: str,
    binaries_filter: Optional[Tuple[int, int, int, int]],
    singles_filter: Optional[Tuple[int, int]],
    scale: float,
    add_noise: bool,
    snr: bool,
    template_diffs: bool,
    chisq: bool,
    bartolez_chisq: bool,
    chisq_std: bool,
) -> pd.DataFrame:
    """
    Load the fully cleaned and preprocessed dataset.

    :param f_name: Filename of raw dataset
    :param binaries_filter: Conditions for filtering binaries by primary and secondary type,
    optional
    :param singles_filter: Conditions for filtering singles by spectral type, optional
    :param scale: Noise scaling (and chi-squared if using)
    :param add_noise: Whether to add noise
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
    if binaries_filter:
        (
            primaries_min,
            primaries_max,
            secondaries_min,
            secondaries_max,
        ) = binaries_filter
        _binaries = filter_binaries(
            _binaries,
            primaries_min,
            primaries_max,
            secondaries_min,
            secondaries_max,
        )
    if singles_filter:
        type_min, type_max = singles_filter
        _singles = filter_singles(_singles, type_min, type_max)
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
    print(f"DataFrame shape: {_df.shape}")
    return _df


def filter_snr(
    _df: pd.DataFrame, snr_min: float, snr_max: float
) -> pd.DataFrame:
    """
    Get the DF where the SNR is within a smaller range.

    :param _df: Input DF
    :param snr_min: Minimum SNR
    :param snr_max: Maximum SNR
    :return: Prepared DF with appropriate SNRs
    """
    assert "snr" in _df.columns
    return _df[(_df["snr"] >= snr_min) & (_df["snr"] <= snr_max)]


# %%
df = prepare_data(
    f_name=config["fp_july15"],
    binaries_filter=(15, 20, 15, 35),
    singles_filter=None,
    scale=config["noise_scale"],
    add_noise=True,
    snr=True,
    template_diffs=True,
    chisq=True,
    bartolez_chisq=True,
    chisq_std=True,
)
