"""Load the data with specified preprocessing/FE steps and test a classifier."""
import pandas as pd

from preprocess_data import prepare_data
from utils import load_config

config = load_config("config.yaml")


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
    _add_noise=True,
    snr=True,
    template_diffs=True,
    chisq=True,
    bartolez_chisq=True,
    chisq_std=True,
)
