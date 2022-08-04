"""Feature engineering module (add noise, compute chi-squared statistic, etc.)"""
import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from preprocess_data import get_spectral_data
from utils import load_config

config = load_config("config.yaml")

# To get the chi-squared statistic according to Bardalez et. al.
wavegrids = pd.read_hdf(
    os.path.join(config["data_dir"], config["fp_june20"]),
    key=config["wavegrid_key"],
)
range_1 = wavegrids[(wavegrids >= 0.95) & (wavegrids <= 1.35)].dropna().index
range_2 = wavegrids[(wavegrids >= 1.45) & (wavegrids <= 1.80)].dropna().index
range_3 = wavegrids[(wavegrids >= 2.00) & (wavegrids <= 2.35)].dropna().index


def add_noise(
    flux_df: pd.DataFrame, noise_df: pd.DataFrame, scale: float
) -> Union[np.ndarray, int, float, complex]:
    """
    Add Gaussian noise to each star in input DataFrame. Multiply uncertainty by a desired scale.

    :param flux_df: DataFrame of flux measurements
    :param noise_df: DataFrame of uncertainties
    :param scale: Uncertainty scaling factor
    :return: Flux measurements with noise as a NumPy array
    """
    with_noise = np.random.normal(flux_df, noise_df.abs() * scale)
    return with_noise


def compute_snr(
    flux_df: pd.DataFrame, noise_df: pd.DataFrame, scale: float
) -> np.ndarray:
    """
    Compute signal-to-noise ratio for each spectrum in input DataFrame. Note: SNR is not meant to
    be used as a feature.

    :param flux_df: DataFrame of flux measurements
    :param noise_df: DataFrame of uncertainties
    :param scale: Uncertainty scaling factor
    :return: SNR for each spectrum
    """
    return np.nanmedian(
        flux_df.values
        / ((noise_df.abs() + config["zero_div_offset"]).values * scale),
        axis=1,
    )


def compute_chisq(
    diff_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    scale: Optional[float],
    bardalez: bool,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute chi-squared statistics on same wavelengths as in Bardalez et. al. or on all wavelengths

    :param diff_df: DataFrame of spectrum difference with best-fit template
    :param noise_df: DataFrame of uncertainties
    :param scale: Uncertainty scaling factor
    :param bardalez: Whether to use wavelength binning from Bardalez et. al.
    :return: DataFrame of chi-squared statistics for three wavelength ranges
    """
    if bardalez:
        diff_range_1 = diff_df[
            ["flux_" + str(i) + config["diff_suffix"] for i in range_1]
        ]
        diff_range_2 = diff_df[
            ["flux_" + str(i) + config["diff_suffix"] for i in range_2]
        ]
        diff_range_3 = diff_df[
            ["flux_" + str(i) + config["diff_suffix"] for i in range_3]
        ]

        noise_range_1 = noise_df[
            ["flux_" + str(i) + config["noise_suffix"] for i in range_1]
        ]
        noise_range_2 = noise_df[
            ["flux_" + str(i) + config["noise_suffix"] for i in range_2]
        ]
        noise_range_3 = noise_df[
            ["flux_" + str(i) + config["noise_suffix"] for i in range_3]
        ]

        chisq_df = pd.DataFrame()
        chisq_df["chisq_095_135"] = (
            (
                scale
                * diff_range_1.values
                / (noise_range_1.values + config["zero_div_offset"])
            )
            ** 2
        ).sum(axis=1) / config["chisq_rescale"]
        chisq_df["chisq_145_180"] = (
            (
                scale
                * diff_range_2.values
                / (noise_range_2.values + config["zero_div_offset"])
            )
            ** 2
        ).sum(axis=1) / config["chisq_rescale"]
        chisq_df["chisq_200_235"] = (
            (
                scale
                * diff_range_3.values
                / (noise_range_3.values + config["zero_div_offset"])
            )
            ** 2
        ).sum(axis=1) / config["chisq_rescale"]
        return chisq_df

    return pd.Series(
        (
            (diff_df.values / (noise_df.values + config["zero_div_offset"]))
            ** 2
        ).sum(axis=1)
    )


def compute_chisq_std(
    _singles_df: pd.DataFrame, _binaries_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute standard deviation of wavelength chi-squared statistics if using Bardalez binning.

    :param _singles_df: DataFrame of single stars with chi-squared spectrum difference statistics.
    :param _binaries_df: DataFrame of binary stars with chi-squared spectrum difference statistics.
    :return:
    """
    _singles_std = (
        _singles_df[["chisq_095_135", "chisq_145_180", "chisq_200_235"]].std(
            axis=1
        )
        * config["chisq_rescale"]
    )
    _binaries_std = (
        _binaries_df[["chisq_095_135", "chisq_145_180", "chisq_200_235"]].std(
            axis=1
        )
        * config["chisq_rescale"]
    )
    return _singles_std, _binaries_std


def feature_engineering(
    _singles: pd.DataFrame,
    _binaries: pd.DataFrame,
    scale: Optional[float],
    _add_noise: bool,
    snr: bool,
    add_template_diffs: bool,
    chisq: bool,
    bardalez_chisq: bool,
    chisq_std: bool,
) -> pd.DataFrame:
    """
    Add noise to spectra, optionally calculate SNR for binning, add spectrum differences, compute
    chi-squared statistics with differences, and take the standard deviation of the ranges if using
    Bardalez wavelength binning.

    :param _singles: Input singles DataFrame
    :param _binaries: Input binaries DataFrame
    :param scale: Noise scaling (and chi-squared if using)
    :param _add_noise: Whether or not to add noise
    :param snr: Whether to calculate signal-to-noise ratio
    :param add_template_diffs: Whether to add in the difference spectra as a feature
    :param chisq: Whether to calculate the chi-squared statistic for spectra
    :param bardalez_chisq: Whether to use Bardalez binning for chi-squared calculation
    :param chisq_std: Whether to add standard deviation feature (if using Bardalez binning for
    chi-squared calculation)
    :return: Preprocessed DataFrame
    """
    if chisq_std or bardalez_chisq:
        assert (
            chisq
        ), "You must set chisq to True if you want to compute it with the Bardalez method."

    (
        _singles_type,
        _singles_flux,
        _singles_noise,
        _singles_diffs,
        _binaries_type,
        _binaries_flux,
        _binaries_noise,
        _binaries_diffs,
    ) = get_spectral_data(_singles, _binaries)

    # Add the noise
    if _add_noise:
        assert scale, "You must provide a scale to add noise."
        _singles = add_noise(_singles_flux, _singles_noise, scale)
        _binaries = add_noise(_binaries_flux, _binaries_noise, scale)

    # Convert from NP array to DataFrame
    _singles = pd.DataFrame(_singles, columns=_binaries_flux.columns)
    _binaries = pd.DataFrame(_binaries, columns=_binaries_flux.columns)

    # Compute signal-to-noise ratio (NOT TO USE AS A FEATURE, but binning for different models)
    if snr:
        assert (
            _add_noise and scale
        ), "You must provide values for _add_noise and scale to compute SNR."
        _singles["snr"] = compute_snr(_singles_flux, _singles_noise, scale)
        _binaries["snr"] = compute_snr(_binaries_flux, _binaries_noise, scale)

    # Add the difference
    if add_template_diffs:
        assert (
            _singles_diffs is not None and _binaries_diffs is not None
        ), "You must provide DataFrames for singles_diffs and binaries_diffs."
        _singles = pd.concat([_singles, _singles_diffs], axis=1)
        _binaries = pd.concat([_binaries, _binaries_diffs], axis=1)

    # Calculate chi-squared between difference spectra and noise
    if chisq:
        if bardalez_chisq:
            _singles = pd.concat(
                [
                    _singles,
                    compute_chisq(_singles_diffs, _singles_noise, scale, True),
                ],
                axis=1,
            )
            _binaries = pd.concat(
                [
                    _binaries,
                    compute_chisq(
                        _binaries_diffs, _binaries_noise, scale, True
                    ),
                ],
                axis=1,
            )
        else:
            print(
                compute_chisq(
                    _singles_diffs, _singles_noise, scale, bardalez=False
                )
            )
            _singles.loc[:, "chisq"] = compute_chisq(
                _singles_diffs, _singles_noise, scale, bardalez=False
            )
            _binaries["chisq"] = compute_chisq(
                _binaries_diffs, _binaries_noise, scale, bardalez=False
            )

        # Calculate the standard deviation between the chi-squared statistics at the three
        # wavelength ranges
        if chisq_std:
            assert bardalez_chisq, (
                "You must set bardalez_chisq to True if you want to compute the chisq standard "
                "deviation."
            )
            singles_std, binaries_std = compute_chisq_std(_singles, _binaries)
            _singles["chisq_std"] = singles_std
            _binaries["chisq_std"] = binaries_std

    _singles = pd.concat([_singles, _singles_type], axis=1)
    _binaries = pd.concat([_binaries, _binaries_type], axis=1)

    _singles[config["target_col"]] = 1
    _binaries[config["target_col"]] = 0
    df_preprocessed = (
        pd.concat([_binaries, _singles]).sample(frac=1).reset_index(drop=True)
    )
    return df_preprocessed
