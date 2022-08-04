"""Feature engineering module (add noise, compute chi-squared statistic, etc.)"""
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd

from preprocess_data import get_spectral_data
from utils import load_config

config = load_config("config.yaml")

# To get the chi-squared statistic according to Bardalez et. al.
wavegrids = pd.read_hdf(
    os.path.join(config["data_dir"], config["fp_aug3"]),
    key=config["wavegrid_key"],
)
range_1 = wavegrids[(wavegrids >= 0.95) & (wavegrids <= 1.35)].dropna().index
range_2 = wavegrids[(wavegrids >= 1.45) & (wavegrids <= 1.80)].dropna().index
range_3 = wavegrids[(wavegrids >= 2.00) & (wavegrids <= 2.35)].dropna().index


def add_noise(
    flux_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    scale_range: Tuple[float, float] = (
        config["noise_scale_low"],
        config["noise_scale_high"],
    ),
) -> Union[np.ndarray, int, float, complex]:
    """
    Add Gaussian noise to each star in input DataFrame. Multiply uncertainty by a desired scale.

    :param flux_df: DataFrame of flux measurements
    :param noise_df: DataFrame of uncertainties
    :return: Flux measurements with noise as a NumPy array
    """
    noise_low, noise_high = scale_range
    scale = pd.Series(
        np.random.uniform(noise_low, noise_high, size=len(noise_df))
    )
    with_noise = np.random.normal(flux_df, noise_df.abs().mul(scale, axis=0))
    with_noise = pd.DataFrame(with_noise, columns=flux_df.columns)
    with_noise["scale"] = scale
    return with_noise


def compute_snr(flux_df: pd.DataFrame, noise_df: pd.DataFrame) -> np.ndarray:
    """
    Compute signal-to-noise ratio for each spectrum in input DataFrame. Note: SNR is not meant to
    be used as a feature.

    :param flux_df: DataFrame of flux measurements
    :param noise_df: DataFrame of uncertainties
    :return: SNR for each spectrum
    """
    assert (
        "scale" in noise_df.columns
    ), "A scale column must be used to compute SNR."
    scale = noise_df["scale"]
    return np.nanmedian(
        flux_df.values
        / (
            (
                noise_df.drop(columns="scale").abs()
                + config["zero_div_offset"]
            ).mul(scale, axis=0)
        ).values,
        axis=1,
    )


def compute_chisq(
    diff_df: pd.DataFrame, noise_df: pd.DataFrame, new_version: bool
) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute chi-squared statistics on same wavelengths as in Bardalez et. al.

    :param diff_df: DataFrame of spectrum difference with best-fit template
    :param noise_df: DataFrame of uncertainties
    :return: DataFrame of chi-squared statistics for three wavelength ranges
    """
    assert (
        "scale" in noise_df.columns
    ), "A scale column must be used to compute the chi-squared statistic."
    if new_version:
        diff_range_1 = diff_df[
            [config["diff_prefix"] + str(i) for i in range_1]
        ]
        diff_range_2 = diff_df[
            [config["diff_prefix"] + str(i) for i in range_2]
        ]
        diff_range_3 = diff_df[
            [config["diff_prefix"] + str(i) for i in range_3]
        ]
        noise_range_1 = noise_df[
            [config["noise_prefix"] + str(i) for i in range_1]
        ]
        noise_range_2 = noise_df[
            [config["noise_prefix"] + str(i) for i in range_2]
        ]
        noise_range_3 = noise_df[
            [config["noise_prefix"] + str(i) for i in range_3]
        ]
    else:
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

    diffs = pd.concat([diff_range_1, diff_range_2, diff_range_3], axis=1)
    noises = pd.concat([noise_range_1, noise_range_2, noise_range_3], axis=1)
    scale = noise_df["scale"]

    return pd.Series(
        (
            (
                diffs.mul(scale, axis=0).values
                / (
                    noises.mul(scale, axis=0).values
                    + config["zero_div_offset"]
                )
            )
            ** 2
        ).sum(axis=1)
    )


def feature_engineering(
    _singles: pd.DataFrame,
    _binaries: pd.DataFrame,
    _add_noise: bool,
    snr: bool,
    add_template_diffs: bool,
    chisq: bool,
    new_version: bool,
) -> pd.DataFrame:
    """
    Add noise to spectra, optionally calculate SNR for binning, add spectrum differences, compute
    chi-squared statistics with differences.

    :param _singles: Input singles DataFrame
    :param _binaries: Input binaries DataFrame
    :param _add_noise: Whether or not to add noise
    :param snr: Whether to calculate signal-to-noise ratio
    :param add_template_diffs: Whether to add in the difference spectra as a feature
    :param chisq: Whether to calculate the chi-squared statistic for spectra
    chi-squared calculation)
    :param new_version: Whether we are using the August 3 dataset
    :return: Preprocessed DataFrame
    """
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
        _singles = add_noise(_singles_flux, _singles_noise)
        _binaries = add_noise(_binaries_flux, _binaries_noise)

    # Add the difference
    if add_template_diffs:
        assert (
            _singles_diffs is not None and _binaries_diffs is not None
        ), "You must provide DataFrames for singles_diffs and binaries_diffs."
        _singles = pd.concat([_singles, _singles_diffs], axis=1)
        _binaries = pd.concat([_binaries, _binaries_diffs], axis=1)

    # Compute signal-to-noise ratio (NOT TO USE AS A FEATURE, but binning for different models)
    if snr:
        assert (
            _add_noise
        ), "You must provide values for _add_noise to compute SNR."
        _singles["snr"] = compute_snr(
            _singles_flux,
            pd.concat([_singles_noise, _singles["scale"]], axis=1),
        )
        _binaries["snr"] = compute_snr(
            _binaries_flux,
            pd.concat([_binaries_noise, _binaries["scale"]], axis=1),
        )

    # Calculate chi-squared between difference spectra and noise
    if chisq:
        assert (
            _add_noise
        ), "You must provide values for _add_noise to compute the chi-squared statistic."
        _singles["chisq"] = compute_chisq(
            _singles_diffs,
            pd.concat([_singles_noise, _singles["scale"]], axis=1),
            new_version=new_version,
        )
        _binaries["chisq"] = compute_chisq(
            _binaries_diffs,
            pd.concat([_binaries_noise, _binaries["scale"]], axis=1),
            new_version=new_version,
        )

    _singles = pd.concat([_singles, _singles_type], axis=1)
    _binaries = pd.concat([_binaries, _binaries_type], axis=1)

    _singles[config["target_col"]] = 1
    _binaries[config["target_col"]] = 0
    df_preprocessed = (
        pd.concat([_binaries, _singles]).sample(frac=1).reset_index(drop=True)
    )

    if "scale" in df_preprocessed.columns:
        df_preprocessed = df_preprocessed.drop(columns="scale")
    return df_preprocessed
