"""Load and preprocess data (works for July 15th version)."""
import os
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

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


def type_to_num(val: str) -> Union[Union[int, str], Any]:
    """
    Helper function to map spectral types to numbers.

    :param val: Input spectral type
    :return: Output number
    """
    type_map = {"M": 10, "L": 20, "T": 30}
    if isinstance(val, str):
        return int(val[1]) + type_map[val[0]]
    return val


def get_binary_single_dfs(_fp: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, merge, and deduplicate data. Meant to work on July 15th version.

    :param _fp: path of data file
    :return: Merged single and binary DataFrames
    """
    _singles_flux = pd.read_hdf(_fp, key=config["singles_flux_key"])
    _binaries_flux = pd.read_hdf(_fp, key=config["binaries_flux_key"])

    _singles_noise = pd.read_hdf(_fp, key=config["singles_noise_key"])
    _binaries_noise = pd.read_hdf(_fp, key=config["binaries_noise_key"])

    _singles_difference_spectrum = pd.read_hdf(
        _fp, key=config["singles_diff_key"]
    )
    _binaries_difference_spectrum = pd.read_hdf(
        _fp, key=config["binaries_diff_key"]
    )

    _singles_flux_dedup = _singles_flux.drop_duplicates(
        subset=config["merge_key"]
    )
    _singles_noise_dedup = _singles_noise.drop_duplicates(
        subset=config["merge_key"]
    )
    _singles_difference_spectrum_dedup = (
        _singles_difference_spectrum.drop_duplicates(
            subset=config["merge_key"]
        )
    )

    _singles_merged = _singles_noise_dedup.merge(
        _singles_difference_spectrum_dedup,
        on=config["merge_key"],
        suffixes=(config["noise_suffix"], config["diff_suffix"]),
    )
    _singles_merged = _singles_merged.merge(
        _singles_flux_dedup, on=config["merge_key"]
    )
    _singles_merged = _singles_merged.drop(
        columns=["spectral_type_noise", "spectral_type_diff"]
    )

    _binaries_merged = _binaries_noise.merge(
        _binaries_difference_spectrum,
        on=config["merge_key"],
        suffixes=(config["noise_suffix"], config["diff_suffix"]),
    )
    _binaries_merged = _binaries_merged.merge(
        _binaries_flux, on=config["merge_key"]
    )

    _binaries_merged = _binaries_merged.rename(
        columns={config["binary_rename_col"]: config["spectral_type_col"]}
    )
    _binaries_merged = _binaries_merged.drop(columns=config["merge_key"])

    return _singles_merged, _binaries_merged


def filter_binaries(
    _binaries_df: pd.DataFrame,
    primary_min: int,
    primary_max: int,
    secondary_min: int,
    secondary_max: int,
) -> pd.DataFrame:
    """
    Filter binary stars by primary and secondary spectral types.

    :param _binaries_df: DataFrame containing binary stars with spectral types
    :param primary_min: Minimum desired primary type
    :param primary_max: Maximum desired primary type
    :param secondary_min: Minimum desired secondary type
    :param secondary_max: Maximum desire secondary type
    :return: Filtered binary DataFrame
    """
    return _binaries_df[
        (_binaries_df[config["primary_type_col"]] >= primary_min)
        & (_binaries_df[config["primary_type_col"]] <= primary_max)
        & (_binaries_df[config["secondary_type_col"]] >= secondary_min)
        & (_binaries_df[config["secondary_type_col"]] <= secondary_max)
    ]


def filter_singles(
    _singles_df: pd.DataFrame, min_type: int, max_type: int
) -> pd.DataFrame:
    """
    Filter single stars by spectral type.

    :param _singles_df: DataFrame containing single stars with spectral type
    :param min_type: Minimum desired spectral type
    :param max_type: Maximum desired spectral type
    :return: Filtered singles DataFrame
    """
    _singles_df[config["spectral_type_col"]] = _singles_df[
        config["spectral_type_col"]
    ].apply(type_to_num)

    return _singles_df[
        (_singles_df[config["spectral_type_col"]] >= min_type)
        & (_singles_df[config["spectral_type_col"]] <= max_type)
    ]


def get_spectral_data(
    _singles_df: pd.DataFrame, _binaries_df: pd.DataFrame
) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
    """
    Get different measurements from merged and deduplicated data.

    :param _singles_df: DataFrame of single stars
    :param _binaries_df: DataFrame of binary stars
    :return: Spectral type, flux, noise, difference spectrum DataFrames for
    binaries and singles
    """
    _singles_type = _singles_df[config["spectral_type_col"]]
    if _singles_type.dtypes in (object, np.object_):
        _singles_type = _singles_type.apply(type_to_num)

    _singles_flux = _singles_df.loc[
        :, ~_singles_df.columns.str.contains("noise|diff|type|name")
    ]
    _singles_noise = _singles_df.loc[
        :, _singles_df.columns.str.contains("noise")
    ]
    _singles_diff = _singles_df.loc[
        :, _singles_df.columns.str.contains("diff")
    ]

    # Assuming binaries are already filtered by primary and secondary type groupings
    _binaries_type = _binaries_df[config["spectral_type_col"]]
    _binaries_flux = _binaries_df.loc[
        :, ~_binaries_df.columns.str.contains("noise|diff|type|name")
    ]
    _binaries_noise = _binaries_df.loc[
        :, _binaries_df.columns.str.contains("noise")
    ]
    _binaries_diff = _binaries_df.loc[
        :, _binaries_df.columns.str.contains("diff")
    ]

    return (
        _singles_type,
        _singles_flux,
        _singles_noise,
        _singles_diff,
        _binaries_type,
        _binaries_flux,
        _binaries_noise,
        _binaries_diff,
    )


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
    bartolez: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute chi-squared statistics on same wavelengths as in Bartolez et. al. or on all wavelengths

    :param diff_df: DataFrame of spectrum difference with best-fit template
    :param noise_df: DataFrame of uncertainties
    :param scale: Uncertainty scaling factor
    :param bartolez: Whether to use wavelength binning from Bartolez et. al.
    :return: DataFrame of chi-squared statistics for three wavelength ranges
    """
    if bartolez:
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
    Compute standard deviation of wavelength chi-squared statistics if using Bartolez binning.

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
    bartolez_chisq: bool,
    chisq_std: bool,
) -> pd.DataFrame:
    """
    Add noise to spectra, optionally calculate SNR for binning, add spectrum differences, compute
    chi-squared statistics with differences, and take the standard deviation of the ranges if using
    Bartolez wavelength binning.

    :param _singles: Input singles DataFrame
    :param _binaries: Input binaries DataFrame
    :param scale: Noise scaling (and chi-squared if using)
    :param _add_noise: Whether or not to add noise
    :param snr: Whether to calculate signal-to-noise ratio
    :param add_template_diffs: Whether to add in the difference spectra as a feature
    :param chisq: Whether to calculate the chi-squared statistic for spectra
    :param bartolez_chisq: Whether to use Bartolez binning for chi-squared calculation
    :param chisq_std: Whether to add standard deviation feature (if using Bartolez binning for
    chi-squared calculation)
    :return: Preprocessed DataFrame
    """
    if chisq_std or bartolez_chisq:
        assert chisq

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
        assert scale
        _singles = add_noise(_singles_flux, _singles_noise, scale)
        _binaries = add_noise(_binaries_flux, _binaries_noise, scale)

    # Convert from NP array to DataFrame
    _singles = pd.DataFrame(_singles, columns=_binaries_flux.columns)
    _binaries = pd.DataFrame(_binaries, columns=_binaries_flux.columns)

    # Compute signal-to-noise ratio (NOT TO USE AS A FEATURE, but binning for different models)
    if snr:
        assert _add_noise and scale
        _singles["snr"] = compute_snr(_singles_flux, _singles_noise, scale)
        _binaries["snr"] = compute_snr(_binaries_flux, _binaries_noise, scale)

    # Add the difference
    if add_template_diffs:
        assert _binaries_diffs is not None and _singles_diffs is not None
        _singles = pd.concat([_singles, _singles_diffs], axis=1)
        _binaries = pd.concat([_binaries, _binaries_diffs], axis=1)

    # Calculate chi-squared between difference spectra and noise
    if chisq:
        if bartolez_chisq:
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
            _singles["chisq"] = compute_chisq(
                _singles_diffs, _singles_noise, scale
            )
            _binaries["chisq"] = compute_chisq(
                _binaries_diffs, _binaries_noise, scale
            )

        # Calculate the standard deviation between the chi-squared statistics at the three
        # wavelength ranges
        if chisq_std:
            assert bartolez_chisq
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


def prepare_data(
    f_name: str,
    binaries_filter: Optional[Tuple[int, int, int, int]],
    singles_filter: Optional[Tuple[int, int]],
    scale: float,
    _add_noise: bool,
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
    :param _add_noise: Whether to add noise
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
        _add_noise=_add_noise,
        snr=snr,
        add_template_diffs=template_diffs,
        chisq=chisq,
        bartolez_chisq=bartolez_chisq,
        chisq_std=chisq_std,
    )
    print(f"DataFrame shape: {_df.shape}")
    return _df
