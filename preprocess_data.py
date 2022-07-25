"""Load and clean data (works for July 15th version)."""
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

from utils import load_config

config = load_config("config.yaml")


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
