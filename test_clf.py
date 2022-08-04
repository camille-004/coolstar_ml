"""Load the data with specified preprocessing/FE steps and test a classifier."""
import os
from typing import Dict, Optional, Tuple

import pandas as pd

# from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from feature_engineering import feature_engineering
from preprocess_data import (
    get_binaries_single_dfs_aug_3,
    get_binary_single_dfs,
    type_to_num,
)
from utils import load_config

config = load_config("config.yaml")


def filter_binaries(
    _binaries_df: pd.DataFrame,
    primary_min: int = config["primary_min"],
    primary_max: int = config["primary_max"],
    secondary_min: int = config["secondary_min"],
    secondary_max: int = config["secondary_max"],
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


def prepare_data(
    f_name: str,
    _add_noise: bool,
    snr: bool,
    template_diffs: bool,
    chisq: bool,
    binaries_filter: [Tuple[str]] = (
        config["primary_min"],
        config["primary_max"],
        config["primary_min"],
        config["primary_max"],
    ),
    singles_filter: [Tuple[str]] = (
        config["single_min"],
        config["single_max"],
    ),
    new_version: bool = True,
) -> pd.DataFrame:
    """
    Load the fully cleaned and preprocessed dataset.

    :param f_name: Filename of raw dataset
    :param binaries_filter: Conditions for filtering binaries by primary and secondary type,
    :param singles_filter: Conditions for filtering singles by spectral type
    :param _add_noise: Whether to add noise
    :param snr: Whether to calculate signal-to-noise ratio
    :param template_diffs: Whether to add in the difference spectra as a feature
    :param chisq: Whether to calculate the chi-squared statistic for spectra
    :param new_version: Whether to load the August 3 version of the dataset
    :return: Prepared DataFrame
    """
    assert f_name in os.listdir(config["data_dir"])
    if new_version:
        _singles, _binaries = get_binaries_single_dfs_aug_3(
            os.path.join(config["data_dir"], f_name)
        )
    else:
        _singles, _binaries = get_binary_single_dfs(
            os.path.join(config["data_dir"], f_name)
        )

    binaries_filter = map(type_to_num, binaries_filter)
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
    _binaries = _binaries.drop(
        columns=[config["primary_type_col"], config["secondary_type_col"]]
    )
    singles_filter = map(type_to_num, singles_filter)
    type_min, type_max = singles_filter
    _singles = filter_singles(_singles, type_min, type_max)
    _df = feature_engineering(
        _singles,
        _binaries,
        _add_noise=_add_noise,
        snr=snr,
        add_template_diffs=template_diffs,
        chisq=chisq,
        new_version=new_version,
    )
    print(f"DataFrame shape: {_df.shape}")
    return _df


# def random_undersample(_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Randomly under-sample the majority class (binaries) to balance the training dataset.
#
#     :param _df: Input training DataFrame
#     :return: Randomly under-sampled DataFrame
#     """
#     counts = _df[config["target_col"]].value_counts()
#     print("Before random undersampling:")
#     print(f"{counts[0]} binaries, {counts[1]} singles")
#     undersample = RandomUnderSampler(sampling_strategy="majority")
#     X = _df.drop(columns=config["target_col"])
#     y = _df[config["target_col"]]
#     X_under, y_under = undersample.fit_resample(X, y)
#     _df = pd.concat([X_under, y_under], axis=1)
#     counts = _df[config["target_col"]].value_counts()
#     print("After random undersampling:")
#     print(f"{counts[0]} binaries, {counts[1]} singles")
#     return _df


def split_data(
    _df: pd.DataFrame, undersample: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into training and test sets.

    :param _df: Input DataFrame to split into training and test sets
    :param undersample: Whether or not to randomly undersample the majority class in the training
    set
    :return: Training and testing X and y as a tuple
    """
    _df = _df.dropna()
    X = _df.drop(columns=config["target_col"])
    y = _df[config["target_col"]]
    _X_train, _X_test, _y_train, _y_test = train_test_split(
        X, y, test_size=config["test_size"], stratify=y, random_state=42
    )

    if undersample:
        df_train = pd.concat([_X_train, _y_train], axis=1)
        # df_train = random_undersample(df_train)
        _X_train = df_train.drop(columns=config["target_col"])
        _y_train = df_train[config["target_col"]]

    return _X_train, _X_test, _y_train, _y_test


def test_rf(
    _X_train: pd.DataFrame,
    _X_test: pd.DataFrame,
    _y_train: pd.Series,
    _y_test: pd.Series,
    report: bool = True,
    rf_params: Optional[Dict] = None,
) -> Tuple[RandomForestClassifier, float, float, float]:
    """
    Fit a RandomForestClassifier on the training data and print the classification report on the
    training and test sets.
    :param _X_train: Input training features
    :param _X_test: Input testing features
    :param _y_train: Input training label
    :param _y_test: Input testing label
    :param report: Whether to print the train and test classification_reports
    :param rf_params: Dictionary of hyperparameters for the RandomForestClassifier
    :return: Test precision, recall, and F1-score
    """
    if rf_params:
        clf = RandomForestClassifier(**rf_params)
    else:
        clf = RandomForestClassifier()
    clf.fit(_X_train, _y_train)
    pred_train = clf.predict(_X_train)
    pred_test = clf.predict(_X_test)

    if report:
        print("Train:")
        print(classification_report(_y_train, pred_train) + "\n")
        print("Test:")
        print(classification_report(_y_test, pred_test))

    return (
        clf,
        precision_score(_y_test, pred_test),
        recall_score(_y_test, pred_test),
        f1_score(_y_test, pred_test),
    )


df = prepare_data(
    config["fp_aug3"],
    _add_noise=True,
    snr=True,
    template_diffs=True,
    chisq=True,
    new_version=True,
)
