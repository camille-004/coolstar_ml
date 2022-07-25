"""Load the data with specified preprocessing/FE steps and test a classifier."""
from typing import Tuple

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

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


def random_undersample(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly under-sample the majority class (binaries) to balance the training dataset.

    :param _df: Input training DataFrame
    :return: Randomly under-sampled DataFrame
    """
    counts = _df[config["target_col"]].value_counts()
    print("Before random undersampling:")
    print(f"{counts[0]} binaries, {counts[1]} singles")
    undersample = RandomUnderSampler(sampling_strategy="majority")
    X = _df.drop(columns=config["target_col"])
    y = _df[config["target_col"]]
    X_under, y_under = undersample.fit_resample(X, y)
    _df = pd.concat([X_under, y_under], axis=1)
    counts = _df[config["target_col"]].value_counts()
    print("After random undersampling:")
    print(f"{counts[0]} binaries, {counts[1]} singles")
    return _df


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
    X = _df.drop(columns=config["target_col"])
    y = _df[config["target_col"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], stratify=y
    )

    if undersample:
        df_train = pd.concat([X_train, y_train], axis=1)
        df_train = random_undersample(df_train)
        X_train = df_train.drop(columns=config["target_col"])
        y_train = df_train[config["target_col"]]

    return X_train, X_test, y_train, y_test


# def test_rf(X_train, y_train, X_test, y_test):


# %%
df = prepare_data(
    f_name=config["fp_july15"],
    binaries_filter=(17, 24, 25, 31),
    singles_filter=(17, 24),
    scale=config["noise_scale"],
    _add_noise=True,
    snr=True,
    template_diffs=True,
    chisq=True,
    bartolez_chisq=True,
    chisq_std=True,
)
