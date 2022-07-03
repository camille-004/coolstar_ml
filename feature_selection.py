"""Perform feature selection on spectra."""
from typing import Tuple

import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

from utils import load_config

config = load_config("config.yaml")


def feature_selection(
    method: str,
    _X_train: pd.DataFrame,
    _y_train: pd.Series,
    _X_val: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform specified feature selection on training data and validation
    data, if provided."""
    assert method in config["feature_selection_methods"]
    _k = config["feature_selection_k"]
    if method == "rfe":  # Recursive feature elimination
        rfe = RFE(
            estimator=DecisionTreeClassifier(),
            n_features_to_select=_k,
        )
        rfe.fit(_X_train, _y_train)
        _X_train_new = rfe.transform(_X_train)
        if _X_val is not None:
            _X_val_new = rfe.transform(_X_val)
    elif method == "fisher":  # Fisher score
        score = fisher_score.fisher_score(
            _X_train.values, _y_train.values, mode="index"
        )
        _X_train_new = _X_train.iloc[:, score[0:_k]]
        if _X_val is not None:
            _X_val_new = _X_val.iloc[:, score[0:_k]]
    else:  # ANOVA
        selector = SelectKBest(f_classif, k=_k)
        _X_train_new = selector.fit_transform(_X_train, _y_train)
        if _X_val is not None:
            _X_val_new = selector.transform(_X_val)

    if _X_val is not None:
        return _X_train_new, _X_val_new
    return _X_train_new
