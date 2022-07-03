from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_selection import feature_selection
from load_data import get_all_spectra
from resampling import oversample
from utils import load_config

config = load_config("config.yaml")


def train_test_val_split(
    _X: pd.DataFrame, _y: pd.Series
) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """Split the data into train, validation, and test sets."""
    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _X, _y, test_size=config["test_size"], random_state=1, stratify=_y
    )
    _X_train, _X_val, _y_train, _y_val = train_test_split(
        _X_train,
        _y_train,
        test_size=config["val_size"],
        random_state=1,
        stratify=_y_train,
    )
    return _X_train, _X_val, _X_test, _y_train, _y_val, _y_test


def baseline_classifier(
    classifier: str,
    _X_train: pd.DataFrame,
    _y_train: pd.Series,
    _X_test: pd.DataFrame,
    _y_test: pd.Series,
) -> Tuple[List, List]:
    """Fit a specified classifier out of SVC, KNN, and RF, and return the
    train and test predictions."""
    assert classifier in config["classifiers"]
    if classifier == "svc":  # Support vector classifier
        clf = make_pipeline(StandardScaler(), SVC())
    elif classifier == "knn":  # KNN classifier
        clf = KNeighborsClassifier(algorithm="kd_tree")  # KD Tree to speed up
    elif classifier == "rf":  # Random forest classifier
        clf = RandomForestClassifier()
    else:  # SGD Classifier
        clf = make_pipeline(
            StandardScaler(), SGDClassifier(loss="hinge")
        )  # Linear SVM
    clf.fit(_X_train, _y_train)
    return clf.predict(_X_train).tolist(), clf.predict(_X_test).tolist()


def eval_clf(y_true: List, y_pred: List) -> None:
    """Print a classification report from true values and predicted values."""
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    df = get_all_spectra()
    flux_df = df.drop(columns=config["spectral_type_col"])
    X = flux_df.drop(columns=config["target_col"])
    y = flux_df[config["target_col"]]
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(X, y)
    X_train, y_train = oversample("smote", X_train, y_train)
    print("Selecting features...")
    X_train, X_test = feature_selection("anova", X_train, y_train, X_test)
    print("Training classifier...")
    train_pred, test_pred = baseline_classifier(
        "rf", X_train, y_train, X_test, y_test
    )
    eval_clf(y_train, train_pred)
    eval_clf(y_test, test_pred)
