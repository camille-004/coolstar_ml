"""Functions to undersample or oversample spectra."""
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.decomposition import PCA

from load_data import get_all_spectra
from utils import load_config

config = load_config("config.yaml")


def get_spectral_type(
    _X: pd.DataFrame,
    _y: pd.Series,
    lower: int,
    upper: int,
    drop_type: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Get the data of starts between two spectral types, inclusive."""
    spectral_type_col = _X[config["spectral_type_col"]]
    assert (
        lower in spectral_type_col.unique()
        and upper in spectral_type_col.unique()
    )
    _X = _X[(spectral_type_col >= lower) & (spectral_type_col <= upper)]
    idx = _X.index
    if drop_type:
        _X = _X.drop(columns=config["spectral_type_col"])
    return _X, _y.iloc[idx]


def cluster_undersample(
    _X: pd.DataFrame, _y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """Use clustering to undersample training data."""
    cc = ClusterCentroids()
    _X_res, _y_res = cc.fit_resample(_X, _y)
    print(f"New class distribution: {_y_res.value_counts().tolist()}")
    return _X_res, _y_res


def oversample(
    method: str, _X: pd.DataFrame, _y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    assert method in config["oversample_methods"]
    if method == "adasyn":  # ADASYN
        over = ADASYN()
    else:  # SMOTE
        over = SMOTE()
    _X_res, _y_res = over.fit_resample(_X, _y)
    print(f"New class distribution: {_y_res.value_counts().tolist()}")
    return _X_res, _y_res


def viz_pca(_X: pd.DataFrame, _y: pd.Series, file_name: str = "") -> None:
    """Visualize the sampling with a PCA embedding."""
    pca = PCA(n_components=2)
    z = pca.fit_transform(_X)
    pca_df = pd.DataFrame()
    pca_df["y"] = _y
    pca_df["comp-1"] = z[:, 0]
    pca_df["comp-2"] = z[:, 1]
    pca_df = pca_df.sort_values(by="y", ascending=False)

    ax = sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue=pca_df["y"].tolist(),
        palette=sns.color_palette("hls", 2),
        data=pca_df,
    )
    ax.set(title="Flux Data PCA Projection")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Single", "Binary"], loc="upper right")
    if file_name != "":
        plt.savefig(
            os.path.join(config["plots_dir"], file_name),
            dpi=config["savefig_dpi"],
        )
    plt.show()


if __name__ == "__main__":
    X, y = get_all_spectra()
    X, y = get_spectral_type(X, y, 39, 40, False)
    viz_pca(X, y, "39_44_pca.png")
