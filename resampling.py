# pylint: disable=C0103
"""Functions to undersample or oversample spectra."""
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import ClusterCentroids
from sklearn.decomposition import PCA

from load_data import get_all_spectra
from utils import load_config

config = load_config("config.yaml")


def cluster_undersample(
    _X: pd.DataFrame, _y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """Use clustering to undersample training data."""
    cc = ClusterCentroids()
    _X_res, _y_res = cc.fit_resample(_X, _y)
    print(f"New class distribution: {_y_res.value_counts().tolist()}")
    return _X_res, _y_res


def viz_pca(_X: pd.DataFrame, _y: pd.Series) -> None:
    """Visualize the sampling with a PCA embedding."""
    pca = PCA(n_components=2)
    z = pca.fit_transform(_X)
    pca_df = pd.DataFrame()
    pca_df["y"] = _y
    pca_df["comp-1"] = z[:, 0]
    pca_df["comp-2"] = z[:, 1]

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
    plt.show()


if __name__ == "__main__":
    df = get_all_spectra()
    flux_df = df.drop(columns=config["spectral_type_col"])
    X = flux_df.drop(columns=config["target_col"])
    y = flux_df[config["target_col"]]
    viz_pca(X, y)
    X_res, y_res = cluster_undersample(X, y)
    viz_pca(X_res, y_res)
