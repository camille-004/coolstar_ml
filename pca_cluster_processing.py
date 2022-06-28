"""Perform PCA and cluster-based upsampling on the single star spectra."""
# %%
import numpy as np
import pandas as pd
from keras import layers, models
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import binary, get_master_df, single
from utils import load_config

config = load_config("config.yaml")
PCA_N_COMPONENTS = config["pca_n_components"]
N_CLUSTERS = config["kmeans_n_clusters"]
TARGET_COL = config["target_col"]
SPECTRAL_TYPE_COL = config["spectral_type_col"]
N_SUBSAMPLE = config["subsample"]
N_FOLDS = config["n_folds"]


df = get_master_df()
X = df.drop(columns=[TARGET_COL])  # , SPECTRAL_TYPE_COL])
y = df[TARGET_COL]

skf = StratifiedKFold(n_splits=N_FOLDS)


def upsample(single_flux: pd.DataFrame) -> pd.DataFrame:
    """Upsample the single star spectra and return a DataFrame of only new
    samples."""
    _n = len(binary) - len(single)
    upsampled_groups = []
    upsample_group_sizes = (
        _n * single_flux.groupby("cluster").size() / len(single_flux)
    )
    upsample_group_sizes = upsample_group_sizes.astype(int)

    for i in range(N_CLUSTERS):
        upsampled_seqs = []
        cluster_flux = single_flux[single_flux["cluster"] == i]
        upsample_num = upsample_group_sizes.iloc[i]

        for _ in range(upsample_num):
            sample = cluster_flux.sample(N_SUBSAMPLE)
            result_seq = sample.mean(axis=0)
            upsampled_seqs.append(result_seq)

        upsampled_groups.append(pd.DataFrame(upsampled_seqs))

    upsample_seq_df = pd.concat(upsampled_groups).drop(columns="cluster")
    upsample_seq_df[TARGET_COL] = 0
    return (
        upsample_seq_df.drop(columns=TARGET_COL),
        upsample_seq_df[TARGET_COL],
    )


def build_model(X: pd.DataFrame) -> models.Sequential:
    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            input_shape=(X.shape[1], X.shape[2]),
        )
    )
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(layers.Dropout(0.5))
    # model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics="accuracy",
    )

    return model


def train_model(
    _model: models.Sequential,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fold_n: int,
) -> float:
    callback = EarlyStopping(
        monitor="accuracy", mode="min", min_delta=0.01, patience=4
    )
    _model.fit(X_train, y_train, epochs=10, callbacks=[callback])
    y_pred = _model.predict(X_test)
    y_pred = (y_pred > 0.5 * 1.0).ravel()

    score = accuracy_score(y_test, y_pred)
    print(f"Fold {fold_n} - Test accuracy: {score}")
    return score


fold_n = 1
score_folds = []

for train_idx, test_idx in skf.split(X, y):
    print(f"TRAIN INDEX: {train_idx}, TEST INDEX: {test_idx}")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    df_train = pd.concat([X_train, y_train], axis=1)
    X_single_train = df_train[df_train[TARGET_COL] == 0].drop(
        columns=TARGET_COL
    )
    y_single_train = df_train[df_train[TARGET_COL] == 0][TARGET_COL]

    df_test = pd.concat([X_test, y_test], axis=1)
    X_single_test = df_test[df_test[TARGET_COL] == 0].drop(columns=TARGET_COL)
    y_single_test = df_test[df_test[TARGET_COL] == 0][TARGET_COL]

    pca = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_N_COMPONENTS)),
        ]
    )

    X_train_pca = pd.DataFrame(
        pca.fit_transform(X_single_train, y_single_train)
    )
    X_test_pca = pd.DataFrame(pca.transform(X_single_test))

    km = KMeans(n_clusters=N_CLUSTERS)
    X_single_train["cluster"] = km.fit_predict(X_train_pca)
    X_single_test["cluster"] = km.predict(X_test_pca)
    upsampled_flux_single, upsampled_y_single_train = upsample(X_single_train)
    upsampled_flux_single_test, upsampled_y_single_test = upsample(
        X_single_test
    )

    X_train = pd.concat([X_train, upsampled_flux_single])

    y_train = pd.concat([y_train, upsampled_y_single_train])

    X_test = pd.concat([X_test, upsampled_flux_single_test])

    y_test = pd.concat([y_test, upsampled_y_single_test])

    df_train = (
        pd.concat([X_train, y_train], axis=1)
        .sample(frac=1)
        .reset_index(drop=True)
    )
    df_test = (
        pd.concat([X_test, y_test], axis=1)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    X_train = df_train.drop(columns=TARGET_COL)
    X_test = df_test.drop(columns=TARGET_COL)
    y_train = df_train[TARGET_COL]
    y_test = df_test[TARGET_COL]

    fs = SelectKBest(score_func=f_classif, k=5)
    X_train = fs.fit_transform(X=X_train, y=y_train)
    X_test = fs.transform(X=X_test)

    # X_train_np = X_train.values
    X_train_reshaped = X_train[:, :, np.newaxis]

    # X_test_np = X_test.values
    X_test_reshaped = X_test[:, :, np.newaxis]

    model = build_model(X_train_reshaped)
    score = train_model(
        model, X_train_reshaped, X_test_reshaped, y_train, y_test, fold_n
    )
    score_folds.append(score)
    fold_n += 1

print(f"\nMean Stratified K-Fold CV Accuracy: {np.mean(score_folds)}")
