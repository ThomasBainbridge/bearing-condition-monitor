from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLUMNS = [
    "mean",
    "std",
    "rms",
    "peak_to_peak",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "skewness",
    "kurtosis",
    "dominant_frequency",
]


def add_load_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["load_id"] = df["source_file"].str.extract(r"_(\d)\.mat$").astype(int)
    return df


def split_by_load(
    df: pd.DataFrame,
    train_loads: list[int],
    test_loads: list[int],
    feature_columns: list[str] | None = None,
):
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    df = add_load_id_column(df)

    train_df = df[df["load_id"].isin(train_loads)].copy()
    test_df = df[df["load_id"].isin(test_loads)].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["label"]

    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    return train_df, test_df, X_train, X_test, y_train, y_test


def build_logistic_regression_model(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])


def build_random_forest_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
    )


def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def save_model(model, file_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path = models_dir / file_name
    joblib.dump(model, output_path)
    return output_path