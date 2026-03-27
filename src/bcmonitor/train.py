from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from bcmonitor.features import get_feature_columns


BASELINE_FEATURE_COLUMNS = get_feature_columns("baseline")
ENHANCED_FEATURE_COLUMNS = get_feature_columns("enhanced")
DEFAULT_FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS.copy()


def add_load_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "load_id" in df.columns and df["load_id"].notna().all():
        df["load_id"] = df["load_id"].astype(int)
        return df

    extracted_loads = df["source_file"].str.extract(r"_(\d)\.mat$").astype(int)
    df["load_id"] = extracted_loads.iloc[:, 0]
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


def build_xgboost_model(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )


def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def fit_xgboost_model(model, X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    model.fit(X_train, y_train_encoded)
    return model, label_encoder


def predict_xgboost(model, label_encoder, X_test):
    y_pred_encoded = model.predict(X_test)
    return label_encoder.inverse_transform(y_pred_encoded)


def save_model(model, file_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    output_path = models_dir / file_name
    joblib.dump(model, output_path)
    return output_path