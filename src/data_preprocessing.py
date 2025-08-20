from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, List, Dict

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

TARGET_COLUMN = "target"


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Standardize column names if common variants are present
    rename_map = {
        "trestbps": "trestbps", "resting_bp_s": "trestbps",
        "chol": "chol", "cholesterol": "chol",
        "thalach": "thalach", "max_hr": "thalach",
        "oldpeak": "oldpeak", "st_depression": "oldpeak",
        "cp": "cp", "chest_pain_type": "cp",
        "fbs": "fbs", "fasting_blood_sugar": "fbs",
        "restecg": "restecg", "rest_ecg": "restecg",
        "exang": "exang", "exercise_angina": "exang",
        "slope": "slope", "st_slope": "slope",
        "ca": "ca", "num_vessels": "ca",
        "thal": "thal", "thalassemia": "thal",
        "sex": "sex",
        "age": "age",
        "target": "target",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y
