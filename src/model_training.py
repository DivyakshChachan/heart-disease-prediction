from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from .data_preprocessing import load_dataset, get_feature_lists, build_preprocessor, split_X_y

RANDOM_STATE = 42


def train_models(csv_path: str, models_dir: str) -> Dict[str, Any]:
    df = load_dataset(csv_path)
    numeric, categorical = get_feature_lists(df)
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric, categorical)

    # Logistic Regression pipeline and grid
    lr_pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=1, random_state=RANDOM_STATE))
    ])
    lr_grid = {
        "clf__C": [0.1, 1.0, 5.0, 10.0],
        "clf__penalty": ["l2"],
    }

    # XGBoost pipeline and grid
    xgb_pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
            reg_lambda=1.0,
        ))
    ])
    xgb_grid = {
        "clf__max_depth": [3, 4, 5],
        "clf__n_estimators": [300, 500],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def fit_and_score(name: str, pipeline: Pipeline, grid: Dict[str, Any]) -> Tuple[Any, float]:
        logger.info(f"Tuning {name}...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"{name} best AUC: {auc:.3f}")
        return best_model, auc

    lr_model, lr_auc = fit_and_score("LogisticRegression", lr_pipeline, lr_grid)
    xgb_model, xgb_auc = fit_and_score("XGBoost", xgb_pipeline, xgb_grid)

    os.makedirs(models_dir, exist_ok=True)
    lr_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    joblib.dump(lr_model, lr_path)
    joblib.dump(xgb_model, xgb_path)

    return {
        "lr_model_path": lr_path,
        "xgb_model_path": xgb_path,
        "lr_auc": lr_auc,
        "xgb_auc": xgb_auc,
    }
