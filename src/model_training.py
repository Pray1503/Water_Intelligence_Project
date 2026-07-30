"""
Stage 9: Model Training.

Trains machine learning models to predict future Water Stress Index (WSI)
at 7, 15, and 30-day horizons.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns_to_exclude: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target y by dropping NaNs in the target column
    and removing non-feature variables (like dates and alternative targets).
    """
    if feature_columns_to_exclude is None:
        feature_columns_to_exclude = [
            "Date",
            "District",
            "wsi_lead_7",
            "wsi_lead_15",
            "wsi_lead_30"
        ]

    # Drop rows where the target is missing (occurs at the end of the time series due to lead shifting)
    df_clean = df.dropna(subset=[target_column]).copy()

    # Drop non-feature columns
    X = df_clean.drop(columns=feature_columns_to_exclude, errors="ignore")
    y = df_clean[target_column]

    # Convert any remaining object/string columns to category or drop them
    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        # If it's the district column or similar, we can map it to numeric codes
        X[col] = X[col].astype("category").cat.codes

    # Fill any remaining NaNs in features with column median (imputation)
    # This is a fallback to prevent Scikit-Learn training errors
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def split_chronologically(
    df: pd.DataFrame,
    split_date: str = "2025-01-01",
    date_col: str = "Date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically into train and test sets to prevent temporal leakage.
    """
    dates = pd.to_datetime(df[date_col])
    split_dt = pd.to_datetime(split_date)
    
    train_mask = dates < split_dt
    test_mask = dates >= split_dt
    
    return df[train_mask].copy(), df[test_mask].copy()


def train_lead_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    feature_columns_to_exclude: List[str] = None
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train a RandomForestRegressor model for a given target lead and evaluate its performance.
    """
    start_time = time.perf_counter()
    logger.info("Preparing data for target: %s", target_column)
    
    X_train, y_train = prepare_training_data(train_df, target_column, feature_columns_to_exclude)
    X_test, y_test = prepare_training_data(test_df, target_column, feature_columns_to_exclude)

    logger.info(
        "Train set: X=%s, y=%s | Test set: X=%s, y=%s",
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
    )

    # Instantiate and train RandomForestRegressor
    # Keeping hyperparameters reasonable for fast execution and robust generalizability
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Training Random Forest model for %s...", target_column)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_preds)),
        "train_rmse": float(root_mean_squared_error(y_train, train_preds)),
        "train_r2": float(r2_score(y_train, train_preds)),
        "test_mae": float(mean_absolute_error(y_test, test_preds)),
        "test_rmse": float(root_mean_squared_error(y_test, test_preds)),
        "test_r2": float(r2_score(y_test, test_preds)),
        "training_duration_seconds": time.perf_counter() - start_time
    }
    
    logger.info(
        "Evaluation results for %s:\n"
        "  Train R2: %.4f | Test R2: %.4f\n"
        "  Train MAE: %.4f | Test MAE: %.4f",
        target_column, metrics["train_r2"], metrics["test_r2"],
        metrics["train_mae"], metrics["test_mae"]
    )
    
    return model, metrics


def save_model_artifact(model: RandomForestRegressor, filepath: Path) -> None:
    """Save the model binary to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Saved model artifact to: %s", filepath)
