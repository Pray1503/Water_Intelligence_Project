"""Stage 10 - Shared Utilities.

Houses logic used by more than one Stage 10 layer, so it is written
exactly once. Imports only from exceptions.py -- never from any layer
module (dataset_bundle, validation, model_registry, training,
evaluation, selection, persistence) -- so it can never participate in a
circular import regardless of which layer needs it.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.stage10.exceptions import FeatureExtractionError
from src.stage10.logging_utils import log_call

# Dedicated logger utilized for granular execution tracing
logger = logging.getLogger("stage10")

try:
    from sklearn.metrics import root_mean_squared_error as _sklearn_rmse
except (
    ImportError
):  # pragma: no cover - fallback for older scikit-learn versions (<1.4)
    from sklearn.metrics import mean_squared_error

    def _sklearn_rmse(
        y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
    ) -> float:
        return float(mean_squared_error(y_true, y_pred) ** 0.5)


class ExecutionMode(str, Enum):
    """Execution mode shared by every layer that iterates over multiple
    candidate models (evaluation, selection, persistence).

    STRICT:
        Fail immediately on the first error.
    BEST_EFFORT:
        Log and skip the failing model; continue processing the rest.
    """

    STRICT = "strict"
    BEST_EFFORT = "best_effort"


@log_call
def extract_features_and_target(
    df: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice *df* into (X, y) using the exact feature order given.

    Canonical shared implementation. Behaviorally identical to the
    Layer-4-local copy in training.py (which predates this module and
    is left untouched since Layer 4 is frozen) -- new code should use
    this version, not that one.

    Parameters
    ----------
    df:
        Source DataFrame (already validated by Layer 2).
    feature_columns:
        Ordered feature column names. If empty, falls back to using
        all non-target columns.
    target_column:
        Target column name.

    Returns
    -------
    (X, y) : Defensive explicit copies of features DataFrame and target Series.

    Raises
    ------
    FeatureExtractionError
        If df is not a DataFrame, if df is empty, if target column is missing,
        if requested features are missing, or no feature columns remain.
    """
    if not isinstance(df, pd.DataFrame):
        raise FeatureExtractionError(
            f"Expected pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty:
        raise FeatureExtractionError(
            "Input DataFrame is empty; cannot extract features and target."
        )

    if target_column not in df.columns:
        raise FeatureExtractionError(
            f"Cannot extract target: column '{target_column}' not found in DataFrame."
        )

    if not feature_columns:
        feature_columns = df.columns.drop(target_column).tolist()

    if not feature_columns:
        raise FeatureExtractionError(
            f"No feature columns are available after excluding target column '{target_column}'."
        )

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise FeatureExtractionError(
            f"Cannot extract features: missing column(s) {missing_features}"
        )

    logger.debug(
        "Extracting %d feature(s) and target '%s' from DataFrame (%d rows)",
        len(feature_columns),
        target_column,
        len(df),
    )

    X = df.loc[:, feature_columns].copy()
    y = df.loc[:, target_column].copy()
    return X, y


@log_call
def compute_regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> tuple[float, float, float]:
    """Compute (rmse, mae, r2) for a set of predictions.

    Parameters
    ----------
    y_true:
        Ground truth target values.
    y_pred:
        Predicted values, same length and order as y_true.

    Returns
    -------
    (rmse, mae, r2)

    Raises
    ------
    FeatureExtractionError
        If y_true and y_pred have mismatched lengths or contain
        non-finite (NaN, Inf, -Inf) numerical values.
    """
    if len(y_true) != len(y_pred):
        raise FeatureExtractionError(
            f"y_true (len={len(y_true)}) and y_pred (len={len(y_pred)}) "
            f"have mismatched lengths."
        )

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if not np.all(np.isfinite(y_true_arr)):
        n_bad = int((~np.isfinite(y_true_arr)).sum())
        raise FeatureExtractionError(
            f"y_true contains {n_bad} non-finite (NaN/Inf) value(s)."
        )

    if not np.all(np.isfinite(y_pred_arr)):
        n_bad = int((~np.isfinite(y_pred_arr)).sum())
        raise FeatureExtractionError(
            f"y_pred contains {n_bad} non-finite (NaN/Inf) value(s)."
        )

    rmse = float(_sklearn_rmse(y_true_arr, y_pred_arr))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return rmse, mae, r2
