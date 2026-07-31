"""Stage 10 - Layer 4: Model Training Orchestration.

Fits every model in a Layer 3 model registry against the train split of a
Layer 1/2 DatasetBundle, producing structured, timed results.

No evaluation, no persistence, no hyperparameter search -- those belong
to later layers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from sklearn.base import BaseEstimator

from src.stage10.dataset_bundle import DatasetBundle
from src.stage10.exceptions import Stage10Error
from src.stage10.logging_utils import log_call
from src.stage10.model_registry import ModelRegistry
from src.stage10.validation import ValidationConfig

logger = logging.getLogger("stage10")


class EmptyModelRegistryError(Stage10Error):
    """Raised when train_all_models is called with an empty model
    registry (nothing to train)."""


class ModelTrainingError(Stage10Error):
    """Raised when a model fails to fit on the training data."""


class FeatureExtractionError(Stage10Error):
    """Raised when features/target cannot be extracted from a split
    DataFrame."""


@dataclass(frozen=True)
class TrainedModelResult:
    """Result of fitting a single model.

    Attributes
    ----------
    name:
        Registered model name (matches the key in the model registry).
    model:
        The fitted model instance.
    feature_columns:
        Ordered feature columns the model was trained on.
    target_column:
        Name of the target column the model was trained to predict.
    train_rows:
        Number of rows used for training.
    training_duration_seconds:
        Wall-clock time spent inside model.fit().
    trained_at:
        UTC timestamp recorded when training completed.
    """

    name: str
    model: BaseEstimator
    feature_columns: list[str]
    target_column: str
    train_rows: int
    training_duration_seconds: float
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class TrainingReport:
    """Aggregate result of training every model in a registry.

    Attributes
    ----------
    results:
        Mapping of model name -> TrainedModelResult, for every model
        that trained successfully.
    total_duration_seconds:
        Wall-clock time for the entire train_all_models call.
    trained_at:
        UTC timestamp recorded when the training run completed.
    """

    results: dict[str, TrainedModelResult]
    total_duration_seconds: float
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@log_call
def extract_features_and_target(
    df: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice *df* into (X, y) using the exact feature order given.

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
    (X, y)

    Raises
    ------
    FeatureExtractionError
        If target column is missing, missing requested features, or no
        feature columns remain.
    """
    if not feature_columns:
        feature_columns = [col for col in df.columns if col != target_column]

    if not feature_columns:
        raise FeatureExtractionError("No feature columns are available for training.")

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise FeatureExtractionError(
            f"Cannot extract features: missing column(s) {missing_features}"
        )

    if target_column not in df.columns:
        raise FeatureExtractionError(
            f"Cannot extract target: column '{target_column}' not found."
        )

    X = df.loc[:, feature_columns]
    y = df.loc[:, target_column]
    return X, y


@log_call
def train_single_model(
    name: str,
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> TrainedModelResult:
    """Fit a single unfitted model on (X, y).

    Parameters
    ----------
    name:
        Model name, used for logging and error messages.
    model:
        Unfitted model instance (must implement .fit(X, y)).
    X:
        Feature matrix.
    y:
        Target vector.

    Returns
    -------
    TrainedModelResult

    Raises
    ------
    ModelTrainingError
        If model.fit() raises any exception.
    """
    start_time = time.perf_counter()

    try:
        model.fit(X, y)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed for model '%s'", name)
        raise ModelTrainingError(
            f"Model '{name}' failed to fit on {len(X)} rows / "
            f"{len(X.columns)} features: {exc}"
        ) from exc

    duration = time.perf_counter() - start_time

    logger.info(
        "Trained model '%s' on %s rows / %s features in %.4fs",
        name,
        len(X),
        len(X.columns),
        duration,
    )

    return TrainedModelResult(
        name=name,
        model=model,
        feature_columns=list(X.columns),
        target_column=y.name if y.name is not None else "target",
        train_rows=len(X),
        training_duration_seconds=duration,
    )


@log_call
def train_all_models(
    model_registry: ModelRegistry | dict[str, BaseEstimator],
    bundle: DatasetBundle,
    validation_config: ValidationConfig,
) -> TrainingReport:
    """Fit every model in *model_registry* on the train split of *bundle*.

    Parameters
    ----------
    model_registry:
        Layer 3 ModelRegistry instance or dictionary mapping model name ->
        unfitted model instance.
    bundle:
        DatasetBundle produced by Layer 1 and already validated by
        Layer 2.
    validation_config:
        The same ValidationConfig used for Layer 2 validation; supplies
        target_column and feature_columns so training uses the exact
        same, already-verified schema.

    Returns
    -------
    TrainingReport
        Contains a TrainedModelResult for every model that trained
        successfully.

    Raises
    ------
    EmptyModelRegistryError
        If model_registry contains no models.
    FeatureExtractionError
        If the train split does not contain expected columns.
    ModelTrainingError
        If any model fails to fit.
    """
    raw_registry = (
        model_registry.models
        if isinstance(model_registry, ModelRegistry)
        else model_registry
    )

    if not raw_registry:
        raise EmptyModelRegistryError(
            "model_registry is empty; there are no models to train. "
            "Check that at least one model is enabled in configuration."
        )

    overall_start = time.perf_counter()

    X_train, y_train = extract_features_and_target(
        bundle.train, validation_config.feature_columns, validation_config.target_column
    )

    results: dict[str, TrainedModelResult] = {}
    for name, model in raw_registry.items():
        results[name] = train_single_model(name, model, X_train, y_train)

    total_duration = time.perf_counter() - overall_start

    report = TrainingReport(results=results, total_duration_seconds=total_duration)

    logger.info(
        "Layer 4 training complete: %s model(s) trained in %.4fs",
        len(results),
        total_duration,
    )

    return report
