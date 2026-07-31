"""Module for model training, evaluation, and artifact orchestration.

Provides data structures, exceptions, configuration handlers, and public interface
declarations for training machine learning regression models on preprocessed water
intelligence datasets.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

# Configure logger for the model training domain module
logger = logging.getLogger(__name__)

# =============================================================================
# IMPLEMENTATION ROADMAP
# =============================================================================
#
# Layer 1 ✓ Foundation (Exceptions, Dataclasses, Constants, Signatures)
# Layer 2 ✓ Dataset Loading & Input Validation (FROZEN)
# Layer 3   Configuration-Driven Model Registry
# Layer 4   Sequential Training Engine
# Layer 5   Evaluation Engine & Model Selection
# Layer 6   Artifact Persistence Engine
# Layer 7   Reporting & Metadata Generation
#
# =============================================================================

# =============================================================================
# LOGGING CONVENTION
# =============================================================================
#
# INFO:
#     High-level pipeline progress (e.g., stage start/stop, champion selection).
#
# DEBUG:
#     Dataset shapes, feature counts, execution timings, configuration details.
#
# ERROR:
#     Failures captured with logger.exception() prior to raising custom domain exceptions.
#
# =============================================================================

# =============================================================================
# REUSABLE MODULE CONSTANTS & TYPE ALIASES
# =============================================================================

RMSE: str = "rmse"
MAE: str = "mae"
R2: str = "r2"

DEFAULT_SELECTION_METRIC: str = RMSE

TRAIN_DATASET_FILENAME: str = "train.parquet"
VALIDATION_DATASET_FILENAME: str = "validation.parquet"

# Type alias for candidate model results dictionary
ModelResults = dict[str, "ModelResult"]


# =============================================================================
# DOMAIN EXCEPTIONS
# =============================================================================


class ModelTrainingBaseError(Exception):
    """Base exception for all errors raised within the model training domain."""


class ConfigurationError(ModelTrainingBaseError):
    """Raised when configuration parameters are missing, malformed, or invalid."""


class DatasetValidationError(ModelTrainingBaseError):
    """Raised when input datasets fail schema, dimension, or content validations."""


class ModelTrainingError(ModelTrainingBaseError):
    """Raised when model instantiation, fitting, evaluation, or saving fails."""


# =============================================================================
# DATACLASSES & DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Holds preprocessed feature matrices and target vectors for training splits.

    Attributes:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_validation: Validation feature DataFrame.
        y_validation: Validation target Series.
        feature_names: List of preprocessed feature names present in X splits.
        target_name: Name of the target variable column.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    feature_names: list[str]
    target_name: str

    @property
    def n_train_rows(self) -> int:
        """Return the number of rows in the training split."""
        return len(self.X_train)

    @property
    def n_validation_rows(self) -> int:
        """Return the number of rows in the validation split."""
        return len(self.X_validation)

    @property
    def n_features(self) -> int:
        """Return the total number of features used for model input."""
        return len(self.feature_names)


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Encapsulates execution outcome, fitted artifact, and evaluation metrics for a model.

    Attributes:
        model_name: Unique identifier/name of the model algorithm.
        estimator: The trained scikit-learn compatible estimator instance.
        rmse: Root Mean Squared Error computed on the validation split.
        mae: Mean Absolute Error computed on the validation split.
        r2: R-squared coefficient of determination on the validation split.
        training_time: Execution time in seconds required to fit the model.
        prediction_time: Execution time in seconds required to generate predictions.
    """

    model_name: str
    estimator: BaseEstimator
    rmse: float
    mae: float
    r2: float
    training_time: float
    prediction_time: float


@dataclass(frozen=True, slots=True)
class TrainingMetadata:
    """Provenance and runtime statistics for the model training stage.

    Attributes:
        training_timestamp: ISO-8601 formatted timestamp of training execution.
        python_version: Version of the active Python interpreter.
        sklearn_version: Installed scikit-learn library version.
        random_state: Seed utilized across random generation operations.
        preprocessing_sha256: Hash of the upstream preprocessing pipeline binary.
        train_rows: Number of rows in the training split.
        validation_rows: Number of rows in the validation split.
        feature_count: Total count of features supplied to candidate models.
    """

    training_timestamp: str
    python_version: str
    sklearn_version: str
    random_state: int
    preprocessing_sha256: str
    train_rows: int
    validation_rows: int
    feature_count: int


# =============================================================================
# LAYER 2 — PRIVATE HELPERS
# =============================================================================


def _read_parquet_split(path: Path, split_name: str) -> pd.DataFrame:
    """Read a single preprocessed parquet split file with timing and validation.

    Args:
        path: Filesystem path to the parquet file.
        split_name: Human-readable split identifier used in log/error messages.

    Returns:
        The loaded, non-empty DataFrame.

    Raises:
        DatasetValidationError: If the file does not exist, is not a file,
            is unreadable, or is empty.
    """
    if not path.exists():
        raise DatasetValidationError(
            f"Required '{split_name}' parquet file not found: {path}"
        )
    if not path.is_file():
        raise DatasetValidationError(
            f"Expected '{split_name}' path to be a file, not a directory: {path}"
        )

    logger.debug("Reading '%s' parquet file: %s", split_name, path)
    start = time.perf_counter()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error below
        logger.exception("Failed to read '%s' parquet file: %s", split_name, path)
        raise DatasetValidationError(
            f"Failed to read '{split_name}' parquet file at {path}: {exc}"
        ) from exc
    elapsed = time.perf_counter() - start
    logger.debug(
        "Read '%s' parquet file in %.4fs, shape=%s", split_name, elapsed, df.shape
    )

    if df.empty:
        raise DatasetValidationError(
            f"'{split_name}' dataset at {path} is empty (0 rows)"
        )

    return df


def _split_features_target(
    df: pd.DataFrame, target_column: str, split_name: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Isolate the target column from a preprocessed split, returning (X, y).

    Args:
        df: The loaded split DataFrame.
        target_column: Name of the target column to isolate.
        split_name: Human-readable split identifier used in error messages.

    Returns:
        A tuple of (feature DataFrame, target Series).

    Raises:
        DatasetValidationError: If target_column is not present in df.
    """
    if target_column not in df.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' not found in '{split_name}' dataset. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]
    logger.debug(
        "Split '%s' dataset into X shape=%s, y length=%d", split_name, X.shape, len(y)
    )
    return X, y


def _check_no_duplicate_columns(df: pd.DataFrame, split_name: str) -> None:
    """Verify a DataFrame has no duplicate column names.

    Raises:
        DatasetValidationError: If duplicate column names are found.
    """
    duplicate_columns = df.columns[df.columns.duplicated()].unique().tolist()
    logger.debug(
        "'%s' dataset duplicate column check: %d duplicate(s) found",
        split_name,
        len(duplicate_columns),
    )
    if duplicate_columns:
        raise DatasetValidationError(
            f"'{split_name}' dataset contains duplicate column names: {duplicate_columns}"
        )


def _check_no_nan_values(df: pd.DataFrame, split_name: str) -> None:
    """Verify a DataFrame contains no NaN values in any column.

    Raises:
        DatasetValidationError: If any NaN values are present, listing the
            offending columns and their null counts.
    """
    null_counts = df.isna().sum()
    total_nulls = int(null_counts.sum())
    logger.debug("'%s' dataset missing value count: %d", split_name, total_nulls)

    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise DatasetValidationError(
            f"'{split_name}' dataset contains NaN values in columns: "
            f"{columns_with_nulls.to_dict()}"
        )


def _check_feature_schemas_match(
    X_train: pd.DataFrame, X_validation: pd.DataFrame
) -> None:
    """Verify train/validation feature sets share identical columns and dtypes.

    Column order is intentionally not checked here -- see
    _check_feature_order_matches for that. This checks set membership and
    per-column dtype agreement only.

    Raises:
        DatasetValidationError: If columns differ, or a shared column's
            dtype differs between splits.
    """
    train_columns = set(X_train.columns)
    validation_columns = set(X_validation.columns)

    if train_columns != validation_columns:
        only_in_train = sorted(train_columns - validation_columns)
        only_in_validation = sorted(validation_columns - train_columns)
        raise DatasetValidationError(
            "Train/validation feature schemas do not match. "
            f"Columns only in train: {only_in_train}. "
            f"Columns only in validation: {only_in_validation}."
        )

    dtype_mismatches = {
        column: (str(X_train[column].dtype), str(X_validation[column].dtype))
        for column in X_train.columns
        if X_train[column].dtype != X_validation[column].dtype
    }
    if dtype_mismatches:
        raise DatasetValidationError(
            f"Train/validation dtype mismatch for shared columns: {dtype_mismatches}"
        )

    logger.debug("Train/validation feature schema check passed")


def _check_feature_order_matches(
    X_train: pd.DataFrame, X_validation: pd.DataFrame
) -> None:
    """Verify train/validation feature columns appear in identical order.

    Raises:
        DatasetValidationError: If column order differs between splits.
    """
    train_order = list(X_train.columns)
    validation_order = list(X_validation.columns)
    if train_order != validation_order:
        raise DatasetValidationError(
            "Train/validation feature column order does not match. "
            f"Train order: {train_order}. Validation order: {validation_order}."
        )
    logger.debug("Train/validation feature order check passed")


def _load_preprocessing_metadata(path: Path) -> dict[str, Any]:
    """Load the preprocessing metadata JSON file produced by Stage 9.

    Args:
        path: Path to the preprocessing metadata JSON file.

    Returns:
        The parsed metadata as a dict.

    Raises:
        ConfigurationError: If the file does not exist, is unreadable, or
            does not contain a valid JSON object.
    """
    if not path.exists():
        raise ConfigurationError(f"Preprocessing metadata file not found: {path}")

    logger.debug("Loading preprocessing metadata: %s", path)
    start = time.perf_counter()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except OSError as exc:
        logger.exception("Failed to read preprocessing metadata file: %s", path)
        raise ConfigurationError(
            f"Preprocessing metadata file at {path} could not be read: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse preprocessing metadata file: %s", path)
        raise ConfigurationError(
            f"Preprocessing metadata file at {path} does not contain valid JSON: {exc}"
        ) from exc
    elapsed = time.perf_counter() - start
    logger.debug("Loaded preprocessing metadata in %.4fs", elapsed)

    if not isinstance(metadata, dict):
        raise ConfigurationError(
            f"Preprocessing metadata file at {path} must contain a JSON object, "
            f"got {type(metadata).__name__}"
        )

    return metadata


# =============================================================================
# LAYER 2 — PUBLIC FUNCTIONS
# =============================================================================


def load_training_datasets(
    preprocessed_dir: Path,
    target_column: str,
) -> DatasetBundle:
    """Load preprocessed train and validation parquet splits into a DatasetBundle.

    Args:
        preprocessed_dir: Directory containing preprocessed parquet split files.
        target_column: Name of the target variable to isolate from features.

    Returns:
        DatasetBundle: Encapsulated features and targets for train and validation splits.

    Raises:
        DatasetValidationError: If files are missing, empty, or lack expected target columns.
        ConfigurationError: If paths or column parameters are invalid.
    """
    logger.info("Starting dataset loading")
    start = time.perf_counter()

    if not isinstance(target_column, str) or not target_column.strip():
        raise ConfigurationError("target_column must be a non-empty string")

    logger.debug(
        "preprocessed_dir=%s, target_column=%s", preprocessed_dir, target_column
    )

    if not preprocessed_dir.exists():
        raise DatasetValidationError(
            f"Preprocessed dataset directory not found: {preprocessed_dir}"
        )
    if not preprocessed_dir.is_dir():
        raise DatasetValidationError(
            f"Expected preprocessed dataset path to be a directory: {preprocessed_dir}"
        )

    train_path = preprocessed_dir / TRAIN_DATASET_FILENAME
    validation_path = preprocessed_dir / VALIDATION_DATASET_FILENAME

    train_df = _read_parquet_split(train_path, "train")
    validation_df = _read_parquet_split(validation_path, "validation")

    _check_no_duplicate_columns(train_df, "train")
    _check_no_duplicate_columns(validation_df, "validation")

    _check_no_nan_values(train_df, "train")
    _check_no_nan_values(validation_df, "validation")

    X_train, y_train = _split_features_target(train_df, target_column, "train")
    X_validation, y_validation = _split_features_target(
        validation_df, target_column, "validation"
    )

    _check_feature_schemas_match(X_train, X_validation)
    _check_feature_order_matches(X_train, X_validation)

    bundle = DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        feature_names=list(X_train.columns),
        target_name=target_column,
    )

    elapsed = time.perf_counter() - start
    logger.debug(
        "Dataset bundle assembled in %.4fs: n_train_rows=%d, n_validation_rows=%d, n_features=%d",
        elapsed,
        bundle.n_train_rows,
        bundle.n_validation_rows,
        bundle.n_features,
    )
    logger.info("Successfully loaded datasets")
    logger.debug(
        "Training Dataset Summary:\n"
        "Train rows: %d\n"
        "Validation rows: %d\n"
        "Features: %d\n"
        "Target: %s",
        bundle.n_train_rows,
        bundle.n_validation_rows,
        bundle.n_features,
        bundle.target_name,
    )
    return bundle


def validate_training_inputs(
    datasets: DatasetBundle,
    preprocessing_metadata_path: Path,
) -> None:
    """Validate data consistency between preprocessed datasets and pipeline metadata.

    Args:
        datasets: The loaded dataset bundle to validate.
        preprocessing_metadata_path: Path to the preprocessing metadata JSON file.

    Raises:
        DatasetValidationError: If feature schemas mismatch or NaNs exist in preprocessed data.
        ConfigurationError: If metadata file is unreadable or malformed.
    """
    logger.info("Starting validation")
    start = time.perf_counter()

    metadata = _load_preprocessing_metadata(preprocessing_metadata_path)

    metadata_features = metadata.get("feature_names_out")
    if metadata_features is None:
        raise ConfigurationError(
            f"Preprocessing metadata at {preprocessing_metadata_path} is missing "
            f"required key 'feature_names_out'"
        )
    if not isinstance(metadata_features, list) or not all(
        isinstance(f, str) for f in metadata_features
    ):
        raise ConfigurationError(
            f"Metadata field 'feature_names_out' at {preprocessing_metadata_path} "
            "must be a list of non-empty strings."
        )
    if len(metadata_features) != len(set(metadata_features)):
        raise ConfigurationError(
            f"Metadata field 'feature_names_out' at {preprocessing_metadata_path} "
            "contains duplicate feature names."
        )

    logger.debug("Metadata declares %d feature(s)", len(metadata_features))

    if list(datasets.feature_names) != list(metadata_features):
        only_in_datasets = sorted(set(datasets.feature_names) - set(metadata_features))
        only_in_metadata = sorted(set(metadata_features) - set(datasets.feature_names))
        raise DatasetValidationError(
            "Loaded dataset features do not match preprocessing metadata features. "
            f"Present only in loaded datasets: {only_in_datasets}. "
            f"Present only in preprocessing metadata: {only_in_metadata}. "
            "This usually means non-feature columns were not excluded upstream, "
            "or the wrong preprocessing artifact is being referenced."
        )
    logger.debug("Feature name check passed")

    if datasets.n_features != len(metadata_features):
        raise DatasetValidationError(
            f"Feature count mismatch: loaded datasets have {datasets.n_features} "
            f"features, preprocessing metadata declares {len(metadata_features)}"
        )
    logger.debug("Feature count check passed")

    metadata_train_rows = metadata.get("n_train_rows")
    if metadata_train_rows is None:
        raise ConfigurationError(
            f"Preprocessing metadata at {preprocessing_metadata_path} is missing "
            f"required key 'n_train_rows'"
        )
    if not isinstance(metadata_train_rows, int):
        raise ConfigurationError(
            f"Metadata field 'n_train_rows' at {preprocessing_metadata_path} must be an integer."
        )
    if datasets.n_train_rows != metadata_train_rows:
        raise DatasetValidationError(
            f"Train row count mismatch: loaded {datasets.n_train_rows} rows, "
            f"preprocessing metadata declares {metadata_train_rows} rows"
        )
    logger.debug("Train row count check passed")

    metadata_validation_rows = metadata.get("n_validation_rows")
    if metadata_validation_rows is not None:
        if not isinstance(metadata_validation_rows, int):
            raise ConfigurationError(
                f"Metadata field 'n_validation_rows' at {preprocessing_metadata_path} must be an integer."
            )
        if datasets.n_validation_rows != metadata_validation_rows:
            raise DatasetValidationError(
                f"Validation row count mismatch: loaded {datasets.n_validation_rows} rows, "
                f"preprocessing metadata declares {metadata_validation_rows} rows"
            )
        logger.debug("Validation row count check passed")

    preprocessing_hash = metadata.get("preprocessing_sha256")
    if preprocessing_hash:
        logger.debug("Preprocessing SHA256 found: %s", preprocessing_hash)
    else:
        logger.warning(
            "No preprocessing_sha256 found in preprocessing metadata at %s. "
            "Skipping provenance validation until Stage 9 provides this field.",
            preprocessing_metadata_path,
        )

    elapsed = time.perf_counter() - start
    logger.debug("Validation checkpoints completed in %.4fs", elapsed)
    logger.info("Validation completed successfully")


# =============================================================================
# UNIMPLEMENTED PUBLIC API DECLARATIONS (LAYERS 3-7)
# =============================================================================


def build_model_registry(
    config: dict[str, Any],
) -> dict[str, BaseEstimator]:
    """Instantiate candidate estimators based on configuration parameters.

    Args:
        config: Parsed model configurations including hyperparameters and enablement flags.

    Returns:
        dict[str, BaseEstimator]: Mapping of model names to un-fitted estimator instances.

    Raises:
        ConfigurationError: If model hyperparameters or requested model drivers are invalid.
    """
    ...


def train_models(
    datasets: DatasetBundle,
    registry: dict[str, BaseEstimator],
    config: dict[str, Any],
) -> ModelResults:
    """Train enabled candidate models sequentially, evaluate them, and return results.

    Args:
        datasets: The dataset bundle containing preprocessed features and targets.
        registry: Initialized dictionary of candidate model estimators.
        config: Training execution configuration parameters.

    Returns:
        ModelResults: Mapping of candidate model names to their fitted ModelResults.

    Raises:
        ModelTrainingError: If training fails for an estimator or execution breaks down.
    """
    ...


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, float], float]:
    """Compute standard regression performance metrics and measure prediction latency.

    Args:
        model: A fitted estimator with a `predict` method.
        X: Feature matrix DataFrame.
        y: Ground truth target Series.

    Returns:
        tuple[dict[str, float], float]: A dictionary of metrics ("rmse", "mae", "r2")
            and the total prediction time in seconds.

    Raises:
        ModelTrainingError: If inference fails or predictions contain NaNs/Infs.
    """
    ...


def select_best_model(
    results: ModelResults,
    metric: str = DEFAULT_SELECTION_METRIC,
) -> ModelResult:
    """Select the top-performing candidate model based on a specified evaluation metric.

    Args:
        results: Dictionary mapping candidate model names to their ModelResult objects.
        metric: Evaluation metric name used for selection. Defaults to "rmse".

    Returns:
        ModelResult: The winning candidate model result.

    Raises:
        ModelTrainingError: If results dictionary is empty or specified metric is unsupported.
    """
    ...


def save_model(
    model_result: ModelResult,
    output_dir: Path,
) -> Path:
    """Serialize a fitted candidate model artifact to disk.

    Args:
        model_result: The ModelResult instance containing the fitted estimator.
        output_dir: Destination directory for the serialized model binary.

    Returns:
        Path: Resolved filepath of the saved candidate model.

    Raises:
        ModelTrainingError: If serialization or writing fails.
    """
    ...


def save_best_model(
    best_model_result: ModelResult,
    output_dir: Path,
) -> Path:
    """Persist the selected champion model to the designated production model directory.

    Args:
        best_model_result: The winning candidate model result.
        output_dir: Destination directory for champion model persistence.

    Returns:
        Path: Resolved filepath of the saved champion model.

    Raises:
        ModelTrainingError: If serialization or writing fails.
    """
    ...


def generate_training_report(
    results: ModelResults,
    best_model_result: ModelResult,
    metadata: TrainingMetadata,
    output_path: Path,
) -> Path:
    """Assemble and write the comprehensive model training and evaluation report as JSON.

    Args:
        results: Dictionary of all candidate model results.
        best_model_result: Result object corresponding to the winning champion model.
        metadata: Pipeline provenance and run metadata object.
        output_path: Destination JSON file path for the report.

    Returns:
        Path: Filepath to the saved report.

    Raises:
        ModelTrainingError: If report generation or disk writing fails.
    """
    ...


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants & Type Aliases
    "RMSE",
    "MAE",
    "R2",
    "DEFAULT_SELECTION_METRIC",
    "TRAIN_DATASET_FILENAME",
    "VALIDATION_DATASET_FILENAME",
    "ModelResults",
    # Dataclasses
    "DatasetBundle",
    "ModelResult",
    "TrainingMetadata",
    # Exceptions
    "ModelTrainingBaseError",
    "ConfigurationError",
    "DatasetValidationError",
    "ModelTrainingError",
    # Public Functions
    "load_training_datasets",
    "validate_training_inputs",
    "build_model_registry",
    "train_models",
    "evaluate_model",
    "select_best_model",
    "save_model",
    "save_best_model",
    "generate_training_report",
]
