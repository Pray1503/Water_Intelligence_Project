"""Module for training preprocessing.

Provides data cleaning, scaling, encoding, and metadata creation logic for
train/validation/test datasets.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class TrainingPreprocessingError(Exception):
    """Raised when dataset preprocessing or artifact saving fails."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration options for dataset preprocessing."""

    target_column: str = "target"
    group_column: str = "District LGD Code"
    date_column: str = "Date"
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scale_numeric: bool = True
    onehot_min_frequency: float = 0.01
    random_state: int = 42


@dataclass(frozen=True)
class PreprocessingMetadata:
    """Metadata describing the fitted preprocessing transformer."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_names_out: list[str]
    dropped_unsupported_columns: list[str]
    excluded_columns: list[str]
    numeric_impute_strategy: str
    categorical_impute_strategy: str
    scale_numeric: bool
    random_state: int
    pipeline_sha256: str | None = None
    fitted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary, including Stage 10 compatibility aliases."""
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "feature_names_out": self.feature_names_out,
            "dropped_unsupported_columns": self.dropped_unsupported_columns,
            "excluded_columns": self.excluded_columns,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "categorical_impute_strategy": self.categorical_impute_strategy,
            "scale_numeric": self.scale_numeric,
            "random_state": self.random_state,
            "pipeline_sha256": self.pipeline_sha256,
            "fitted_at": self.fitted_at,
            # Stage 10 Compatibility Aliases
            "feature_columns": self.feature_names_out,
            "preprocessing_hash": self.pipeline_sha256,
        }


class PreprocessedDatasets(NamedTuple):
    """Container holding preprocessed splits, transformer, and metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    transformer: ColumnTransformer
    metadata: PreprocessingMetadata


def _compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while chunk := fh.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def _build_column_transformer(
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: PreprocessingConfig,
) -> ColumnTransformer:
    """Construct the scikit-learn ColumnTransformer instance."""
    transformers = []

    if numeric_cols:
        num_steps = [
            ("imputer", SimpleImputer(strategy=config.numeric_impute_strategy))
        ]
        if config.scale_numeric:
            num_steps.append(("scaler", StandardScaler()))

        from sklearn.pipeline import Pipeline

        transformers.append(("numeric", Pipeline(num_steps), numeric_cols))

    if categorical_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy=config.categorical_impute_strategy)),
            (
                "onehot",
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    min_frequency=config.onehot_min_frequency,
                ),
            ),
        ]

        from sklearn.pipeline import Pipeline

        transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def preprocess_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PreprocessingConfig,
) -> PreprocessedDatasets:
    """Preprocess training, validation, and testing DataFrames.

    Fits transformations solely on the training split to avoid data leakage,
    then transforms all three splits using the fitted transformer.
    """
    if train_df.empty:
        raise TrainingPreprocessingError("Training dataset is empty.")

    excluded_cols = [
        col
        for col in [config.target_column, config.group_column, config.date_column]
        if col
    ]

    feature_cols = [c for c in train_df.columns if c not in excluded_cols]

    numeric_cols = []
    categorical_cols = []
    dropped_cols = []

    for col in feature_cols:
        dtype = train_df[col].dtype

        if is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif (
            is_string_dtype(dtype)
            or is_object_dtype(dtype)
            or is_categorical_dtype(dtype)
        ):
            categorical_cols.append(col)
        else:
            dropped_cols.append(col)

    transformer = _build_column_transformer(numeric_cols, categorical_cols, config)

    try:
        transformer.fit(train_df[feature_cols])
    except Exception as exc:  # noqa: BLE001
        raise TrainingPreprocessingError(
            f"Failed to fit preprocessing transformer: {exc}"
        ) from exc

    try:
        feature_names_out = list(transformer.get_feature_names_out())
    except Exception:  # noqa: BLE001
        feature_names_out = numeric_cols + categorical_cols

    def _transform_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=feature_names_out)

        try:
            # Transform only the feature columns.
            arr = transformer.transform(df[feature_cols])
            res_df = pd.DataFrame(
                arr,
                columns=feature_names_out,
                index=df.index,
            )

            # Preserve ONLY the target column for model training.
            # The group column (District LGD Code) and the date column (Date)
            # are intentionally excluded from the final preprocessed dataset.
            if config.target_column in df.columns:
                res_df[config.target_column] = df[config.target_column]

            return res_df

        except Exception as exc:  # noqa: BLE001
            raise TrainingPreprocessingError(
                f"Failed to transform '{name}' split: {exc}"
            ) from exc

    train_transformed = _transform_split(train_df, "train")
    val_transformed = _transform_split(validation_df, "validation")
    test_transformed = _transform_split(test_df, "test")

    metadata = PreprocessingMetadata(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        feature_names_out=feature_names_out,
        dropped_unsupported_columns=dropped_cols,
        excluded_columns=excluded_cols,
        numeric_impute_strategy=config.numeric_impute_strategy,
        categorical_impute_strategy=config.categorical_impute_strategy,
        scale_numeric=config.scale_numeric,
        random_state=config.random_state,
        fitted_at=datetime.now(timezone.utc).isoformat(),
    )

    return PreprocessedDatasets(
        train=train_transformed,
        validation=val_transformed,
        test=test_transformed,
        transformer=transformer,
        metadata=metadata,
    )


def save_preprocessing_pipeline(
    pipeline: ColumnTransformer,
    metadata: PreprocessingMetadata,
    output_path: str | Path,
) -> PreprocessingMetadata:
    """Save the fitted preprocessing pipeline and associated metadata.

    Serializes the ColumnTransformer pipeline using joblib, computes its SHA-256
    checksum, updates the metadata with the checksum, and persists the metadata
    as JSON in the same directory.

    Args:
        pipeline: Fitted ColumnTransformer instance to save.
        metadata: Associated PreprocessingMetadata object.
        output_path: Path where the pipeline joblib file should be written.

    Returns:
        PreprocessingMetadata: The updated metadata object containing the
            computed pipeline SHA-256 hash.

    Raises:
        TrainingPreprocessingError: If serialization or writing fails.
    """
    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, output_path)
    except Exception as exc:  # noqa: BLE001
        raise TrainingPreprocessingError(
            f"Failed to save preprocessing pipeline to {output_path}: {exc}"
        ) from exc

    try:
        pipeline_sha256 = _compute_file_sha256(output_path)
    except Exception as exc:  # noqa: BLE001
        raise TrainingPreprocessingError(
            f"Failed to compute SHA-256 for saved pipeline at {output_path}: {exc}"
        ) from exc

    updated_metadata = replace(metadata, pipeline_sha256=pipeline_sha256)

    metadata_path = output_path.parent / "preprocessing_metadata.json"
    try:
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(updated_metadata.to_dict(), fh, indent=2)
    except Exception as exc:  # noqa: BLE001
        raise TrainingPreprocessingError(
            f"Failed to save preprocessing metadata to {metadata_path}: {exc}"
        ) from exc

    logger.info(
        "training_preprocessing.pipeline_saved",
        extra={
            "pipeline_path": str(output_path),
            "metadata_path": str(metadata_path),
            "sha256": pipeline_sha256,
        },
    )

    return updated_metadata
