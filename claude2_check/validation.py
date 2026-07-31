"""Stage 10 - Layer 2: Dataset Schema and Metadata Validation.

Validates that the input DatasetBundle strictly conforms to expected schema,
feature ordering, target existence, non-null requirements, variance checks,
and preprocessing metadata integrity before model instantiation or training begins.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.stage10.dataset_bundle import DatasetBundle
from src.stage10.exceptions import (
    DataValidationError,
    MetadataValidationError,
    SchemaValidationError,
)
from src.stage10.logging_utils import log_call

logger = logging.getLogger("stage10")


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration requirements for dataset schema and metadata validation.

    Attributes
    ----------
    target_column:
        Name of the target column expected in each dataset split.
    feature_columns:
        Exact list and order of expected feature column names.
    preprocessing_metadata_path:
        Path to the Stage 9 preprocessing_metadata.json file.
    expected_preprocessing_hash:
        Optional hash (pipeline_sha256 or preprocessing_hash) to verify.
    check_variance_splits:
        Tuple of split names ("train", "validation", "test") to verify feature variance.
    """

    target_column: str
    feature_columns: list[str]
    preprocessing_metadata_path: Path
    expected_preprocessing_hash: str | None = None
    check_variance_splits: tuple[str, ...] = field(default_factory=lambda: ("train",))


@log_call
def validate_preprocessing_metadata(
    metadata_path: Path,
    expected_hash: str | None = None,
) -> dict:
    """Validate existence, JSON syntax, and hash consistency of preprocessing metadata.

    Parameters
    ----------
    metadata_path:
        Path to preprocessing_metadata.json.
    expected_hash:
        Optional expected pipeline hash.

    Returns
    -------
    dict
        Parsed metadata JSON contents.

    Raises
    ------
    MetadataValidationError
        If file does not exist, is invalid JSON, or hash mismatches.
    """
    if not metadata_path.exists():
        raise MetadataValidationError(
            f"Preprocessing metadata file not found at: {metadata_path}"
        )

    try:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except Exception as exc:
        raise MetadataValidationError(
            f"Failed to parse preprocessing metadata JSON at {metadata_path}: {exc}"
        ) from exc

    if expected_hash is not None:
        actual_hash = metadata.get("preprocessing_hash") or metadata.get(
            "pipeline_sha256"
        )
        if actual_hash != expected_hash:
            raise MetadataValidationError(
                f"Preprocessing hash mismatch! Expected: '{expected_hash}', "
                f"found in metadata: '{actual_hash}'."
            )

    return metadata


@log_call
def validate_split_schema(
    df: pd.DataFrame,
    split_name: str,
    target_column: str,
    expected_features: Sequence[str],
) -> None:
    """Validate column existence, types, non-emptiness, and exact ordering for a split.

    Parameters
    ----------
    df:
        DataFrame split to validate.
    split_name:
        Name of the split (e.g., 'train', 'validation', 'test').
    target_column:
        Name of target column.
    expected_features:
        Sequence of expected feature column names.

    Raises
    ------
    SchemaValidationError
    """
    if df.empty:
        raise SchemaValidationError(f"Dataset split '{split_name}' is empty.")

    # Check target presence
    if target_column not in df.columns:
        raise SchemaValidationError(
            f"Target column '{target_column}' missing in split '{split_name}'. "
            f"Available columns: {list(df.columns[:5])}..."
        )

    # Check target nulls
    if df[target_column].isnull().any():
        null_count = df[target_column].isnull().sum()
        raise DataValidationError(
            f"Target column '{target_column}' in split '{split_name}' "
            f"contains {null_count} null/NaN values."
        )

    # Check features presence and exact column ordering
    actual_features = [col for col in df.columns if col != target_column]
    expected_features_list = list(expected_features)

    missing_features = set(expected_features_list) - set(df.columns)
    if missing_features:
        raise SchemaValidationError(
            f"Split '{split_name}' is missing {len(missing_features)} expected features: "
            f"{sorted(missing_features)[:5]}"
        )

    if actual_features != expected_features_list:
        raise SchemaValidationError(
            f"Feature column order mismatch in split '{split_name}'. "
            f"Expected {len(expected_features_list)} features in specific order."
        )


@log_call
def validate_feature_variance(
    df: pd.DataFrame,
    split_name: str,
    feature_columns: Sequence[str],
) -> None:
    """Ensure no feature column in the specified split has zero variance (constant value).

    Parameters
    ----------
    df:
        DataFrame split to validate.
    split_name:
        Name of the split.
    feature_columns:
        Features to inspect for variance.

    Raises
    ------
    DataValidationError
        If zero-variance features are found.
    """
    zero_variance_cols = []
    for col in feature_columns:
        if col in df.columns and df[col].nunique(dropna=False) <= 1:
            zero_variance_cols.append(col)

    if zero_variance_cols:
        raise DataValidationError(
            f"Found {len(zero_variance_cols)} feature(s) with zero variance "
            f"in split '{split_name}': {zero_variance_cols[:5]}"
        )


@log_call
def validate_dataset_bundle(
    bundle: DatasetBundle,
    config: ValidationConfig,
) -> None:
    """Perform comprehensive validation of a DatasetBundle against ValidationConfig.

    Checks:
    1. Preprocessing metadata existence and hash validation.
    2. Schema, column presence, ordering, and nulls for train, validation, and test splits.
    3. Feature variance checks on designated splits.

    Parameters
    ----------
    bundle:
        The DatasetBundle instance containing train, validation, and test DataFrames.
    config:
        ValidationConfig specifying expected schema and metadata requirements.

    Raises
    ------
    MetadataValidationError
    SchemaValidationError
    DataValidationError
    """
    logger.info("Validating Stage 9 preprocessing metadata...")
    validate_preprocessing_metadata(
        metadata_path=config.preprocessing_metadata_path,
        expected_hash=config.expected_preprocessing_hash,
    )

    splits = {
        "train": bundle.train,
        "validation": bundle.validation,
        "test": bundle.test,
    }

    for split_name, df in splits.items():
        logger.info("Validating schema for split '%s'...", split_name)
        validate_split_schema(
            df=df,
            split_name=split_name,
            target_column=config.target_column,
            expected_features=config.feature_columns,
        )

        if split_name in config.check_variance_splits:
            logger.info("Validating feature variance for split '%s'...", split_name)
            validate_feature_variance(
                df=df,
                split_name=split_name,
                feature_columns=config.feature_columns,
            )

    logger.info("DatasetBundle schema and metadata validation completed successfully.")
