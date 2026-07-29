"""
Feature Dataset Validation Module.

Performs comprehensive quality assurance and schema validation
for the engineered feature dataset before export.

Pipeline Phase:
    Stage 7 - Feature Engineering

Compatibility:
    Stage 6 Validation Framework
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "validate_feature_dataset",
    "generate_validation_report",
]

logger = logging.getLogger(__name__)


def _validate_validation_inputs(
    df: pd.DataFrame,
    required_columns: List[str],
) -> List[str]:
    """
    Validate validation inputs.

    Returns
    -------
    List[str]
        Normalized required column list.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if len(df.columns) == 0:
        raise ValueError("Input DataFrame contains no columns.")

    if not isinstance(required_columns, (list, tuple)):
        raise TypeError("'required_columns' must be a list or tuple.")

    normalized_columns = sorted(set(required_columns))

    for column in normalized_columns:

        if not isinstance(column, str):

            raise TypeError("Every required column name must be a string.")

    return normalized_columns


def validate_feature_dataset(
    df: pd.DataFrame,
    required_columns: List[str],
    date_column: str = "Date",
    strict: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the engineered feature dataset.

    Returns
    -------
    (
        Validation status,
        Validation metrics
    )
    """

    start_time = time.perf_counter()

    logger.info("Starting feature dataset validation...")

    required_columns = _validate_validation_inputs(
        df,
        required_columns,
    )

    df_out = df.copy()

    # -------------------------------------------------------------
    # Datetime validation
    # -------------------------------------------------------------

    if date_column in df_out.columns:

        if not pd.api.types.is_datetime64_any_dtype(df_out[date_column]):

            logger.debug(
                "Converting '%s' to datetime...",
                date_column,
            )

            try:

                df_out[date_column] = pd.to_datetime(
                    df_out[date_column],
                    errors="raise",
                )

            except Exception as exc:

                logger.exception("Datetime validation failed.")

                raise ValueError(
                    f"Unable to parse '{date_column}' " "as datetime."
                ) from exc

    # -------------------------------------------------------------
    # Validation metrics
    # -------------------------------------------------------------

    issues: Dict[str, Any] = {}

    total_rows = len(df_out)
    total_columns = len(df_out.columns)

    missing_columns = [
        column for column in required_columns if column not in df_out.columns
    ]

    issues["missing_required_columns"] = missing_columns

    if missing_columns:

        message = "Missing required columns: " f"{missing_columns}"

        if strict:

            logger.error(message)

            raise ValueError(message)

        logger.warning(message)

    duplicate_rows = int(df_out.duplicated().sum())

    issues["duplicate_rows"] = duplicate_rows

    if duplicate_rows:

        logger.warning(
            "Detected %d duplicate rows.",
            duplicate_rows,
        )

    # -------------------------------------------------------------
    # Numeric audits
    # -------------------------------------------------------------

    numeric_df = df_out.select_dtypes(include=[np.number])

    inf_counts: Dict[str, int] = {}

    for column in numeric_df.columns:

        count = int(np.isinf(numeric_df[column]).sum())

        if count > 0:

            inf_counts[column] = count

    issues["infinite_value_counts"] = inf_counts

    if inf_counts:

        message = f"Infinite values detected in " f"{len(inf_counts)} columns."

        if strict:

            logger.error(message)

            raise ValueError(message)

        logger.warning(message)

    # -------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------

    null_counts = {
        column: int(df_out[column].isna().sum())
        for column in df_out.columns
        if df_out[column].isna().sum() > 0
    }

    issues["missing_value_counts"] = null_counts

    if null_counts:

        logger.info(
            "Detected null values in %d columns. "
            "This may be expected for lag and rolling features.",
            len(null_counts),
        )

    # -------------------------------------------------------------
    # Dataset statistics
    # -------------------------------------------------------------

    numeric_columns = len(df_out.select_dtypes(include=[np.number]).columns)

    categorical_columns = len(
        df_out.select_dtypes(include=["object", "category"]).columns
    )

    datetime_columns = len(df_out.select_dtypes(include=["datetime64[ns]"]).columns)

    memory_usage_mb = df_out.memory_usage(deep=True).sum() / (1024**2)

    elapsed = time.perf_counter() - start_time

    is_valid = not missing_columns and duplicate_rows == 0 and not inf_counts

    metrics: Dict[str, Any] = {
        "is_valid": is_valid,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "memory_usage_mb": round(
            memory_usage_mb,
            2,
        ),
        "duplicate_count": duplicate_rows,
        "missing_columns_count": len(missing_columns),
        "columns_with_inf": len(inf_counts),
        "columns_with_nulls": len(null_counts),
        "issues": issues,
        "validation_time_seconds": round(
            elapsed,
            4,
        ),
    }

    # -------------------------------------------------------------
    # Logging summary
    # -------------------------------------------------------------

    logger.info(
        "Rows: %d | Columns: %d",
        total_rows,
        total_columns,
    )

    logger.info(
        "Numeric: %d | Categorical: %d | Datetime: %d",
        numeric_columns,
        categorical_columns,
        datetime_columns,
    )

    logger.info(
        "Memory Usage: %.2f MB",
        memory_usage_mb,
    )

    logger.info(
        "Validation %s in %.3f seconds.",
        "PASSED" if is_valid else "FAILED",
        elapsed,
    )

    return (
        is_valid,
        metrics,
    )


def generate_validation_report(
    df: pd.DataFrame,
    validation_metrics: Dict[str, Any],
    engineered_feature_count: int,
) -> Dict[str, Any]:
    """
    Generate a JSON-serializable validation report.

    Returns
    -------
    Dict[str, Any]
        Structured validation report.
    """

    logger.debug("Generating validation report...")

    report: Dict[str, Any] = {
        "status": (
            "SUCCESS"
            if validation_metrics.get(
                "is_valid",
                False,
            )
            else "FAILED"
        ),
        "summary": {
            "total_records": validation_metrics.get(
                "total_rows",
                len(df),
            ),
            "total_columns": validation_metrics.get(
                "total_columns",
                len(df.columns),
            ),
            "engineered_features_count": engineered_feature_count,
            "memory_usage_mb": validation_metrics.get(
                "memory_usage_mb",
                0.0,
            ),
            "validation_execution_seconds": validation_metrics.get(
                "validation_time_seconds",
                0.0,
            ),
        },
        "audit_results": validation_metrics.get(
            "issues",
            {},
        ),
        "column_dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }

    return report
