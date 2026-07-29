"""
Feature Engineering Pipeline Orchestrator.

Coordinates all Stage 7 feature engineering modules and validates
the final engineered dataset.

Pipeline Phase:
    Stage 7 - Feature Engineering

Compatibility:
    Stage 6 Master Dataset
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

import pandas as pd

from src.feature_validation import (
    generate_validation_report,
    validate_feature_dataset,
)
from src.lag_features import generate_lag_features
from src.rolling_features import generate_rolling_features
from src.temporal_features import generate_temporal_features
from src.trend_features import generate_trend_features

__all__ = ["build_feature_dataset"]

logger = logging.getLogger(__name__)


def _validate_orchestrator_inputs(
    df: pd.DataFrame,
    group_column: str,
    date_column: str,
    feature_config: Dict[str, Any],
) -> None:
    """
    Validate orchestrator inputs.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if len(df.columns) == 0:
        raise ValueError("Input DataFrame contains no columns.")

    if group_column not in df.columns:
        raise ValueError(f"Missing required group column '{group_column}'.")

    if date_column not in df.columns:
        raise ValueError(f"Missing required date column '{date_column}'.")

    if not isinstance(feature_config, dict):
        raise TypeError("feature_config must be a dictionary.")

    if not feature_config:
        raise ValueError("feature_config cannot be empty.")

    for section in ("lags", "rolling", "trends"):
        if section in feature_config:
            if not isinstance(
                feature_config[section],
                dict,
            ):
                raise TypeError(
                    f"Configuration section '{section}' must be a dictionary."
                )


def build_feature_dataset(
    df: pd.DataFrame,
    feature_config: Dict[str, Any],
    group_column: str = "District LGD Code",
    date_column: str = "Date",
    strict: bool = True,
) -> Tuple[pd.DataFrame, Tuple[int, Dict[str, Any]]]:
    """
    Orchestrate the complete Stage 7 feature engineering pipeline.

    Args:
        df: Input master DataFrame from Stage 6.
        feature_config: Central configuration dictionary defining sub-module rules.
        group_column: District identifier column name. Defaults to 'District LGD Code'.
        date_column: Datetime identifier column name. Defaults to 'Date'.
        strict: If True, enforces strict validation behavior across all feature modules.

    Returns:
        Tuple[
            pd.DataFrame,
            Tuple[
                int,
                Dict[str, Any],
            ],
        ]
        Returns the engineered dataset, created feature count, and validation report.

    Raises:
        ValueError: If validation fails or critical feature pipeline error occurs.
        TypeError: If input parameter types are invalid.
    """
    start_time = time.perf_counter()
    logger.info("Starting Master Feature Engineering Orchestration Pipeline...")

    # 1. Validation
    _validate_orchestrator_inputs(df, group_column, date_column, feature_config)

    # 2. Copy DataFrame
    df_out = df.copy()

    # 3. Datetime validation
    # -------------------------------------------------------------
    # Datetime validation
    # -------------------------------------------------------------
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
            logger.exception("Datetime conversion failed.")
            raise ValueError(f"Unable to parse '{date_column}'.") from exc

    # 4. Sorting
    logger.debug(
        "Sorting DataFrame by ['%s', '%s']...",
        group_column,
        date_column,
    )
    df_out = df_out.sort_values(
        by=[
            group_column,
            date_column,
        ]
    ).reset_index(drop=True)

    # 5. Feature Generation Pipeline Setup
    feature_family_counts: Dict[str, int] = {}
    initial_columns = len(df_out.columns)

    # Module 1: Temporal Features
    try:
        logger.info("=" * 60)
        logger.info("Running Temporal Feature Engineering...")
        logger.info("=" * 60)

        df_out, temporal_count = generate_temporal_features(
            df=df_out,
            date_column=date_column,
        )
        feature_family_counts["temporal_features"] = temporal_count

    except Exception:
        logger.exception("Temporal feature generation failed.")
        raise

    # Module 2: Lag Features
    if "lags" in feature_config:
        try:
            logger.info("=" * 60)
            logger.info("Running Lag Feature Engineering...")
            logger.info("=" * 60)

            df_out, lag_count = generate_lag_features(
                df=df_out,
                lag_config=feature_config["lags"],
                group_column=group_column,
                date_column=date_column,
                strict=strict,
            )
            feature_family_counts["lag_features"] = lag_count

        except Exception:
            logger.exception("Lag feature generation failed.")
            raise

    # Module 3: Rolling Features
    if "rolling" in feature_config:
        try:
            logger.info("=" * 60)
            logger.info("Running Rolling Feature Engineering...")
            logger.info("=" * 60)

            df_out, rolling_count = generate_rolling_features(
                df=df_out,
                rolling_config=feature_config["rolling"],
                group_column=group_column,
                date_column=date_column,
                strict=strict,
            )
            feature_family_counts["rolling_features"] = rolling_count

        except Exception:
            logger.exception("Rolling feature generation failed.")
            raise

    # Module 4: Trend Features
    if "trends" in feature_config:
        try:
            logger.info("=" * 60)
            logger.info("Running Trend Feature Engineering...")
            logger.info("=" * 60)

            df_out, trend_count = generate_trend_features(
                df=df_out,
                trend_config=feature_config["trends"],
                group_column=group_column,
                date_column=date_column,
                strict=strict,
            )
            feature_family_counts["trend_features"] = trend_count

        except Exception:
            logger.exception("Trend feature generation failed.")
            raise

    # 6. Feature Count Verification
    total_features_created = len(df_out.columns) - initial_columns

    counted_features = sum(feature_family_counts.values())

    if counted_features != total_features_created:
        logger.warning(
            "Feature count mismatch detected " "(Module Total=%d, Actual=%d).",
            counted_features,
            total_features_created,
        )

    # 7. Quality Audit & Validation
    logger.info("=" * 60)
    logger.info("Running Feature Dataset Validation...")
    logger.info("=" * 60)

    is_valid, validation_metrics = validate_feature_dataset(
        df=df_out,
        required_columns=[
            group_column,
            date_column,
        ],
        date_column=date_column,
        strict=strict,
    )

    if strict and not is_valid:
        raise ValueError("Feature dataset validation failed.")

    # 8. Execution Time calculation (for metrics reporting)
    elapsed_time = time.perf_counter() - start_time

    report = generate_validation_report(
        df=df_out,
        validation_metrics=validation_metrics,
        engineered_feature_count=total_features_created,
    )

    report["feature_breakdown_by_family"] = feature_family_counts

    report["execution_summary"] = {
        "execution_time_seconds": round(
            elapsed_time,
            4,
        ),
        "validation_passed": is_valid,
        "total_features_created": total_features_created,
        "final_dataset_columns": len(df_out.columns),
    }

    # 9. Logging Summary
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 60)

    for family, count in feature_family_counts.items():
        logger.info(
            "%-25s %5d",
            family,
            count,
        )

    logger.info("-" * 60)

    logger.info(
        "Rows                     : %d",
        len(df_out),
    )

    logger.info(
        "Total Features Created   : %d",
        total_features_created,
    )

    logger.info(
        "Final Dataset Columns    : %d",
        len(df_out.columns),
    )

    logger.info(
        "Validation Status        : %s",
        "PASSED" if is_valid else "FAILED",
    )

    logger.info(
        "Execution Time           : %.3f seconds",
        elapsed_time,
    )

    logger.info("=" * 60)

    # 10. Return
    return df_out, (total_features_created, report)
