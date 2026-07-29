"""
Trend and Hydrological Indicator Feature Engineering Module.

Generates trend, difference, percentage change, dry/wet spell,
and hydro-climatic indicator features.

Pipeline Phase:
    Stage 7 - Feature Engineering

Compatibility:
    Stage 6 Master Dataset
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["generate_trend_features"]

logger = logging.getLogger(__name__)


def _calculate_consecutive_spells(
    condition_series: pd.Series,
) -> pd.Series:
    """
    Calculate consecutive occurrences of a boolean condition.
    """

    groups = (~condition_series).cumsum()

    return condition_series.groupby(groups).cumsum()


def _validate_trend_inputs(
    df: pd.DataFrame,
    group_column: str,
    date_column: str,
    trend_config: Dict[str, Any],
) -> None:
    """
    Validate trend feature engineering inputs.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if group_column not in df.columns:
        raise ValueError(f"Missing required group column '{group_column}'.")

    if date_column not in df.columns:
        raise ValueError(f"Missing required date column '{date_column}'.")

    if not isinstance(trend_config, dict):
        raise TypeError("trend_config must be a dictionary.")

    target_columns = trend_config.get(
        "target_columns",
        [],
    )

    if not isinstance(target_columns, (list, tuple)):
        raise TypeError("'target_columns' must be a list or tuple.")

    diff_intervals = trend_config.get(
        "diff_intervals",
        [1, 7],
    )

    if not isinstance(diff_intervals, (list, tuple)):
        raise TypeError("'diff_intervals' must be a list or tuple.")

    for interval in diff_intervals:

        if not isinstance(interval, int):

            raise ValueError(f"Invalid interval '{interval}'.")

        if interval <= 0:

            raise ValueError("Intervals must be positive.")

    threshold = trend_config.get(
        "dry_threshold_mm",
        1.0,
    )

    if not isinstance(threshold, (int, float)):
        raise TypeError("'dry_threshold_mm' must be numeric.")

    if threshold < 0:
        raise ValueError("'dry_threshold_mm' cannot be negative.")


def generate_trend_features(
    df: pd.DataFrame,
    trend_config: Dict[str, Any],
    group_column: str = "District LGD Code",
    date_column: str = "Date",
    strict: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Generate trend-based engineered features.

    Returns
    -------
    (
        Engineered DataFrame,
        Number of features created
    )
    """

    start_time = time.perf_counter()

    logger.info("Starting trend feature engineering...")

    _validate_trend_inputs(
        df,
        group_column,
        date_column,
        trend_config,
    )

    df_out = df.copy()

    # -------------------------------------------------------------
    # Ensure datetime
    # -------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(df_out[date_column]):

        logger.debug(
            "Converting '%s' to datetime...",
            date_column,
        )

        df_out[date_column] = pd.to_datetime(
            df_out[date_column],
            errors="raise",
        )

    # -------------------------------------------------------------
    # Sort dataset
    # -------------------------------------------------------------

    logger.debug(
        "Sorting by '%s' and '%s'...",
        group_column,
        date_column,
    )

    df_out = df_out.sort_values(by=[group_column, date_column]).reset_index(drop=True)

    grouped = df_out.groupby(
        group_column,
        sort=False,
    )

    initial_columns = len(df_out.columns)

    feature_summary: Dict[str, int] = {}

    target_columns = trend_config.get(
        "target_columns",
        [],
    )

    diff_intervals = sorted(
        set(
            trend_config.get(
                "diff_intervals",
                [1, 7],
            )
        )
    )

    rainfall_col: Optional[str] = trend_config.get(
        "rainfall_col",
        "rainfall_mm",
    )

    temperature_col: Optional[str] = trend_config.get(
        "temperature_col",
        "air_temperature",
    )

    dry_threshold = trend_config.get(
        "dry_threshold_mm",
        1.0,
    )

    # -------------------------------------------------------------
    # Difference and Percentage Change
    # -------------------------------------------------------------

    for column in target_columns:

        if column not in df_out.columns:

            message = f"Configured column '{column}' " "does not exist."

            if strict:
                raise ValueError(message)

            logger.warning(message)

            continue

        created = 0

        for interval in diff_intervals:

            lagged = grouped[column].shift(interval)

            change_name = f"{column}_change_{interval}d"

            pct_name = f"{column}_pct_change_{interval}d"

            df_out[change_name] = df_out[column] - lagged

            df_out[pct_name] = (df_out[column] - lagged) / (np.abs(lagged) + 1e-5)

            created += 2

        feature_summary[column] = created

    # -------------------------------------------------------------
    # Rain / Temperature Ratio
    # -------------------------------------------------------------

    if trend_config.get(
        "compute_rain_temp_ratio",
        True,
    ):

        if rainfall_col in df_out.columns and temperature_col in df_out.columns:

            logger.debug("Generating rain_temp_ratio...")

            df_out["rain_temp_ratio"] = df_out[rainfall_col] / (
                df_out[temperature_col] + 1.0
            )

        else:

            logger.warning("Skipping rain_temp_ratio.")

    # -------------------------------------------------------------
    # Dry / Wet Spells
    # -------------------------------------------------------------

    if (
        trend_config.get(
            "compute_dry_wet_spells",
            True,
        )
        and rainfall_col in df_out.columns
    ):

        logger.debug("Generating dry/wet spell features...")

        df_out["consecutive_dry_days"] = (
            df_out.groupby(
                group_column,
                sort=False,
            )[rainfall_col]
            .transform(lambda x: _calculate_consecutive_spells(x < dry_threshold))
            .astype(int)
        )

        df_out["consecutive_wet_days"] = (
            df_out.groupby(
                group_column,
                sort=False,
            )[rainfall_col]
            .transform(lambda x: _calculate_consecutive_spells(x >= dry_threshold))
            .astype(int)
        )

    # -------------------------------------------------------------
    # Rainfall Anomaly
    # -------------------------------------------------------------

    if (
        trend_config.get(
            "compute_anomaly",
            True,
        )
        and rainfall_col in df_out.columns
    ):

        mean_col = f"{rainfall_col}_rolling_30d_mean"

        std_col = f"{rainfall_col}_rolling_30d_std"

        if mean_col in df_out.columns and std_col in df_out.columns:

            logger.debug("Generating rainfall anomaly...")

            df_out["rainfall_30d_anomaly"] = (
                df_out[rainfall_col] - df_out[mean_col]
            ) / (df_out[std_col] + 1e-5)

        else:

            logger.warning(
                "Skipping rainfall anomaly because "
                "required rolling features are missing."
            )

    features_created = len(df_out.columns) - initial_columns

    elapsed = time.perf_counter() - start_time

    # -------------------------------------------------------------
    # Logging summary
    # -------------------------------------------------------------

    for feature, count in feature_summary.items():

        logger.info(
            "%s -> %d trend features",
            feature,
            count,
        )

    logger.info("Trend feature engineering completed successfully.")

    logger.info(
        "Created %d trend features in %.3f seconds.",
        features_created,
        elapsed,
    )

    return (
        df_out,
        features_created,
    )
