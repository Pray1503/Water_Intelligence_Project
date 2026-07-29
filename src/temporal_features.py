"""
Temporal Feature Engineering Module.

Extracts calendar, seasonal, and cyclical features from the primary datetime
column in the Water Intelligence Platform master dataset.

Pipeline Phase:
    Stage 7 - Feature Engineering

Compatibility:
    Stage 6 Master Dataset Schema
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import numpy as np
import pandas as pd

__all__ = ["generate_temporal_features"]

logger = logging.getLogger(__name__)


def _validate_temporal_input(
    df: pd.DataFrame,
    date_column: str,
) -> None:
    """
    Validate the temporal feature engineering inputs.

    Args:
        df:
            Input DataFrame.

        date_column:
            Name of the datetime column.

    Raises:
        TypeError:
            If df is not a pandas DataFrame.

        ValueError:
            If the DataFrame is empty or the date column is missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot generate temporal features.")

    if date_column not in df.columns:
        raise ValueError(
            f"Required date column '{date_column}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )


def generate_temporal_features(
    df: pd.DataFrame,
    date_column: str = "Date",
) -> Tuple[pd.DataFrame, int]:
    """
    Generate temporal features from the Stage 6 master dataset.

    Features Created
    ----------------
    Calendar Features
        year
        month
        quarter
        week
        day
        day_of_week
        day_of_year
        is_month_start
        is_month_end
        is_weekend

    Cyclical Features
        month_sin
        month_cos
        day_of_year_sin
        day_of_year_cos

    Seasonal Features
        is_monsoon
        is_summer
        is_post_monsoon
        is_winter

    Args:
        df:
            Stage 6 master dataset.

        date_column:
            Datetime column.
            Defaults to "Date".

    Returns:
        Tuple consisting of

        (
            Engineered DataFrame,
            Number of features created
        )
    """

    start_time = time.perf_counter()

    logger.info("Starting temporal feature engineering...")

    _validate_temporal_input(df, date_column)

    df_out = df.copy()

    # ------------------------------------------------------------------
    # Convert to datetime
    # ------------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(df_out[date_column]):
        logger.debug("Converting '%s' to datetime...", date_column)

        try:
            df_out[date_column] = pd.to_datetime(
                df_out[date_column],
                errors="raise",
            )
        except Exception as exc:
            logger.exception("Datetime conversion failed.")
            raise ValueError(f"Unable to convert '{date_column}' to datetime.") from exc

    initial_columns = len(df_out.columns)

    dt = df_out[date_column].dt

    # ------------------------------------------------------------------
    # Calendar Features
    # ------------------------------------------------------------------

    logger.debug("Generating calendar features...")

    df_out["year"] = dt.year.astype(int)
    df_out["month"] = dt.month.astype(int)
    df_out["quarter"] = dt.quarter.astype(int)
    df_out["week"] = dt.isocalendar().week.astype(int)
    df_out["day"] = dt.day.astype(int)
    df_out["day_of_week"] = dt.dayofweek.astype(int)
    df_out["day_of_year"] = dt.dayofyear.astype(int)

    df_out["is_month_start"] = dt.is_month_start.astype(int)
    df_out["is_month_end"] = dt.is_month_end.astype(int)

    df_out["is_weekend"] = (df_out["day_of_week"] >= 5).astype(int)

    # ------------------------------------------------------------------
    # Cyclical Features
    # ------------------------------------------------------------------

    logger.debug("Generating cyclical features...")

    df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12)

    df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12)

    days_in_year = np.where(
        dt.is_leap_year,
        366,
        365,
    )

    df_out["day_of_year_sin"] = np.sin(2 * np.pi * df_out["day_of_year"] / days_in_year)

    df_out["day_of_year_cos"] = np.cos(2 * np.pi * df_out["day_of_year"] / days_in_year)

    # ------------------------------------------------------------------
    # Indian Meteorological Seasons
    # ------------------------------------------------------------------

    logger.debug("Generating seasonal indicators...")

    df_out["is_monsoon"] = (df_out["month"].isin([6, 7, 8, 9])).astype(int)

    df_out["is_summer"] = (df_out["month"].isin([3, 4, 5])).astype(int)

    df_out["is_post_monsoon"] = (df_out["month"].isin([10, 11])).astype(int)

    df_out["is_winter"] = (df_out["month"].isin([12, 1, 2])).astype(int)

    features_created = len(df_out.columns) - initial_columns

    elapsed = time.perf_counter() - start_time

    logger.info("Temporal feature engineering completed successfully.")

    logger.info(
        "Created %d features in %.3f seconds.",
        features_created,
        elapsed,
    )

    return df_out, features_created
