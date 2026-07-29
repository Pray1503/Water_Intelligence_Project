"""
Lag Feature Engineering Module.

Generates historical lag features grouped by administrative districts.

Pipeline Phase:
    Stage 7 - Feature Engineering

Compatibility:
    Stage 6 Master Dataset
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import pandas as pd

__all__ = ["generate_lag_features"]

logger = logging.getLogger(__name__)


def _validate_lag_inputs(
    df: pd.DataFrame,
    group_column: str,
    date_column: str,
    lag_config: Dict[str, List[int]],
) -> None:
    """
    Validate lag feature generation inputs.

    Args:
        df:
            Input DataFrame.

        group_column:
            District grouping column.

        date_column:
            Datetime column.

        lag_config:
            Dictionary mapping feature columns to lag horizons.

    Raises:
        TypeError:
            If input types are invalid.

        ValueError:
            If required columns are missing or lag values are invalid.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if group_column not in df.columns:
        raise ValueError(f"Missing required group column '{group_column}'.")

    if date_column not in df.columns:
        raise ValueError(f"Missing required date column '{date_column}'.")

    if not isinstance(lag_config, dict):
        raise TypeError("lag_config must be a dictionary.")

    for feature, lags in lag_config.items():

        if not isinstance(lags, (list, tuple)):
            raise TypeError(
                f"Lag configuration for '{feature}' " "must be a list or tuple."
            )

        for lag in lags:

            if not isinstance(lag, int):

                raise ValueError(f"Lag '{lag}' for '{feature}' " "must be an integer.")

            if lag <= 0:

                raise ValueError(
                    f"Lag '{lag}' for '{feature}' " "must be greater than zero."
                )


def generate_lag_features(
    df: pd.DataFrame,
    lag_config: Dict[str, List[int]],
    group_column: str = "District LGD Code",
    date_column: str = "Date",
    strict: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Generate lag features grouped by district.

    Args
    ----
    df
        Stage 6 master dataset.

    lag_config
        Dictionary of lag definitions.

    group_column
        District grouping column.

    date_column
        Datetime column.

    strict
        If True, missing configured columns raise an exception.
        Otherwise they are skipped with a warning.

    Returns
    -------
    (
        Engineered DataFrame,
        Number of lag features created
    )
    """

    start_time = time.perf_counter()

    logger.info("Starting lag feature engineering...")

    _validate_lag_inputs(
        df,
        group_column,
        date_column,
        lag_config,
    )

    df_out = df.copy()

    # ---------------------------------------------------------------
    # Ensure datetime
    # ---------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(df_out[date_column]):

        logger.debug(
            "Converting '%s' to datetime...",
            date_column,
        )

        df_out[date_column] = pd.to_datetime(
            df_out[date_column],
            errors="raise",
        )

    # ---------------------------------------------------------------
    # Sort data
    # ---------------------------------------------------------------

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

    skipped_columns: List[str] = []

    feature_summary: Dict[str, int] = {}

    # ---------------------------------------------------------------
    # Generate lag features
    # ---------------------------------------------------------------

    for source_column, lag_steps in lag_config.items():

        if source_column not in df_out.columns:

            message = f"Configured column '{source_column}' " "does not exist."

            if strict:
                raise ValueError(message)

            logger.warning(message)

            skipped_columns.append(source_column)

            continue

        lag_steps = sorted(set(lag_steps))

        created = 0

        for lag in lag_steps:

            feature_name = f"{source_column}_lag_{lag}"

            logger.debug(
                "Generating %s",
                feature_name,
            )

            df_out[feature_name] = grouped[source_column].shift(lag)

            created += 1

        feature_summary[source_column] = created

    features_created = len(df_out.columns) - initial_columns

    elapsed = time.perf_counter() - start_time

    # ---------------------------------------------------------------
    # Logging summary
    # ---------------------------------------------------------------

    for feature, count in feature_summary.items():

        logger.info(
            "%s -> %d lag features",
            feature,
            count,
        )

    if skipped_columns:

        logger.warning(
            "Skipped configured columns: %s",
            skipped_columns,
        )

    logger.info("Lag feature engineering completed successfully.")

    logger.info(
        "Created %d lag features in %.3f seconds.",
        features_created,
        elapsed,
    )

    return (
        df_out,
        features_created,
    )
