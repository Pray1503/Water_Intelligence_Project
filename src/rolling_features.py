"""
Rolling Window Feature Engineering Module.

Generates leakage-free rolling window statistics grouped by
administrative districts.

Pipeline Phase:
    Stage 7 - Feature Engineering

Compatibility:
    Stage 6 Master Dataset
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

__all__ = ["generate_rolling_features"]

logger = logging.getLogger(__name__)


def _validate_rolling_inputs(
    df: pd.DataFrame,
    group_column: str,
    date_column: str,
    rolling_config: Dict[str, Dict[str, Any]],
) -> None:
    """
    Validate rolling feature engineering inputs.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if group_column not in df.columns:
        raise ValueError(f"Missing required group column '{group_column}'.")

    if date_column not in df.columns:
        raise ValueError(f"Missing required date column '{date_column}'.")

    if not isinstance(rolling_config, dict):
        raise TypeError("rolling_config must be a dictionary.")

    valid_stats = {
        "mean",
        "median",
        "std",
        "min",
        "max",
        "sum",
    }

    for feature, config in rolling_config.items():

        if not isinstance(config, dict):
            raise TypeError(f"Configuration for '{feature}' must be a dictionary.")

        windows = config.get("windows", [])
        stats = config.get("stats", [])

        if not isinstance(windows, (list, tuple)):
            raise TypeError(f"'windows' for '{feature}' must be a list or tuple.")

        for window in windows:

            if not isinstance(window, int) or window <= 0:

                raise ValueError(
                    f"Invalid rolling window '{window}' " f"for '{feature}'."
                )

        if not isinstance(stats, (list, tuple)):
            raise TypeError(f"'stats' for '{feature}' must be a list or tuple.")

        for stat in stats:

            if stat not in valid_stats:

                raise ValueError(
                    f"Unsupported statistic '{stat}'. "
                    f"Valid statistics: {sorted(valid_stats)}"
                )


def generate_rolling_features(
    df: pd.DataFrame,
    rolling_config: Dict[str, Dict[str, Any]],
    group_column: str = "District LGD Code",
    date_column: str = "Date",
    strict: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Generate leakage-free rolling window statistics.

    Notes
    -----
    Rolling features are calculated on shifted data
    (`shift(1)`) so that values from the current day are
    never included in the rolling statistics.

    Returns
    -------
    (
        Engineered DataFrame,
        Number of rolling features created
    )
    """

    start_time = time.perf_counter()

    logger.info("Starting rolling feature engineering...")

    _validate_rolling_inputs(
        df,
        group_column,
        date_column,
        rolling_config,
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

    skipped_columns: List[str] = []

    feature_summary: Dict[str, int] = {}

    # -------------------------------------------------------------
    # Generate rolling features
    # -------------------------------------------------------------

    for source_column, config in rolling_config.items():

        if source_column not in df_out.columns:

            message = f"Configured column '{source_column}' " "does not exist."

            if strict:
                raise ValueError(message)

            logger.warning(message)

            skipped_columns.append(source_column)

            continue

        windows: Sequence[int] = sorted(set(config.get("windows", [])))

        stats: Sequence[str] = sorted(set(config.get("stats", ["mean"])))

        shifted = grouped[source_column].shift(1)

        shifted_group = shifted.groupby(
            df_out[group_column],
            sort=False,
        )

        created = 0

        for window in windows:

            rolling = shifted_group.rolling(
                window=window,
                min_periods=1,
            )

            for stat in stats:

                feature_name = f"{source_column}" f"_rolling_{window}d_{stat}"

                logger.debug(
                    "Generating %s",
                    feature_name,
                )

                if stat == "mean":
                    values = rolling.mean()

                elif stat == "median":
                    values = rolling.median()

                elif stat == "std":
                    values = rolling.std()

                elif stat == "min":
                    values = rolling.min()

                elif stat == "max":
                    values = rolling.max()

                elif stat == "sum":
                    values = rolling.sum()

                else:
                    continue

                df_out[feature_name] = values.reset_index(
                    level=0,
                    drop=True,
                )

                created += 1

        feature_summary[source_column] = created

    features_created = len(df_out.columns) - initial_columns

    elapsed = time.perf_counter() - start_time

    # -------------------------------------------------------------
    # Logging summary
    # -------------------------------------------------------------

    for feature, count in feature_summary.items():

        logger.info(
            "%s -> %d rolling features",
            feature,
            count,
        )

    if skipped_columns:

        logger.warning(
            "Skipped configured columns: %s",
            skipped_columns,
        )

    logger.info("Rolling feature engineering completed successfully.")

    logger.info(
        "Created %d rolling features in %.3f seconds.",
        features_created,
        elapsed,
    )

    return (
        df_out,
        features_created,
    )
