"""
Stage 4.5 -- District-Day Aggregation.

Contains all reusable aggregation logic used by
scripts/05_aggregate_district_day.py:

    * dataset category detection (from filename)
    * measurement column detection (from category)
    * aggregation rule selection (sum vs mean)
    * schema validation
    * district-day aggregation
    * sanity checks
    * report generation / JSON persistence
    * debug helpers

No feature engineering, no ML, no GIS. This module produces one row per
(District LGD Code, Date) per category, with a Station Count showing how
many stations actually contributed a valid (non-null) reading to that row.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global debug switch. When True, aggregate_to_district_day() and the
# pipeline runner print detailed per-dataset diagnostics. When False, only
# normal logging statements are emitted.
# ---------------------------------------------------------------------------
DEBUG: bool = True

# Standard separator used throughout debug output.
DEBUG_SEPARATOR = "=" * 70
# ---------------------------------------------------------------------------
# Canonical column names shared by every cleaned dataset.
# ---------------------------------------------------------------------------
COL_DISTRICT_LGD_CODE = "District LGD Code"
COL_DISTRICT = "District"
COL_STATION = "Station"
COL_TIMESTAMP = "Data Acquisition Time"
COL_DATE = "Date"
COL_MEASUREMENT = "Measurement"
COL_STATION_COUNT = "Station Count"

REQUIRED_SHARED_COLUMNS: List[str] = [
    COL_DISTRICT_LGD_CODE,
    COL_DISTRICT,
    COL_STATION,
    COL_TIMESTAMP,
]

# ---------------------------------------------------------------------------
# Category registry -- one entry per real-world measurement category.
# Keys are the canonical, internal category names used throughout Stage 4.5.
# ---------------------------------------------------------------------------
CATEGORY_MEASUREMENT_COLUMNS: Dict[str, str] = {
    "groundwater": "Groundwater Level Telemetry 6 Hourly (meter)",
    "humidity": "Telemetry Hourly Relative Humidity (%)",
    "rainfall": "Telemetry Hourly Rainfall (mm)",
    "river_level": "River Water Level Telemetry Hourly (meter)",
    "temperature": "Air Temperature Telemetry Hourly (AoC)",
}

CATEGORY_AGGREGATION_METHOD: Dict[str, str] = {
    "groundwater": "mean",
    "humidity": "mean",
    "rainfall": "sum",
    "river_level": "mean",
    "temperature": "mean",
}

CATEGORY_OUTPUT_FILENAMES: Dict[str, str] = {
    "groundwater": "groundwater_daily_district.csv",
    "humidity": "humidity_daily_district.csv",
    "rainfall": "rainfall_daily_district.csv",
    "river_level": "river_level_daily_district.csv",
    "temperature": "temperature_daily_district.csv",
}

# Filename keywords used to detect which category a cleaned CSV belongs to.
# Matching is case-insensitive substring matching against the file stem.
# Order matters: more specific keywords are checked first to avoid
# accidental collisions (none currently collide, but this keeps the
# detector safe if new categories are added later).
_CATEGORY_FILENAME_KEYWORDS: List[Tuple[str, str]] = [
    ("gwl", "groundwater"),
    ("humid", "humidity"),
    ("rainfall", "rainfall"),
    ("rwl", "river_level"),
    ("temperature", "temperature"),
]


class SchemaValidationError(Exception):
    """Raised when a cleaned dataset is missing a required column."""


class AggregationSanityCheckError(Exception):
    """Raised when an aggregated output fails a pre-save sanity check."""


class CategoryDetectionError(Exception):
    """Raised when a cleaned CSV's category cannot be determined from its
    filename."""


def detect_dataset_category(filename: str) -> str:
    """Detect which measurement category a cleaned CSV belongs to, based on
    keyword matching against the filename.

    Parameters
    ----------
    filename:
        The file name (or stem) of a cleaned CSV, e.g.
        "gwl_tel_6_hourly_gujarat_sw_gw_gj_2021_2025_cleaned.csv".

    Returns
    -------
    The canonical category name, one of: "groundwater", "humidity",
    "rainfall", "river_level", "temperature".

    Raises
    ------
    CategoryDetectionError
        If no known keyword is found in the filename.
    """
    lowered = filename.lower()
    for keyword, category in _CATEGORY_FILENAME_KEYWORDS:
        if keyword in lowered:
            return category

    raise CategoryDetectionError(
        f"Could not detect a measurement category for file '{filename}'. "
        f"Expected one of the following keywords in the filename: "
        f"{[kw for kw, _ in _CATEGORY_FILENAME_KEYWORDS]}."
    )


def get_measurement_column(category: str) -> str:
    """Return the canonical measurement column name for *category*.

    Raises
    ------
    KeyError
        If *category* is not a registered category.
    """
    if category not in CATEGORY_MEASUREMENT_COLUMNS:
        raise KeyError(
            f"Unknown category '{category}'. Registered categories: "
            f"{sorted(CATEGORY_MEASUREMENT_COLUMNS)}"
        )
    return CATEGORY_MEASUREMENT_COLUMNS[category]


def get_aggregation_method(category: str) -> str:
    """Return the aggregation method ("sum" or "mean") for *category*.

    Raises
    ------
    KeyError
        If *category* is not a registered category.
    """
    if category not in CATEGORY_AGGREGATION_METHOD:
        raise KeyError(
            f"Unknown category '{category}'. Registered categories: "
            f"{sorted(CATEGORY_AGGREGATION_METHOD)}"
        )
    return CATEGORY_AGGREGATION_METHOD[category]


def detect_measurement_column(df: pd.DataFrame, category: str) -> str:
    """Look up the expected measurement column for *category* and verify it
    is actually present in *df*.

    Raises
    ------
    SchemaValidationError
        If the expected measurement column is missing from *df*.
    """
    measurement_column = get_measurement_column(category)
    if measurement_column not in df.columns:
        raise SchemaValidationError(
            f"Expected measurement column '{measurement_column}' for "
            f"category '{category}' was not found in the dataset. "
            f"Available columns: {list(df.columns)}"
        )
    return measurement_column


def validate_dataset_schema(df: pd.DataFrame, category: str, source_name: str) -> str:
    """Validate that *df* contains every column Stage 4.5 needs before it
    can be aggregated: the shared key columns plus the category's
    measurement column.

    Also verifies that ``District LGD Code`` contains no nulls, since it is
    the primary aggregation key and a null key would silently drop rows
    from every downstream group.

    Parameters
    ----------
    df:
        The cleaned dataset to validate.
    category:
        The canonical category name (e.g. "rainfall").
    source_name:
        A human-readable name for *df*, used in error messages.

    Returns
    -------
    The validated measurement column name for *category*.

    Raises
    ------
    SchemaValidationError
        If any required column is missing, or if ``District LGD Code``
        contains null values.
    """
    missing_columns = [
        column for column in REQUIRED_SHARED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise SchemaValidationError(
            f"Dataset '{source_name}' (category='{category}') is missing "
            f"required column(s): {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    measurement_column = detect_measurement_column(df, category)

    null_district_codes = int(df[COL_DISTRICT_LGD_CODE].isna().sum())
    if null_district_codes > 0:
        raise SchemaValidationError(
            f"Dataset '{source_name}' (category='{category}') has "
            f"{null_district_codes} row(s) with a null '{COL_DISTRICT_LGD_CODE}'. "
            "This column is the primary aggregation key and must not be null."
        )

    if DEBUG:
        print(f"[VALIDATION] '{source_name}' (category='{category}') PASSED")
        print(f"[VALIDATION]   required columns present : {REQUIRED_SHARED_COLUMNS}")
        print(f"[VALIDATION]   measurement column        : {measurement_column}")

    return measurement_column


def extract_date_column(
    df: pd.DataFrame, timestamp_col: str = COL_TIMESTAMP
) -> pd.Series:
    """Derive a calendar-date column from *timestamp_col*.

    The cleaned datasets already contain normalized timestamps from
    Stage 4. Parse them defensively and fail if any value cannot be
    interpreted as a valid datetime.

    Returns
    -------
    A pandas Series of ``datetime.date`` objects.

    Raises
    ------
    SchemaValidationError
        If any timestamp value cannot be parsed.
    """

    parsed = pd.to_datetime(
        df[timestamp_col].astype(str).str.strip(),
        errors="coerce",
        format="mixed",
    )

    unparseable = int(parsed.isna().sum())

    if unparseable > 0:
        raise SchemaValidationError(
            f"Column '{timestamp_col}' contains {unparseable} value(s) that "
            "could not be parsed as a datetime. Stage 4.5 requires every "
            "timestamp to be valid."
        )

    return parsed.dt.date


def _print_debug_dataset_summary(
    dataset_name: str,
    source_files: List[str],
    measurement_column: str,
    aggregation_rule: str,
    rows_before: int,
    rows_after: int,
    stations_used: int,
    missing_measurements: int,
    district_count: int,
    date_range: Tuple[str, str],
    execution_time_seconds: float,
    output_file: str,
) -> None:
    """Print the standardized debug block for one aggregated dataset."""
    print(DEBUG_SEPARATOR)
    print("DATASET")
    print(DEBUG_SEPARATOR)
    print(f"Dataset Name        : {dataset_name}")
    print(f"Input File(s)       : {source_files}")
    print(f"Measurement Column  : {measurement_column}")
    print(f"Aggregation Rule    : {aggregation_rule}")
    print(f"Rows Before         : {rows_before:,}")
    print(f"Rows After          : {rows_after:,}")
    print(f"Stations Used       : {stations_used:,}")
    print(f"Missing Measurements: {missing_measurements:,}")
    print(f"District Count      : {district_count}")
    print(f"Date Range          : {date_range[0]} -> {date_range[1]}")
    print(f"Execution Time      : {execution_time_seconds:.3f}s")
    print(f"Output File         : {output_file}")
    print(DEBUG_SEPARATOR)


def check_no_duplicate_keys(df: pd.DataFrame, dataset_name: str) -> None:
    """Verify that *df* has no duplicate (District LGD Code, Date) rows.

    Raises
    ------
    AggregationSanityCheckError
        If any duplicate key is found.
    """
    duplicate_mask = df.duplicated(subset=[COL_DISTRICT_LGD_CODE, COL_DATE], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        raise AggregationSanityCheckError(
            f"Aggregated dataset '{dataset_name}' contains {duplicate_count} "
            f"row(s) that duplicate a ({COL_DISTRICT_LGD_CODE}, {COL_DATE}) "
            "key. Aggregation must produce exactly one row per key."
        )


def check_station_count_positive(df: pd.DataFrame, dataset_name: str) -> None:
    """Verify that every row has ``Station Count`` > 0.

    Raises
    ------
    AggregationSanityCheckError
        If any row has a non-positive station count.
    """
    non_positive_count = int((df[COL_STATION_COUNT] <= 0).sum())
    if non_positive_count > 0:
        raise AggregationSanityCheckError(
            f"Aggregated dataset '{dataset_name}' contains {non_positive_count} "
            f"row(s) with a non-positive '{COL_STATION_COUNT}'. Every "
            "aggregated row must be backed by at least one contributing "
            "station."
        )


def check_measurement_numeric(df: pd.DataFrame, dataset_name: str) -> None:
    """Verify that the ``Measurement`` column is numeric.

    Raises
    ------
    AggregationSanityCheckError
        If the column is not a numeric dtype.
    """
    if not pd.api.types.is_numeric_dtype(df[COL_MEASUREMENT]):
        raise AggregationSanityCheckError(
            f"Aggregated dataset '{dataset_name}' has a non-numeric "
            f"'{COL_MEASUREMENT}' column (dtype={df[COL_MEASUREMENT].dtype})."
        )


def run_sanity_checks(df: pd.DataFrame, dataset_name: str) -> None:
    """Run every pre-save sanity check on an aggregated dataframe.

    Raises
    ------
    AggregationSanityCheckError
        If any individual check fails. The first failing check's
        exception propagates.
    """
    check_no_duplicate_keys(df, dataset_name)
    check_station_count_positive(df, dataset_name)
    check_measurement_numeric(df, dataset_name)

    if DEBUG:
        print(f"[SANITY CHECK] '{dataset_name}' PASSED all pre-save checks")
        print(f"[SANITY CHECK]   no duplicate (District LGD Code, Date) keys")
        print(f"[SANITY CHECK]   station count > 0 for every row")
        print(f"[SANITY CHECK]   measurement column is numeric")
        print(f"[SANITY CHECK]   output dataframe shape : {df.shape}")


def consolidate_category_files(
    file_frames: List[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Concatenate multiple cleaned datasets belonging to the same category
    (e.g. five yearly humidity files) into a single frame.

    This must run *before* aggregation, not after, so that Stage 4.5
    aggregates the full multi-year history of a category in one pass and
    produces a single, non-duplicated output file per category.

    Parameters
    ----------
    file_frames:
        A list of (source_filename, dataframe) pairs belonging to the same
        category, in any order.

    Returns
    -------
    A single concatenated dataframe.
    """
    if DEBUG:
        print(f"[CONSOLIDATE] merging {len(file_frames)} file(s):")
        for name, frame in file_frames:
            print(f"[CONSOLIDATE]   {name} -> {len(frame):,} rows")

    frames = [frame for _, frame in file_frames]
    consolidated = pd.concat(frames, ignore_index=True, sort=False)

    if DEBUG:
        print(f"[CONSOLIDATE] total rows after consolidation: {len(consolidated):,}")

    return consolidated


def aggregate_to_district_day(
    df: pd.DataFrame,
    category: str,
    dataset_name: str,
    source_files: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate a cleaned dataset to one row per (District LGD Code, Date)
    for the given *category*.

    Rows whose measurement value is null are excluded before grouping, so
    that:
      * NaN values are ignored during aggregation (per spec), and
      * every output row is guaranteed to have Station Count > 0 -- a
        district-date group with zero valid readings simply produces no
        output row at all, rather than a row with a fabricated zero or an
        all-NaN aggregate.
    A district is never dropped entirely just because one of its stations
    is missing a reading -- only the individual missing readings are
    excluded, not the whole district-day group.

    Parameters
    ----------
    df:
        The (already consolidated, if multi-file) cleaned dataset.
    category:
        Canonical category name, e.g. "rainfall".
    dataset_name:
        Human-readable name used in logs, debug output, and reports.
    source_files:
        The filenames that contributed to *df* (for reporting only).

    Returns
    -------
    (aggregated_df, report) where aggregated_df has columns
    [District LGD Code, District, Date, Measurement, Station Count],
    sorted by District LGD Code then Date, and report is a dict of
    aggregation statistics ready to be JSON-serialized.

    Raises
    ------
    SchemaValidationError
        If schema validation fails.
    AggregationSanityCheckError
        If the aggregated output fails a sanity check.
    """
    start_time = time.perf_counter()

    if DEBUG:
        print("\n" + "=" * 70)
        print(f"Starting aggregation for category: {category}")
        print("=" * 70)

    measurement_column = validate_dataset_schema(
        df,
        category,
        dataset_name,
    )
    aggregation_method = get_aggregation_method(category)

    rows_before = len(df)

    working = df.copy()

    # Defensive cleanup in case CSV headers contain leading/trailing spaces.
    working.columns = [str(col).strip() for col in working.columns]
    working[COL_DATE] = extract_date_column(working, COL_TIMESTAMP)

    missing_measurements = int(working[measurement_column].isna().sum())

    # Ignore NaN measurements entirely -- they contribute neither to the
    # aggregate value nor to the station count. This also guarantees every
    # resulting group has Station Count > 0, since a group only exists if
    # at least one valid measurement fed into it.

    valid = working.loc[working[measurement_column].notna()].copy()

    if DEBUG:
        print(f"Valid measurement rows : {len(valid):,}")
        print(f"Dropped missing rows   : {missing_measurements:,}")

    grouped = valid.groupby([COL_DISTRICT_LGD_CODE, COL_DATE], dropna=False)

    measurement_agg = grouped[measurement_column].agg(aggregation_method)
    station_count = grouped[COL_STATION].nunique()
    # District name is constant within a District LGD Code, so the first
    # observed value in each group is used as the representative label.
    district_name = grouped[COL_DISTRICT].first()

    aggregated = pd.DataFrame(
        {
            COL_DISTRICT_LGD_CODE: measurement_agg.index.get_level_values(
                COL_DISTRICT_LGD_CODE
            ),
            COL_DATE: measurement_agg.index.get_level_values(COL_DATE),
            COL_MEASUREMENT: measurement_agg.values,
            COL_STATION_COUNT: station_count.values,
        }
    )
    aggregated[COL_DISTRICT] = district_name.values
    aggregated = aggregated[
        [
            COL_DISTRICT_LGD_CODE,
            COL_DISTRICT,
            COL_DATE,
            COL_MEASUREMENT,
            COL_STATION_COUNT,
        ]
    ]

    aggregated = aggregated.sort_values(
        by=[COL_DISTRICT_LGD_CODE, COL_DATE], kind="mergesort"
    ).reset_index(drop=True)

    run_sanity_checks(aggregated, dataset_name)

    execution_time_seconds = time.perf_counter() - start_time

    rows_after = len(aggregated)
    district_count = int(aggregated[COL_DISTRICT_LGD_CODE].nunique())

    if DEBUG:
        print(f"Unique district-day groups : {rows_after:,}")

    if rows_after > 0:
        date_range = (
            str(aggregated[COL_DATE].min()),
            str(aggregated[COL_DATE].max()),
        )
        station_count_stats = {
            "min": int(aggregated[COL_STATION_COUNT].min()),
            "max": int(aggregated[COL_STATION_COUNT].max()),
            "mean": float(aggregated[COL_STATION_COUNT].mean()),
            "median": float(aggregated[COL_STATION_COUNT].median()),
        }
    else:
        date_range = ("N/A", "N/A")
        station_count_stats = {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

    report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "category": category,
        "source_files": source_files,
        "input_rows": rows_before,
        "output_rows": rows_after,
        "aggregation_method": aggregation_method,
        "measurement_column": measurement_column,
        "missing_measurements": missing_measurements,
        "station_count_statistics": station_count_stats,
        "district_count": district_count,
        "date_range": {"start": date_range[0], "end": date_range[1]},
        "execution_time_seconds": execution_time_seconds,
    }

    if DEBUG:
        _print_debug_dataset_summary(
            dataset_name=dataset_name,
            source_files=source_files,
            measurement_column=measurement_column,
            aggregation_rule=aggregation_method,
            rows_before=rows_before,
            rows_after=rows_after,
            stations_used=int(valid[COL_STATION].nunique()),
            missing_measurements=missing_measurements,
            district_count=district_count,
            date_range=date_range,
            execution_time_seconds=execution_time_seconds,
            output_file=CATEGORY_OUTPUT_FILENAMES[category],
        )
    else:
        logger.info(
            "Aggregated %s: %s rows -> %s rows (%s districts, %.3fs)",
            dataset_name,
            f"{rows_before:,}",
            f"{rows_after:,}",
            district_count,
            execution_time_seconds,
        )

    if DEBUG:
        compression_ratio = rows_before / rows_after if rows_after > 0 else 0

        print(f"Compression Ratio : {compression_ratio:.2f}:1")
        print("=" * 70)

    return aggregated, report


def write_aggregation_report(report: Dict[str, Any], output_path: Path) -> None:
    """Write *report* to *output_path* as pretty-printed JSON.

    Creates parent directories if they do not already exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    if DEBUG:
        print(f"[REPORT] saved -> {output_path}")
    else:
        logger.info("Saved aggregation report: %s", output_path)


def check_no_duplicate_output_filenames(filenames: List[str]) -> None:
    """Verify that no two categories would write to the same output
    filename.

    Raises
    ------
    AggregationSanityCheckError
        If a duplicate filename is found.
    """
    seen: Dict[str, int] = {}
    for name in filenames:
        seen[name] = seen.get(name, 0) + 1

    duplicates = {name: count for name, count in seen.items() if count > 1}
    if duplicates:
        raise AggregationSanityCheckError(
            f"Duplicate output filenames detected: {duplicates}. Every "
            "category must write to a unique file."
        )
