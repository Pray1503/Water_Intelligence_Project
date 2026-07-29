"""
Stage 6 -- Master Dataset Construction.

Contains all reusable logic used by scripts/06_build_master_dataset.py:

    * category registry (input filenames, rename targets)
    * per-category load, validate, rename, normalize
    * District-column coalescing
    * pairwise and sequential outer-merge logic with merge statistics
    * final validation of the merged master dataset
    * report generation / JSON persistence
    * debug helpers

Stage 6 reads ONLY the district-day aggregated outputs produced by
Stage 4.5 (data/aggregated/*.csv). It never reads data/cleaned, and it
never depends on the old station-level src/loaders.py or src/merger.py
modules -- those belonged to the retired station-level merge
architecture and have no role here.

The five aggregated inputs share an identical schema:
[District LGD Code, District, Date, Measurement, Station Count].
Because "Measurement" and "Station Count" are generically named and
repeated across all five files, this module renames them to
category-qualified names immediately after loading, and merges the
five datasets on (District LGD Code, Date) only -- never on Station,
Timestamp, Latitude, or Longitude, none of which exist in the
aggregated inputs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
MODULE_NAME = "master_dataset"

# ---------------------------------------------------------------------------
# Global debug switch. When True, this module prints detailed per-dataset
# and per-merge-step diagnostics, matching Stage 4.5's DEBUG convention.
# When False, only normal logging statements are emitted.
# ---------------------------------------------------------------------------
DEBUG: bool = True

# Standard separator used throughout debug output.
DEBUG_SEPARATOR = "=" * 70

# ---------------------------------------------------------------------------
# Canonical column names as produced by Stage 4.5's aggregated CSVs.
# ---------------------------------------------------------------------------
COL_DISTRICT_LGD_CODE = "District LGD Code"
COL_DISTRICT = "District"
COL_DATE = "Date"
COL_MEASUREMENT = "Measurement"
COL_STATION_COUNT = "Station Count"

REQUIRED_AGGREGATED_COLUMNS: List[str] = [
    COL_DISTRICT_LGD_CODE,
    COL_DISTRICT,
    COL_DATE,
    COL_MEASUREMENT,
    COL_STATION_COUNT,
]

# The only columns Stage 6 is ever allowed to merge on. Station, Timestamp,
# Latitude, and Longitude do not exist in the aggregated inputs at all --
# this constant exists so the merge key is defined in exactly one place and
# is trivially auditable.
MERGE_KEYS: List[str] = [COL_DISTRICT_LGD_CODE, COL_DATE]

# ---------------------------------------------------------------------------
# Category registry -- one entry per real-world measurement category.
# Keys are the canonical, internal category names, consistent with
# Stage 4.5's category registry in src/aggregation.py.
# ---------------------------------------------------------------------------
CATEGORY_INPUT_FILENAMES: Dict[str, str] = {
    "groundwater": "groundwater_daily_district.csv",
    "humidity": "humidity_daily_district.csv",
    "rainfall": "rainfall_daily_district.csv",
    "river_level": "river_level_daily_district.csv",
    "temperature": "temperature_daily_district.csv",
}

# Target name for each category's renamed "Measurement" column.
CATEGORY_MEASUREMENT_RENAME: Dict[str, str] = {
    "groundwater": "groundwater_level",
    "humidity": "relative_humidity",
    "rainfall": "rainfall_mm",
    "river_level": "river_level",
    "temperature": "air_temperature",
}

# Target name for each category's renamed "Station Count" column.
CATEGORY_STATION_COUNT_RENAME: Dict[str, str] = {
    "groundwater": "groundwater_station_count",
    "humidity": "humidity_station_count",
    "rainfall": "rainfall_station_count",
    "river_level": "river_station_count",
    "temperature": "temperature_station_count",
}

# Fixed, deterministic merge order. Using insertion order of the category
# registry (rather than e.g. sorted() or dict iteration at call time) keeps
# every run's merge sequence -- and therefore every intermediate suffix
# collision, if any -- identical and reproducible.
CATEGORY_MERGE_ORDER: List[str] = list(CATEGORY_INPUT_FILENAMES)

# All measurement / station-count columns the finished master dataset must
# contain, derived from the registries above so they can never drift apart.
MEASUREMENT_COLUMNS: List[str] = [
    CATEGORY_MEASUREMENT_RENAME[category] for category in CATEGORY_MERGE_ORDER
]
STATION_COUNT_COLUMNS: List[str] = [
    CATEGORY_STATION_COUNT_RENAME[category] for category in CATEGORY_MERGE_ORDER
]

# The complete expected schema of the finished master dataset.
EXPECTED_MASTER_COLUMNS: List[str] = (
    [COL_DISTRICT_LGD_CODE, COL_DISTRICT, COL_DATE]
    + MEASUREMENT_COLUMNS
    + STATION_COUNT_COLUMNS
)


class AggregatedFileNotFoundError(Exception):
    """Raised when an expected Stage 4.5 aggregated CSV is missing."""


class SchemaValidationError(Exception):
    """Raised when an aggregated dataset (or the merged master dataset)
    fails a schema, dtype, or null-key check."""


class DuplicateKeyError(Exception):
    """Raised when a dataset contains duplicate (District LGD Code, Date)
    keys where exactly one row per key is required."""


class MergeValidationError(Exception):
    """Raised when a merge step produces an invalid result, or when merge
    inputs are insufficient to proceed."""


class FinalValidationError(Exception):
    """Raised when the finished master dataset fails final validation."""


# ---------------------------------------------------------------------------
# File / schema / dtype validation
# ---------------------------------------------------------------------------


def validate_file_exists(path: Path, category: str) -> None:
    """Verify that the aggregated CSV for *category* exists on disk.

    Raises
    ------
    AggregatedFileNotFoundError
        If *path* does not exist or is not a file.
    """
    if not path.exists() or not path.is_file():
        raise AggregatedFileNotFoundError(
            f"Expected aggregated dataset for category '{category}' at "
            f"'{path}', but no such file exists. Stage 6 reads only from "
            "data/aggregated -- has Stage 4.5 been run?"
        )


def validate_aggregated_schema(
    df: pd.DataFrame, category: str, source_name: str
) -> None:
    """Validate that *df* contains every column Stage 6 expects from a
    Stage 4.5 aggregated CSV, and that the merge key has no nulls.

    Raises
    ------
    SchemaValidationError
        If any required column is missing, or if ``District LGD Code``
        contains null values.
    """
    missing_columns = [
        column for column in REQUIRED_AGGREGATED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise SchemaValidationError(
            f"Aggregated dataset '{source_name}' (category='{category}') is "
            f"missing required column(s): {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    null_district_codes = int(df[COL_DISTRICT_LGD_CODE].isna().sum())
    if null_district_codes > 0:
        raise SchemaValidationError(
            f"Aggregated dataset '{source_name}' (category='{category}') has "
            f"{null_district_codes} row(s) with a null '{COL_DISTRICT_LGD_CODE}'. "
            "This column is part of the merge key and must not be null."
        )

    if DEBUG:
        print(f"[VALIDATION] '{source_name}' (category='{category}') PASSED")
        print(
            f"[VALIDATION]   required columns present : {REQUIRED_AGGREGATED_COLUMNS}"
        )


def check_column_numeric(df: pd.DataFrame, column: str, dataset_name: str) -> None:
    """Verify that *column* in *df* is numeric.

    Raises
    ------
    SchemaValidationError
        If the column is not a numeric dtype.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise SchemaValidationError(
            f"Dataset '{dataset_name}' has a non-numeric '{column}' column "
            f"(dtype={df[column].dtype})."
        )


def check_no_duplicate_keys(
    df: pd.DataFrame, dataset_name: str, keys: List[str] = MERGE_KEYS
) -> None:
    """Verify that *df* has no duplicate rows on *keys*.

    Raises
    ------
    DuplicateKeyError
        If any duplicate key is found.
    """
    duplicate_mask = df.duplicated(subset=keys, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        raise DuplicateKeyError(
            f"Dataset '{dataset_name}' contains {duplicate_count} row(s) "
            f"that duplicate a ({', '.join(keys)}) key. Exactly one row per "
            "key is required."
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_district_lgd_code(series: pd.Series, dataset_name: str) -> pd.Series:
    """Normalize a ``District LGD Code`` column to nullable ``Int64``.

    Raises
    ------
    SchemaValidationError
        If normalization introduces new nulls that were not already
        present in *series* (i.e. a value could not be coerced to an
        integer).
    """
    original_null_count = int(series.isna().sum())
    coerced = pd.to_numeric(series, errors="coerce").astype("Int64")
    new_null_count = int(coerced.isna().sum())

    if new_null_count > original_null_count:
        raise SchemaValidationError(
            f"Dataset '{dataset_name}': normalizing '{COL_DISTRICT_LGD_CODE}' "
            f"to Int64 introduced {new_null_count - original_null_count} new "
            "null value(s), meaning at least one value could not be "
            "interpreted as an integer."
        )

    return coerced


def normalize_date_column(series: pd.Series, dataset_name: str) -> pd.Series:
    """Normalize a ``Date`` column to Python ``datetime.date`` objects.

    Raises
    ------
    SchemaValidationError
        If normalization introduces new nulls that were not already
        present in *series* (i.e. a value could not be parsed as a date).
    """
    original_null_count = int(series.isna().sum())
    parsed = pd.to_datetime(
        series.astype(str).str.strip(), errors="coerce", format="mixed"
    )
    new_null_count = int(parsed.isna().sum())

    if new_null_count > original_null_count:
        raise SchemaValidationError(
            f"Dataset '{dataset_name}': normalizing '{COL_DATE}' to date "
            f"introduced {new_null_count - original_null_count} new null "
            "value(s), meaning at least one value could not be parsed as "
            "a date."
        )

    return parsed.dt.date


# ---------------------------------------------------------------------------
# Per-category load / prepare
# ---------------------------------------------------------------------------


def load_and_prepare_category_dataset(path: Path, category: str) -> pd.DataFrame:
    """Load one Stage 4.5 aggregated CSV, validate it, and reshape it into
    the form Stage 6 merges.

    Concretely: validates the file exists and matches the expected
    schema, checks for duplicate (District LGD Code, Date) keys and
    non-numeric measurement/station-count columns, normalizes
    ``District LGD Code`` to nullable Int64 and ``Date`` to
    ``datetime.date``, and renames ``Measurement`` and ``Station Count``
    to category-qualified names so that all five categories can later be
    merged without name collisions.

    Parameters
    ----------
    path:
        Path to the aggregated CSV for *category*.
    category:
        Canonical category name, e.g. "rainfall".

    Returns
    -------
    A dataframe with columns
    [District LGD Code, District, Date, <category measurement column>,
    <category station count column>].

    Raises
    ------
    AggregatedFileNotFoundError
        If the file does not exist.
    SchemaValidationError
        If schema, dtype, or normalization validation fails.
    DuplicateKeyError
        If the file contains duplicate (District LGD Code, Date) keys.
    """
    start_time = time.perf_counter()
    dataset_name = f"{category}_daily_district"

    if DEBUG:
        print("\n" + DEBUG_SEPARATOR)
        print(f"Loading category: {category}")
        print(DEBUG_SEPARATOR)

    validate_file_exists(path, category)

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]

    if DEBUG:
        print(f"[LOAD] {path.name}")
        print(f"[ROWS] {len(df):,}")

    validate_aggregated_schema(df, category, dataset_name)
    check_no_duplicate_keys(df, dataset_name, keys=MERGE_KEYS)
    check_column_numeric(df, COL_MEASUREMENT, dataset_name)
    check_column_numeric(df, COL_STATION_COUNT, dataset_name)

    prepared = df.copy()
    prepared[COL_DISTRICT_LGD_CODE] = normalize_district_lgd_code(
        prepared[COL_DISTRICT_LGD_CODE], dataset_name
    )
    prepared[COL_DATE] = normalize_date_column(prepared[COL_DATE], dataset_name)

    measurement_target = CATEGORY_MEASUREMENT_RENAME[category]
    station_count_target = CATEGORY_STATION_COUNT_RENAME[category]

    prepared = prepared.rename(
        columns={
            COL_MEASUREMENT: measurement_target,
            COL_STATION_COUNT: station_count_target,
        }
    )
    prepared = prepared[
        [
            COL_DISTRICT_LGD_CODE,
            COL_DISTRICT,
            COL_DATE,
            measurement_target,
            station_count_target,
        ]
    ]

    execution_time_seconds = time.perf_counter() - start_time

    if DEBUG:
        print(f"[RENAME] {COL_MEASUREMENT!r} -> {measurement_target!r}")
        print(f"[RENAME] {COL_STATION_COUNT!r} -> {station_count_target!r}")
        print(f"[NORMALIZE] {COL_DISTRICT_LGD_CODE!r} -> Int64")
        print(f"[NORMALIZE] {COL_DATE!r} -> datetime.date")
        print(f"[PREPARED ROWS] {len(prepared):,}")
        print(f"[EXECUTION TIME] {execution_time_seconds:.3f}s")
        print(DEBUG_SEPARATOR)
    else:
        logger.info(
            "Prepared %s: %s rows (%.3fs)",
            dataset_name,
            f"{len(prepared):,}",
            execution_time_seconds,
        )

    return prepared


# ---------------------------------------------------------------------------
# District coalescing
# ---------------------------------------------------------------------------


def coalesce_district_column(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicated ``District`` columns produced by a merge into a
    single ``District`` column.

    When two frames that each carry a ``District`` column are merged on
    keys that do not include ``District`` itself, pandas suffixes the
    colliding column as ``District_x`` / ``District_y``. This function
    detects that pattern and coalesces the two into one ``District``
    column, preferring the left frame's (``_x``) value and falling back
    to the right frame's (``_y``) value wherever the left is null. If no
    suffixed pair is present, *df* is returned unchanged.

    Parameters
    ----------
    df:
        A dataframe immediately after a pairwise outer merge.

    Returns
    -------
    A dataframe with exactly one ``District`` column.
    """
    left_col = f"{COL_DISTRICT}_x"
    right_col = f"{COL_DISTRICT}_y"

    if left_col not in df.columns or right_col not in df.columns:
        return df

    coalesced = df.copy()
    coalesced[COL_DISTRICT] = coalesced[left_col].combine_first(coalesced[right_col])
    coalesced = coalesced.drop(columns=[left_col, right_col])

    # Restore District next to the merge keys for readability.
    ordered_columns = [COL_DISTRICT_LGD_CODE, COL_DATE, COL_DISTRICT] + [
        column
        for column in coalesced.columns
        if column not in {COL_DISTRICT_LGD_CODE, COL_DATE, COL_DISTRICT}
    ]
    return coalesced[ordered_columns]


# ---------------------------------------------------------------------------
# Merge statistics
# ---------------------------------------------------------------------------


def compute_null_statistics(df: pd.DataFrame) -> Dict[str, int]:
    """Return a mapping of column name -> null count for *df*."""
    return {column: int(df[column].isna().sum()) for column in df.columns}


def merge_two_category_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Outer-merge two prepared category datasets on (District LGD Code,
    Date) only, coalesce any duplicated District column, validate the
    result, and return merge statistics.

    Parameters
    ----------
    left_df, right_df:
        Prepared category dataframes (as returned by
        load_and_prepare_category_dataset, or the accumulated result of
        prior merge steps).
    left_name, right_name:
        Human-readable names for *left_df* and *right_df*, used in logs,
        debug output, and the returned step statistics.

    Returns
    -------
    (merged_df, step_stats) where step_stats records rows before/after,
    matched/left-only/right-only row counts, and post-merge null
    statistics.

    Raises
    ------
    DuplicateKeyError
        If the merged result contains duplicate (District LGD Code,
        Date) keys.
    """
    start_time = time.perf_counter()

    if DEBUG:
        print("\n" + DEBUG_SEPARATOR)
        print(f"Merging: {left_name}  +  {right_name}")
        print(DEBUG_SEPARATOR)

    rows_before_left = len(left_df)
    rows_before_right = len(right_df)

    merged = pd.merge(
        left_df,
        right_df,
        on=MERGE_KEYS,
        how="outer",
        suffixes=("_x", "_y"),
        indicator=True,
    )

    matched_rows = int((merged["_merge"] == "both").sum())
    left_only_rows = int((merged["_merge"] == "left_only").sum())
    right_only_rows = int((merged["_merge"] == "right_only").sum())
    merged = merged.drop(columns=["_merge"])

    merged = coalesce_district_column(merged)

    step_dataset_name = f"{left_name}+{right_name}"
    check_no_duplicate_keys(merged, step_dataset_name, keys=MERGE_KEYS)

    null_statistics = compute_null_statistics(merged)
    rows_after = len(merged)
    execution_time_seconds = time.perf_counter() - start_time

    step_stats: Dict[str, Any] = {
        "left_dataset": left_name,
        "right_dataset": right_name,
        "merge_keys": MERGE_KEYS,
        "merge_strategy": "outer",
        "rows_before_left": rows_before_left,
        "rows_before_right": rows_before_right,
        "rows_after": rows_after,
        "matched_rows": matched_rows,
        "left_only_rows": left_only_rows,
        "right_only_rows": right_only_rows,
        "null_statistics": null_statistics,
        "execution_time_seconds": execution_time_seconds,
    }

    if DEBUG:
        _print_debug_merge_step(step_stats)
    else:
        logger.info(
            "Merged %s + %s -> %s rows (matched=%s left_only=%s right_only=%s, %.3fs)",
            left_name,
            right_name,
            f"{rows_after:,}",
            f"{matched_rows:,}",
            f"{left_only_rows:,}",
            f"{right_only_rows:,}",
            execution_time_seconds,
        )

    return merged, step_stats


def merge_all_category_datasets(
    category_frames: Dict[str, pd.DataFrame],
    merge_order: List[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Sequentially outer-merge every prepared category dataset in
    *category_frames* on (District LGD Code, Date).

    Datasets are merged one at a time, in *merge_order* (defaulting to
    ``CATEGORY_MERGE_ORDER``), coalescing the District column after each
    step so that at most one pair of suffixed District columns ever
    exists at a time.

    Parameters
    ----------
    category_frames:
        Mapping of category name -> prepared dataframe, as returned by
        load_and_prepare_category_dataset.
    merge_order:
        Optional explicit merge order. Defaults to
        ``CATEGORY_MERGE_ORDER`` filtered to the categories actually
        present in *category_frames*.

    Returns
    -------
    (merged_df, merge_summary) where merged_df is sorted by
    (District LGD Code, Date), and merge_summary records the merge keys,
    strategy, category order, and every pairwise step's statistics.

    Raises
    ------
    MergeValidationError
        If fewer than two category datasets are provided.
    """
    if merge_order is None:
        merge_order = [
            category for category in CATEGORY_MERGE_ORDER if category in category_frames
        ]

    missing = [category for category in merge_order if category not in category_frames]
    if missing:
        raise MergeValidationError(
            f"merge_order references categories not present in "
            f"category_frames: {missing}"
        )

    if len(merge_order) < 2:
        raise MergeValidationError(
            f"At least two category datasets are required to merge; got "
            f"{len(merge_order)} ({merge_order})."
        )

    if DEBUG:
        print("\n" + DEBUG_SEPARATOR)
        print("MERGE PLAN")
        print(DEBUG_SEPARATOR)
        print(f"Merge keys    : {MERGE_KEYS}")
        print(f"Merge strategy: outer")
        print(f"Merge order   : {merge_order}")
        print(DEBUG_SEPARATOR)

    steps: List[Dict[str, Any]] = []

    merged_name = merge_order[0]
    merged = category_frames[merged_name]

    for next_category in merge_order[1:]:
        merged, step = merge_two_category_datasets(
            merged, category_frames[next_category], merged_name, next_category
        )
        steps.append(step)
        merged_name = f"{merged_name}+{next_category}"

    merged = merged.sort_values(by=MERGE_KEYS, kind="mergesort").reset_index(drop=True)

    merge_summary: Dict[str, Any] = {
        "merge_keys": MERGE_KEYS,
        "merge_strategy": "outer",
        "categories_merged": merge_order,
        "steps": steps,
        "final_rows": int(len(merged)),
        "final_columns": int(len(merged.columns)),
    }

    if DEBUG:
        print("\n" + DEBUG_SEPARATOR)
        print(f"MERGE COMPLETE -- final shape: {merged.shape}")
        print(DEBUG_SEPARATOR)

    return merged, merge_summary


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------


def run_final_validation(
    df: pd.DataFrame, dataset_name: str = "master_dataset"
) -> Dict[str, Any]:
    """Run every post-merge validation check on the finished master
    dataset and return a report of its final shape, schema, dtypes, and
    missing-value statistics.

    Checks that raise:
      * no duplicate (District LGD Code, Date) keys
      * every expected column is present
      * District LGD Code is Int64
      * every measurement and station-count column is numeric

    Missing values are expected (an outer join across categories with
    different coverage will produce them) and are therefore reported,
    not treated as a failure.

    Raises
    ------
    DuplicateKeyError
        If the merge key is duplicated.
    SchemaValidationError
        If the schema or dtypes are wrong.
    """
    if DEBUG:
        print("\n" + DEBUG_SEPARATOR)
        print("FINAL VALIDATION")
        print(DEBUG_SEPARATOR)

    check_no_duplicate_keys(df, dataset_name, keys=MERGE_KEYS)

    missing_columns = [
        column for column in EXPECTED_MASTER_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise SchemaValidationError(
            f"'{dataset_name}' is missing expected column(s): {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    unexpected_district_columns = [
        column
        for column in df.columns
        if column in {f"{COL_DISTRICT}_x", f"{COL_DISTRICT}_y"}
    ]
    if unexpected_district_columns:
        raise SchemaValidationError(
            f"'{dataset_name}' still contains uncoalesced District column(s): "
            f"{unexpected_district_columns}. Every merge step must coalesce "
            f"District into a single '{COL_DISTRICT}' column."
        )

    if str(df[COL_DISTRICT_LGD_CODE].dtype) != "Int64":
        raise SchemaValidationError(
            f"'{dataset_name}' has '{COL_DISTRICT_LGD_CODE}' dtype "
            f"{df[COL_DISTRICT_LGD_CODE].dtype}, expected nullable Int64."
        )

    for column in MEASUREMENT_COLUMNS + STATION_COUNT_COLUMNS:
        check_column_numeric(df, column, dataset_name)

    missing_value_statistics = compute_null_statistics(df)

    report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_value_statistics": missing_value_statistics,
        "duplicate_keys": 0,
    }

    if DEBUG:
        _print_debug_final_validation(report)
    else:
        logger.info(
            "Final validation PASSED for '%s': %s rows, %s columns",
            dataset_name,
            f"{report['total_rows']:,}",
            report["total_columns"],
        )

    return report


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------


def write_json_report(report: Dict[str, Any], output_path: Path) -> None:
    """Write *report* to *output_path* as pretty-printed JSON.

    Creates parent directories if they do not already exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    if DEBUG:
        print(f"[REPORT] saved -> {output_path}")
    else:
        logger.info("Saved report: %s", output_path)


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------


def _print_debug_merge_step(step_stats: Dict[str, Any]) -> None:
    """Print the standardized debug block for one pairwise merge step."""
    print(DEBUG_SEPARATOR)
    print("MERGE STEP")
    print(DEBUG_SEPARATOR)
    print(f"Left Dataset        : {step_stats['left_dataset']}")
    print(f"Right Dataset       : {step_stats['right_dataset']}")
    print(f"Merge Keys          : {step_stats['merge_keys']}")
    print(f"Merge Strategy      : {step_stats['merge_strategy']}")
    print(f"Rows Before (Left)  : {step_stats['rows_before_left']:,}")
    print(f"Rows Before (Right) : {step_stats['rows_before_right']:,}")
    print(f"Rows After          : {step_stats['rows_after']:,}")
    print(f"Matched Rows        : {step_stats['matched_rows']:,}")
    print(f"Left Only Rows      : {step_stats['left_only_rows']:,}")
    print(f"Right Only Rows     : {step_stats['right_only_rows']:,}")
    print(f"Null Statistics     :")
    for column, null_count in step_stats["null_statistics"].items():
        print(f"    {column:<35}: {null_count:,}")
    print(f"Execution Time      : {step_stats['execution_time_seconds']:.3f}s")
    print(DEBUG_SEPARATOR)


def _print_debug_final_validation(report: Dict[str, Any]) -> None:
    """Print the standardized debug block for the final validation report."""
    print(DEBUG_SEPARATOR)
    print("FINAL VALIDATION RESULT")
    print(DEBUG_SEPARATOR)
    print(f"Dataset Name        : {report['dataset_name']}")
    print(f"Total Rows          : {report['total_rows']:,}")
    print(f"Total Columns       : {report['total_columns']}")
    print(f"Duplicate Keys      : {report['duplicate_keys']}")
    print(f"Columns             : {report['columns']}")
    print(f"Dtypes              :")
    for column, dtype in report["dtypes"].items():
        print(f"    {column:<35}: {dtype}")
    print(f"Missing Value Stats :")
    for column, null_count in report["missing_value_statistics"].items():
        print(f"    {column:<35}: {null_count:,}")
    print("[VALIDATION] PASSED all final checks")
    print(f"[VALIDATION]   no duplicate (District LGD Code, Date) keys")
    print(f"[VALIDATION]   expected schema present")
    print(f"[VALIDATION]   District LGD Code is nullable Int64")
    print(f"[VALIDATION]   all measurement/station-count columns numeric")
    print(DEBUG_SEPARATOR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_master_dataset(
    aggregated_directory: Path,
    processed_directory: Path,
    reports_directory: Path,
) -> pd.DataFrame:
    """
    Build the Stage 6 master dataset.

    Workflow
    --------
    1. Load all aggregated datasets.
    2. Validate every dataset.
    3. Rename Measurement / Station Count columns.
    4. Merge all datasets.
    5. Run final validation.
    6. Save CSV + Parquet.
    7. Write merge reports.
    8. Return the finished dataframe.

    Parameters
    ----------
    aggregated_directory
        data/aggregated

    processed_directory
        data/processed

    reports_directory
        reports/master_dataset

    Returns
    -------
    pd.DataFrame
        Final merged master dataset.
    """

    overall_start = time.perf_counter()

    if DEBUG:
        print("\n" + DEBUG_SEPARATOR)
        print("STAGE 6 - MASTER DATASET BUILD")
        print(DEBUG_SEPARATOR)

    category_frames: Dict[str, pd.DataFrame] = {}

    # ----------------------------------------------------
    # Load every aggregated dataset
    # ----------------------------------------------------

    for category in CATEGORY_MERGE_ORDER:

        filename = CATEGORY_INPUT_FILENAMES[category]

        dataset_path = aggregated_directory / filename

        category_frames[category] = load_and_prepare_category_dataset(
            dataset_path,
            category,
        )

    # ----------------------------------------------------
    # Merge
    # ----------------------------------------------------

    master_df, merge_statistics = merge_all_category_datasets(category_frames)

    # ----------------------------------------------------
    # Final validation
    # ----------------------------------------------------

    summary_report = run_final_validation(master_df)

    # ----------------------------------------------------
    # Create output folders
    # ----------------------------------------------------

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------
    # Save outputs
    # ----------------------------------------------------

    csv_output = processed_directory / "master_dataset.csv"

    parquet_output = processed_directory / "master_dataset.parquet"

    if DEBUG:
        print(f"[SAVE] CSV      -> {csv_output}")

    master_df.to_csv(
        csv_output,
        index=False,
    )

    if DEBUG:
        print(f"[SAVE] Parquet  -> {parquet_output}")

    try:
        master_df.to_parquet(
            parquet_output,
            index=False,
        )
    except ImportError as exc:
        raise ImportError(
            "Saving Parquet requires 'pyarrow' or 'fastparquet'. "
            "Install one of them before running Stage 6."
        ) from exc

    # ----------------------------------------------------
    # Reports
    # ----------------------------------------------------

    merge_report_path = reports_directory / "merge_statistics.json"

    summary_report_path = reports_directory / "master_dataset_summary.json"

    write_json_report(
        merge_statistics,
        merge_report_path,
    )

    write_json_report(
        summary_report,
        summary_report_path,
    )

    elapsed = time.perf_counter() - overall_start

    if DEBUG:
        print(DEBUG_SEPARATOR)
        print("MASTER DATASET BUILD COMPLETE")
        print(DEBUG_SEPARATOR)
        print(f"Rows        : {len(master_df):,}")
        print(f"Columns     : {len(master_df.columns)}")
        print(f"CSV         : {csv_output}")
        print(f"Parquet     : {parquet_output}")
        print(f"Reports     : {reports_directory}")
        print(f"Time        : {elapsed:.2f} seconds")
        print(DEBUG_SEPARATOR)
    else:
        logger.info(
            "Stage 6 completed successfully in %.2f seconds.",
            elapsed,
        )

    return master_df


__all__ = [
    "build_master_dataset",
]
