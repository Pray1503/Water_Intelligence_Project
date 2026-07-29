"""
Stage 4: Data Cleaning & Consolidation
=======================================
Water Intelligence Platform

Reads every raw CSV dataset discovered in data/raw/, applies a sequence of
common and dataset-specific cleaning rules, saves cleaned CSVs into data/cleaned/,
and emits one JSON cleaning report per dataset into reports/cleaning/.

All configuration paths are read from config/config.yaml.
"""

import logging
import sys
from pathlib import Path

# Remove the script directory from sys.path to prevent shadowing standard
# libraries (e.g. inspect.py, csv.py) — identical guard used in Stages 1-3.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging — same format as Stages 1-3: plain message only.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# CONFIGURATION
# ===========================================================================


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and return the YAML configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception as exc:
        logger.error(f"Failed to load config from {config_path}: {exc}")
        sys.exit(1)


def load_inventory(inventory_path: Path) -> pd.DataFrame:
    """Load the dataset inventory produced by Stage 1."""
    if not inventory_path.exists():
        logger.error(
            f"Dataset inventory not found at {inventory_path}. "
            "Run Stage 1 (01_discover_datasets.py) first."
        )
        sys.exit(1)
    try:
        df = pd.read_csv(inventory_path)
    except Exception as exc:
        logger.error(f"Failed to read inventory: {exc}")
        sys.exit(1)
    if df.empty:
        logger.error("Dataset inventory is empty — nothing to clean.")
        sys.exit(1)
    return df


# ===========================================================================
# CATEGORY DETECTION  (mirrors the logic already in Stage 1)
# ===========================================================================


def detect_category(filename: str) -> str:
    """Infer the dataset category from the filename (case-insensitive)."""
    name = filename.lower()
    if "gwl" in name or "groundwater" in name:
        return "Groundwater"
    if "humid" in name:
        return "Humidity"
    if "rainfall" in name:
        return "Rainfall"
    if "rwl" in name or "river" in name:
        return "River Level"
    if "temperature" in name or "temp" in name:
        return "Temperature"
    return "Unknown"


# ===========================================================================
# COMMON CLEANING HELPERS
# ===========================================================================

# Sentinel strings that should all map to NaN.
_MISSING_SENTINELS: List[str] = [
    "NA",
    "N/A",
    "NULL",
    "null",
    "na",
    "n/a",
    "None",
    "none",
    "NONE",
    "nan",
    "NaN",
    "--",
    "-",
]


def clean_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Strip leading/trailing whitespace from every column name.

    Returns:
        df: DataFrame with cleaned column names.
        cleaned: List of original column names that were changed.
    """
    cleaned: List[str] = []
    new_cols: List[str] = []
    for col in df.columns:
        stripped = col.strip()
        if stripped != col:
            cleaned.append(col)
        new_cols.append(stripped)
    df.columns = new_cols  # type: ignore[assignment]
    return df, cleaned


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace known sentinel strings and blank / whitespace-only strings with
    NaN across every column.  Operates only on object (string) columns to
    avoid touching already-numeric data.
    """
    for col in df.select_dtypes(include=["object", "string"]).columns:
        # Trim string whitespace first
        df[col] = df[col].astype(str).str.strip()
        # Replace sentinels
        df[col] = df[col].replace(_MISSING_SENTINELS, pd.NA)
        # Blank strings that survived trimming
        df[col] = df[col].replace({"": pd.NA, " ": pd.NA})
        # Restore actual NaN type (replace turns to <NA> with StringDtype)
        df[col] = df[col].where(df[col].notna(), other=float("nan"))
    return df


def convert_numeric_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Attempt pd.to_numeric(errors='coerce') on every column whose dtype is
    object.  Returns the mutated DataFrame and a list of columns that were
    successfully converted.
    """
    converted: List[str] = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        attempted = pd.to_numeric(df[col], errors="coerce")
        # Accept the conversion only when at least 50% of non-null values
        # parse successfully — avoids accidentally nuking text columns.
        original_non_null = df[col].notna().sum()
        converted_non_null = attempted.notna().sum()
        if original_non_null == 0 or (converted_non_null / original_non_null) >= 0.5:
            df[col] = attempted
            converted.append(col)
    return df, converted


def convert_datetime_columns(
    df: pd.DataFrame,
    datetime_hints: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Try pd.to_datetime(errors='coerce') on columns whose names contain
    common datetime keywords, or on any object column supplied in
    *datetime_hints*.

    Returns the mutated DataFrame and a list of columns that were converted.
    """
    _DATETIME_KEYWORDS = ("time", "date", "timestamp", "acquisition")
    converted: List[str] = []
    hints = set(datetime_hints or [])

    for col in df.columns:
        col_lower = col.lower()
        is_hint = col in hints
        is_keyword_match = any(kw in col_lower for kw in _DATETIME_KEYWORDS)
        if not (is_hint or is_keyword_match):
            continue
        # Skip columns that are already parsed as datetime.
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        result = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        df[col] = result
        converted.append(col)

    return df, converted


def remove_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove fully-duplicated rows. Returns cleaned df and count removed."""
    before = len(df)
    df = df.drop_duplicates()
    return df.reset_index(drop=True), before - len(df)


def remove_empty_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove rows where every column is NaN. Returns cleaned df and count."""
    before = len(df)
    df = df.dropna(how="all")
    return df.reset_index(drop=True), before - len(df)


def validate_coordinates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows where Latitude is outside [-90, 90] or
    Longitude is outside [-180, 180].  Columns are located by
    case-insensitive name match.  Returns cleaned df and count removed.
    """
    lat_col: Optional[str] = None
    lon_col: Optional[str] = None

    for col in df.columns:
        cl = col.lower()
        if "latitude" in cl or cl == "lat":
            lat_col = col
        if "longitude" in cl or cl == "lon":
            lon_col = col

    if lat_col is None or lon_col is None:
        return df, 0

    before = len(df)
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    valid_mask = (
        lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)
    )
    df = df[valid_mask].reset_index(drop=True)
    return df, before - len(df)


# ===========================================================================
# DATASET-SPECIFIC CLEANING RULES
# ===========================================================================


def _find_sensor_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Return the first column whose lower-case name contains any keyword."""
    for col in df.columns:
        cl = col.lower()
        if any(kw in cl for kw in keywords):
            return col
    return None


def clean_humidity(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Relative Humidity must be in [0, 100].
    Values outside this range are replaced with NaN.
    Returns the mutated df and the number of values invalidated.
    """
    col = _find_sensor_column(df, ["humidity", "relative humidity", "humid"])
    if col is None:
        return df, 0

    df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_mask = df[col].notna() & ~df[col].between(0, 100)
    count = int(invalid_mask.sum())
    df.loc[invalid_mask, col] = float("nan")
    return df, count


def clean_temperature(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Air temperature must be in [-60, 60] degrees Celsius.
    Values outside this range are replaced with NaN.
    Returns the mutated df and the number of values invalidated.
    """
    col = _find_sensor_column(df, ["temperature", "temp", "aoc"])
    if col is None:
        return df, 0

    df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_mask = df[col].notna() & ~df[col].between(-60, 60)
    count = int(invalid_mask.sum())
    df.loc[invalid_mask, col] = float("nan")
    return df, count


def clean_rainfall(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Rainfall cannot be negative.
    Negative values are replaced with NaN.
    Returns the mutated df and the number of values invalidated.
    """
    col = _find_sensor_column(df, ["rainfall", "rain"])
    if col is None:
        return df, 0

    df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_mask = df[col].notna() & (df[col] < 0)
    count = int(invalid_mask.sum())
    df.loc[invalid_mask, col] = float("nan")
    return df, count


def clean_groundwater(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Groundwater levels below -1000 m or above +200 m are physically impossible
    and are replaced with NaN.
    Returns the mutated df and the number of values invalidated.
    """
    col = _find_sensor_column(df, ["groundwater", "gwl", "ground water"])
    if col is None:
        return df, 0

    df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_mask = df[col].notna() & ~df[col].between(-1000, 200)
    count = int(invalid_mask.sum())
    df.loc[invalid_mask, col] = float("nan")
    return df, count


def clean_river_level(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    River water levels are physically bounded:
        lower: -50 m   (tidal gauges can read slightly below datum)
        upper: 10 000 m (Himalayan headwater absolute ceiling)
    Values outside this range are replaced with NaN.
    Returns the mutated df and the number of values invalidated.
    """
    col = _find_sensor_column(
        df, ["river water level", "rwl", "river level", "water level"]
    )
    if col is None:
        return df, 0

    df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_mask = df[col].notna() & ~df[col].between(-50, 10_000)
    count = int(invalid_mask.sum())
    df.loc[invalid_mask, col] = float("nan")
    return df, count


def apply_dataset_specific_rules(
    df: pd.DataFrame,
    category: str,
) -> Tuple[pd.DataFrame, int]:
    """
    Dispatch to the correct category-specific cleaning function.

    Returns:
        df: Cleaned DataFrame.
        invalid_sensor_count: Number of sensor values replaced with NaN.
    """
    category_lower = category.lower()
    if "humidity" in category_lower:
        return clean_humidity(df)
    if "temperature" in category_lower:
        return clean_temperature(df)
    if "rainfall" in category_lower:
        return clean_rainfall(df)
    if "groundwater" in category_lower:
        return clean_groundwater(df)
    if "river" in category_lower:
        return clean_river_level(df)
    return df, 0


# ===========================================================================
# REPORT & PERSISTENCE
# ===========================================================================


def generate_cleaning_report(
    dataset_name: str,
    dataset_category: str,
    rows_before: int,
    rows_after: int,
    duplicates_removed: int,
    empty_rows_removed: int,
    rows_with_missing: int,
    columns_cleaned: List[str],
    columns_converted: List[str],
    invalid_coordinates_removed: int,
    invalid_sensor_values_removed: int,
    status: str,
) -> Dict[str, Any]:
    """
    Assemble and return the standardised cleaning report dictionary.
    All field names match the spec in the task brief.
    """
    return {
        "Dataset Name": dataset_name,
        "Dataset Category": dataset_category,
        "Rows Before": rows_before,
        "Rows After": rows_after,
        "Duplicate Rows Removed": duplicates_removed,
        "Completely Empty Rows Removed": empty_rows_removed,
        "Rows With Missing Values": rows_with_missing,
        "Rows Removed": rows_before - rows_after,
        "Columns Cleaned": columns_cleaned,
        "Columns Converted": columns_converted,
        "Invalid Coordinates Removed": invalid_coordinates_removed,
        "Invalid Sensor Values Removed": invalid_sensor_values_removed,
        "Cleaning Timestamp": datetime.now().isoformat(),
        "Overall Cleaning Status": status,
    }


def save_cleaned_dataset(
    df: pd.DataFrame,
    cleaned_dir: Path,
    dataset_name: str,
    force: bool = False,
) -> Path:
    """
    Save a cleaned dataset.

    Existing cleaned datasets are preserved unless force=True.

    Returns
    -------
    Path
        Path to the cleaned dataset.
    """

    cleaned_dir.mkdir(parents=True, exist_ok=True)

    out_path = cleaned_dir / f"{dataset_name}_cleaned.csv"

    if out_path.exists() and not force:
        logger.info(f"[SAVE SKIP] Cleaned dataset already exists: {out_path.name}")
        return out_path

    if out_path.exists() and force:
        logger.info(f"[OVERWRITE] Existing cleaned dataset: {out_path.name}")

    df.to_csv(out_path, index=False, encoding="utf-8")

    logger.info(f"[SAVED] {out_path.name}")

    return out_path


def save_cleaning_report(
    report: Dict[str, Any],
    cleaning_reports_dir: Path,
    dataset_name: str,
) -> Path:
    """
    Serialise *report* as pretty-printed JSON into *cleaning_reports_dir*.
    The output filename is <dataset_name>_cleaning_report.json.
    """
    cleaning_reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = cleaning_reports_dir / f"{dataset_name}_cleaning_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)
    return out_path


# ===========================================================================
# PER-DATASET ORCHESTRATION
# ===========================================================================


def process_dataset(
    row: pd.Series,
    cleaned_dir: Path,
    cleaning_reports_dir: Path,
    project_root: Path,
    force: bool = False,
) -> bool:
    """
    Execute the full cleaning pipeline for a single dataset entry from the
    inventory.  Returns True on success, False if the dataset was skipped or
    an unrecoverable error occurred (the caller continues regardless).

    Cleaning sequence
    -----------------
    1.  Load CSV (chunked for large files).
    2.  Strip column-name whitespace.
    3.  Replace sentinel / blank missing values.
    4.  Remove fully-duplicated rows.
    5.  Remove completely-empty rows.
    6.  Convert datetime columns.
    7.  Convert remaining numeric-looking object columns.
    8.  Validate coordinate bounds.
    9.  Apply dataset-specific sensor-value rules.
    10. Preserve original column order.
    11. Save cleaned CSV.
    12. Save cleaning report.
    """
    dataset_name: str = str(row.get("Dataset Name", "unknown"))
    category: str = str(row.get("Dataset Category", "Unknown"))
    abs_path_str: str = str(row.get("Absolute Path", ""))

    # -------------------------------------------------------------
    # Skip datasets that have already been cleaned
    # -------------------------------------------------------------
    out_csv = cleaned_dir / f"{dataset_name}_cleaned.csv"

    if out_csv.exists() and not force:
        logger.info(f"[SKIP] {dataset_name} already cleaned.")
        return True

    logger.info(f"{'=' * 60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Category: {category}")

    if not abs_path_str:
        logger.warning(f"Skipping {dataset_name}: Absolute Path missing in inventory.")
        return False

    abs_path = Path(abs_path_str)
    if not abs_path.exists():
        logger.warning(f"Skipping {dataset_name}: file not found at {abs_path}")
        return False

    # ------------------------------------------------------------------
    # 1. Load dataset (chunked to handle multi-GB files gracefully)
    # ------------------------------------------------------------------
    logger.info("Loading dataset...")
    try:
        chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(
            abs_path,
            chunksize=200_000,
            low_memory=False,
            na_values=_MISSING_SENTINELS,
            keep_default_na=True,
        ):
            chunks.append(chunk)
        if not chunks:
            logger.warning(f"Skipping {dataset_name}: file is empty.")
            return False
        df: pd.DataFrame = pd.concat(chunks, ignore_index=True)
    except Exception as exc:
        logger.error(f"Failed to load {dataset_name}: {exc}")
        return False

    rows_before: int = len(df)
    original_columns: List[str] = df.columns.tolist()
    logger.info(f"Loaded {rows_before:,} rows x {len(df.columns)} columns.")

    # ------------------------------------------------------------------
    # 2. Strip column-name whitespace
    # ------------------------------------------------------------------
    logger.info("Cleaning column names...")
    df, columns_name_cleaned = clean_column_names(df)

    # ------------------------------------------------------------------
    # 3. Replace sentinel / blank missing values
    # ------------------------------------------------------------------
    logger.info("Standardising missing values...")
    df = clean_missing_values(df)

    # ------------------------------------------------------------------
    # 4. Remove duplicates
    # ------------------------------------------------------------------
    logger.info("Removing duplicate rows...")
    df, duplicates_removed = remove_duplicate_rows(df)
    logger.info(f"  Duplicates removed: {duplicates_removed:,}")

    # ------------------------------------------------------------------
    # 5. Remove completely-empty rows
    # ------------------------------------------------------------------
    logger.info("Removing completely empty rows...")
    df, empty_rows_removed = remove_empty_rows(df)
    logger.info(f"  Empty rows removed: {empty_rows_removed:,}")

    # ------------------------------------------------------------------
    # 6. Convert datetime columns
    # ------------------------------------------------------------------
    logger.info("Converting datetime columns...")
    df, datetime_converted = convert_datetime_columns(df)
    logger.info(f"  Datetime columns converted: {datetime_converted}")

    # ------------------------------------------------------------------
    # 7. Convert numeric columns
    # ------------------------------------------------------------------
    logger.info("Converting numeric columns...")
    df, numeric_converted = convert_numeric_columns(df)
    logger.info(f"  Numeric columns converted: {numeric_converted}")

    columns_converted: List[str] = datetime_converted + numeric_converted

    # ------------------------------------------------------------------
    # 8. Validate coordinates
    # ------------------------------------------------------------------
    logger.info("Validating coordinates...")
    df, invalid_coordinates_removed = validate_coordinates(df)
    logger.info(f"  Invalid coordinates removed: {invalid_coordinates_removed:,}")

    # ------------------------------------------------------------------
    # 9. Dataset-specific sensor-value rules
    # ------------------------------------------------------------------
    logger.info("Applying dataset-specific cleaning rules...")
    df, invalid_sensor_values_removed = apply_dataset_specific_rules(df, category)
    logger.info(
        f"  Invalid sensor values replaced with NaN: {invalid_sensor_values_removed:,}"
    )

    # ------------------------------------------------------------------
    # 10. Preserve original column order (names may have been whitespace-stripped)
    # ------------------------------------------------------------------
    stripped_original = [c.strip() for c in original_columns]
    ordered_cols = [c for c in stripped_original if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining]

    # ------------------------------------------------------------------
    # 11. Rows with at least one missing value (informational only)
    # ------------------------------------------------------------------
    rows_with_missing: int = int(df.isnull().any(axis=1).sum())

    rows_after: int = len(df)

    # ------------------------------------------------------------------
    # 12. Determine overall status
    # ------------------------------------------------------------------
    rows_removed = rows_before - rows_after
    if rows_after == 0:
        status = "EMPTY_AFTER_CLEANING"
    elif rows_removed == 0 and invalid_sensor_values_removed == 0:
        status = "CLEAN"
    else:
        status = "CLEANED"

    # ------------------------------------------------------------------
    # 13. Save cleaned CSV
    # ------------------------------------------------------------------
    logger.info("Saving cleaned dataset...")
    try:
        out_csv = save_cleaned_dataset(df, cleaned_dir, dataset_name, force=force)
        logger.info(f"  Saved: {out_csv.relative_to(project_root).as_posix()}")
    except Exception as exc:
        logger.error(f"Failed to save cleaned CSV for {dataset_name}: {exc}")
        return False

    # ------------------------------------------------------------------
    # 14. Build and save cleaning report
    # ------------------------------------------------------------------
    report = generate_cleaning_report(
        dataset_name=dataset_name,
        dataset_category=category,
        rows_before=rows_before,
        rows_after=rows_after,
        duplicates_removed=duplicates_removed,
        empty_rows_removed=empty_rows_removed,
        rows_with_missing=rows_with_missing,
        columns_cleaned=columns_name_cleaned,
        columns_converted=columns_converted,
        invalid_coordinates_removed=invalid_coordinates_removed,
        invalid_sensor_values_removed=invalid_sensor_values_removed,
        status=status,
    )

    logger.info("Saving cleaning report...")
    try:
        out_json = save_cleaning_report(report, cleaning_reports_dir, dataset_name)
        logger.info(f"  Saved: {out_json.relative_to(project_root).as_posix()}")
    except Exception as exc:
        logger.error(f"Failed to save cleaning report for {dataset_name}: {exc}")

    logger.info(
        f"Cleaning completed -- {rows_before:,} -> {rows_after:,} rows "
        f"({rows_removed:,} removed). Status: {status}"
    )
    return True


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================


def clean_datasets(config: Dict[str, Any], project_root: Path) -> None:
    """
    Top-level orchestrator for Stage 4.

    Reads paths from *config*, loads the dataset inventory produced by Stage 1,
    and processes every dataset in turn.  A failure in one dataset never aborts
    the loop — all exceptions are caught and logged, then processing continues
    with the next dataset.
    """
    paths: Dict[str, str] = config.get("paths", {})

    inventory_path = project_root / paths.get(
        "dataset_inventory", "reports/dataset_inventory.csv"
    )
    cleaned_dir = project_root / paths.get("data_cleaned", "data/cleaned")
    cleaning_reports_dir = project_root / paths.get("cleaning_dir", "reports/cleaning")

    logger.info("=" * 60)
    logger.info("Stage 4: Data Cleaning & Consolidation")
    logger.info("=" * 60)

    inventory_df = load_inventory(inventory_path)

    # Create output directories up-front so partial failures still produce
    # whatever outputs they can.
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaning_reports_dir.mkdir(parents=True, exist_ok=True)

    total = len(inventory_df)
    success_count = 0
    failure_count = 0

    for idx, row in inventory_df.iterrows():
        logger.info(f"\n[{int(idx) + 1}/{total}]")  # type: ignore[arg-type]
        try:
            ok = process_dataset(
                row=row,
                cleaned_dir=cleaned_dir,
                cleaning_reports_dir=cleaning_reports_dir,
                project_root=project_root,
            )
            if ok:
                success_count += 1
            else:
                failure_count += 1
        except Exception as exc:
            # Belt-and-suspenders: process_dataset already catches errors
            # internally, but this guard prevents any unexpected exception from
            # breaking the outer loop.
            dataset_name = str(row.get("Dataset Name", "unknown"))
            logger.error(f"Unexpected error while processing {dataset_name}: {exc}")
            failure_count += 1

    logger.info("\n" + "=" * 60)
    logger.info("Stage 4 complete.")
    logger.info(f"  Datasets processed successfully : {success_count}")
    logger.info(f"  Datasets skipped / failed       : {failure_count}")
    logger.info(f"  Cleaned CSVs saved to           : {cleaned_dir}")
    logger.info(f"  Cleaning reports saved to        : {cleaning_reports_dir}")
    logger.info("=" * 60)


# ===========================================================================
# ENTRY POINT
# ===========================================================================


def main() -> None:
    """Entry point — resolve project root, load config, run Stage 4."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"

    config = load_config(config_path)
    clean_datasets(config, project_root)


if __name__ == "__main__":
    main()
