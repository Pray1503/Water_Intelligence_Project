"""
Stage 5: Dataset Audit & Quality Reporting
==========================================
Water Intelligence Platform

Reads every cleaned CSV discovered under data/cleaned/, computes a
comprehensive set of quality metrics for each one, writes a per-dataset
JSON report into reports/dataset_audit/, and produces a consolidated
audit_summary.csv.

All configuration paths are read from config/config.yaml.
No dataset is ever modified — this stage is strictly read-only.

Output artefacts
----------------
  reports/dataset_audit/<dataset_name>_report.json   (one per dataset)
  reports/dataset_audit/audit_summary.csv            (consolidated view)
"""

import logging
import sys
from pathlib import Path

# Remove the script directory from sys.path to prevent shadowing standard
# libraries (e.g. inspect.py, csv.py) — identical guard used in Stages 1-4.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging — same format as Stages 1-4: plain message only.
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


# ===========================================================================
# DATASET DISCOVERY
# ===========================================================================


def discover_cleaned_csvs(cleaned_dir: Path) -> List[Path]:
    """
    Recursively scan *cleaned_dir* and return a sorted list of all CSV files.

    Sorting ensures deterministic processing order regardless of file system.
    """
    if not cleaned_dir.exists():
        logger.warning(f"Cleaned data directory does not exist: {cleaned_dir}")
        return []
    paths = sorted(cleaned_dir.rglob("*.csv"))
    logger.info(f"Found {len(paths)} cleaned CSV(s) in {cleaned_dir}")
    return paths


# ===========================================================================
# COLUMN DETECTION HELPERS
# ===========================================================================


def _find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """
    Return the first column whose lower-cased name contains any of the
    supplied *keywords*.  Returns None if no match is found.
    """
    for col in df.columns:
        cl = col.lower()
        if any(kw in cl for kw in keywords):
            return col
    return None


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Locate a timestamp / date column by keyword matching."""
    return _find_column(df, ["time", "date", "timestamp", "acquisition"])


def _find_latitude_column(df: pd.DataFrame) -> Optional[str]:
    """Locate a latitude column by keyword matching."""
    return _find_column(df, ["latitude", "lat"])


def _find_longitude_column(df: pd.DataFrame) -> Optional[str]:
    """Locate a longitude column by keyword matching."""
    return _find_column(df, ["longitude", "lon"])


def _find_station_column(df: pd.DataFrame) -> Optional[str]:
    """Locate a station identifier column by keyword matching."""
    return _find_column(df, ["station", "site", "sensor"])


def _find_district_column(df: pd.DataFrame) -> Optional[str]:
    """Locate a district / administrative region column by keyword matching."""
    # Prefer an exact column name of 'District' if present, case-insensitive.
    for col in df.columns:
        if col.strip().lower() == "district":
            return col
    return _find_column(df, ["district", "taluka", "taluk", "division"])


# ===========================================================================
# METRIC COMPUTATION
# ===========================================================================


def _compute_timestamp_range(
    df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (earliest_timestamp, latest_timestamp) as ISO-8601 strings, or
    (None, None) if no timestamp column is present or it holds no valid values.
    """
    col = _find_timestamp_column(df)
    if col is None:
        return None, None

    # Attempt coercion in case the column was not already parsed as datetime.
    ts_series = pd.to_datetime(df[col], errors="coerce")
    ts_series = ts_series.dropna()
    if ts_series.empty:
        return None, None

    return ts_series.min().isoformat(), ts_series.max().isoformat()


def _compute_lat_range(
    df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (lat_min, lat_max) or (None, None) if no latitude column exists."""
    col = _find_latitude_column(df)
    if col is None:
        return None, None

    numeric = pd.to_numeric(df[col], errors="coerce").dropna()
    if numeric.empty:
        return None, None
    return float(numeric.min()), float(numeric.max())


def _compute_lon_range(
    df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (lon_min, lon_max) or (None, None) if no longitude column exists."""
    col = _find_longitude_column(df)
    if col is None:
        return None, None

    numeric = pd.to_numeric(df[col], errors="coerce").dropna()
    if numeric.empty:
        return None, None
    return float(numeric.min()), float(numeric.max())


def _count_unique(df: pd.DataFrame, col_finder) -> Optional[int]:
    """
    Call *col_finder(df)* to locate a column; return the number of unique
    non-null values in that column, or None if the column is absent.
    """
    col = col_finder(df)
    if col is None:
        return None
    return int(df[col].dropna().nunique())


def _district_list(df: pd.DataFrame) -> Optional[List[str]]:
    """
    Return a sorted, deduplicated list of district names found in the district
    column, or None if no district column is present.
    """
    col = _find_district_column(df)
    if col is None:
        return None
    values = df[col].dropna().astype(str).str.strip().unique().tolist()
    return sorted(values)


def _ahmedabad_present(districts: Optional[List[str]]) -> Optional[bool]:
    """
    Return True if 'Ahmedabad' appears in *districts* (case-insensitive),
    False if districts were found but Ahmedabad is absent,
    None if no district column existed.
    """
    if districts is None:
        return None
    return any("ahmedabad" in d.lower() for d in districts)


def _missing_values_per_column(df: pd.DataFrame) -> Dict[str, int]:
    """Return a dict mapping each column name to its count of missing values."""
    return {col: int(df[col].isna().sum()) for col in df.columns}


def _memory_usage_bytes(df: pd.DataFrame) -> int:
    """Return total in-memory size of the DataFrame in bytes (deep count)."""
    return int(df.memory_usage(deep=True).sum())


def _data_types_map(df: pd.DataFrame) -> Dict[str, str]:
    """Return a dict mapping column name to its pandas dtype string."""
    return {col: str(df[col].dtype) for col in df.columns}


# ===========================================================================
# STATUS DETERMINATION
# ===========================================================================

# Thresholds used to decide PASS / WARNING / FAIL.
_WARN_MISSING_RATIO = 0.20  # >20 % of values missing triggers WARNING
_FAIL_MISSING_RATIO = 0.50  # >50 % of values missing triggers FAIL
_WARN_DUPLICATE_RATIO = 0.05  # >5 % duplicate rows triggers WARNING
_FAIL_DUPLICATE_RATIO = 0.20  # >20 % duplicate rows triggers FAIL


def determine_status(
    rows: int,
    total_cells: int,
    total_missing: int,
    duplicate_rows: int,
) -> str:
    """
    Assign an overall audit status of PASS, WARNING, or FAIL.

    Rules (evaluated in order; first match wins):
      FAIL    — dataset has zero rows, OR >= 50 % of all cells are missing,
                OR >= 20 % of rows are duplicates.
      WARNING — >= 20 % of all cells are missing, OR >= 5 % of rows are duplicates.
      PASS    — none of the above conditions.
    """
    if rows == 0:
        return "FAIL"

    missing_ratio = total_missing / total_cells if total_cells > 0 else 0.0
    dup_ratio = duplicate_rows / rows if rows > 0 else 0.0

    if missing_ratio >= _FAIL_MISSING_RATIO or dup_ratio >= _FAIL_DUPLICATE_RATIO:
        return "FAIL"
    if missing_ratio >= _WARN_MISSING_RATIO or dup_ratio >= _WARN_DUPLICATE_RATIO:
        return "WARNING"
    return "PASS"


# ===========================================================================
# REPORT ASSEMBLY
# ===========================================================================


def build_audit_report(
    csv_path: Path,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute all quality metrics for *df* and return the standardised audit
    report dictionary.  This function is deterministic: given the same
    DataFrame and file path it will always produce identical output.

    Parameters
    ----------
    csv_path : Path
        Path to the cleaned CSV file (used for name and size).
    df : pd.DataFrame
        The dataset loaded from *csv_path*.

    Returns
    -------
    Dict[str, Any]
        Audit report dictionary ready for JSON serialisation.
    """
    dataset_name = csv_path.stem
    file_size_bytes = csv_path.stat().st_size

    rows, cols = df.shape
    column_names: List[str] = df.columns.tolist()
    dtypes_map = _data_types_map(df)
    memory_bytes = _memory_usage_bytes(df)

    missing_per_col = _missing_values_per_column(df)
    total_missing = sum(missing_per_col.values())
    total_cells = rows * cols

    duplicate_rows = int(df.duplicated().sum())

    earliest_ts, latest_ts = _compute_timestamp_range(df)
    lat_min, lat_max = _compute_lat_range(df)
    lon_min, lon_max = _compute_lon_range(df)

    unique_stations = _count_unique(df, _find_station_column)
    districts = _district_list(df)
    num_districts = len(districts) if districts is not None else None
    ahmedabad_present = _ahmedabad_present(districts)

    status = determine_status(rows, total_cells, total_missing, duplicate_rows)

    return {
        "Dataset Name": dataset_name,
        "File Size (bytes)": file_size_bytes,
        "Rows": rows,
        "Columns": cols,
        "Column Names": column_names,
        "Data Types": dtypes_map,
        "Memory Usage (bytes)": memory_bytes,
        "Missing Values Per Column": missing_per_col,
        "Duplicate Rows": duplicate_rows,
        "Earliest Timestamp": earliest_ts,
        "Latest Timestamp": latest_ts,
        "Latitude Min": lat_min,
        "Latitude Max": lat_max,
        "Longitude Min": lon_min,
        "Longitude Max": lon_max,
        "Unique Stations": unique_stations,
        "Number of Districts": num_districts,
        "Ahmedabad District Present": ahmedabad_present,
        "Audit Status": status,
        "Audit Timestamp": datetime.now().isoformat(),
    }


# ===========================================================================
# PERSISTENCE
# ===========================================================================


def save_audit_report(
    report: Dict[str, Any],
    audit_dir: Path,
    dataset_name: str,
) -> Path:
    """
    Serialise *report* as pretty-printed JSON into *audit_dir*.
    Output filename: <dataset_name>_report.json.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_dir / f"{dataset_name}_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)
    return out_path


def save_audit_summary(
    summary_rows: List[Dict[str, Any]],
    audit_dir: Path,
) -> Path:
    """
    Write *summary_rows* as a UTF-8 CSV to *audit_dir/audit_summary.csv*.

    Each row is a flattened, scalar view of one dataset's audit report
    suitable for quick spreadsheet inspection.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_dir / "audit_summary.csv"
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def _flatten_report_for_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an audit report dict into a single flat row suitable for the
    summary CSV.  Nested structures (lists, dicts) are omitted or summarised.
    """
    missing_per_col: Dict[str, int] = report.get("Missing Values Per Column", {})
    total_missing = sum(missing_per_col.values())

    return {
        "Dataset Name": report.get("Dataset Name"),
        "File Size (bytes)": report.get("File Size (bytes)"),
        "Rows": report.get("Rows"),
        "Columns": report.get("Columns"),
        "Memory Usage (bytes)": report.get("Memory Usage (bytes)"),
        "Total Missing Values": total_missing,
        "Duplicate Rows": report.get("Duplicate Rows"),
        "Earliest Timestamp": report.get("Earliest Timestamp"),
        "Latest Timestamp": report.get("Latest Timestamp"),
        "Latitude Min": report.get("Latitude Min"),
        "Latitude Max": report.get("Latitude Max"),
        "Longitude Min": report.get("Longitude Min"),
        "Longitude Max": report.get("Longitude Max"),
        "Unique Stations": report.get("Unique Stations"),
        "Number of Districts": report.get("Number of Districts"),
        "Ahmedabad District Present": report.get("Ahmedabad District Present"),
        "Audit Status": report.get("Audit Status"),
        "Audit Timestamp": report.get("Audit Timestamp"),
    }


# ===========================================================================
# PER-DATASET ORCHESTRATION
# ===========================================================================


def audit_dataset(
    csv_path: Path,
    audit_dir: Path,
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """
    Run the full audit pipeline for a single cleaned CSV.

    Returns the audit report dict on success, or None if the dataset could
    not be loaded (the caller continues with the next dataset regardless).

    This function never modifies the source file.

    Steps
    -----
    1. Load the cleaned CSV (chunked for large files).
    2. Compute all quality metrics.
    3. Determine overall status (PASS / WARNING / FAIL).
    4. Persist the per-dataset JSON report.
    """
    dataset_name = csv_path.stem
    logger.info(f"{'=' * 60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Path   : {csv_path.relative_to(project_root).as_posix()}")

    # ------------------------------------------------------------------
    # 1. Load dataset (chunked to handle large files gracefully).
    #    na_values / keep_default_na are intentionally left as defaults so
    #    we audit what was actually saved by Stage 4 — no extra parsing.
    # ------------------------------------------------------------------
    logger.info("Loading dataset...")
    try:
        chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(
            csv_path,
            chunksize=200_000,
            low_memory=False,
        ):
            chunks.append(chunk)

        if not chunks:
            logger.warning(f"Skipping {dataset_name}: file is empty.")
            return None

        df: pd.DataFrame = pd.concat(chunks, ignore_index=True)
    except Exception as exc:
        logger.error(f"Failed to load {dataset_name}: {exc}")
        return None

    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns.")

    # ------------------------------------------------------------------
    # 2. Compute metrics and assemble report.
    # ------------------------------------------------------------------
    logger.info("Computing quality metrics...")
    try:
        report = build_audit_report(csv_path, df)
    except Exception as exc:
        logger.error(f"Failed to compute metrics for {dataset_name}: {exc}")
        return None

    # Log a concise summary of key findings.
    total_missing = sum(report["Missing Values Per Column"].values())
    logger.info(f"  Rows            : {report['Rows']:,}")
    logger.info(f"  Columns         : {report['Columns']}")
    logger.info(f"  Missing values  : {total_missing:,}")
    logger.info(f"  Duplicate rows  : {report['Duplicate Rows']:,}")
    logger.info(f"  Earliest ts     : {report['Earliest Timestamp']}")
    logger.info(f"  Latest ts       : {report['Latest Timestamp']}")
    logger.info(f"  Unique stations : {report['Unique Stations']}")
    logger.info(f"  Districts       : {report['Number of Districts']}")
    logger.info(f"  Ahmedabad       : {report['Ahmedabad District Present']}")
    logger.info(f"  Audit Status    : {report['Audit Status']}")

    # ------------------------------------------------------------------
    # 3. Persist per-dataset JSON report.
    # ------------------------------------------------------------------
    logger.info("Saving audit report...")
    try:
        out_json = save_audit_report(report, audit_dir, dataset_name)
        logger.info(f"  Saved: {out_json.relative_to(project_root).as_posix()}")
    except Exception as exc:
        logger.error(f"Failed to save audit report for {dataset_name}: {exc}")
        return None

    return report


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================


def audit_datasets(config: Dict[str, Any], project_root: Path) -> None:
    """
    Top-level orchestrator for Stage 5.

    Reads paths from *config*, auto-discovers all cleaned CSVs, audits each
    one, and writes per-dataset JSON reports and a consolidated summary CSV.
    A failure in one dataset never aborts the loop — all exceptions are caught
    and logged, then processing continues with the next dataset.
    """
    paths: Dict[str, str] = config.get("paths", {})

    cleaned_dir = project_root / paths.get("data_cleaned", "data/cleaned")
    audit_dir = project_root / paths.get("audit_dir", "reports/dataset_audit")

    logger.info("=" * 60)
    logger.info("Stage 5: Dataset Audit & Quality Reporting")
    logger.info("=" * 60)
    logger.info(f"Cleaned data directory : {cleaned_dir}")
    logger.info(f"Audit reports directory: {audit_dir}")

    # Ensure output directory exists before we start (partial failures still
    # produce whatever outputs they can).
    audit_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = discover_cleaned_csvs(cleaned_dir)
    if not csv_paths:
        logger.warning("No cleaned CSVs found. Run Stage 4 first.")
        return

    total = len(csv_paths)
    success_count = 0
    failure_count = 0
    summary_rows: List[Dict[str, Any]] = []

    for idx, csv_path in enumerate(csv_paths, start=1):
        logger.info(f"\n[{idx}/{total}]")
        try:
            report = audit_dataset(csv_path, audit_dir, project_root)
            if report is not None:
                summary_rows.append(_flatten_report_for_summary(report))
                success_count += 1
            else:
                failure_count += 1
        except Exception as exc:
            # Belt-and-suspenders: audit_dataset already catches errors
            # internally, but this guard prevents any unexpected exception from
            # breaking the outer loop.
            logger.error(f"Unexpected error while auditing {csv_path.name}: {exc}")
            failure_count += 1

    # ------------------------------------------------------------------
    # Write consolidated summary CSV (sorted deterministically by name).
    # ------------------------------------------------------------------
    if summary_rows:
        summary_rows.sort(key=lambda r: str(r.get("Dataset Name", "")))
        logger.info("\nSaving audit summary CSV...")
        try:
            summary_path = save_audit_summary(summary_rows, audit_dir)
            logger.info(f"  Saved: {summary_path.relative_to(project_root).as_posix()}")
        except Exception as exc:
            logger.error(f"Failed to save audit summary: {exc}")

    logger.info("\n" + "=" * 60)
    logger.info("Stage 5 complete.")
    logger.info(f"  Datasets audited successfully : {success_count}")
    logger.info(f"  Datasets skipped / failed     : {failure_count}")
    logger.info(f"  Audit reports saved to        : {audit_dir}")
    logger.info("=" * 60)


# ===========================================================================
# ENTRY POINT
# ===========================================================================


def main() -> None:
    """Entry point -- resolve project root, load config, run Stage 5."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"

    config = load_config(config_path)
    audit_datasets(config, project_root)


if __name__ == "__main__":
    main()
