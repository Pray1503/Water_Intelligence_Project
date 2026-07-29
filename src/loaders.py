from pathlib import Path
from typing import List, Optional

import pandas as pd


def discover_cleaned_csvs(cleaned_dir: Path) -> List[Path]:
    """Return all cleaned CSV files in a deterministic sorted order."""
    if not cleaned_dir.exists():
        return []
    return sorted(cleaned_dir.rglob("*.csv"))


def load_cleaned_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a cleaned CSV into a DataFrame with trimmed column names."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Return the first column whose name contains any keyword (case-insensitive)."""
    for col in df.columns:
        column_name = str(col).strip().lower()
        if any(keyword in column_name for keyword in keywords):
            return col
    return None


def _find_exact_column(df: pd.DataFrame, target: str) -> Optional[str]:
    """Return an exact column match after stripping whitespace and lowercasing."""
    target_name = target.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == target_name:
            return col
    return None


def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the timestamp column by keyword matching."""
    return _find_exact_column(df, "Data Acquisition Time") or _find_column(
        df, ["time", "date", "timestamp", "acquisition"]
    )


def find_district_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the district column, preferring an exact District match."""
    return _find_exact_column(df, "District") or _find_column(
        df, ["district", "taluka", "taluk", "division"]
    )


def find_station_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the station column, preferring an exact Station match."""
    return _find_exact_column(df, "Station") or _find_column(
        df, ["station", "site", "sensor"]
    )


def find_latitude_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the latitude column, preferring an exact Latitude match."""
    return _find_exact_column(df, "Latitude") or _find_column(df, ["latitude", "lat"])


def find_longitude_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the longitude column, preferring an exact Longitude match."""
    return _find_exact_column(df, "Longitude") or _find_column(df, ["longitude", "lon"])


def standardize_dataset(
    df: pd.DataFrame, dataset_name: str, source_file: str
) -> pd.DataFrame:
    """Add standardized shared columns without removing any existing measurement columns."""
    standardized = df.copy()
    standardized.columns = [str(col).strip() for col in standardized.columns]

    timestamp_col = find_timestamp_column(standardized)
    district_col = find_district_column(standardized)
    station_col = find_station_column(standardized)
    latitude_col = find_latitude_column(standardized)
    longitude_col = find_longitude_column(standardized)

    standardized["timestamp"] = pd.to_datetime(
        (
            standardized[timestamp_col]
            if timestamp_col is not None
            else pd.Series([pd.NaT] * len(standardized))
        ),
        errors="coerce",
        dayfirst=True,
    )
    standardized["district"] = (
        standardized[district_col].astype("string").str.strip()
        if district_col is not None
        else pd.Series([pd.NA] * len(standardized), dtype="string")
    )
    standardized["station"] = (
        standardized[station_col].astype("string").str.strip()
        if station_col is not None
        else pd.Series([pd.NA] * len(standardized), dtype="string")
    )
    standardized["latitude"] = (
        pd.to_numeric(standardized[latitude_col], errors="coerce")
        if latitude_col is not None
        else pd.Series([pd.NA] * len(standardized), dtype="float64")
    )
    standardized["longitude"] = (
        pd.to_numeric(standardized[longitude_col], errors="coerce")
        if longitude_col is not None
        else pd.Series([pd.NA] * len(standardized), dtype="float64")
    )
    standardized["source_dataset"] = dataset_name
    standardized["source_file"] = source_file
    return standardized
