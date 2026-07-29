"""
Stage 6: Build Master Dataset
============================

Builds a reusable master dataset from all cleaned CSV files under
data/cleaned. Standardizes shared key columns (timestamp, district,
station, latitude, longitude), merges datasets under a strict merge-key
policy (timestamp+station, else timestamp+district, else raise -- never
timestamp alone), and writes:

  * data/processed/master_dataset.parquet -- the merged table
  * reports/master_dataset/merge_statistics.json -- full merge statistics

Explicitly out of scope for this stage (see project instructions):
no feature engineering, no labels, no ML, no GeoJSON/ward mapping, no
spatial joins. Those happen in later stages.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import yaml

from src.loaders import discover_cleaned_csvs, load_cleaned_dataset, standardize_dataset
from src.merger import MergeExplosionError, MergeKeyError, merge_datasets

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def select_measurement_columns(df: pd.DataFrame) -> List[str]:
    shared_columns = {
        "timestamp",
        "district",
        "station",
        "latitude",
        "longitude",
        "source_dataset",
        "source_file",
    }
    return [
        col
        for col in df.columns
        if col not in shared_columns and col not in {"Unnamed: 0"}
    ]


def build_master_dataset(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    paths = config.get("paths", {})
    # Keys follow the same naming convention as the rest of the pipeline
    # (data_raw, data_cleaned, ...) so Stage 6 stays in sync if config.yaml
    # changes -- it previously read a differently-named key that only
    # happened to work by matching the hardcoded default.
    cleaned_dir = project_root / paths.get("data_cleaned", "data/cleaned")
    output_dir = project_root / paths.get("data_processed", "data/processed")
    reports_dir = project_root / paths.get("reports_dir", "reports")
    master_dataset_report_dir = reports_dir / "master_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    master_dataset_report_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_cleaned_csvs(cleaned_dir)
    if not csv_files:
        raise FileNotFoundError(f"No cleaned datasets found in {cleaned_dir}")

    logger.info(f"Found {len(csv_files)} cleaned datasets")
    standardized_datasets: List[pd.DataFrame] = []
    dataset_names: List[str] = []

    for csv_path in csv_files:
        dataset_name = csv_path.stem.replace("_cleaned", "")
        logger.info(f"Loading {csv_path.name}")
        df = load_cleaned_dataset(csv_path)
        standardized = standardize_dataset(
            df, dataset_name=dataset_name, source_file=csv_path.name
        )
        standardized_datasets.append(standardized)
        dataset_names.append(dataset_name)

    logger.info("Merging standardized datasets")
    try:
        merged_df, merge_summary = merge_datasets(
            standardized_datasets, dataset_names=dataset_names
        )
    except (MergeKeyError, MergeExplosionError) as exc:
        logger.error(f"Stage 6 merge halted: {exc}")
        raise

    measurement_columns = select_measurement_columns(merged_df)
    logger.info(f"Measurement columns retained: {len(measurement_columns)}")

    output_path = output_dir / "master_dataset.parquet"
    merged_df.to_parquet(output_path, index=False)

    result = {
        "output_path": str(output_path),
        "row_count": int(len(merged_df)),
        "column_count": int(len(merged_df.columns)),
        "measurement_columns": measurement_columns,
        "merge_summary": merge_summary,
    }

    stats_path = master_dataset_report_dir / "merge_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    result["stats_path"] = str(stats_path)

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)

    try:
        result = build_master_dataset(config, project_root)
    except Exception as exc:
        logger.error(f"Failed to build master dataset: {exc}")
        raise

    logger.info("Master dataset built successfully")
    logger.info(f"Output: {result['output_path']}")
    logger.info(f"Rows: {result['row_count']}")
    logger.info(f"Columns: {result['column_count']}")
    logger.info(f"Merge statistics written to: {result['stats_path']}")
    for step in result["merge_summary"].get("steps", []):
        logger.info(
            f"  {step['left_dataset']} + {step['right_dataset']} "
            f"-> keys={step['merge_keys']} strategy={step['merge_strategy']} "
            f"rows={step['rows_after']} matched={step['matched_rows']} "
            f"left_only={step['left_only_rows']} right_only={step['right_only_rows']}"
        )


if __name__ == "__main__":
    main()
